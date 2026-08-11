"""Pure helpers for the ``exclusive_hierarchy_v1`` training target.

The training route is intentionally not imported here.  This module operates on one
image at a time and accepts either binary masks or contour-like objects which can be
rasterised without a database.  Keeping the transform here makes it possible to test
the representation before the toolbox contract and the route orchestration are
available.

For an exported node ``n`` the emitted mask is::

    original(n) - union(original(d) for each exported descendant d of n)

The result is a normal COCO annotation whose ``segmentation`` is compressed RLE.  The
small ``hierarchy`` sidecar records the selected annotation tree; it is deliberately
plain JSON so the future route can attach it to a dataset document or artifact.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import cv2
import numpy as np
from pycocotools import mask as coco_mask


EXCLUSIVE_HIERARCHY_V1 = "exclusive_hierarchy_v1"
STRICT_HIERARCHY_POLICY = "strict"
NORMALIZE_HIERARCHY_POLICY = "normalize"
MIN_AUTO_PARENT_CONTAINMENT = 0.5


class HierarchyValidationError(ValueError):
    """Raised when annotations cannot be represented by the exclusive target.

    ``code`` and ``details`` keep validation failures useful to a future route without
    forcing it to parse human-readable text.  Details are deliberately limited to
    JSON-safe values so the error can be returned to a client without exposing an ORM
    object or an implementation traceback.
    """

    def __init__(
            self,
            message: str,
            *,
            code: str = "invalid_hierarchy",
            details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.details = {
            str(key): _safe_error_value(value)
            for key, value in (details or {}).items()
        }
        self.message = str(message)
        super().__init__(self.message)

    @property
    def error_code(self) -> str:
        """Compatibility alias for API layers that call the field ``error_code``."""
        return self.code

    def as_dict(self) -> dict[str, Any]:
        """Return a safe, JSON-compatible error document."""
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }

    to_dict = as_dict


@dataclass(frozen=True, slots=True)
class HierarchyNode:
    """A database-independent input node for the exclusive encoder.

    ``mask`` is expected to be a two-dimensional array with shape ``(height, width)``.
    It may be omitted when the caller passes a contour-like object directly to
    :func:`encode_exclusive_hierarchy_v1`; the object then needs ``to_binary_mask`` or
    ``x``/``y`` coordinates.  The aliases below are kept for callers that prefer a
    name tied to the training use case.
    """

    id: int
    label_id: int
    parent_id: int | None = None
    mask: np.ndarray | None = None
    image_id: int | None = None
    label_name: str | None = None


TrainingContour = HierarchyNode
ContourMask = HierarchyNode
ExclusiveHierarchyNode = HierarchyNode


ExclusiveHierarchyPayload: TypeAlias = dict[str, Any]

_MISSING = object()

_PAYLOAD_FIELDS = {
    "target_encoding",
    "encoding",
    "width",
    "height",
    "categories",
    "annotations",
    "image_id",
    "images",
    "hierarchy",
}
_IMAGE_FIELDS = {"id", "width", "height"}
_CATEGORY_FIELDS = {"id", "name", "supercategory"}
_ANNOTATION_FIELDS = {
    "id",
    "image_id",
    "category_id",
    "segmentation",
    "area",
    "bbox",
    "iscrowd",
    "original_annotation_id",
    "original_parent_annotation_id",
    "parent_annotation_id",
    "parent_id",
}
_HIERARCHY_FIELDS = {"encoding", "selected_label_ids", "label_parent_ids", "annotations"}
_LABEL_PARENT_FIELDS = {"label_id", "parent_label_id"}
_SIDECAR_ANNOTATION_FIELDS = {
    "annotation_id",
    "original_annotation_id",
    "parent_annotation_id",
    "label_id",
}


def encode_exclusive_hierarchy_v1(
        contours: Iterable[Any],
        width: int | Sequence[int] | None = None,
        height: int | Mapping[Any, Any] | None = None,
        label_parent_ids: Any = None,
        selected_label_ids: Iterable[int] | None = None,
        *,
        image_size: tuple[int, int] | None = None,
        image_shape: tuple[int, int] | None = None,
        label_hierarchy: Any = None,
        labels: Any = None,
        image_id: int | None = None,
        min_residual_area: int = 1,
        minimum_residual_area: int | None = None,
        containment_threshold: float = 1.0,
        max_peer_overlap_fraction: float = 0.0,
        max_sibling_overlap_fraction: float | None = None,
        overlap_threshold: float | None = None,
) -> ExclusiveHierarchyPayload:
    """Encode one image's selected contours as mutually exclusive COCO RLE masks.

    Parameters
    ----------
    contours:
        An iterable of :class:`HierarchyNode`, mappings, ORM rows, toolbox ``Contour``
        objects, or equivalent duck-typed objects.  A node must expose ``id``,
        ``label_id`` and ``parent_id``.  A binary ``mask`` is preferred; toolbox-style
        ``to_binary_mask(height, width)`` and normalized ``x``/``y`` coordinates are
        also supported for integration convenience.  Any non-null ``parent_id`` must
        resolve to another node in this iterable, including when that parent is outside
        the selected label scope; label metadata alone is not proof that a contour ID
        exists.
    width, height:
        Native image dimensions in pixels.  ``image_size``/``image_shape`` may instead
        be supplied as ``(height, width)``.  Explicit dimensions are recommended so a
        stale database thumbnail size cannot silently change the target.
    label_parent_ids:
        Mapping from database label ID to its immediate parent label ID (or ``None``).
        A toolbox ``LabelHierarchy`` or an iterable of label-like objects is accepted
        too.  It is required for every multi-label selection so skipped hierarchy
        levels cannot pass unvalidated.  A single-label selection may omit it.
    selected_label_ids:
        Labels exported by this run.  A single label is valid even when its parent
        label is outside the model scope.  When omitted, labels represented by the
        input nodes are selected.
    image_id:
        The one COCO image ID.  It may be supplied once here, or derived when every
        input node carries the same non-null ID.  Missing or mixed IDs are rejected.
    min_residual_area:
        Minimum number of native pixels that every exported residual must contain.
    containment_threshold:
        Kept as an explicit API setting for callers that already provide the
        prediction-placement configuration, but this encoder requires it to be exactly
        ``1.0``.  Exact containment is necessary for mathematical reversibility; a
        tolerant threshold would let escaped child pixels expand a reconstructed
        parent.
    max_peer_overlap_fraction:
        Must be zero.  Siblings and exported roots have no reversible relationship from
        which an overlap could be reconstructed, so every shared target pixel is
        rejected.  The legacy alias parameters are accepted only to produce an
        explicit validation error for nonzero values.

    Returns
    -------
        dict
        A JSON-compatible, single-image COCO payload with exactly one ``images``
        entry, an ``image_id`` on every annotation, dimensions, ``target_encoding``
        and a versioned hierarchy sidecar.  Each annotation's ``segmentation`` is
        ``{"size": [height, width], "counts": <ASCII RLE>}``.

    Raises
    ------
    HierarchyValidationError
        If references, labels, geometry, dimensions, or residual areas are invalid.
    """
    if minimum_residual_area is not None:
        min_residual_area = minimum_residual_area
    overlap_aliases = [
        value
        for value in (max_sibling_overlap_fraction, overlap_threshold)
        if value is not None
    ]
    if overlap_aliases and any(value != overlap_aliases[0] for value in overlap_aliases[1:]):
        raise HierarchyValidationError(
            "peer-overlap threshold aliases disagree",
            code="conflicting_peer_overlap_threshold",
        )
    if overlap_aliases:
        max_peer_overlap_fraction = overlap_aliases[0]

    # Support the compact positional form ``encode(nodes, (height, width),
    # label_parent_ids)`` in addition to the explicit keyword form.  A mapping cannot
    # be an image height, so the shift is unambiguous.
    if (
            isinstance(width, Sequence)
            and not isinstance(width, (str, bytes))
            and isinstance(height, Mapping)
            and label_parent_ids is None
    ):
        label_parent_ids = height
        height = None

    resolved_height, resolved_width = _resolve_dimensions(
        width=width,
        height=height,
        image_size=image_size,
        image_shape=image_shape,
    )
    _validate_threshold("containment_threshold", containment_threshold)
    _validate_threshold("max_peer_overlap_fraction", max_peer_overlap_fraction)
    if containment_threshold != 1.0:
        raise HierarchyValidationError(
            "containment_threshold must be exactly 1.0 for reversible exclusive encoding",
            code="non_exact_containment_threshold",
            details={"containment_threshold": containment_threshold},
        )
    if max_peer_overlap_fraction != 0.0:
        raise HierarchyValidationError(
            "max_peer_overlap_fraction must be exactly 0.0 for exclusive encoding",
            code="nonzero_peer_overlap_threshold",
            details={"max_peer_overlap_fraction": max_peer_overlap_fraction},
        )
    if not isinstance(min_residual_area, (int, np.integer)) or isinstance(min_residual_area, bool):
        raise HierarchyValidationError("minimum residual area must be a non-negative integer")
    if min_residual_area < 1:
        raise HierarchyValidationError("minimum residual area must be at least one pixel")

    if label_parent_ids is not None and (label_hierarchy is not None or labels is not None):
        raise HierarchyValidationError(
            "provide only one of label_parent_ids, label_hierarchy, or labels"
        )
    if label_parent_ids is None:
        label_parent_ids = label_hierarchy if label_hierarchy is not None else labels
    normalized_label_parents = _normalize_label_parent_ids(label_parent_ids)

    selected_labels = _normalize_selected_label_ids(selected_label_ids)
    requested_image_id = (
        None
        if image_id is None
        else _coerce_int(image_id, "image ID")
    )

    raw_nodes = _materialize_contours(contours)
    nodes = [
        _normalize_node(raw_node, resolved_height, resolved_width, requested_image_id, index)
        for index, raw_node in enumerate(raw_nodes)
    ]
    resolved_image_id = _resolve_image_id(nodes, requested_image_id)
    _validate_unique_node_ids(nodes)
    nodes_by_id = {node.id: node for node in nodes}
    _validate_parent_references(nodes, nodes_by_id)
    _validate_parent_cycles(nodes, nodes_by_id)

    represented_labels = {node.label_id for node in nodes}
    if normalized_label_parents is not None:
        unknown_node_labels = represented_labels - set(normalized_label_parents)
        if unknown_node_labels:
            raise HierarchyValidationError(
                "label hierarchy is missing label IDs "
                f"{sorted(unknown_node_labels)} used by the image"
            )

    if selected_labels is None:
        selected_labels = set(represented_labels)
    elif not selected_labels:
        raise HierarchyValidationError("selected label IDs must contain at least one label")

    if normalized_label_parents is None and len(selected_labels) > 1:
        raise HierarchyValidationError(
            "label_parent_ids is required for multi-label selection so intermediate "
            "hierarchy levels can be validated",
            code="label_parent_metadata_required",
            details={"selected_label_ids": sorted(selected_labels)},
        )

    if normalized_label_parents is not None:
        unknown_selected_labels = selected_labels - set(normalized_label_parents)
        if unknown_selected_labels:
            raise HierarchyValidationError(
                "selected label IDs are missing from the label hierarchy: "
                f"{sorted(unknown_selected_labels)}"
            )
        _validate_selected_label_paths(selected_labels, normalized_label_parents)

    missing_selected_labels = selected_labels - represented_labels
    selected_nodes = [node for node in nodes if node.label_id in selected_labels]
    if not selected_nodes:
        if missing_selected_labels:
            raise HierarchyValidationError(
                "selected label IDs are absent from this image's contour set: "
                f"{sorted(missing_selected_labels)}; select labels represented by the image",
                code="selected_labels_missing_from_image",
                details={
                    "image_id": resolved_image_id,
                    "missing_label_ids": sorted(missing_selected_labels),
                },
            )
        raise HierarchyValidationError(
            "selected label IDs do not match any contour in the image"
        )
    selected_node_ids = {node.id for node in selected_nodes}

    _validate_selected_label_relationships(
        selected_nodes,
        nodes_by_id,
        normalized_label_parents,
        selected_node_ids,
        selected_labels,
        containment_threshold,
    )
    if missing_selected_labels:
        raise HierarchyValidationError(
            "selected label IDs are absent from this image's contour set: "
            f"{sorted(missing_selected_labels)}; select labels represented by the image",
            code="selected_labels_missing_from_image",
            details={
                "image_id": resolved_image_id,
                "missing_label_ids": sorted(missing_selected_labels),
            },
        )
    _validate_peer_overlap(selected_nodes, nodes_by_id, max_peer_overlap_fraction)
    selected_nodes = _order_hierarchy_nodes(selected_nodes)

    # Traverse the complete input forest so every selected descendant is removed,
    # even when an intermediate node was filtered out of the selected scope.  When
    # label metadata is supplied, the selected-path validator rejects skipped
    # levels before this point; the complete traversal keeps the pure encoder's
    # disjointness guarantee explicit for metadata-free callers as well.
    all_children_by_parent: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        if node.parent_id in nodes_by_id:
            all_children_by_parent[node.parent_id].append(node.id)

    descendants_by_id = {
        node.id: [
            descendant_id
            for descendant_id in _descendants(node.id, all_children_by_parent)
            if descendant_id in selected_node_ids
        ]
        for node in selected_nodes
    }
    residual_masks: dict[int, np.ndarray] = {}
    for node in selected_nodes:
        residual = node.mask.copy()
        descendants = descendants_by_id[node.id]
        if descendants:
            descendant_union = np.zeros((resolved_height, resolved_width), dtype=bool)
            for descendant_id in descendants:
                descendant_union |= nodes_by_id[descendant_id].mask
            residual &= ~descendant_union

        residual_area = int(residual.sum())
        if residual_area < min_residual_area:
            raise HierarchyValidationError(
                "empty or too-small exported residual for "
                f"image {resolved_image_id}, "
                f"contour {node.id}, label {node.label_id}: "
                f"{residual_area} pixels < minimum {min_residual_area}",
                code="empty_or_small_residual",
                details={
                    "image_id": resolved_image_id,
                    "contour_id": node.id,
                    "label_id": node.label_id,
                    "residual_area": residual_area,
                    "minimum_residual_area": min_residual_area,
                },
            )
        residual_masks[node.id] = residual

    _assert_exclusive_residuals(selected_nodes, residual_masks)
    return _build_payload(
        selected_nodes=selected_nodes,
        residual_masks=residual_masks,
        label_parent_ids=normalized_label_parents,
        selected_label_ids=selected_labels,
        width=resolved_width,
        height=resolved_height,
        image_id=resolved_image_id,
    )


def decode_coco_rle(segmentation: Mapping[str, Any]) -> np.ndarray:
    """Decode a compressed COCO RLE mapping into a boolean ``(height, width)`` mask."""
    if not isinstance(segmentation, Mapping):
        raise HierarchyValidationError(
            "COCO segmentation must be an RLE mapping",
            code="invalid_coco_segmentation",
        )
    unknown_fields = set(segmentation) - {"size", "counts"}
    if unknown_fields:
        raise HierarchyValidationError(
            "COCO RLE contains unknown fields",
            code="invalid_coco_rle",
            details={"unknown_fields": sorted(str(field) for field in unknown_fields)},
        )
    if "size" not in segmentation or "counts" not in segmentation:
        raise HierarchyValidationError(
            "COCO RLE must contain size and counts",
            code="invalid_coco_rle",
        )

    size = segmentation["size"]
    if not isinstance(size, Sequence) or len(size) != 2:
        raise HierarchyValidationError(
            "COCO RLE size must be [height, width]",
            code="invalid_coco_rle",
        )
    height = _coerce_positive_int(size[0], "RLE height")
    width = _coerce_positive_int(size[1], "RLE width")
    try:
        counts = segmentation["counts"]
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        elif isinstance(counts, bytes):
            counts.decode("ascii")
        else:
            raise TypeError("compressed COCO RLE counts must be ASCII text")
        if not counts:
            raise ValueError("compressed COCO RLE counts cannot be empty")
        raw_rle = {"size": [height, width], "counts": counts}
        decoded = coco_mask.decode(raw_rle)
    except Exception as exc:  # pycocotools raises several implementation-specific types
        raise HierarchyValidationError(
            "invalid COCO RLE counts",
            code="invalid_coco_rle",
            details={"cause": type(exc).__name__},
        ) from exc
    if decoded.ndim == 3:
        if decoded.shape[2] != 1:
            raise HierarchyValidationError(
                "only one binary mask may be decoded at a time",
                code="invalid_coco_rle",
            )
        decoded = decoded[:, :, 0]
    decoded = decoded.astype(bool, copy=False)
    canonical_rle = coco_mask.encode(np.asfortranarray(decoded.astype(np.uint8)))
    canonical_counts = canonical_rle["counts"]
    if isinstance(canonical_counts, bytes):
        canonical_counts = canonical_counts.decode("ascii")
    supplied_counts = segmentation["counts"]
    if isinstance(supplied_counts, bytes):
        supplied_counts = supplied_counts.decode("ascii")
    if supplied_counts != canonical_counts:
        raise HierarchyValidationError(
            "COCO RLE counts are malformed or non-canonical",
            code="invalid_coco_rle",
        )
    return decoded


def _validate_coco_hierarchy_payload(
        payload: Mapping[str, Any],
) -> tuple[
        dict[int, Mapping[str, Any]],
        dict[int, np.ndarray],
        dict[int, int | None],
]:
    """Validate the emitted single-image COCO contract and decode its annotations."""
    if not isinstance(payload, Mapping):
        raise HierarchyValidationError(
            "encoded hierarchy payload must be a mapping",
            code="invalid_coco_payload",
        )
    unknown_payload_fields = set(payload) - _PAYLOAD_FIELDS
    if unknown_payload_fields:
        raise HierarchyValidationError(
            "encoded hierarchy payload contains unknown fields",
            code="invalid_coco_payload",
            details={"unknown_fields": sorted(str(field) for field in unknown_payload_fields)},
        )

    target_encoding = payload.get("target_encoding")
    encoding = payload.get("encoding")
    if target_encoding != EXCLUSIVE_HIERARCHY_V1 or encoding != EXCLUSIVE_HIERARCHY_V1:
        raise HierarchyValidationError(
            "unsupported or mismatched exclusive hierarchy encoding",
            code="unsupported_target_encoding",
            details={
                "target_encoding": target_encoding,
                "encoding": encoding,
                "expected": EXCLUSIVE_HIERARCHY_V1,
            },
        )

    hierarchy = payload.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise HierarchyValidationError(
            "exclusive hierarchy payload is missing its hierarchy sidecar",
            code="invalid_hierarchy_sidecar",
        )
    unknown_hierarchy_fields = set(hierarchy) - _HIERARCHY_FIELDS
    if unknown_hierarchy_fields:
        raise HierarchyValidationError(
            "hierarchy sidecar contains unknown fields",
            code="invalid_hierarchy_sidecar",
            details={"unknown_fields": sorted(str(field) for field in unknown_hierarchy_fields)},
        )
    if hierarchy.get("encoding") != EXCLUSIVE_HIERARCHY_V1:
        raise HierarchyValidationError(
            "hierarchy sidecar has an unsupported encoding version",
            code="unsupported_target_encoding",
            details={"encoding": hierarchy.get("encoding")},
        )

    height = _coerce_positive_int(payload.get("height", _MISSING), "payload height")
    width = _coerce_positive_int(payload.get("width", _MISSING), "payload width")
    images = payload.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)) or len(images) != 1:
        raise HierarchyValidationError(
            "exclusive hierarchy COCO payload must contain exactly one image",
            code="single_image_required",
        )
    image = images[0]
    if not isinstance(image, Mapping):
        raise HierarchyValidationError(
            "COCO image entry must be a mapping",
            code="invalid_coco_image",
        )
    unknown_image_fields = set(image) - _IMAGE_FIELDS
    if unknown_image_fields:
        raise HierarchyValidationError(
            "COCO image entry contains unknown fields",
            code="invalid_coco_image",
            details={"unknown_fields": sorted(str(field) for field in unknown_image_fields)},
        )
    image_id = _coerce_int(image.get("id", _MISSING), "COCO image ID")
    declared_image_width = _coerce_positive_int(
        image.get("width", _MISSING), "COCO image width"
    )
    declared_image_height = _coerce_positive_int(
        image.get("height", _MISSING), "COCO image height"
    )
    if (declared_image_height, declared_image_width) != (height, width):
        raise HierarchyValidationError(
            "COCO image dimensions do not match payload dimensions",
            code="image_dimension_mismatch",
            details={
                "image_id": image_id,
                "image_dimensions": [declared_image_height, declared_image_width],
                "payload_dimensions": [height, width],
            },
        )
    payload_image_id = _coerce_int(payload.get("image_id", _MISSING), "payload image ID")
    if payload_image_id != image_id:
        raise HierarchyValidationError(
            "payload image_id does not match its single COCO image",
            code="mixed_image_ids",
            details={"payload_image_id": payload_image_id, "image_id": image_id},
        )

    categories = payload.get("categories")
    if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
        raise HierarchyValidationError(
            "COCO payload is missing its categories list",
            code="invalid_coco_categories",
        )
    category_ids: set[int] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, Mapping):
            raise HierarchyValidationError(
                f"category {index} is not a mapping",
                code="invalid_coco_categories",
            )
        unknown_category_fields = set(category) - _CATEGORY_FIELDS
        if unknown_category_fields:
            raise HierarchyValidationError(
                f"category {index} contains unknown fields",
                code="invalid_coco_categories",
                details={"unknown_fields": sorted(str(field) for field in unknown_category_fields)},
            )
        if not isinstance(category.get("name"), str) or not isinstance(
                category.get("supercategory"), str
        ):
            raise HierarchyValidationError(
                f"category {index} must contain string name and supercategory",
                code="invalid_coco_categories",
            )
        category_id = _coerce_int(
            category.get("id", _MISSING),
            f"category {index} ID",
        )
        if category_id in category_ids:
            raise HierarchyValidationError(
                f"duplicate COCO category ID {category_id}",
                code="duplicate_category_id",
            )
        category_ids.add(category_id)

    selected_label_ids = hierarchy.get("selected_label_ids")
    if (
            not isinstance(selected_label_ids, Sequence)
            or isinstance(selected_label_ids, (str, bytes))
            or not selected_label_ids
    ):
        raise HierarchyValidationError(
            "hierarchy sidecar must contain selected_label_ids",
            code="invalid_hierarchy_sidecar",
        )
    normalized_selected_labels = [
        _coerce_int(value, "selected label ID") for value in selected_label_ids
    ]
    if len(set(normalized_selected_labels)) != len(normalized_selected_labels):
        raise HierarchyValidationError(
            "hierarchy sidecar contains duplicate selected label IDs",
            code="invalid_hierarchy_sidecar",
        )
    if set(normalized_selected_labels) != category_ids:
        raise HierarchyValidationError(
            "selected label IDs do not match COCO categories",
            code="invalid_hierarchy_sidecar",
        )

    sidecar_label_entries = hierarchy.get("label_parent_ids")
    if not isinstance(sidecar_label_entries, Mapping):
        raise HierarchyValidationError(
            "hierarchy sidecar is missing or has invalid label_parent_ids",
            code="invalid_hierarchy_sidecar",
        )
    sidecar_label_parents: dict[int, int | None] = {}
    for raw_label_id, raw_parent_id in sidecar_label_entries.items():
        label_id = _coerce_int(raw_label_id, "sidecar label ID")
        parent_id = None if raw_parent_id is None else _coerce_int(
            raw_parent_id,
            f"parent label ID for {label_id}",
        )
        sidecar_label_parents[label_id] = parent_id
    if not sidecar_label_parents and len(normalized_selected_labels) > 1:
        raise HierarchyValidationError(
            "hierarchy sidecar label_parent_ids cannot be empty for multi-label encoding",
            code="invalid_hierarchy_sidecar",
        )
    if sidecar_label_parents and not set(normalized_selected_labels) <= set(sidecar_label_parents):
        raise HierarchyValidationError(
            "hierarchy sidecar is missing a selected label mapping",
            code="invalid_hierarchy_sidecar",
        )
    for label_id, parent_id in sidecar_label_parents.items():
        if parent_id is not None and parent_id not in sidecar_label_parents:
            raise HierarchyValidationError(
                f"sidecar label {label_id} references missing parent label {parent_id}",
                code="missing_label_parent",
            )
    _validate_label_cycles(sidecar_label_parents)
    for selected_label_id in normalized_selected_labels:
        if selected_label_id not in sidecar_label_parents:
            continue
        parent_id = sidecar_label_parents[selected_label_id]
        while parent_id is not None:
            if parent_id not in sidecar_label_parents:
                raise HierarchyValidationError(
                    f"selected label {selected_label_id} has an incomplete parent path",
                    code="missing_label_parent",
                )
            parent_id = sidecar_label_parents[parent_id]

    annotations = payload.get("annotations")
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)) or not annotations:
        raise HierarchyValidationError(
            "encoded hierarchy payload must contain a non-empty annotations list",
            code="invalid_coco_annotations",
        )

    by_id: dict[int, Mapping[str, Any]] = {}
    decoded: dict[int, np.ndarray] = {}
    parent_by_id: dict[int, int | None] = {}
    original_id_by_id: dict[int, int] = {}
    annotation_category_by_id: dict[int, int] = {}
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise HierarchyValidationError(
                f"annotation {index} is not a mapping",
                code="invalid_coco_annotation",
            )
        unknown_annotation_fields = set(annotation) - _ANNOTATION_FIELDS
        if unknown_annotation_fields:
            raise HierarchyValidationError(
                f"annotation {index} contains unknown fields",
                code="invalid_coco_annotation",
                details={"unknown_fields": sorted(str(field) for field in unknown_annotation_fields)},
            )
        missing_annotation_fields = _ANNOTATION_FIELDS - set(annotation)
        if missing_annotation_fields:
            readable_missing = [
                "image ID" if field == "image_id" else field
                for field in sorted(missing_annotation_fields)
            ]
            raise HierarchyValidationError(
                f"annotation {index} is missing required fields: {', '.join(readable_missing)}",
                code="invalid_coco_annotation",
                details={"missing_fields": sorted(missing_annotation_fields)},
            )
        annotation_id = _coerce_int(
            annotation.get("id", _MISSING),
            f"annotation {index} ID",
        )
        if annotation_id in by_id:
            raise HierarchyValidationError(
                f"duplicate encoded annotation ID {annotation_id}",
                code="duplicate_annotation_id",
            )
        annotation_image_id = _coerce_int(
            annotation.get("image_id", _MISSING),
            f"image ID for annotation {annotation_id}",
        )
        if annotation_image_id != image_id:
            raise HierarchyValidationError(
                f"annotation {annotation_id} has image {annotation_image_id}, expected image {image_id}",
                code="mixed_image_ids",
                details={"annotation_id": annotation_id, "image_id": annotation_image_id},
            )
        category_id = _coerce_int(
            annotation.get("category_id", _MISSING),
            f"category ID for annotation {annotation_id}",
        )
        if annotation["iscrowd"] != 0:
            raise HierarchyValidationError(
                f"annotation {annotation_id} iscrowd must be 0",
                code="invalid_coco_annotation",
            )
        original_annotation_id = _coerce_int(
            annotation.get("original_annotation_id", _MISSING),
            f"original annotation ID for {annotation_id}",
        )
        if original_annotation_id in original_id_by_id.values():
            raise HierarchyValidationError(
                f"duplicate original annotation ID {original_annotation_id}",
                code="duplicate_original_annotation_id",
            )
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, Mapping):
            raise HierarchyValidationError(
                f"annotation {annotation_id} segmentation must be an RLE mapping",
                code="invalid_coco_segmentation",
            )
        segmentation_size = segmentation.get("size")
        if (
                not isinstance(segmentation_size, Sequence)
                or isinstance(segmentation_size, (str, bytes))
                or len(segmentation_size) != 2
        ):
            raise HierarchyValidationError(
                f"annotation {annotation_id} RLE size must be [height, width]",
                code="invalid_coco_rle",
            )
        rle_height = _coerce_positive_int(
            segmentation_size[0],
            f"RLE height for annotation {annotation_id}",
        )
        rle_width = _coerce_positive_int(
            segmentation_size[1],
            f"RLE width for annotation {annotation_id}",
        )
        if (rle_height, rle_width) != (height, width):
            raise HierarchyValidationError(
                f"annotation {annotation_id} RLE dimensions {(rle_height, rle_width)} "
                f"do not match image dimensions {(height, width)}; all annotations "
                "must use the same mask dimensions",
                code="image_dimension_mismatch",
                details={"annotation_id": annotation_id},
            )
        if category_id not in category_ids:
            raise HierarchyValidationError(
                f"annotation {annotation_id} references missing category {category_id}",
                code="missing_category",
            )
        decoded_mask = decode_coco_rle(segmentation)
        actual_area = int(decoded_mask.sum())
        if actual_area < 1:
            raise HierarchyValidationError(
                f"annotation {annotation_id} has an empty RLE mask",
                code="empty_annotation_mask",
            )

        raw_area = annotation.get("area", _MISSING)
        if (
                raw_area is _MISSING
                or isinstance(raw_area, (bool, np.bool_))
                or not isinstance(raw_area, (int, float, np.integer, np.floating))
                or not np.isfinite(raw_area)
                or float(raw_area) != actual_area
        ):
            raise HierarchyValidationError(
                f"annotation {annotation_id} area does not match its RLE mask",
                code="invalid_coco_annotation",
                details={"annotation_id": annotation_id, "expected_area": actual_area},
            )

        bbox = annotation.get("bbox", _MISSING)
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise HierarchyValidationError(
                f"annotation {annotation_id} bbox must contain four numbers",
                code="invalid_coco_annotation",
            )
        try:
            bbox_values = [float(value) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise HierarchyValidationError(
                f"annotation {annotation_id} bbox must contain finite numbers",
                code="invalid_coco_annotation",
            ) from exc
        if not np.isfinite(bbox_values).all():
            raise HierarchyValidationError(
                f"annotation {annotation_id} bbox must contain finite numbers",
                code="invalid_coco_annotation",
            )
        bbox_x, bbox_y, bbox_width, bbox_height = bbox_values
        if (
                bbox_x < 0
                or bbox_y < 0
                or bbox_width <= 0
                or bbox_height <= 0
                or bbox_x + bbox_width > width + 1e-6
                or bbox_y + bbox_height > height + 1e-6
        ):
            raise HierarchyValidationError(
                f"annotation {annotation_id} bbox is outside image bounds",
                code="out_of_bounds_annotation",
                details={"annotation_id": annotation_id, "image_id": image_id},
            )

        expected_bbox = [
            float(value)
            for value in coco_mask.toBbox(
                coco_mask.encode(np.asfortranarray(decoded_mask.astype(np.uint8)))
            ).tolist()
        ]
        if bbox_values != expected_bbox:
            raise HierarchyValidationError(
                f"annotation {annotation_id} bbox does not match its RLE mask",
                code="invalid_coco_annotation",
                details={"annotation_id": annotation_id, "expected_bbox": expected_bbox},
            )

        if (
                "original_parent_annotation_id" not in annotation
                or "parent_annotation_id" not in annotation
                or "parent_id" not in annotation
        ):
            raise HierarchyValidationError(
                f"annotation {annotation_id} is missing its hierarchy parent fields",
                code="invalid_hierarchy_sidecar",
            )
        raw_parent_annotation_id = annotation["parent_annotation_id"]
        raw_parent_id = annotation["parent_id"]
        raw_original_parent_id = annotation["original_parent_annotation_id"]
        original_parent_id = (
            None
            if raw_original_parent_id is None
            else _coerce_int(raw_original_parent_id, "original parent annotation ID")
        )
        parent_annotation_id = (
            None
            if raw_parent_annotation_id is None
            else _coerce_int(raw_parent_annotation_id, "parent annotation ID")
        )
        parent_id = None if raw_parent_id is None else _coerce_int(raw_parent_id, "parent ID")
        if parent_annotation_id != parent_id:
            raise HierarchyValidationError(
                f"annotation {annotation_id} has conflicting hierarchy parent fields",
                code="conflicting_parent_reference",
            )

        decoded[annotation_id] = decoded_mask
        by_id[annotation_id] = annotation
        parent_by_id[annotation_id] = parent_annotation_id
        original_id_by_id[annotation_id] = original_annotation_id
        annotation_category_by_id[annotation_id] = category_id

    original_ids = set(original_id_by_id.values())
    for annotation_id, parent_id in parent_by_id.items():
        annotation = by_id[annotation_id]
        original_parent_id = (
            None
            if annotation["original_parent_annotation_id"] is None
            else _coerce_int(
                annotation["original_parent_annotation_id"],
                f"original parent annotation ID for {annotation_id}",
            )
        )
        if parent_id is not None:
            if parent_id not in original_id_by_id:
                raise HierarchyValidationError(
                    f"encoded annotation {annotation_id} references missing parent {parent_id}",
                    code="missing_parent_reference",
                )
            expected_original_parent = original_id_by_id[parent_id]
            if original_parent_id != expected_original_parent:
                raise HierarchyValidationError(
                    f"annotation {annotation_id} has a mismatched original parent path",
                    code="conflicting_parent_reference",
                )
        elif original_parent_id in original_ids:
            raise HierarchyValidationError(
                f"annotation {annotation_id} skips an encoded original parent",
                code="conflicting_parent_reference",
            )

    decoded_items = list(decoded.items())
    for index, (left_id, left_mask) in enumerate(decoded_items):
        for right_id, right_mask in decoded_items[index + 1:]:
            if np.logical_and(left_mask, right_mask).any():
                raise HierarchyValidationError(
                    f"encoded annotations {left_id} and {right_id} overlap",
                    code="residual_overlap",
                    details={
                        "left_annotation_id": left_id,
                        "right_annotation_id": right_id,
                    },
                )

    _validate_encoded_parent_cycles(parent_by_id)

    if set(category_ids) != {
            _coerce_int(annotation["category_id"], "annotation category ID")
            for annotation in by_id.values()
    }:
        raise HierarchyValidationError(
            "COCO categories do not match annotation categories",
            code="invalid_coco_categories",
        )

    for annotation_id, parent_id in parent_by_id.items():
        if parent_id is None:
            continue
        child_label_id = annotation_category_by_id[annotation_id]
        expected_parent_label = sidecar_label_parents.get(child_label_id)
        parent_label_id = annotation_category_by_id[parent_id]
        if expected_parent_label != parent_label_id:
            raise HierarchyValidationError(
                f"annotation {annotation_id} parent label {parent_label_id} does not match "
                f"the label hierarchy parent {expected_parent_label}",
                code="invalid_hierarchy_sidecar",
            )
    for annotation_id, parent_id in parent_by_id.items():
        if parent_id is None:
            expected_parent_label = sidecar_label_parents.get(annotation_category_by_id[annotation_id])
            if expected_parent_label in category_ids:
                raise HierarchyValidationError(
                    f"annotation {annotation_id} is missing its selected label parent",
                    code="missing_parent_reference",
                )

    sidecar_annotations = hierarchy.get("annotations")
    if not isinstance(sidecar_annotations, Sequence) or isinstance(sidecar_annotations, (str, bytes)):
        raise HierarchyValidationError(
            "hierarchy sidecar is missing annotation metadata",
            code="invalid_hierarchy_sidecar",
        )
    sidecar_by_id: dict[int, Mapping[str, Any]] = {}
    sidecar_original_ids: dict[int, int] = {}
    for index, sidecar_annotation in enumerate(sidecar_annotations):
        if not isinstance(sidecar_annotation, Mapping):
            raise HierarchyValidationError(
                f"sidecar annotation {index} is not a mapping",
                code="invalid_hierarchy_sidecar",
            )
        unknown_sidecar_fields = set(sidecar_annotation) - _SIDECAR_ANNOTATION_FIELDS
        if unknown_sidecar_fields:
            raise HierarchyValidationError(
                f"sidecar annotation {index} contains unknown fields",
                code="invalid_hierarchy_sidecar",
                details={"unknown_fields": sorted(str(field) for field in unknown_sidecar_fields)},
            )
        missing_sidecar_fields = _SIDECAR_ANNOTATION_FIELDS - set(sidecar_annotation)
        if missing_sidecar_fields:
            raise HierarchyValidationError(
                f"sidecar annotation {index} is missing required fields",
                code="invalid_hierarchy_sidecar",
                details={"missing_fields": sorted(missing_sidecar_fields)},
            )
        sidecar_id = _coerce_int(
            sidecar_annotation.get("annotation_id", _MISSING),
            f"sidecar annotation {index} ID",
        )
        if sidecar_id in sidecar_by_id:
            raise HierarchyValidationError(
                f"duplicate sidecar annotation ID {sidecar_id}",
                code="duplicate_annotation_id",
            )
        sidecar_original_id = _coerce_int(
            sidecar_annotation["original_annotation_id"],
            f"sidecar original annotation ID for {sidecar_id}",
        )
        if sidecar_original_id in sidecar_original_ids.values():
            raise HierarchyValidationError(
                f"duplicate sidecar original annotation ID {sidecar_original_id}",
                code="duplicate_original_annotation_id",
            )
        sidecar_by_id[sidecar_id] = sidecar_annotation
        sidecar_original_ids[sidecar_id] = sidecar_original_id
    if set(sidecar_by_id) != set(by_id):
        raise HierarchyValidationError(
            "hierarchy sidecar annotations do not match COCO annotations",
            code="invalid_hierarchy_sidecar",
        )
    for annotation_id, annotation in by_id.items():
        sidecar_annotation = sidecar_by_id[annotation_id]
        sidecar_label_id = _coerce_int(
            sidecar_annotation.get("label_id", _MISSING),
            f"sidecar label ID for annotation {annotation_id}",
        )
        annotation_label_id = _coerce_int(
            annotation["category_id"],
            f"category ID for annotation {annotation_id}",
        )
        if sidecar_label_id != annotation_label_id:
            raise HierarchyValidationError(
                f"sidecar annotation {annotation_id} label does not match category",
                code="invalid_hierarchy_sidecar",
            )
        if sidecar_original_ids[annotation_id] != original_id_by_id[annotation_id]:
            raise HierarchyValidationError(
                f"sidecar annotation {annotation_id} original ID does not match COCO annotation",
                code="invalid_hierarchy_sidecar",
            )
        sidecar_parent_value = sidecar_annotation.get("parent_annotation_id", _MISSING)
        sidecar_parent_id = (
            None
            if sidecar_parent_value is None
            else _coerce_int(sidecar_parent_value, "sidecar parent annotation ID")
        )
        if sidecar_parent_id != parent_by_id[annotation_id]:
            raise HierarchyValidationError(
                f"sidecar annotation {annotation_id} parent does not match COCO annotation",
                code="conflicting_parent_reference",
            )

    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for annotation_id, parent_id in parent_by_id.items():
        if parent_id is not None:
            if parent_id not in by_id:
                raise HierarchyValidationError(
                    f"encoded annotation {annotation_id} references missing parent {parent_id}",
                    code="missing_parent_reference",
                    details={"annotation_id": annotation_id, "parent_id": parent_id},
                )
            children_by_parent[parent_id].append(annotation_id)

    return by_id, decoded, parent_by_id


def reconstruct_hierarchy_masks(payload: Mapping[str, Any]) -> dict[int, np.ndarray]:
    """Reconstruct original masks from an encoded payload's residual annotations.

    The helper is intentionally pure and consumes only the emitted RLEs and parent
    annotation IDs.  It is useful for exact round-trip tests and is also a bounded
    intermediate for the later inference decoder.
    """
    by_id, decoded, parent_by_id = _validate_coco_hierarchy_payload(payload)

    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for annotation_id, parent_id in parent_by_id.items():
        if parent_id is not None:
            if parent_id not in by_id:
                raise HierarchyValidationError(
                    f"encoded annotation {annotation_id} references missing parent {parent_id}"
                )
            children_by_parent[parent_id].append(annotation_id)

    state: dict[int, int] = {}
    reconstructed: dict[int, np.ndarray] = {}

    def build(annotation_id: int) -> np.ndarray:
        status = state.get(annotation_id, 0)
        if status == 1:
            raise HierarchyValidationError("cycle in encoded hierarchy sidecar")
        if status == 2:
            return reconstructed[annotation_id]
        state[annotation_id] = 1
        result = decoded[annotation_id].copy()
        for child_id in children_by_parent.get(annotation_id, []):
            result |= build(child_id)
        state[annotation_id] = 2
        reconstructed[annotation_id] = result
        return result

    for annotation_id in by_id:
        build(annotation_id)
    return reconstructed


# Names used by integration code and tests can remain descriptive without coupling the
# later route to one private implementation spelling.
decode_exclusive_rle = decode_coco_rle
reconstruct_original_masks = reconstruct_hierarchy_masks


def _resolve_dimensions(
        *,
        width: int | Sequence[int] | None,
        height: int | Mapping[Any, Any] | None,
        image_size: tuple[int, int] | None,
        image_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if image_size is not None and image_shape is not None:
        raise HierarchyValidationError("provide only one of image_size and image_shape")
    if image_shape is not None:
        image_size = image_shape

    # Also accept encode(nodes, (height, width), label_parent_ids) as a small
    # convenience for callers working directly from numpy masks.
    if isinstance(width, Sequence) and not isinstance(width, (str, bytes)):
        if len(width) != 2:
            raise HierarchyValidationError("image dimensions must contain height and width")
        if image_size is not None:
            raise HierarchyValidationError("image dimensions were provided twice")
        if isinstance(height, Mapping):
            # The positional third argument is handled by the caller below; this
            # branch only documents the accepted shape and lets the type checker know
            # the sequence is not a scalar.
            raise HierarchyValidationError(
                "when passing image dimensions positionally, pass label_parent_ids by keyword"
            )
        image_size = (width[0], width[1])
        width = None
        height = None

    if image_size is not None:
        if not isinstance(image_size, Sequence) or len(image_size) != 2:
            raise HierarchyValidationError("image dimensions must be (height, width)")
        resolved_height = _coerce_positive_int(image_size[0], "image height")
        resolved_width = _coerce_positive_int(image_size[1], "image width")
        return resolved_height, resolved_width

    if isinstance(width, bool) or isinstance(height, bool) or width is None or height is None:
        raise HierarchyValidationError("positive image width and height are required")
    if isinstance(height, Mapping):
        raise HierarchyValidationError("image height must be a positive integer")
    resolved_width = _coerce_positive_int(width, "image width")
    resolved_height = _coerce_positive_int(height, "image height")
    return resolved_height, resolved_width


def _materialize_contours(contours: Iterable[Any]) -> list[Any]:
    if isinstance(contours, Mapping):
        values = list(contours.values())
    else:
        try:
            values = list(contours)
        except TypeError as exc:
            raise HierarchyValidationError("contours must be an iterable of nodes") from exc
    if not values:
        raise HierarchyValidationError("at least one contour is required")
    return values


def _normalize_node(
        raw_node: Any,
        height: int,
        width: int,
        image_id: int | None,
        index: int,
) -> HierarchyNode:
    node_id = _coerce_int(
        _value(raw_node, "id", "contour_id", "annotation_id", default=_MISSING),
        f"contour {index} ID",
    )
    label_id = _coerce_int(
        _value(raw_node, "label_id", "category_id", default=_MISSING),
        f"contour {node_id} label ID",
    )
    raw_parent_id = _value(raw_node, "parent_id", "parent_annotation_id", default=None)
    parent_id = None if raw_parent_id is None else _coerce_int(raw_parent_id, f"contour {node_id} parent ID")
    raw_image_id = _value(raw_node, "image_id", default=image_id)
    resolved_image_id = None if raw_image_id is None else _coerce_int(raw_image_id, f"contour {node_id} image ID")
    if image_id is not None and resolved_image_id is not None and resolved_image_id != image_id:
        raise HierarchyValidationError(
            f"contour {node_id} belongs to image {resolved_image_id}, expected image {image_id}"
        )

    raw_mask = _value(
        raw_node,
        "mask",
        "binary_mask",
        "raster",
        "segmentation",
        default=_MISSING,
    )
    if raw_mask is _MISSING or raw_mask is None:
        raw_mask = _rasterize_contour_like(raw_node, height, width, node_id)
    mask = _normalize_mask(raw_mask, height, width, node_id)
    raw_label_name = _value(raw_node, "label_name", "category_name", default=None)
    return HierarchyNode(
        id=node_id,
        label_id=label_id,
        parent_id=parent_id,
        mask=mask,
        image_id=resolved_image_id,
        label_name=None if raw_label_name is None else str(raw_label_name),
    )


def _resolve_image_id(
        nodes: Sequence[HierarchyNode],
        requested_image_id: int | None,
) -> int:
    """Resolve one image ID and reject ambiguous or unidentifiable input."""
    present_ids = {node.image_id for node in nodes if node.image_id is not None}
    if len(present_ids) > 1:
        raise HierarchyValidationError(
            "contours belong to multiple image IDs",
            code="mixed_image_ids",
            details={"image_ids": sorted(present_ids)},
        )
    if requested_image_id is None:
        if any(node.image_id is None for node in nodes):
            raise HierarchyValidationError(
                "image_id is required on every contour when no image_id argument is provided",
                code="missing_image_id",
            )
        if len(present_ids) != 1:
            raise HierarchyValidationError(
                "exactly one image_id must be derivable from the contours",
                code="missing_image_id",
            )
        return next(iter(present_ids))
    if present_ids and next(iter(present_ids)) != requested_image_id:
        raise HierarchyValidationError(
            "explicit image_id does not match contour image IDs",
            code="mixed_image_ids",
            details={
                "explicit_image_id": requested_image_id,
                "contour_image_id": next(iter(present_ids)),
            },
        )
    return requested_image_id


def _rasterize_contour_like(raw_node: Any, height: int, width: int, node_id: int) -> Any:
    to_binary_mask = _value(raw_node, "to_binary_mask", default=_MISSING)
    if callable(to_binary_mask):
        try:
            return to_binary_mask(height, width)
        except Exception as exc:
            raise HierarchyValidationError(
                f"could not rasterize contour {node_id} with to_binary_mask",
                code="rasterization_failed",
                details={"contour_id": node_id, "cause": type(exc).__name__},
            ) from exc

    x_values = _value(raw_node, "x", default=_MISSING)
    y_values = _value(raw_node, "y", default=_MISSING)
    if x_values is _MISSING or y_values is _MISSING:
        raise HierarchyValidationError(
            f"contour {node_id} must provide a 2-D mask or rasterizable x/y coordinates"
        )
    try:
        x = np.asarray(x_values, dtype=np.float64)
        y = np.asarray(y_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HierarchyValidationError(f"contour {node_id} has invalid x/y coordinates") from exc
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise HierarchyValidationError(
            f"contour {node_id} must have at least three paired x/y coordinates"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise HierarchyValidationError(f"contour {node_id} has non-finite x/y coordinates")

    normalized = bool(np.max(np.abs(x)) <= 1.5 and np.max(np.abs(y)) <= 1.5)
    if normalized:
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            raise HierarchyValidationError(
                f"contour {node_id} has normalized coordinates outside [0, 1]"
            )
        points = np.column_stack((x * width, y * height))
    else:
        points = np.column_stack((x, y))
    if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] > width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] > height)
    ):
        raise HierarchyValidationError(
            f"contour {node_id} coordinates are outside image bounds",
            code="out_of_bounds_contour",
            details={"contour_id": node_id, "image_dimensions": [height, width]},
        )
    polygon = np.rint(points).astype(np.int32)
    raster = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(raster, [polygon.reshape((-1, 1, 2))], 1)
    return raster


def _normalize_mask(raw_mask: Any, height: int, width: int, node_id: int) -> np.ndarray:
    if isinstance(raw_mask, Mapping) and "counts" in raw_mask:
        mask = decode_coco_rle(raw_mask)
    else:
        try:
            mask = np.asarray(raw_mask)
        except (TypeError, ValueError) as exc:
            raise HierarchyValidationError(f"contour {node_id} has an invalid mask") from exc
    if mask.ndim != 2:
        raise HierarchyValidationError(
            f"contour {node_id} mask must be 2-D, got shape {mask.shape}"
        )
    if tuple(mask.shape) != (height, width):
        raise HierarchyValidationError(
            f"contour {node_id} mask dimensions {tuple(mask.shape)} do not match "
            f"image dimensions {(height, width)}"
        )
    if mask.dtype != np.dtype(bool):
        if not np.issubdtype(mask.dtype, np.number) or np.iscomplexobj(mask):
            raise HierarchyValidationError(
                f"contour {node_id} mask must contain only binary numeric values"
            )
        if not np.isfinite(mask).all():
            raise HierarchyValidationError(f"contour {node_id} mask contains non-finite values")
        if not np.logical_or(mask == 0, mask == 1).all():
            raise HierarchyValidationError(
                f"contour {node_id} mask must contain only 0/1 values"
            )
    binary = mask.astype(bool, copy=True)
    if not binary.any():
        raise HierarchyValidationError(f"contour {node_id} mask is empty")
    return binary


def _normalize_label_parent_ids(raw_labels: Any) -> dict[int, int | None] | None:
    if raw_labels is None:
        return None
    if hasattr(raw_labels, "id_to_label_object"):
        raw_labels = raw_labels.id_to_label_object

    if isinstance(raw_labels, Mapping):
        items = list(raw_labels.items())
    else:
        try:
            items = [
                (
                    _value(item, "id", "label_id", default=_MISSING),
                    _value(item, "parent_id", "parent_label_id", default=None),
                )
                for item in raw_labels
            ]
        except TypeError as exc:
            raise HierarchyValidationError(
                "label hierarchy must be a mapping or iterable of label-like objects"
            ) from exc

    normalized: dict[int, int | None] = {}
    for raw_id, raw_parent in items:
        # ``LabelHierarchy.id_to_label_object`` maps each ID to a full Label object;
        # iterable label inputs do the same.  In that shape the object's *parent_id*,
        # rather than its own ``id``, is the value needed for this map.
        if isinstance(raw_parent, Mapping) or hasattr(raw_parent, "parent_id"):
            raw_parent = _value(raw_parent, "parent_id", "parent_label_id", default=_MISSING)
        elif hasattr(raw_parent, "id"):
            raw_parent = _value(raw_parent, "id", "label_id", default=_MISSING)
        label_id = _coerce_int(raw_id, "label ID")
        parent_id = None if raw_parent is None else _coerce_int(raw_parent, f"parent label ID for {label_id}")
        if label_id in normalized:
            if normalized[label_id] != parent_id:
                raise HierarchyValidationError(
                    f"label {label_id} has conflicting parent references",
                    code="conflicting_label_parent",
                )
            raise HierarchyValidationError(
                f"label {label_id} has duplicate parent references",
                code="duplicate_label_parent",
            )
        normalized[label_id] = parent_id

    if not normalized:
        raise HierarchyValidationError("label hierarchy cannot be empty")
    for label_id, parent_id in normalized.items():
        if parent_id is not None and parent_id not in normalized:
            raise HierarchyValidationError(
                f"label {label_id} references missing parent label {parent_id}"
            )
    _validate_label_cycles(normalized)
    return normalized


def _normalize_selected_label_ids(selected_label_ids: Iterable[int] | None) -> set[int] | None:
    if selected_label_ids is None:
        return None
    if isinstance(selected_label_ids, (str, bytes)):
        raise HierarchyValidationError("selected label IDs must be iterable integers")
    try:
        result: set[int] = set()
        for label_id in selected_label_ids:
            normalized_label_id = _coerce_int(label_id, "selected label ID")
            if normalized_label_id in result:
                raise HierarchyValidationError(
                    f"selected label ID {normalized_label_id} is duplicated",
                    code="duplicate_selected_label_id",
                )
            result.add(normalized_label_id)
    except TypeError as exc:
        raise HierarchyValidationError("selected label IDs must be iterable") from exc
    return result


def _validate_encoded_parent_cycles(parent_by_id: Mapping[int, int | None]) -> None:
    state: dict[int, int] = {}

    def visit(annotation_id: int) -> None:
        status = state.get(annotation_id, 0)
        if status == 1:
            raise HierarchyValidationError(
                f"cycle detected in encoded annotation parent references at annotation {annotation_id}",
                code="cycle_in_parent_references",
                details={"annotation_id": annotation_id},
            )
        if status == 2:
            return
        state[annotation_id] = 1
        parent_id = parent_by_id[annotation_id]
        if parent_id is not None:
            visit(parent_id)
        state[annotation_id] = 2

    for annotation_id in parent_by_id:
        visit(annotation_id)


def _validate_unique_node_ids(nodes: Sequence[HierarchyNode]) -> None:
    seen: set[int] = set()
    for node in nodes:
        if node.id in seen:
            raise HierarchyValidationError(f"duplicate contour ID {node.id}")
        seen.add(node.id)


def _validate_parent_references(
        nodes: Sequence[HierarchyNode],
        nodes_by_id: Mapping[int, HierarchyNode],
) -> None:
    for node in nodes:
        if node.parent_id is None:
            continue
        if node.parent_id not in nodes_by_id:
            raise HierarchyValidationError(
                f"contour {node.id} references missing parent contour {node.parent_id}",
                code="missing_parent_reference",
                details={"contour_id": node.id, "parent_id": node.parent_id},
            )
        if node.parent_id == node.id:
            raise HierarchyValidationError(
                f"contour {node.id} cannot be its own parent",
                code="cycle_in_parent_references",
                details={"contour_id": node.id},
            )


def _validate_parent_cycles(
        nodes: Sequence[HierarchyNode],
        nodes_by_id: Mapping[int, HierarchyNode],
) -> None:
    state: dict[int, int] = {}

    def visit(node_id: int) -> None:
        status = state.get(node_id, 0)
        if status == 1:
            raise HierarchyValidationError(
                f"cycle detected in contour parent references at contour {node_id}",
                code="cycle_in_parent_references",
                details={"contour_id": node_id},
            )
        if status == 2:
            return
        state[node_id] = 1
        parent_id = nodes_by_id[node_id].parent_id
        if parent_id in nodes_by_id:
            visit(parent_id)
        state[node_id] = 2

    for node in nodes:
        visit(node.id)


def _validate_label_cycles(label_parent_ids: Mapping[int, int | None]) -> None:
    state: dict[int, int] = {}

    def visit(label_id: int) -> None:
        status = state.get(label_id, 0)
        if status == 1:
            raise HierarchyValidationError(
                f"cycle detected in label parent references at label {label_id}",
                code="cycle_in_label_references",
                details={"label_id": label_id},
            )
        if status == 2:
            return
        state[label_id] = 1
        parent_id = label_parent_ids[label_id]
        if parent_id is not None:
            visit(parent_id)
        state[label_id] = 2

    for label_id in label_parent_ids:
        visit(label_id)


def _validate_selected_label_paths(
        selected_label_ids: set[int],
        label_parent_ids: Mapping[int, int | None],
) -> None:
    """Reject a selected ancestor/descendant pair with an omitted label level."""
    for descendant_id in selected_label_ids:
        path: list[int] = []
        parent_id = label_parent_ids[descendant_id]
        while parent_id is not None:
            path.append(parent_id)
            parent_id = label_parent_ids[parent_id]
        selected_ancestors = [label_id for label_id in path if label_id in selected_label_ids]
        for selected_ancestor in selected_ancestors:
            ancestor_index = path.index(selected_ancestor)
            between = path[:ancestor_index]
            omitted = [label_id for label_id in between if label_id not in selected_label_ids]
            if omitted:
                raise HierarchyValidationError(
                    "selected label scope skips intermediate label(s) "
                    f"{omitted} between ancestor label {selected_ancestor} and "
                    f"descendant label {descendant_id}"
                )


def _validate_selected_label_relationships(
        selected_nodes: Sequence[HierarchyNode],
        nodes_by_id: Mapping[int, HierarchyNode],
        label_parent_ids: Mapping[int, int | None] | None,
        selected_node_ids: set[int],
        selected_label_ids: set[int],
        containment_threshold: float,
) -> None:
    for node in selected_nodes:
        expected_parent_label = (
            None if label_parent_ids is None else label_parent_ids.get(node.label_id)
        )
        parent = nodes_by_id.get(node.parent_id) if node.parent_id is not None else None

        if parent is not None:
            if label_parent_ids is not None and expected_parent_label != parent.label_id:
                raise HierarchyValidationError(
                    f"contour {node.id} label {node.label_id} has parent contour {parent.id} "
                    f"with label {parent.label_id}; expected parent label {expected_parent_label}"
                )
            _validate_containment(node, parent, containment_threshold)
            continue

        if expected_parent_label is not None and expected_parent_label in selected_label_ids:
            if parent is None:
                raise HierarchyValidationError(
                    f"contour {node.id} label {node.label_id} is missing its selected "
                    f"parent label {expected_parent_label} contour"
                )
            raise HierarchyValidationError(
                f"contour {node.id} label {node.label_id} does not point to a selected "
                f"parent contour with label {expected_parent_label}"
            )

def _validate_containment(
        child: HierarchyNode,
        parent: HierarchyNode,
        threshold: float,
) -> None:
    child_area = int(child.mask.sum())
    if child_area <= 0:
        raise HierarchyValidationError(
            f"contour {child.id} has an empty mask",
            code="empty_contour_mask",
            details={"contour_id": child.id},
        )
    escaped_pixels = np.logical_and(child.mask, np.logical_not(parent.mask))
    escaped_count = int(escaped_pixels.sum())
    if escaped_count:
        raise HierarchyValidationError(
            f"child contour {child.id} is not exactly contained by parent {parent.id}: "
            f"{escaped_count} escaped pixels are outside the parent",
            code="child_outside_parent",
            details={
                "child_contour_id": child.id,
                "parent_contour_id": parent.id,
                "escaped_pixels": escaped_count,
            },
        )
    ratio = 1.0
    if ratio < threshold:
        raise HierarchyValidationError(
            f"child contour {child.id} is insufficiently contained by parent {parent.id}: "
            f"{ratio:.3f} of child pixels are inside the parent, minimum is {threshold:.3f}",
            code="insufficient_containment",
            details={
                "child_contour_id": child.id,
                "parent_contour_id": parent.id,
                "contained_fraction": ratio,
                "minimum_fraction": threshold,
            },
        )


def _validate_peer_overlap(
        selected_nodes: Sequence[HierarchyNode],
        nodes_by_id: Mapping[int, HierarchyNode],
        max_overlap_fraction: float,
) -> None:
    groups: dict[int | None, list[HierarchyNode]] = defaultdict(list)
    selected_ids = {node.id for node in selected_nodes}
    for node in selected_nodes:
        exported_parent_id = node.parent_id if node.parent_id in selected_ids else None
        groups[exported_parent_id].append(node)

    for peers in groups.values():
        for index, left in enumerate(peers):
            for right in peers[index + 1:]:
                intersection = int(np.logical_and(left.mask, right.mask).sum())
                if intersection == 0:
                    continue
                denominator = min(int(left.mask.sum()), int(right.mask.sum()))
                overlap_fraction = intersection / denominator if denominator else 1.0
                if overlap_fraction > max_overlap_fraction:
                    raise HierarchyValidationError(
                        "sibling/root contours overlap: "
                        f"contours {left.id} and {right.id} share {intersection} pixels "
                        f"({overlap_fraction:.3f} of the smaller mask)",
                        code="peer_overlap",
                        details={
                            "left_contour_id": left.id,
                            "right_contour_id": right.id,
                            "intersection_pixels": intersection,
                            "overlap_fraction": overlap_fraction,
                        },
                    )


def _descendants(node_id: int, children_by_parent: Mapping[int, Sequence[int]]) -> list[int]:
    result: list[int] = []
    stack = list(children_by_parent.get(node_id, ()))
    while stack:
        child_id = stack.pop()
        result.append(child_id)
        stack.extend(children_by_parent.get(child_id, ()))
    return result


def _order_hierarchy_nodes(nodes: Sequence[HierarchyNode]) -> list[HierarchyNode]:
    """Return a stable forest order with every parent before its children."""
    by_id = {node.id: node for node in nodes}
    depths: dict[int, int] = {}

    def depth(node_id: int) -> int:
        if node_id in depths:
            return depths[node_id]
        parent_id = by_id[node_id].parent_id
        if parent_id is None or parent_id not in by_id:
            depths[node_id] = 0
        else:
            depths[node_id] = depth(parent_id) + 1
        return depths[node_id]

    for node in nodes:
        depth(node.id)
    return sorted(nodes, key=lambda node: (depths[node.id], node.id))


def _assert_exclusive_residuals(
        selected_nodes: Sequence[HierarchyNode],
        residual_masks: Mapping[int, np.ndarray],
) -> None:
    for index, left in enumerate(selected_nodes):
        for right in selected_nodes[index + 1:]:
            if np.logical_and(residual_masks[left.id], residual_masks[right.id]).any():
                raise HierarchyValidationError(
                    f"exclusive residual masks overlap for contours {left.id} and {right.id}"
                )


def _build_payload(
        *,
        selected_nodes: Sequence[HierarchyNode],
        residual_masks: Mapping[int, np.ndarray],
        label_parent_ids: Mapping[int, int | None] | None,
        selected_label_ids: set[int],
        width: int,
        height: int,
        image_id: int,
) -> ExclusiveHierarchyPayload:
    annotations: list[dict[str, Any]] = []
    selected_node_ids = {node.id for node in selected_nodes}
    category_names: dict[int, str] = {}
    for node in selected_nodes:
        if node.label_name is not None:
            previous_name = category_names.get(node.label_id)
            if previous_name is not None and previous_name != node.label_name:
                raise HierarchyValidationError(
                    f"label {node.label_id} has conflicting category names "
                    f"{previous_name!r} and {node.label_name!r}",
                    code="conflicting_category_name",
                )
            category_names[node.label_id] = node.label_name
        raw_rle = coco_mask.encode(np.asfortranarray(residual_masks[node.id].astype(np.uint8)))
        rle = {
            "size": [height, width],
            "counts": raw_rle["counts"].decode("ascii")
            if isinstance(raw_rle["counts"], bytes)
            else raw_rle["counts"],
        }
        area = int(coco_mask.area(raw_rle | {"counts": raw_rle["counts"]}))
        bbox = [float(value) for value in coco_mask.toBbox(raw_rle).tolist()]
        parent_annotation_id = (
            node.parent_id if node.parent_id in selected_node_ids else None
        )
        annotation: dict[str, Any] = {
            "id": node.id,
            "category_id": node.label_id,
            "segmentation": rle,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "original_annotation_id": node.id,
            "original_parent_annotation_id": node.parent_id,
            "parent_annotation_id": parent_annotation_id,
            "parent_id": parent_annotation_id,
        }
        annotation["image_id"] = image_id
        annotations.append(annotation)

    categories = [
        {
            "id": label_id,
            "name": category_names.get(label_id, str(label_id)),
            "supercategory": "none",
        }
        for label_id in sorted({node.label_id for node in selected_nodes})
    ]
    sidecar_labels = {
        int(label_id): label_parent_ids[label_id]
        for label_id in sorted(label_parent_ids or {})
    }
    sidecar_annotations = [
        {
            "annotation_id": node.id,
            "original_annotation_id": node.id,
            "parent_annotation_id": (
                node.parent_id if node.parent_id in selected_node_ids else None
            ),
            "label_id": node.label_id,
        }
        for node in selected_nodes
    ]
    payload: ExclusiveHierarchyPayload = {
        "target_encoding": EXCLUSIVE_HIERARCHY_V1,
        "encoding": EXCLUSIVE_HIERARCHY_V1,
        "width": width,
        "height": height,
        "categories": categories,
        "annotations": annotations,
        "image_id": image_id,
        "images": [{"id": image_id, "width": width, "height": height}],
        "hierarchy": {
            "encoding": EXCLUSIVE_HIERARCHY_V1,
            "selected_label_ids": sorted(selected_label_ids),
            "label_parent_ids": sidecar_labels,
            "annotations": sidecar_annotations,
        },
    }
    return payload


def _safe_error_value(value: Any) -> Any:
    """Convert error detail values to small JSON-safe primitives."""
    if value is _MISSING:
        return "<missing>"
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else "<non-finite>"
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        converted = value.item()
        return (
            converted
            if not isinstance(converted, float) or np.isfinite(converted)
            else "<non-finite>"
        )
    if isinstance(value, Mapping):
        return {
            str(key): _safe_error_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_safe_error_value(item) for item in value]
    return f"<{type(value).__name__}>"


def _value(obj: Any, *names: str, default: Any = _MISSING) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _mapping_value(mapping: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _coerce_int(value: Any, field: str) -> int:
    if value is _MISSING or value is None or isinstance(value, bool):
        raise HierarchyValidationError(f"{field} must be an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise HierarchyValidationError(f"{field} must be an integer") from exc
    raise HierarchyValidationError(f"{field} must be an integer")


def _coerce_positive_int(value: Any, field: str) -> int:
    result = _coerce_int(value, field)
    if result <= 0:
        raise HierarchyValidationError(f"{field} must be positive")
    return result


def _validate_threshold(name: str, value: float) -> None:
    if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
    ):
        raise HierarchyValidationError(f"{name} must be a finite number in [0, 1]")
    if value < 0 or value > 1:
        raise HierarchyValidationError(f"{name} must be in [0, 1]")


from sqlalchemy.orm import Session
from app.database.contours import Contours
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.datasets import Datasets
from app.services.database_access.datasets import _native_image_dimensions


def _rasterize_training_contour(
        contour: Contours,
        width: int,
        height: int,
) -> tuple[np.ndarray, bool]:
    """Rasterize a stored contour, clipping crop-boundary overshoot for export.

    Imported geometric annotations such as circles may legitimately extend beyond a
    cropped image. Those pixels do not exist and are clipped by image viewers anyway;
    the strict pure encoder remains unchanged, while the explicit normalized training
    path records that clipping was required.
    """
    x = np.asarray(contour.x, dtype=np.float64)
    y = np.asarray(contour.y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise HierarchyValidationError(
            f"contour {contour.id} must have at least three paired x/y coordinates"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise HierarchyValidationError(f"contour {contour.id} has non-finite x/y coordinates")

    normalized = bool(np.max(np.abs(x)) <= 1.5 and np.max(np.abs(y)) <= 1.5)
    if normalized:
        clipped = bool(np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1))
        points = np.column_stack((np.clip(x, 0, 1) * width, np.clip(y, 0, 1) * height))
    else:
        clipped = bool(
            np.any(x < 0) or np.any(x > width) or np.any(y < 0) or np.any(y > height)
        )
        points = np.column_stack((np.clip(x, 0, width), np.clip(y, 0, height)))

    polygon = np.rint(points).astype(np.int32)
    raster = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(raster, [polygon.reshape((-1, 1, 2))], 1)
    return raster.astype(bool), clipped


def _partition_peer_masks_by_centroid(
        peers: Sequence[HierarchyNode],
        masks_by_id: dict[int, np.ndarray],
) -> None:
    """Give every shared peer pixel to the nearest instance centroid.

    Contour IDs provide a stable tie-break: peers are processed in ascending ID order
    and an equal-distance claimant never replaces the existing owner. The operation is
    export-only; callers pass copies of the stored contour masks.
    """
    if len(peers) < 2:
        return

    shape = masks_by_id[peers[0].id].shape
    owner = np.full(shape, -1, dtype=np.int64)
    best_distance = np.full(shape, np.inf, dtype=np.float64)

    for node in sorted(peers, key=lambda item: item.id):
        mask = masks_by_id[node.id]
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
        current_distances = best_distance[ys, xs]
        current_owners = owner[ys, xs]
        take = (current_owners < 0) | (distances < current_distances)
        if np.any(take):
            take_y = ys[take]
            take_x = xs[take]
            owner[take_y, take_x] = node.id
            best_distance[take_y, take_x] = distances[take]

    for node in peers:
        masks_by_id[node.id] = owner == node.id


def _prepare_training_image_nodes(
        *,
        raw_nodes: Sequence[Mapping[str, Any]],
        width: int,
        height: int,
        image_id: int,
        file_name: str,
        selected_label_ids: Sequence[int],
) -> tuple[list[HierarchyNode], dict[str, Any]]:
    """Analyse one image and build deterministic export-only normalized masks."""
    nodes = [
        _normalize_node(raw_node, height, width, image_id, index)
        for index, raw_node in enumerate(raw_nodes)
    ]
    _validate_unique_node_ids(nodes)
    nodes_by_id = {node.id: node for node in nodes}
    _validate_parent_references(nodes, nodes_by_id)
    _validate_parent_cycles(nodes, nodes_by_id)

    selected_ids = set(selected_label_ids)
    selected_nodes = [node for node in nodes if node.label_id in selected_ids]
    selected_node_ids = {node.id for node in selected_nodes}
    boundary_clipped_contour_ids = sorted(
        node.id for node, raw_node in zip(nodes, raw_nodes)
        if bool(_value(raw_node, "boundary_clipped", default=False))
        and node.id in selected_node_ids
    )

    adjustable_children: list[dict[str, Any]] = []
    unsafe_children: list[dict[str, Any]] = []
    for child in selected_nodes:
        if child.parent_id is None:
            continue
        parent = nodes_by_id.get(child.parent_id)
        if parent is None:
            continue
        child_area = int(child.mask.sum())
        if child_area <= 0:
            continue
        inside_pixels = int(np.logical_and(child.mask, parent.mask).sum())
        if inside_pixels == child_area:
            continue
        contained_fraction = inside_pixels / child_area
        item = {
            "child_contour_id": child.id,
            "parent_contour_id": parent.id,
            "contained_fraction": round(contained_fraction, 6),
            "escaped_pixels": child_area - inside_pixels,
        }
        if contained_fraction >= MIN_AUTO_PARENT_CONTAINMENT:
            adjustable_children.append(item)
        else:
            unsafe_children.append(item)

    peer_groups: dict[int | None, list[HierarchyNode]] = defaultdict(list)
    for node in selected_nodes:
        exported_parent_id = node.parent_id if node.parent_id in selected_node_ids else None
        peer_groups[exported_parent_id].append(node)

    overlapping_pairs: list[dict[str, Any]] = []
    for peers in peer_groups.values():
        ordered_peers = sorted(peers, key=lambda item: item.id)
        for index, left in enumerate(ordered_peers):
            for right in ordered_peers[index + 1:]:
                intersection = int(np.logical_and(left.mask, right.mask).sum())
                if intersection:
                    overlapping_pairs.append({
                        "left_contour_id": left.id,
                        "right_contour_id": right.id,
                        "intersection_pixels": intersection,
                    })

    adjustable_children.sort(key=lambda item: int(item["child_contour_id"]))
    unsafe_children.sort(key=lambda item: int(item["child_contour_id"]))
    overlapping_pairs.sort(key=lambda item: (
        int(item["left_contour_id"]), int(item["right_contour_id"])
    ))

    issue = {
        "image_id": image_id,
        "file_name": file_name,
        "adjustable_child_count": len(adjustable_children) if not unsafe_children else 0,
        "excluded_child_count": len(unsafe_children),
        "sibling_overlap_pair_count": len(overlapping_pairs),
        "adjustable_children": adjustable_children,
        "unsafe_children": unsafe_children,
        "sibling_overlaps": overlapping_pairs,
        "boundary_clipped_contour_ids": boundary_clipped_contour_ids,
        "empty_after_normalization": [],
        "action": "exclude" if unsafe_children else "normalize",
    }
    issue["requires_normalization"] = bool(
        adjustable_children or unsafe_children or overlapping_pairs
        or boundary_clipped_contour_ids
    )

    if unsafe_children or not issue["requires_normalization"]:
        return nodes, issue

    masks_by_id = {node.id: node.mask.copy() for node in nodes}
    for peers in peer_groups.values():
        _partition_peer_masks_by_centroid(peers, masks_by_id)

    empty_after_partition = sorted(
        node.id for node in selected_nodes if not np.any(masks_by_id[node.id])
    )
    if empty_after_partition:
        issue["empty_after_normalization"] = empty_after_partition
        issue["action"] = "exclude"
        return nodes, issue

    depths: dict[int, int] = {}

    def depth(node_id: int) -> int:
        if node_id in depths:
            return depths[node_id]
        parent_id = nodes_by_id[node_id].parent_id
        depths[node_id] = (
            0 if parent_id is None or parent_id not in nodes_by_id
            else depth(parent_id) + 1
        )
        return depths[node_id]

    for node in nodes:
        depth(node.id)

    # Only selected nodes and their ancestor chain participate. An unselected child
    # must never enlarge a selected parent merely because it exists in the database.
    propagating_ids = set(selected_node_ids)
    for node in selected_nodes:
        parent_id = node.parent_id
        while parent_id in nodes_by_id:
            propagating_ids.add(parent_id)
            parent_id = nodes_by_id[parent_id].parent_id

    # Propagate normalized child masks into ancestors from leaves to roots. The
    # 50% decision above intentionally uses the original stored masks; propagation
    # only makes relationships already accepted by that rule exact for export.
    for child in sorted(nodes, key=lambda item: (depths[item.id], item.id), reverse=True):
        if child.id not in propagating_ids:
            continue
        if child.parent_id in nodes_by_id:
            masks_by_id[child.parent_id] |= masks_by_id[child.id]

    selected_children: dict[int, list[int]] = defaultdict(list)
    for node in selected_nodes:
        if node.parent_id in selected_node_ids:
            selected_children[node.parent_id].append(node.id)

    empty_residual_ids: list[int] = []
    for node in selected_nodes:
        descendant_ids = _descendants(node.id, selected_children)
        residual = masks_by_id[node.id].copy()
        for descendant_id in descendant_ids:
            residual &= ~masks_by_id[descendant_id]
        if int(residual.sum()) < 1:
            empty_residual_ids.append(node.id)
    if empty_residual_ids:
        issue["empty_after_normalization"] = sorted(empty_residual_ids)
        issue["action"] = "exclude"
        return nodes, issue

    normalized_nodes = [replace(node, mask=masks_by_id[node.id]) for node in nodes]
    return normalized_nodes, issue


def _normalization_summary(issues: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    affected = sorted(
        (dict(issue) for issue in issues if issue.get("requires_normalization")),
        key=lambda issue: int(issue["image_id"]),
    )
    retained = [issue for issue in affected if issue.get("action") == "normalize"]
    excluded = [issue for issue in affected if issue.get("action") == "exclude"]
    return {
        "policy": NORMALIZE_HIERARCHY_POLICY,
        "minimum_parent_containment": MIN_AUTO_PARENT_CONTAINMENT,
        "affected_image_count": len(affected),
        "adjusted_image_count": len(retained),
        "adjusted_child_count": sum(int(issue["adjustable_child_count"]) for issue in retained),
        "excluded_image_count": len(excluded),
        "excluded_child_count": sum(int(issue["excluded_child_count"]) for issue in excluded),
        "empty_after_normalization_count": sum(
            len(issue.get("empty_after_normalization", [])) for issue in excluded
        ),
        "sibling_overlap_pair_count": sum(
            int(issue["sibling_overlap_pair_count"]) for issue in retained
        ),
        "boundary_clipped_contour_count": sum(
            len(issue.get("boundary_clipped_contour_ids", [])) for issue in retained
        ),
        "affected_images": affected,
        "source_annotations_modified": False,
    }

def export_training_hierarchy_dataset(
        dataset_id: int,
        db: Session,
        selected_label_ids: list[int] | None = None,
        min_residual_area: int = 1,
        write_to_disk: bool = False,
        output_file_path: str | None = None,
        hierarchy_conflict_policy: str = STRICT_HIERARCHY_POLICY,
) -> dict[str, Any]:
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found.", "dataset_id": dataset_id}

    labels = db.query(Labels).filter_by(dataset_id=dataset_id).all()
    label_parent_ids = {label.id: label.parent_id for label in labels}
    labels_by_id = {label.id: label for label in labels}
    
    if selected_label_ids is None:
        selected_label_ids = list(label_parent_ids.keys())

    missing = set(selected_label_ids) - set(label_parent_ids.keys())
    if missing:
        return {"success": False, "message": f"Selected labels not in dataset: {sorted(missing)}"}
    if hierarchy_conflict_policy not in {
        STRICT_HIERARCHY_POLICY,
        NORMALIZE_HIERARCHY_POLICY,
    }:
        return {
            "success": False,
            "message": f"Unknown hierarchy conflict policy: {hierarchy_conflict_policy}.",
            "error_code": "invalid_hierarchy_conflict_policy",
        }

    query = (
        db.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .filter(Masks.fully_annotated == True)
        .filter(Contours.reviewed_by.any())
    )
    rows = query.all()
    
    contours_by_image = defaultdict(list)
    image_by_id = {}
    
    for contour, image in rows:
        contours_by_image[image.id].append(contour)
        image_by_id[image.id] = image
        
    all_images = []
    all_annotations = []
    all_categories = []
    seen_category_ids = set()
    all_sidecar_annotations = []
    exported_image_ids = set()
    prepared_images: list[tuple[int, list[HierarchyNode], list[int], int, int]] = []
    normalization_issues: list[dict[str, Any]] = []

    for image_id in sorted(contours_by_image):
        contours = sorted(contours_by_image[image_id], key=lambda contour: contour.id)
        image_present_labels = {c.label_id for c in contours}
        image_selected_labels = [lid for lid in selected_label_ids if lid in image_present_labels]
        if not image_selected_labels:
            continue

        native_width, native_height = _native_image_dimensions(image_by_id[image_id])
        
        raw_nodes = []
        try:
            for contour in contours:
                contour_mask, boundary_clipped = _rasterize_training_contour(
                    contour, native_width, native_height
                )
                raw_nodes.append({
                    "id": contour.id,
                    "label_id": contour.label_id,
                    "parent_id": contour.parent_id,
                    "mask": contour_mask,
                    "boundary_clipped": boundary_clipped,
                    "label_name": labels_by_id[contour.label_id].name,
                })
        except HierarchyValidationError as exc:
            return {
                "success": False,
                "message": exc.message,
                "error_code": exc.code,
                "details": exc.details | {"image_id": image_id},
            }

        try:
            prepared_nodes, issue = _prepare_training_image_nodes(
                raw_nodes=raw_nodes,
                width=native_width,
                height=native_height,
                image_id=image_id,
                file_name=str(image_by_id[image_id].file_name),
                selected_label_ids=image_selected_labels,
            )
        except HierarchyValidationError as exc:
            return {
                "success": False,
                "message": exc.message,
                "error_code": exc.code,
                "details": exc.details,
            }

        normalization_issues.append(issue)
        if hierarchy_conflict_policy == NORMALIZE_HIERARCHY_POLICY and issue["action"] == "exclude":
            continue
        prepared_images.append((
            image_id,
            prepared_nodes if hierarchy_conflict_policy == NORMALIZE_HIERARCHY_POLICY else [
                _normalize_node(raw_node, native_height, native_width, image_id, index)
                for index, raw_node in enumerate(raw_nodes)
            ],
            image_selected_labels,
            native_width,
            native_height,
        ))

    normalization_summary = _normalization_summary(normalization_issues)
    if (
        hierarchy_conflict_policy == STRICT_HIERARCHY_POLICY
        and normalization_summary["affected_image_count"]
    ):
        return {
            "success": False,
            "message": (
                "Some hierarchy annotations need export-only normalization before training."
            ),
            "error_code": "hierarchy_normalization_required",
            "details": {"summary": normalization_summary},
        }

    for image_id, nodes, image_selected_labels, native_width, native_height in prepared_images:
        try:
            payload = encode_exclusive_hierarchy_v1(
                contours=nodes,
                width=native_width,
                height=native_height,
                label_parent_ids=label_parent_ids,
                selected_label_ids=image_selected_labels,
                image_id=image_id,
                min_residual_area=min_residual_area,
            )
            # The AI-side COCO loader resolves each image relative to the image
            # folder using ``images[].file_name``.  The pure encoder cannot infer
            # a database image name, so attach it at this integration boundary.
            payload["images"][0]["file_name"] = str(image_by_id[image_id].file_name)
        except HierarchyValidationError as exc:
            return {
                "success": False,
                "message": exc.message,
                "error_code": exc.code,
                "details": exc.details | {"image_id": image_id},
            }

        all_images.extend(payload["images"])
        all_annotations.extend(payload["annotations"])
        all_sidecar_annotations.extend(payload["hierarchy"]["annotations"])
        
        for category in payload["categories"]:
            if category["id"] not in seen_category_ids:
                all_categories.append(category)
                seen_category_ids.add(category["id"])
                
        exported_image_ids.add(image_id)
                
    annotated_category_ids = {ann["category_id"] for ann in all_annotations}
    missing_labels = [lbl_id for lbl_id in selected_label_ids if lbl_id not in annotated_category_ids]
    if missing_labels:
        return {
            "success": False,
            "message": f"Labels {missing_labels} have zero reviewed annotations in dataset {dataset_id}.",
            "error_code": "missing_label_annotations",
            "details": {"missing_label_ids": missing_labels},
        }

    if not exported_image_ids:
        return {
            "success": False,
            "message": "No eligible contours found.",
            "error_code": "empty_export",
        }


    all_images.sort(key=lambda img: img["id"])
    all_annotations.sort(key=lambda ann: ann["id"])
    all_sidecar_annotations.sort(key=lambda ann: ann["annotation_id"])
    all_categories.sort(key=lambda cat: cat["id"])

    sidecar_labels = {
        int(label_id): label_parent_ids[label_id]
        for label_id in sorted(label_parent_ids)
    }
    
    coco_payload = {
        "target_encoding": EXCLUSIVE_HIERARCHY_V1,
        "encoding": EXCLUSIVE_HIERARCHY_V1,
        "images": all_images,
        "annotations": all_annotations,
        "categories": all_categories,
        "hierarchy": {
            "encoding": EXCLUSIVE_HIERARCHY_V1,
            "selected_label_ids": sorted(selected_label_ids),
            "label_parent_ids": sidecar_labels,
            "annotations": all_sidecar_annotations,
        },
    }
    
    result = {
        "success": True,
        "message": "Hierarchy export created.",
        "dataset_id": dataset_id,
        "coco_payload": coco_payload,
        "image_ids": exported_image_ids,
        "num_images": len(all_images),
        "num_annotations": len(all_annotations),
        "num_categories": len(all_categories),
        "normalization_summary": normalization_summary,
    }

    if write_to_disk:
        import os
        import json
        import uuid
        if output_file_path is None:
            exports_dir = os.path.join(str(dataset.folder_path), "training_exports")
            output_file_path = os.path.join(exports_dir, f"export_{dataset_id}_{uuid.uuid4()}.json")
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                tmp_file_path = f"{output_file_path}.tmp"
                with open(tmp_file_path, "w", encoding="utf-8") as fp:
                    json.dump(coco_payload, fp, indent=2)
                    fp.flush()
                    os.fsync(fp.fileno())
                os.replace(tmp_file_path, output_file_path)
                result["output_file_path"] = output_file_path
            except OSError as e:
                return {
                    "success": False,
                    "message": "Failed to write artifact to disk.",
                    "error_code": "disk_error",
                    "details": {"error": str(e)}
                }

    return result
