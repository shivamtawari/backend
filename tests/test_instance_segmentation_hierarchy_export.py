"""Focused tests for the pure ``exclusive_hierarchy_v1`` encoder."""

import json

import numpy as np
import pytest

from app.services.instance_segmentation_training import (
    EXCLUSIVE_HIERARCHY_V1,
    HierarchyValidationError,
    decode_coco_rle,
    encode_exclusive_hierarchy_v1,
    reconstruct_hierarchy_masks,
)


def _rect(height, width, top, left, bottom, right):
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _node(contour_id, label_id, mask, parent_id=None, image_id=1):
    node = {
        "id": contour_id,
        "label_id": label_id,
        "parent_id": parent_id,
        "mask": mask,
    }
    if image_id is not None:
        node["image_id"] = image_id
    return node


def _annotations_by_id(payload):
    return {annotation["id"]: annotation for annotation in payload["annotations"]}


def test_two_level_masks_are_rle_disjoint_and_round_trip():
    parent = _rect(16, 16, 1, 1, 15, 15)
    child = _rect(16, 16, 5, 5, 10, 10)

    payload = encode_exclusive_hierarchy_v1(
        [_node(101, 10, parent, image_id=7), _node(102, 20, child, parent_id=101, image_id=7)],
        width=16,
        height=16,
        label_parent_ids={10: None, 20: 10},
        image_id=7,
    )

    assert payload["target_encoding"] == EXCLUSIVE_HIERARCHY_V1
    assert len(payload["annotations"]) == 2  # the parent was not leaf-filtered away
    annotations = _annotations_by_id(payload)
    parent_residual = decode_coco_rle(annotations[101]["segmentation"])
    child_residual = decode_coco_rle(annotations[102]["segmentation"])
    assert not np.logical_and(parent_residual, child_residual).any()
    assert annotations[101]["area"] == int(parent.sum() - child.sum())
    assert annotations[102]["area"] == int(child.sum())
    assert annotations[102]["parent_annotation_id"] == 101
    assert annotations[101]["bbox"] == pytest.approx([1.0, 1.0, 14.0, 14.0])
    assert annotations[101]["image_id"] == 7
    assert isinstance(annotations[101]["segmentation"]["counts"], str)
    assert payload["images"] == [{"id": 7, "width": 16, "height": 16}]
    assert [category["id"] for category in payload["categories"]] == [10, 20]

    reconstructed = reconstruct_hierarchy_masks(payload)
    assert np.array_equal(reconstructed[101], parent)
    assert np.array_equal(reconstructed[102], child)


def test_non_square_payload_json_round_trip_preserves_single_image_and_parent_first_order():
    parent = _rect(6, 10, 1, 1, 5, 9)
    child = _rect(6, 10, 2, 3, 4, 7)

    payload = encode_exclusive_hierarchy_v1(
        [
            _node(22, 20, child, parent_id=11, image_id=42),
            _node(11, 10, parent, image_id=42),
        ],
        image_size=(6, 10),
        label_parent_ids={10: None, 20: 10},
    )
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["images"] == [{"id": 42, "width": 10, "height": 6}]
    assert [annotation["id"] for annotation in round_tripped["annotations"]] == [11, 22]
    assert all(annotation["image_id"] == 42 for annotation in round_tripped["annotations"])
    assert all(annotation["segmentation"]["size"] == [6, 10]
               for annotation in round_tripped["annotations"])
    reconstructed = reconstruct_hierarchy_masks(round_tripped)
    assert np.array_equal(reconstructed[11], parent)
    assert np.array_equal(reconstructed[22], child)


def test_encoding_is_stable_when_input_nodes_are_permuted():
    grandparent = _rect(12, 12, 1, 1, 11, 11)
    parent = _rect(12, 12, 3, 3, 9, 9)
    child = _rect(12, 12, 5, 5, 7, 7)
    nodes = [
        _node(30, 30, child, parent_id=20, image_id=9),
        _node(10, 10, grandparent, image_id=9),
        _node(20, 20, parent, parent_id=10, image_id=9),
    ]

    first = encode_exclusive_hierarchy_v1(
        nodes,
        width=12,
        height=12,
        label_parent_ids={10: None, 20: 10, 30: 20},
    )
    second = encode_exclusive_hierarchy_v1(
        list(reversed(nodes)),
        width=12,
        height=12,
        label_parent_ids={30: 20, 10: None, 20: 10},
    )

    assert first == second
    assert [annotation["id"] for annotation in first["annotations"]] == [10, 20, 30]
    assert first["hierarchy"]["label_parent_ids"] == {10: None, 20: 10, 30: 20}


def test_child_with_one_escaped_boundary_pixel_is_rejected_for_exact_reversibility():
    parent = _rect(10, 10, 1, 1, 9, 9)
    child = _rect(10, 10, 2, 2, 8, 8)
    child[0, 0] = True

    with pytest.raises(HierarchyValidationError, match="escaped pixels") as error:
        encode_exclusive_hierarchy_v1(
            [_node(1, 10, parent), _node(2, 20, child, parent_id=1)],
            width=10,
            height=10,
            label_parent_ids={10: None, 20: 10},
        )

    assert error.value.code == "child_outside_parent"
    assert error.value.as_dict()["details"]["escaped_pixels"] == 1


def test_three_level_masks_subtract_all_exported_descendants_and_round_trip():
    grandparent = _rect(20, 20, 1, 1, 19, 19)
    parent = _rect(20, 20, 4, 4, 16, 16)
    child = _rect(20, 20, 8, 8, 12, 12)

    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 100, grandparent),
            _node(2, 200, parent, parent_id=1),
            _node(3, 300, child, parent_id=2),
        ],
        image_size=(20, 20),
        label_parent_ids={100: None, 200: 100, 300: 200},
    )
    annotations = _annotations_by_id(payload)
    residuals = {
        contour_id: decode_coco_rle(annotation["segmentation"])
        for contour_id, annotation in annotations.items()
    }

    for left_id, left in residuals.items():
        for right_id, right in residuals.items():
            if left_id < right_id:
                assert not np.logical_and(left, right).any()

    assert annotations[1]["area"] == int(grandparent.sum() - parent.sum())
    assert annotations[2]["area"] == int(parent.sum() - child.sum())
    assert annotations[3]["area"] == int(child.sum())
    reconstructed = reconstruct_hierarchy_masks(payload)
    assert np.array_equal(reconstructed[1], grandparent)
    assert np.array_equal(reconstructed[2], parent)
    assert np.array_equal(reconstructed[3], child)


def test_flat_annotations_are_unchanged_by_exclusive_encoding():
    first = _rect(9, 12, 1, 1, 4, 4)
    second = _rect(9, 12, 5, 7, 8, 11)

    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 11, first), _node(2, 22, second)],
        width=12,
        height=9,
        label_parent_ids={11: None, 22: None},
    )
    annotations = _annotations_by_id(payload)
    assert np.array_equal(decode_coco_rle(annotations[1]["segmentation"]), first)
    assert np.array_equal(decode_coco_rle(annotations[2]["segmentation"]), second)
    assert annotations[1]["parent_annotation_id"] is None
    assert annotations[2]["parent_annotation_id"] is None


def test_selected_nodes_are_preserved_and_scoped_parent_is_detached():
    parent = _rect(12, 12, 1, 1, 11, 11)
    child = _rect(12, 12, 3, 3, 8, 8)

    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 10, parent), _node(2, 20, child, parent_id=1)],
        width=12,
        height=12,
        label_parent_ids={10: None, 20: 10},
        selected_label_ids=[20],
    )

    assert [annotation["id"] for annotation in payload["annotations"]] == [2]
    assert payload["annotations"][0]["parent_annotation_id"] is None
    assert payload["hierarchy"]["selected_label_ids"] == [20]


def test_single_label_selection_is_valid_without_label_parent_metadata():
    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 20, _rect(12, 12, 2, 2, 6, 6))],
        width=12,
        height=12,
        selected_label_ids=[20],
    )

    assert payload["hierarchy"]["label_parent_ids"] == {}
    assert payload["images"] == [{"id": 1, "width": 12, "height": 12}]
    reconstructed = reconstruct_hierarchy_masks(payload)
    assert np.array_equal(reconstructed[1], _rect(12, 12, 2, 2, 6, 6))


def test_multi_label_selection_requires_label_parent_metadata():
    with pytest.raises(HierarchyValidationError, match="label_parent_ids is required") as error:
        encode_exclusive_hierarchy_v1(
            [
                _node(1, 10, _rect(12, 12, 1, 1, 4, 4)),
                _node(2, 20, _rect(12, 12, 6, 6, 9, 9)),
            ],
            width=12,
            height=12,
            selected_label_ids=[10, 20],
        )

    assert error.value.code == "label_parent_metadata_required"


def test_duplicate_selected_label_ids_are_rejected_before_set_normalization():
    with pytest.raises(HierarchyValidationError, match="selected label ID 10 is duplicated") as error:
        encode_exclusive_hierarchy_v1(
            [_node(1, 10, _rect(12, 12, 1, 1, 4, 4))],
            width=12,
            height=12,
            label_parent_ids={10: None},
            selected_label_ids=[10, 10],
        )

    assert error.value.code == "duplicate_selected_label_id"


def test_duplicate_label_parent_entries_are_rejected_even_when_parent_matches():
    with pytest.raises(HierarchyValidationError, match="duplicate parent references") as error:
        encode_exclusive_hierarchy_v1(
            [_node(1, 10, _rect(12, 12, 1, 1, 4, 4))],
            width=12,
            height=12,
            label_parent_ids=[
                {"id": 10, "parent_id": None},
                {"id": 10, "parent_id": None},
            ],
        )

    assert error.value.code == "duplicate_label_parent"


def test_conflicting_label_parent_entries_remain_rejected():
    with pytest.raises(HierarchyValidationError, match="conflicting parent references") as error:
        encode_exclusive_hierarchy_v1(
            [_node(1, 10, _rect(12, 12, 1, 1, 4, 4))],
            width=12,
            height=12,
            label_parent_ids=[
                {"id": 10, "parent_id": None},
                {"id": 10, "parent_id": 1},
                {"id": 1, "parent_id": None},
            ],
        )

    assert error.value.code == "conflicting_label_parent"


def test_explicit_selected_label_absent_from_image_is_rejected():
    with pytest.raises(HierarchyValidationError, match="absent from this image") as error:
        encode_exclusive_hierarchy_v1(
            [_node(1, 10, _rect(12, 12, 1, 1, 4, 4))],
            width=12,
            height=12,
            label_parent_ids={10: None, 20: None},
            selected_label_ids=[10, 20],
        )

    assert error.value.code == "selected_labels_missing_from_image"
    assert error.value.details["missing_label_ids"] == [20]


def test_all_explicitly_selected_labels_absent_from_image_get_actionable_error():
    with pytest.raises(HierarchyValidationError, match="absent from this image") as error:
        encode_exclusive_hierarchy_v1(
            [_node(1, 10, _rect(12, 12, 1, 1, 4, 4))],
            width=12,
            height=12,
            label_parent_ids={10: None, 20: None},
            selected_label_ids=[20],
        )

    assert error.value.code == "selected_labels_missing_from_image"


def test_missing_parent_reference_is_not_ignored_for_a_root_label_scope():
    with pytest.raises(HierarchyValidationError, match="missing parent"):
        encode_exclusive_hierarchy_v1(
            [_node(1, 10, _rect(12, 12, 1, 1, 5, 5), parent_id=999)],
            width=12,
            height=12,
            label_parent_ids={10: None},
            selected_label_ids=[10],
        )


def test_selected_scope_rejects_missing_parent_even_when_parent_label_is_outside_scope():
    with pytest.raises(HierarchyValidationError, match="missing parent") as error:
        encode_exclusive_hierarchy_v1(
            [_node(2, 20, _rect(12, 12, 3, 3, 8, 8), parent_id=999)],
            width=12,
            height=12,
            label_parent_ids={10: None, 20: 10},
            selected_label_ids=[20],
        )

    assert error.value.code == "missing_parent_reference"


@pytest.mark.parametrize(
    ("nodes", "label_parent_ids", "match"),
    [
        (
            [_node(1, 1, _rect(8, 8, 1, 1, 6, 6), parent_id=999)],
            {1: None},
            "missing parent",
        ),
        (
            [
                _node(1, 1, _rect(8, 8, 1, 1, 6, 6), parent_id=2),
                _node(2, 1, _rect(8, 8, 1, 1, 6, 6), parent_id=1),
            ],
            {1: None},
            "cycle",
        ),
        (
            [
                _node(1, 1, _rect(8, 8, 1, 1, 4, 4)),
                _node(2, 2, _rect(8, 8, 3, 3, 7, 7), parent_id=1),
            ],
            {1: None, 2: 1},
            "contained",
        ),
        (
            [
                _node(1, 1, _rect(8, 8, 1, 1, 7, 7)),
                _node(2, 3, _rect(8, 8, 2, 2, 5, 5), parent_id=1),
            ],
            {1: None, 2: 1, 3: 2},
            "skips intermediate",
        ),
        (
            [
                _node(1, 1, _rect(8, 8, 1, 1, 5, 5)),
                _node(2, 1, _rect(8, 8, 4, 4, 7, 7)),
            ],
            {1: None},
            "overlap",
        ),
        (
            [
                _node(1, 1, _rect(8, 8, 1, 1, 6, 6)),
                _node(2, 2, _rect(8, 8, 1, 1, 6, 6), parent_id=1),
            ],
            {1: None, 2: 1},
            "residual",
        ),
    ],
)
def test_invalid_hierarchy_input_is_rejected(nodes, label_parent_ids, match):
    with pytest.raises(HierarchyValidationError, match=match):
        encode_exclusive_hierarchy_v1(
            nodes,
            width=8,
            height=8,
            label_parent_ids=label_parent_ids,
        )


def test_skipped_intermediate_selected_label_is_rejected():
    nodes = [
        _node(1, 1, _rect(12, 12, 1, 1, 11, 11)),
        _node(3, 3, _rect(12, 12, 4, 4, 8, 8), parent_id=1),
    ]
    with pytest.raises(HierarchyValidationError, match="skips intermediate"):
        encode_exclusive_hierarchy_v1(
            nodes,
            width=12,
            height=12,
            label_parent_ids={1: None, 2: 1, 3: 2},
            selected_label_ids=[1, 3],
        )


def test_sibling_overlap_is_rejected_even_when_parent_is_valid():
    parent = _rect(8, 8, 1, 1, 7, 7)
    first_child = _rect(8, 8, 2, 2, 5, 5)
    second_child = _rect(8, 8, 4, 4, 7, 7)

    with pytest.raises(HierarchyValidationError, match="sibling"):
        encode_exclusive_hierarchy_v1(
            [
                _node(1, 1, parent),
                _node(2, 2, first_child, parent_id=1),
                _node(3, 3, second_child, parent_id=1),
            ],
            width=8,
            height=8,
            label_parent_ids={1: None, 2: 1, 3: 1},
        )


def test_nonzero_peer_overlap_threshold_is_rejected_by_the_exclusive_contract():
    with pytest.raises(HierarchyValidationError, match="must be exactly 0.0") as error:
        encode_exclusive_hierarchy_v1(
            [_node(1, 1, _rect(8, 8, 1, 1, 4, 4))],
            width=8,
            height=8,
            label_parent_ids={1: None},
            max_peer_overlap_fraction=0.1,
        )

    assert error.value.code == "nonzero_peer_overlap_threshold"


def test_contour_parent_label_must_match_label_hierarchy():
    with pytest.raises(HierarchyValidationError, match="expected parent label"):
        encode_exclusive_hierarchy_v1(
            [
                _node(1, 1, _rect(12, 12, 1, 1, 11, 11)),
                _node(3, 3, _rect(12, 12, 4, 4, 8, 8), parent_id=1),
            ],
            width=12,
            height=12,
            label_parent_ids={1: None, 2: 1, 3: 2},
            selected_label_ids=[1, 2, 3],
        )


def test_configured_minimum_residual_area_is_enforced():
    parent = _rect(8, 8, 1, 1, 6, 6)
    child = _rect(8, 8, 2, 2, 5, 5)
    with pytest.raises(HierarchyValidationError, match="minimum"):
        encode_exclusive_hierarchy_v1(
            [_node(1, 1, parent), _node(2, 2, child, parent_id=1)],
            width=8,
            height=8,
            label_parent_ids={1: None, 2: 1},
            min_residual_area=20,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_size": (0, 8)}, "positive"),
        ({"image_size": (8, 8), "image_shape": (8, 8)}, "only one"),
    ],
)
def test_image_dimensions_are_positive_and_unambiguous(kwargs, match):
    with pytest.raises(HierarchyValidationError, match=match):
        encode_exclusive_hierarchy_v1(
            [_node(1, 1, _rect(8, 8, 1, 1, 3, 3))],
            label_parent_ids={1: None},
            **kwargs,
        )


@pytest.mark.parametrize(
    ("nodes", "match"),
    [
        ([_node(1, 1, np.zeros((8, 8), dtype=bool))], "empty"),
        ([_node(1, 1, np.zeros((7, 8), dtype=bool))], "dimensions"),
    ],
)
def test_invalid_mask_dimensions_are_rejected(nodes, match):
    with pytest.raises(HierarchyValidationError, match=match):
        encode_exclusive_hierarchy_v1(
            nodes,
            width=8,
            height=8,
            label_parent_ids={1: None},
        )


def test_reconstruction_rejects_mixed_rle_dimensions():
    first = encode_exclusive_hierarchy_v1(
        [_node(1, 1, _rect(8, 8, 1, 1, 3, 3))],
        width=8,
        height=8,
        label_parent_ids={1: None},
    )
    second = encode_exclusive_hierarchy_v1(
        [_node(2, 2, _rect(4, 4, 1, 1, 3, 3))],
        width=4,
        height=4,
        label_parent_ids={2: None},
    )
    first["annotations"].extend(second["annotations"])

    with pytest.raises(HierarchyValidationError, match="same mask dimensions"):
        reconstruct_hierarchy_masks(first)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update({"target_encoding": "standard"}), "encoding"),
        (lambda payload: payload["images"].append(payload["images"][0].copy()), "exactly one image"),
        (lambda payload: payload["annotations"][0].pop("image_id"), "image ID"),
        (lambda payload: payload["annotations"][0].update({"bbox": [-1, 0, 1, 1]}), "outside image bounds"),
    ],
)
def test_reconstruction_rejects_malformed_or_version_mismatched_payloads(mutate, match):
    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 1, _rect(8, 10, 1, 2, 4, 7))],
        width=10,
        height=8,
        label_parent_ids={1: None},
    )
    mutate(payload)

    with pytest.raises(HierarchyValidationError, match=match):
        reconstruct_hierarchy_masks(payload)


def test_missing_and_mixed_contour_image_ids_are_rejected():
    with pytest.raises(HierarchyValidationError, match="multiple image IDs"):
        encode_exclusive_hierarchy_v1(
            [
                _node(1, 1, _rect(8, 8, 1, 1, 3, 3), image_id=10),
                _node(2, 2, _rect(8, 8, 5, 5, 7, 7), image_id=11),
            ],
            width=8,
            height=8,
            label_parent_ids={1: None, 2: None},
        )

    with pytest.raises(HierarchyValidationError, match="image_id is required"):
        encode_exclusive_hierarchy_v1(
            [_node(1, 1, _rect(8, 8, 1, 1, 3, 3), image_id=None)],
            width=8,
            height=8,
            label_parent_ids={1: None},
        )


def test_duplicate_contour_ids_are_rejected():
    with pytest.raises(HierarchyValidationError, match="duplicate contour ID"):
        encode_exclusive_hierarchy_v1(
            [
                _node(1, 1, _rect(8, 8, 1, 1, 3, 3)),
                _node(1, 2, _rect(8, 8, 5, 5, 7, 7)),
            ],
            width=8,
            height=8,
            label_parent_ids={1: None, 2: None},
        )


def test_conflicting_category_names_for_one_label_are_rejected():
    first = _node(1, 1, _rect(8, 8, 1, 1, 3, 3))
    second = _node(2, 1, _rect(8, 8, 5, 5, 7, 7))
    first["label_name"] = "first"
    second["label_name"] = "second"

    with pytest.raises(HierarchyValidationError, match="conflicting category names"):
        encode_exclusive_hierarchy_v1(
            [first, second],
            width=8,
            height=8,
            label_parent_ids={1: None},
        )


@pytest.mark.parametrize("value", [2, -1, np.nan, 0.5])
def test_nonbinary_masks_are_rejected_without_coercion(value):
    mask = np.zeros((8, 8), dtype=float if isinstance(value, float) else int)
    mask[2, 2] = value
    with pytest.raises(HierarchyValidationError, match="0/1|non-finite|negative"):
        encode_exclusive_hierarchy_v1(
            [_node(1, 1, mask)],
            width=8,
            height=8,
            label_parent_ids={1: None},
        )


def test_reconstruction_rejects_wrong_but_in_bounds_bbox():
    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 1, _rect(8, 10, 1, 2, 4, 7))],
        width=10,
        height=8,
        label_parent_ids={1: None},
    )
    payload["annotations"][0]["bbox"] = [3.0, 1.0, 4.0, 3.0]

    with pytest.raises(HierarchyValidationError, match="bbox does not match"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_malformed_compressed_counts():
    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 1, _rect(8, 10, 1, 2, 4, 7))],
        width=10,
        height=8,
        label_parent_ids={1: None},
    )
    payload["annotations"][0]["segmentation"]["counts"] = "not-a-valid-rle"

    with pytest.raises(HierarchyValidationError, match="RLE"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_area_tampering():
    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 1, _rect(8, 10, 1, 2, 4, 7))],
        width=10,
        height=8,
        label_parent_ids={1: None},
    )
    payload["annotations"][0]["area"] += 1

    with pytest.raises(HierarchyValidationError, match="area does not match"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_duplicate_categories():
    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 1, _rect(8, 8, 1, 1, 3, 3)),
            _node(2, 2, _rect(8, 8, 5, 5, 7, 7)),
        ],
        width=8,
        height=8,
        label_parent_ids={1: None, 2: None},
    )
    payload["categories"].append(payload["categories"][0].copy())

    with pytest.raises(HierarchyValidationError, match="duplicate COCO category ID"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_duplicate_sidecar_annotations():
    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 1, _rect(8, 8, 1, 1, 3, 3)),
            _node(2, 2, _rect(8, 8, 5, 5, 7, 7)),
        ],
        width=8,
        height=8,
        label_parent_ids={1: None, 2: None},
    )
    payload["hierarchy"]["annotations"].append(
        payload["hierarchy"]["annotations"][0].copy()
    )

    with pytest.raises(HierarchyValidationError, match="duplicate sidecar annotation ID"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_noncanonical_but_decodable_compressed_counts():
    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 1, _rect(8, 10, 1, 2, 4, 7))],
        width=10,
        height=8,
        label_parent_ids={1: None},
    )
    counts = payload["annotations"][0]["segmentation"]["counts"]
    # pycocotools decodes this alternate compressed spelling to the same binary mask,
    # but the encoder contract requires the canonical compressed representation.
    payload["annotations"][0]["segmentation"]["counts"] = counts[:1] + "p" + counts[2:]

    with pytest.raises(HierarchyValidationError, match="non-canonical"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_missing_encoded_parent():
    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 1, _rect(8, 8, 1, 1, 7, 7)),
            _node(2, 2, _rect(8, 8, 2, 2, 4, 4), parent_id=1),
        ],
        width=8,
        height=8,
        label_parent_ids={1: None, 2: 1},
    )
    child = payload["annotations"][1]
    child["parent_annotation_id"] = 999
    child["parent_id"] = 999
    child["original_parent_annotation_id"] = 999

    with pytest.raises(HierarchyValidationError, match="missing parent"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_encoded_parent_cycle():
    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 1, _rect(8, 8, 1, 1, 3, 3)),
            _node(2, 2, _rect(8, 8, 5, 5, 7, 7)),
        ],
        width=8,
        height=8,
        label_parent_ids={1: None, 2: None},
    )
    first, second = payload["annotations"]
    first["parent_annotation_id"] = second["id"]
    first["parent_id"] = second["id"]
    first["original_parent_annotation_id"] = second["original_annotation_id"]
    second["parent_annotation_id"] = first["id"]
    second["parent_id"] = first["id"]
    second["original_parent_annotation_id"] = first["original_annotation_id"]

    with pytest.raises(HierarchyValidationError, match="cycle detected"):
        reconstruct_hierarchy_masks(payload)


@pytest.mark.parametrize("mutation", [
    lambda payload: payload["hierarchy"]["annotations"][0].pop("original_annotation_id"),
    lambda payload: payload["hierarchy"]["annotations"][0].update({"unexpected": True}),
    lambda payload: payload["hierarchy"]["annotations"].append(
        payload["hierarchy"]["annotations"][0].copy()
    ),
])
def test_reconstruction_rejects_malformed_sidecar_entries(mutation):
    payload = encode_exclusive_hierarchy_v1(
        [_node(1, 1, _rect(8, 8, 1, 1, 3, 3))],
        width=8,
        height=8,
        label_parent_ids={1: None},
    )
    mutation(payload)

    with pytest.raises(HierarchyValidationError, match="sidecar"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_duplicate_original_annotation_ids():
    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 1, _rect(8, 8, 1, 1, 3, 3)),
            _node(2, 2, _rect(8, 8, 5, 5, 7, 7)),
        ],
        width=8,
        height=8,
        label_parent_ids={1: None, 2: None},
    )
    payload["annotations"][1]["original_annotation_id"] = 1

    with pytest.raises(HierarchyValidationError, match="original annotation ID"):
        reconstruct_hierarchy_masks(payload)


def test_reconstruction_rejects_mismatched_parent_path_and_label_parent():
    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 10, _rect(8, 8, 1, 1, 7, 7)),
            _node(2, 20, _rect(8, 8, 2, 2, 4, 4), parent_id=1),
        ],
        width=8,
        height=8,
        label_parent_ids={10: None, 20: 10},
    )
    payload["hierarchy"]["annotations"][1]["parent_annotation_id"] = None

    with pytest.raises(HierarchyValidationError, match="parent"):
        reconstruct_hierarchy_masks(payload)

    payload = encode_exclusive_hierarchy_v1(
        [
            _node(1, 10, _rect(8, 8, 1, 1, 7, 7)),
            _node(2, 20, _rect(8, 8, 2, 2, 4, 4), parent_id=1),
        ],
        width=8,
        height=8,
        label_parent_ids={10: None, 20: 10},
    )
    payload["hierarchy"]["label_parent_ids"][20] = None

    with pytest.raises(HierarchyValidationError, match="label"):
        reconstruct_hierarchy_masks(payload)


import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import database
from app.database.users import Users
from app.database.datasets import Datasets
from app.database.dataset_members import DatasetMembers
from app.database.labels import Labels
from app.database.images import Images
from app.database.masks import Masks
from app.database.contours import Contours
from app.schemas.permissions import DatasetRole
from app.services.instance_segmentation_training import export_training_hierarchy_dataset

@pytest.fixture
def db_ctx(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'export-hierarchy.db'}",
        connect_args={"check_same_thread": False},
    )
    database.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    owner = Users(username="owner", hashed_password="x")
    db.add(owner)
    
    dataset = Datasets(name="test-ds", description="", dataset_type="image",
                       folder_path="/tmp/test-ds", created_by=owner.username)
    db.add(dataset)
    db.flush()
    
    parent_label = Labels(dataset_id=dataset.id, name="parent", value=1)
    db.add(parent_label)
    db.flush()
    
    child_label = Labels(dataset_id=dataset.id, name="child", value=2, parent_id=parent_label.id)
    db.add(child_label)
    db.flush()

    image = Images(
        dataset_id=dataset.id, file_name="img.png", file_path="/tmp/img.png",
        thumbnail_file_path="/tmp/img-thumb.png", width=10, height=10,
        color_mode="RGB",
    )
    db.add(image)
    db.flush()
    
    mask = Masks(image_id=image.id, fully_annotated=True, file_path="/tmp/mask.png")
    db.add(mask)
    db.flush()

    parent_contour = Contours(
        mask_id=mask.id, label_id=parent_label.id, added_by="manual",
        author_username=owner.username, confidence_score=1.0, area=1.0,
        perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9],
    )
    parent_contour.reviewed_by.append(owner)
    db.add(parent_contour)
    db.flush()

    child_contour = Contours(
        mask_id=mask.id, label_id=child_label.id, parent_id=parent_contour.id, added_by="manual",
        author_username=owner.username, confidence_score=1.0, area=1.0,
        perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.4, 0.6, 0.6, 0.4], y=[0.4, 0.4, 0.6, 0.6],
    )
    child_contour.reviewed_by.append(owner)
    db.add(child_contour)
    db.commit()

    yield {
        "db": db,
        "dataset_id": dataset.id,
        "labels": {"parent": parent_label.id, "child": child_label.id},
    }
    db.close()
    engine.dispose()

def test_export_training_hierarchy_dataset_success(db_ctx):
    result = export_training_hierarchy_dataset(
        dataset_id=db_ctx["dataset_id"],
        db=db_ctx["db"],
        selected_label_ids=[db_ctx["labels"]["parent"], db_ctx["labels"]["child"]]
    )
    assert result["success"] is True
    assert result["num_images"] == 1
    assert result["num_annotations"] == 2
    
    payload = result["coco_payload"]
    assert payload["target_encoding"] == "exclusive_hierarchy_v1"
    assert len(payload["annotations"]) == 2
    assert payload["images"] == [{
        "id": db_ctx["db"].query(Images).filter_by(dataset_id=db_ctx["dataset_id"]).one().id,
        "width": 10,
        "height": 10,
        "file_name": "img.png",
    }]


def test_export_training_hierarchy_dataset_comprehensive(db_ctx):
    db = db_ctx["db"]
    dataset_id = db_ctx["dataset_id"]
    owner = db.query(Users).filter_by(username="owner").first()
    
    parent_label = db_ctx["labels"]["parent"]
    child_label = db_ctx["labels"]["child"]
    
    # Add a third label
    other_label = Labels(dataset_id=dataset_id, name="other", value=3)
    db.add(other_label)
    db.flush()
    
    # Image 1 (already exists from fixture) has fully_annotated=True mask, parent/child reviewed contours.
    
    # Image 2: fully_annotated=True, has only 'other' label contours.
    image2 = Images(dataset_id=dataset_id, file_name="img2.png", file_path="/tmp/img2.png", thumbnail_file_path="/tmp/thumb.png", width=10, height=10, color_mode="RGB")
    db.add(image2)
    db.flush()
    mask2 = Masks(image_id=image2.id, fully_annotated=True, file_path="/tmp/mask2.png")
    db.add(mask2)
    db.flush()
    other_contour = Contours(
        mask_id=mask2.id, label_id=other_label.id, added_by="manual", author_username=owner.username,
        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9]
    )
    other_contour.reviewed_by.append(owner)
    db.add(other_contour)
    
    # Image 3: fully_annotated=False (should be skipped entirely)
    image3 = Images(dataset_id=dataset_id, file_name="img3.png", file_path="/tmp/img3.png", thumbnail_file_path="/tmp/thumb.png", width=10, height=10, color_mode="RGB")
    db.add(image3)
    db.flush()
    mask3 = Masks(image_id=image3.id, fully_annotated=False, file_path="/tmp/mask3.png")
    db.add(mask3)
    db.flush()
    parent_contour3 = Contours(
        mask_id=mask3.id, label_id=parent_label, added_by="manual", author_username=owner.username,
        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9]
    )
    parent_contour3.reviewed_by.append(owner)
    db.add(parent_contour3)
    
    # Image 4: fully_annotated=True, but contour is unreviewed (should be skipped)
    image4 = Images(dataset_id=dataset_id, file_name="img4.png", file_path="/tmp/img4.png", thumbnail_file_path="/tmp/thumb.png", width=10, height=10, color_mode="RGB")
    db.add(image4)
    db.flush()
    mask4 = Masks(image_id=image4.id, fully_annotated=True, file_path="/tmp/mask4.png")
    db.add(mask4)
    db.flush()
    unreviewed_contour = Contours(
        mask_id=mask4.id, label_id=parent_label, added_by="manual", author_username=owner.username,
        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9]
    )
    db.add(unreviewed_contour)
    
    # Image 5: fully_annotated=True, valid reviewed child_label contour
    image5 = Images(dataset_id=dataset_id, file_name="img5.png", file_path="/tmp/img5.png", thumbnail_file_path="/tmp/thumb.png", width=10, height=10, color_mode="RGB")
    db.add(image5)
    db.flush()
    mask5 = Masks(image_id=image5.id, fully_annotated=True, file_path="/tmp/mask5.png")
    db.add(mask5)
    db.flush()
    child_contour5 = Contours(
        mask_id=mask5.id, label_id=child_label, added_by="manual", author_username=owner.username,
        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9]
    )
    child_contour5.reviewed_by.append(owner)
    db.add(child_contour5)
    
    db.commit()

    # Test 1: Selected label filtering & skipping logic
    result = export_training_hierarchy_dataset(
        dataset_id=dataset_id,
        db=db,
        selected_label_ids=[parent_label, other_label.id]
    )
    assert result["success"] is True
    assert result["num_images"] == 2
    assert result["image_ids"] == {1, image2.id}
    
    payload = result["coco_payload"]
    assert [img["id"] for img in payload["images"]] == sorted([1, image2.id])
    assert len(payload["annotations"]) == 2

    # Test 2: Empty dataset validation (missing labels)
    result_empty = export_training_hierarchy_dataset(
        dataset_id=dataset_id,
        db=db,
        selected_label_ids=[9999]
    )
    assert result_empty["success"] is False
    assert "Selected labels not in dataset" in result_empty["message"]

    # Test 3: Valid labels but no eligible contours
    empty_label = Labels(dataset_id=dataset_id, name="empty", value=4)
    db.add(empty_label)
    db.commit()
    
    result_no_contours = export_training_hierarchy_dataset(
        dataset_id=dataset_id,
        db=db,
        selected_label_ids=[empty_label.id]
    )
    assert result_no_contours["success"] is False
    assert result_no_contours["error_code"] == "empty_export"
