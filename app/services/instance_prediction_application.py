from logging import getLogger
from typing import Literal

from sqlalchemy.orm import Session
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.model_info import InstanceSegmentationModelInfo

from app.database.contours import Contours, save_contour_tree
from app.services.database_access import masks as masks_db
from app.services.database_access import labels as labels_db
from app.services.annotation_session.operations import assign_hierarchy_parents, filter_exemplar_overlaps, _iou, DUPLICATE_IOU_THRESHOLD
from app.services.database_access.contours import invalidate_metrics_for_new_contours
from app.services.embedding_lifecycle import enqueue_embed_contours

logger = getLogger(__name__)

async def apply_instance_segmentation_predictions(
    db: Session,
    mask_id: int,
    dataset_id: int,
    author_username: str,
    predictions: list[Contour],
    apply_mode: Literal["patch", "replace"],
    model_info: InstanceSegmentationModelInfo,
) -> dict:
    """Apply instance segmentation predictions transactionally."""
    if apply_mode not in ("patch", "replace"):
        raise ValueError(f"Invalid apply_mode: {apply_mode}")

    if model_info.dataset_id is not None and model_info.dataset_id != dataset_id:
        raise ValueError(f"Model dataset {model_info.dataset_id} does not match target dataset {dataset_id}")

    model_labels = set(model_info.label_ids)

    def validate_node_recursively(node: Contour):
        if node.label_id not in model_labels:
            raise ValueError(f"Prediction label {node.label_id} is outside model's declared scope {model_labels}")
        for child in node.children:
            validate_node_recursively(child)

    for p in predictions:
        validate_node_recursively(p)

    
    # 1. Fetch current hierarchies
    mask_hierarchy = await masks_db.get_contour_hierarchy_of_mask(mask_id, db)
    label_hierarchy = await labels_db.get_label_hierarchy(dataset_id, db)

    # 2. Validate predictions against model scope

    added_count = 0
    suppressed_count = 0
    replaced_count = 0
    unparented_count = 0

    to_insert: list[Contour] = []
    
    if apply_mode == "replace":
        if model_labels:
            # Delete only contours in the model's declared label scope
            contours_to_delete = db.query(Contours.id).filter(
                Contours.mask_id == mask_id,
                Contours.label_id.in_(model_labels)
            ).all()
            to_delete_ids = [c.id for c in contours_to_delete]
            
            if to_delete_ids:
                replaced_count = len(to_delete_ids)
                from sqlalchemy import or_
                # Unparent children that are NOT being deleted to prevent CASCADE from removing unrelated labels
                db.query(Contours).filter(
                    Contours.parent_id.in_(to_delete_ids),
                    or_(~Contours.label_id.in_(model_labels), Contours.label_id.is_(None))
                ).update({"parent_id": None}, synchronize_session='fetch')
                
                db.query(Contours).filter(
                    Contours.id.in_(to_delete_ids)
                ).delete(synchronize_session='fetch')
                
                # Refresh mask_hierarchy since we deleted stuff
                mask_hierarchy = await masks_db.get_contour_hierarchy_of_mask(mask_id, db)
        
        # In replace mode, we just add all predictions (they don't suppress each other against the DB since DB is cleared for those labels)
        to_insert = predictions

    elif apply_mode == "patch":
        # Keep all existing contours. Suppress new same-label predictions above IoU threshold.
        # We check IoU against existing contours of the same label.
        def duplicate_of(prediction: Contour) -> Contour | None:
            for existing in mask_hierarchy.label_id_to_contours.get(prediction.label_id, []):
                if _iou(prediction, existing) >= DUPLICATE_IOU_THRESHOLD:
                    return existing
            return None

        def keep_novel_descendants(prediction: Contour, existing_parent: Contour) -> None:
            """Keep children that add information when their predicted parent is a duplicate.

            Patch mode must not drop a newly predicted polyp merely because its coral
            parent overlaps an existing coral.  A duplicate descendant is suppressed
            recursively; a novel descendant is attached to the matching existing
            parent and saved with its own subtree.
            """
            nonlocal suppressed_count

            for child in prediction.children:
                matching_child = duplicate_of(child)
                if matching_child is not None:
                    suppressed_count += 1
                    keep_novel_descendants(child, matching_child)
                    continue

                child.parent_id = existing_parent.id
                to_insert.append(child)

        for pred in predictions:
            matching_existing = duplicate_of(pred)
            if matching_existing is not None:
                suppressed_count += 1
                keep_novel_descendants(pred, matching_existing)
            else:
                to_insert.append(pred)

    # 3. Handle hierarchy reconstruction persistence
    # We must persist parents before children if predictions have explicit prediction-parent IDs (from reconstruction).
    # Since our predictions are flat list of `Contour` objects (from `run_instance_segmentation`), 
    # we need to build a map of prediction IDs to DB IDs if they have prediction-level parent_id.
    
    # Wait, the predictions already come with `parent_id` set to the prediction ID of the parent?
    # No, `run_instance_segmentation` returns flat instances. 
    # But wait, Phase 3 "Inference reconstruction tasks" says:
    # "Return enough relation information for the backend to persist parents before children and assign final database IDs."
    
    # Let's see what `to_insert` has. We partition into those without parent_id and those with.
    # We map prediction object ID to DB ID.
    
    pred_id_to_db_id = {}
    
    def count_schema_nodes(c: Contour) -> int:
        return 1 + sum(count_schema_nodes(child) for child in c.children)

    def insert_contour_and_children(pred_contour: Contour, db_parent_id: int | None):
        nonlocal added_count
        pred_contour.parent_id = db_parent_id
        # Use save_contour_tree to insert
        db_contour = save_contour_tree(db, pred_contour, mask_id, parent_id=db_parent_id, author_username=author_username, invalidate_metrics=False)
        added_count += count_schema_nodes(pred_contour)
        
        # If the contour had a temp id, map it
        if pred_contour.id is not None:
            pred_id_to_db_id[pred_contour.id] = db_contour.id
            
        return db_contour

    # Group by temp parent_id
    # First insert all roots (parent_id is None or not in the prediction set)
    # Wait, `run_instance_segmentation` might just return a list.
    
    # We will first try to fallback with `assign_hierarchy_parents` for any prediction without an explicit relation.
    # Group by label and fallback.
    # Actually, we can run `assign_hierarchy_parents` on the predictions that don't have a parent.
    unparented_preds = [p for p in to_insert if p.parent_id is None]
    
    # Group unparented by label to run assign_hierarchy_parents
    labels_present = set(p.label_id for p in unparented_preds if p.label_id is not None)
    for lbl in labels_present:
        group = [p for p in unparented_preds if p.label_id == lbl]
        assign_hierarchy_parents(group, mask_hierarchy, label_hierarchy, lbl)
    
    # Now some unparented might have been assigned a parent (from the DB).
    for p in unparented_preds:
        if p.parent_id is None:
            unparented_count += 1
            
    # Now we insert everything. If there's an internal prediction hierarchy, we insert parents first.
    # Build a tree of predictions
    pred_children_map = {}
    roots = []
    
    for p in to_insert:
        if p.parent_id is not None and any(other.id == p.parent_id for other in to_insert):
            # parent is another prediction
            pred_children_map.setdefault(p.parent_id, []).append(p)
        else:
            # parent is None, or refers to a DB contour (from assign_hierarchy_parents)
            roots.append(p)
            
    created_db_contours = []

    def recursive_insert(p: Contour, resolved_parent_id: int | None):
        db_c = insert_contour_and_children(p, resolved_parent_id)
        created_db_contours.append(db_c)
        for child in pred_children_map.get(p.id, []):
            recursive_insert(child, db_c.id)

    for r in roots:
        recursive_insert(r, r.parent_id)
        
    # Finally, metric invalidation for newly inserted contours
    if created_db_contours:
        invalidate_metrics_for_new_contours(db, created_db_contours)

    # 4. Mark mask as incomplete if we replaced or added things? 
    # Yes, delete_all_contours does it, but we should do it explicitly here if anything changed.
    if added_count > 0 or replaced_count > 0:
        mask = db.query(masks_db.Masks).filter_by(id=mask_id).first()
        mask.fully_annotated = False

    return {
        "apply_mode": apply_mode,
        "added_count": added_count,
        "suppressed_count": suppressed_count,
        "replaced_count": replaced_count,
        "unparented_count": unparented_count,
    }
