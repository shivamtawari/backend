import asyncio
import json
import os
from logging import getLogger
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from iquana_toolbox.schemas.user import User
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette import status as http_status

from app.database import get_session
from app.database.contours import Contours
from app.database.images import Images
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.ai_services.instance_segmentation import InstanceSegmentationService
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import labels as labels_db
from app.services.instance_segmentation_training import export_training_hierarchy_dataset
from app.services.model_registry import MODEL_REGISTRY, list_available_models
from app.services.permissions import ensure_permission, require

logger = getLogger(__name__)
router = APIRouter(prefix="/instance_segmentation", tags=["instance_segmentation"])
service = InstanceSegmentationService()

# Default fine-tuning model. Only one instance-segmentation model exists for now.
DEFAULT_MODEL_REGISTRY_KEY = "mask2former"

# Must match instance-segmentation-service/app/tasks.py:TRAINING_EXPERIMENT. The
# worker tags each training run with ``celery_task_id`` so we can map a task id to
# its MLflow run.
TRAINING_EXPERIMENT = "instance-segmentation-training"

# Map MLflow run statuses to the coarse state the frontend renders.
_STATE_BY_MLFLOW_STATUS = {
    "FINISHED": "SUCCESS",
    "FAILED": "FAILED",
    "KILLED": "CANCELLED",
}

_MLFLOW_STATUS_BY_CELERY_STATE = {
    "SUCCESS": "FINISHED",
    "FAILURE": "FAILED",
    "REVOKED": "KILLED",
}
_TERMINAL_STATES = {"SUCCESS", "FAILED", "CANCELLED", "TIMED_OUT"}


def _ai_service_error_detail(exc: httpx.HTTPStatusError) -> Any:
    """Extract the AI service's actual error body instead of httpx's status-only summary."""
    try:
        body = exc.response.json()
    except ValueError:
        return exc.response.text or str(exc)
    return body.get("detail", body) if isinstance(body, dict) else body


class StartTrainingBody(BaseModel):
    """Training configuration sent by the Model Training page."""
    dataset_id: int = Field(..., description="Dataset to train on.")
    label_ids: list[int] = Field(
        default_factory=list,
        description="Labels (classes) to train on. Empty means all labels in the dataset hierarchy (multiclass).",
    )
    model_registry_key: str = Field(DEFAULT_MODEL_REGISTRY_KEY, description="Base model to fine-tune.")
    # Model-declared hyperparameters (keys match the model's training_parameters),
    # forwarded as-is to the training request.
    hyper_parameter: dict = Field(default_factory=dict, description="Hyperparameter overrides keyed by model param key.")
    # Optional human-readable label for the run (e.g. "Cells-FineTuned-v1").  Stored
    # as an MLflow tag so it appears in the run history and can be searched.  Kept
    # optional with no default so the worker's run_name still reads as a useful
    # auto-generated fallback when the field is absent.
    model_run_name: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Optional human-readable name/alias for this training run.",
        pattern=r"^[\w\-\s]{1,80}$",
    )


@router.get("/models")
async def get_models(user: User = Depends(get_current_user)):
    """Retrieve available instance segmentation models directly from MLflow."""
    return await asyncio.to_thread(list_available_models, "instance-segmentation")


@router.get("/training/label-annotation-counts")
async def get_label_annotation_counts(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.AI_TRAIN)),
):
    """Return the number of reviewed, fully annotated annotations per label.

    Used by the training UI to show a pre-flight annotation count beside each
    class selector and disable "Start Training" when no reviewed annotations exist.
    A contour is considered training-ready when at least one user has reviewed it
    and its mask is marked fully annotated, matching the default COCO export.
    """
    from sqlalchemy import func

    from app.database.masks import Masks

    # Contours → Masks → Images to reach dataset_id. These filters match the
    # default COCO export: only fully annotated masks and reviewed contours.
    rows = (
        db.query(Contours.label_id, func.count(Contours.id).label("count"))
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .filter(Masks.fully_annotated.is_(True))
        .filter(Contours.reviewed_by.any())
        .group_by(Contours.label_id)
        .all()
    )
    counts = {row.label_id: row.count for row in rows}
    return {"success": True, "reviewed_annotation_counts": counts}


@router.post("/training/start")
async def start_training(
        body: StartTrainingBody,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user),
):
    """Start training an instance segmentation model on a dataset.

    Exports the dataset's annotations to a COCO file on the shared data volume,
    then dispatches a training job to the instance-segmentation service (which
    delegates to a Celery worker). Returns the task id, which is also the MLflow
    run id used to poll progress.
    """
    # The dataset id only exists after the body is parsed, so this is the
    # imperative form of the require() dependency used elsewhere.
    ensure_permission(user, body.dataset_id, Permission.AI_TRAIN)

    dataset = await datasets_db.get_dataset(body.dataset_id, db=db)
    if not dataset:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    # The COCO file_name is a basename, so the worker needs the directory that
    # actually holds the image files. Derive it from a stored image path rather than
    # assuming it equals the dataset root (images live under e.g. <root>/images/).
    sample_image = db.query(Images).filter_by(dataset_id=body.dataset_id).first()
    if sample_image is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST,
                            detail="The dataset has no images to train on.")
    image_folder_path = os.path.dirname(str(sample_image.file_path))

    # Resolve the labels to train on. Empty selection -> every label in the dataset
    # hierarchy (multiclass, "predict everything"); the single-class case is just a
    # one-element selection.
    hierarchy = await labels_db.get_label_hierarchy(body.dataset_id, db=db)
    if body.label_ids:
        missing = [lid for lid in body.label_ids if lid not in hierarchy.id_to_label_object]
        if missing:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail=f"Labels not found in dataset: {missing}.")
        labels = [hierarchy.id_to_label_object[lid] for lid in body.label_ids]
    else:
        labels = list(hierarchy.id_to_label_object.values())
    if not labels:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST,
                            detail="The dataset has no labels to train on.")

    # Use the exclusive hierarchy encoder to produce mutually exclusive RLE masks
    # for all requested labels. The worker reads this sidecar-enriched payload
    # from the shared volume.
    # Release the read lock held by this route's session so the background
    # thread's session doesn't deadlock against it (critical for SQLite tests).
    db.commit()

    def _export_job(engine, dataset_id, label_ids):
        from sqlalchemy.orm import Session
        with Session(engine) as thread_db:
            return export_training_hierarchy_dataset(
                dataset_id=dataset_id,
                db=thread_db,
                selected_label_ids=label_ids,
                write_to_disk=True,
            )

    export = await run_in_threadpool(
        _export_job,
        engine=db.get_bind(),
        dataset_id=body.dataset_id,
        label_ids=[l.id for l in labels],
    )
    if not export.get("success"):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail={
                "message": export.get("message", "Failed to export annotations."),
                "error_code": export.get("error_code", "export_failed"),
                "details": export.get("details", {}),
            }
        )
    if export.get("num_annotations", 0) == 0:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST,
                            detail="Dataset has no reviewed annotations to train on.")

    request = InstanceSegmentationTrainingRequest(
        dataset_id=body.dataset_id,
        image_folder_path=image_folder_path,
        model_registry_key=body.model_registry_key,
        user_id=user.username,
        labels=labels,
        annotation_file_url=export["output_file_path"],
        hyper_parameter=dict(body.hyper_parameter),
    )

    try:
        result = await service.start_training(request, model_run_name=body.model_run_name)
    except httpx.HTTPStatusError as exc:
        logger.exception("AI service rejected the training request.")
        raise HTTPException(status_code=http_status.HTTP_502_BAD_GATEWAY,
                            detail=_ai_service_error_detail(exc))
    except Exception as exc:
        logger.exception("Failed to start instance segmentation training.")
        raise HTTPException(status_code=http_status.HTTP_502_BAD_GATEWAY,
                            detail=f"Could not start training: {exc}")

    return {"success": True, "message": "Training started.", "task_id": result.get("task_id")}


def _parse_label_ids(raw) -> list[int]:
    """Parse the stringified ``label_ids`` tag (e.g. "[1, 2]") back to a list."""
    if not raw:
        return []
    try:
        import ast
        parsed = ast.literal_eval(raw)
        return [int(v) for v in parsed]
    except (ValueError, SyntaxError, TypeError):
        return []


def _enrich_job_with_mlflow(job: dict) -> dict:
    """Enrich a durable AI job snapshot with MLflow metrics and parameters."""
    run_id = job.get("mlflow_run_id")
    client = MODEL_REGISTRY.client
    
    loss_history = []
    training_parameters = {}
    label_ids = []
    
    if run_id:
        try:
            run = client.get_run(run_id)
            try:
                loss_metric = client.get_metric_history(run_id, "loss")
                loss_history = [{"epoch": int(m.step), "value": m.value} for m in sorted(loss_metric, key=lambda m: m.step)]
            except Exception:
                pass
                
            training_parameters = {
                k: v for k, v in run.data.params.items()
                if k not in {"dataset_id", "selected_database_label_ids"}
            }
            
            label_ids = _parse_label_ids(run.data.tags.get("label_ids"))
        except Exception as e:
            from logging import getLogger
            getLogger(__name__).warning("Could not read MLflow run %s: %s", run_id, e)
            
    epoch = job.get("epoch")
    if epoch is None:
        epoch = loss_history[-1]["epoch"] if loss_history else 0

    # Ensure state matches legacy UI expectations
    state = job.get("state", "PROGRESS")
    if state == "QUEUED":
        state = "starting"
    elif state == "RUNNING":
        state = "PROGRESS"
    elif state == "REGISTERING":
        state = "PROGRESS"
    elif state == "SUCCEEDED":
        state = "SUCCESS"
    elif state == "CANCEL_REQUESTED":
        state = "PROGRESS"
    elif state == "CANCELLED":
        state = "CANCELLED"
    elif state == "FAILED":
        state = "FAILED"
    elif state == "TIMED_OUT":
        state = "TIMED_OUT"
        
    error_dict = job.get("error") or {}
    return {
        "task_id": job.get("task_id"),
        "run_id": run_id,
        "state": state,
        "epoch": epoch,
        "total_epochs": job.get("total_epochs"),
        "loss": loss_history,
        "label_ids": label_ids,
        "run_name": job.get("run_name"),
        "training_parameters": training_parameters,
        "start_time": job.get("started_at"),
        "end_time": job.get("finished_at"),
        "error_code": error_dict.get("code"),
        "error_message": error_dict.get("message"),
    }


async def _read_training_snapshot(task_id: str) -> dict:
    """Read a progress snapshot for a Celery ``task_id`` from the durable AI store."""
    try:
        job = await service.get_training_task_state(task_id)
    except Exception as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 404:
            raise HTTPException(status_code=404, detail="Training job not found")
        raise HTTPException(status_code=502, detail="Durable AI service outage")
        
    return await asyncio.to_thread(_enrich_job_with_mlflow, job)


async def _read_run_snapshot(run_id: str) -> dict:
    """Read a progress snapshot for a specific MLflow ``run_id`` (e.g., past run)."""
    client = MODEL_REGISTRY.client
    run = await asyncio.to_thread(client.get_run, run_id)
    task_id = run.data.tags.get("celery_task_id")
    
    job = None
    if task_id:
        try:
            job = await service.get_training_task_state(task_id)
        except Exception:
            pass
            
    if not job:
        # Fallback for old runs that don't exist in the new durable store
        job = {
            "task_id": task_id,
            "mlflow_run_id": run_id,
            "state": _STATE_BY_MLFLOW_STATUS.get(run.info.status, "PROGRESS"),
            "epoch": run.data.metrics.get("epoch", 0),
            "total_epochs": run.data.params.get("epochs"),
            "run_name": run.data.tags.get("run_name"),
        }
        
    return await asyncio.to_thread(_enrich_job_with_mlflow, job)


async def _list_training_runs(dataset_id: int) -> list[dict]:
    """Return snapshots for all durable training jobs associated with a dataset."""
    jobs = await service.list_training_jobs(dataset_id, limit=100)

    async def snapshot(job):
        return await asyncio.to_thread(_enrich_job_with_mlflow, job)

    return await asyncio.gather(*(snapshot(job) for job in jobs))


@router.get("/training/runs")
async def list_training_runs(dataset_id: int,
                             user: AuthenticatedUser = Depends(require(Permission.AI_TRAIN))):
    """List past + active training runs for a dataset (for the run-history list)."""
    return {"success": True, "runs": await _list_training_runs(dataset_id)}


@router.get("/training/runs/{run_id}")
async def get_run_snapshot(run_id: str, user: User = Depends(get_current_user)):
    """Return a progress snapshot for a specific (e.g. past) MLflow run."""
    return await _read_run_snapshot(run_id)


@router.get("/training/{task_id}")
async def get_training_status(task_id: str, user: User = Depends(get_current_user)):
    """Return a single durable progress snapshot for a training job."""
    return await _read_training_snapshot(task_id)


@router.get("/training/{task_id}/stream")
async def get_training_status_stream(task_id: str, request: Request,
                                     user: User = Depends(get_current_user)):
    """Stream durable progress for a training job as Server-Sent Events.

    Polls the AI service every couple of seconds and emits one ``data:`` event per
    tick until the job reaches a terminal state (FINISHED/FAILED/CANCELLED).
    """

    async def event_generator():
        while True:
            if await request.is_disconnected():
                return
            snapshot = await _read_training_snapshot(task_id)
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["state"] in _TERMINAL_STATES:
                return
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/training/{task_id}")
async def cancel_training_of_model(task_id: str, user: User = Depends(get_current_user)):
    """Cancel a task cooperatively through the durable store."""
    try:
        await service.cancel_training(task_id)
        return await _read_training_snapshot(task_id)
    except Exception as exc:
        logger.exception("Failed to cancel instance segmentation training.")
        raise HTTPException(status_code=http_status.HTTP_502_BAD_GATEWAY,
                            detail=f"Could not cancel training: {exc}")


# The POST /run inference endpoint was removed: nothing called it, and its request
# body carried a raw filesystem path (`image_url`) that was handed straight to
# cv2.imread, which made it both unauthorizable — there is no dataset to resolve
# from a path — and an arbitrary-file-read on the shared volume. Interactive
# inference goes through the annotation-session WebSocket, which resolves the path
# server-side from the image id. If a direct inference API is ever needed, it
# should take an `image_id` so it can be permission-checked.
