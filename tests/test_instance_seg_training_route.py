import pytest
import os
import json
from unittest.mock import patch
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session
from app.database.users import Users
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.dataset_members import DatasetMembers
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.contours import Contours
from app.routes.services.instance_seg_router import router
from app.schemas.auth_user import AuthenticatedUser
from app.services.auth import get_current_user



@pytest.fixture
def test_client(tmp_path):
    from sqlalchemy import event
    engine = create_engine(
        f"sqlite:///{tmp_path / 'router-test.db'}",
        connect_args={"check_same_thread": False},
    )
    
    wal_attempted = {"done": False}
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        if not wal_attempted["done"]:
            wal_attempted["done"] = True
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
        
    database.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    owner = Users(username="owner", hashed_password="x")
    db.add(owner)
    
    dataset = Datasets(name="test-ds", description="", dataset_type="image",
                       folder_path=str(tmp_path / "test-ds"), created_by=owner.username)
    db.add(dataset)
    db.flush()
    db.add(DatasetMembers(dataset_id=dataset.id, username=owner.username,
                          role="owner",
                          extra_permissions=[], denied_permissions=[]))
    
    parent_label = Labels(dataset_id=dataset.id, name="parent", value=1)
    db.add(parent_label)
    db.flush()

    empty_image = Images(dataset_id=dataset.id, file_name="empty.png", file_path="/tmp/empty.png", thumbnail_file_path="/tmp/thumb.png", width=10, height=10, color_mode="RGB")
    db.add(empty_image)
    db.flush()

    db.commit()

    app = FastAPI()
    app.include_router(router)

    def _session_override():
        request_db = SessionLocal()
        try:
            yield request_db
        finally:
            request_db.close()

    def _user_override(session=Depends(_session_override)):
        user = session.query(Users).filter_by(username="owner").one()
        return AuthenticatedUser.from_query(user)

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override

    with TestClient(app) as client:
        yield {
            "client": client,
            "dataset_id": dataset.id,
            "labels": {"parent": parent_label.id},
            "db": db,
            "owner": owner.username
        }

    db.close()
    engine.dispose()

@patch("app.routes.services.instance_seg_router.service.start_training")
def test_start_training_empty_dataset_fails(mock_start_training, test_client):
    # This dataset has an image but no annotations, so export will fail with "empty_export".
    response = test_client["client"].post(
        "/instance_segmentation/training/start",
        json={
            "dataset_id": test_client["dataset_id"],
            "model_run_name": "test run",
            "model_registry_key": "mask2former",
            "hyper_parameter": {},
        }
    )
    assert response.status_code == 400
    assert "error_code" in response.json()["detail"]
    assert response.json()["detail"]["error_code"] in {"missing_label_annotations", "empty_export"}

    
    mock_start_training.assert_not_called()


@patch("app.routes.services.instance_seg_router.service.start_training")
@patch("app.routes.services.instance_seg_router.export_training_hierarchy_dataset")
def test_start_training_returns_complete_normalization_conflict(
        mock_export, mock_start_training, test_client,
):
    summary = {
        "adjusted_image_count": 2,
        "adjusted_child_count": 5,
        "excluded_image_count": 1,
        "excluded_child_count": 1,
        "sibling_overlap_pair_count": 3,
        "affected_images": [{"image_id": 1, "action": "normalize"}],
    }
    mock_export.return_value = {
        "success": False,
        "message": "Hierarchy normalization is required.",
        "error_code": "hierarchy_normalization_required",
        "details": {"summary": summary},
    }

    response = test_client["client"].post(
        "/instance_segmentation/training/start",
        json={"dataset_id": test_client["dataset_id"]},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "message": "Hierarchy normalization is required.",
        "error_code": "hierarchy_normalization_required",
        "details": {"summary": summary},
    }
    assert mock_export.call_args.kwargs["hierarchy_conflict_policy"] == "strict"
    mock_start_training.assert_not_called()


@patch("app.routes.services.instance_seg_router.service.start_training")
@patch("app.routes.services.instance_seg_router.export_training_hierarchy_dataset")
def test_confirmed_normalization_is_forwarded_and_reported(
        mock_export, mock_start_training, test_client,
):
    summary = {
        "adjusted_image_count": 2,
        "excluded_image_count": 1,
        "source_annotations_modified": False,
    }
    mock_export.return_value = {
        "success": True,
        "output_file_path": "/tmp/normalized.json",
        "num_annotations": 3,
        "normalization_summary": summary,
    }
    mock_start_training.return_value = {"task_id": "normalized-task"}

    response = test_client["client"].post(
        "/instance_segmentation/training/start",
        json={
            "dataset_id": test_client["dataset_id"],
            "hierarchy_conflict_policy": "normalize",
        },
    )

    assert response.status_code == 200
    assert response.json()["task_id"] == "normalized-task"
    assert response.json()["normalization_summary"] == summary
    assert mock_export.call_args.kwargs["hierarchy_conflict_policy"] == "normalize"
    mock_start_training.assert_called_once()

@patch("app.routes.services.instance_seg_router.service.start_training")
def test_start_training_success_all_labels(mock_start_training, test_client):
    # Add a valid contour
    db = test_client["db"]
    mask = Masks(image_id=1, fully_annotated=True, file_path="/tmp/mask.png")
    db.add(mask)
    db.flush()
    contour = Contours(
        mask_id=mask.id, label_id=test_client["labels"]["parent"], added_by="manual", author_username=test_client["owner"],
        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9]
    )
    user = db.query(Users).first()
    contour.reviewed_by.append(user)
    db.add(contour)
    db.commit()

    # Mock the return of start_training
    mock_start_training.return_value = {"task_id": "test_task"}

    response = test_client["client"].post(
        "/instance_segmentation/training/start",
        json={
            "dataset_id": test_client["dataset_id"],
            "model_run_name": "test run",
            "model_registry_key": "mask2former",
            "hyper_parameter": {},
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Training started.", "task_id": "test_task"}
    
    mock_start_training.assert_called_once()
    req = mock_start_training.call_args[0][0]
    # Check that empty label_ids resolved to all labels
    assert len(req.labels) == 1
    assert req.labels[0].id == test_client["labels"]["parent"]
    assert req.selected_label_ids == [test_client["labels"]["parent"]]
    # Flat dataset (single label with no parent or children) yields enable_hierarchy=False
    assert req.enable_hierarchy is False


@patch("app.routes.services.instance_seg_router.service.start_training")
def test_start_training_success_explicit_labels(mock_start_training, test_client):
    db = test_client["db"]
    mask = Masks(image_id=1, fully_annotated=True, file_path="/tmp/mask.png")
    db.add(mask)
    db.flush()
    contour = Contours(
        mask_id=mask.id, label_id=test_client["labels"]["parent"], added_by="manual", author_username=test_client["owner"],
        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9]
    )
    user = db.query(Users).first()
    contour.reviewed_by.append(user)
    db.add(contour)
    db.commit()

    mock_start_training.return_value = {"task_id": "test_task_explicit"}

    response = test_client["client"].post(
        "/instance_segmentation/training/start",
        json={
            "dataset_id": test_client["dataset_id"],
            "label_ids": [test_client["labels"]["parent"]],
            "model_run_name": "test explicit run",
            "model_registry_key": "mask2former",
            "hyper_parameter": {},
        }
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Training started.", "task_id": "test_task_explicit"}

    mock_start_training.assert_called_once()
    req = mock_start_training.call_args[0][0]
    assert req.selected_label_ids == [test_client["labels"]["parent"]]
    assert req.enable_hierarchy is False


def test_phase7_end_to_end_integration_proof(test_client, tmp_path, monkeypatch):
    """Phase 7 integration proof covering:
    backend payload -> shared request serialization -> AI worker metadata -> MLflow publication -> backend dataset-scoped discovery
    """
    db = test_client["db"]

    # Add child label so dataset is hierarchical
    child_label = Labels(dataset_id=test_client["dataset_id"], name="child", value=2, parent_id=test_client["labels"]["parent"])
    db.add(child_label)
    db.flush()

    mask = Masks(image_id=1, fully_annotated=True, file_path="/tmp/mask_e2e.png")
    db.add(mask)
    db.flush()
    contour = Contours(
        mask_id=mask.id, label_id=child_label.id, added_by="manual", author_username=test_client["owner"],
        confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.1, 0.9, 0.9, 0.1], y=[0.1, 0.1, 0.9, 0.9]
    )
    user = db.query(Users).first()
    contour.reviewed_by.append(user)
    db.add(contour)
    db.commit()

    captured_request = None

    async def mock_start(req, model_run_name=None):
        nonlocal captured_request
        captured_request = req
        return {"task_id": "integration_task_123"}

    with patch("app.routes.services.instance_seg_router.service.start_training", side_effect=mock_start):
        response = test_client["client"].post(
            "/instance_segmentation/training/start",
            json={
                "dataset_id": test_client["dataset_id"],
                "label_ids": [child_label.id],
                "model_run_name": "E2E Proof Model",
                "model_registry_key": "mask2former",
                "hyper_parameter": {},
            }
        )
        assert response.status_code == 200

    assert captured_request is not None

    # 1. Backend payload -> shared request serialization
    serialized_json = captured_request.model_dump_json()

    # 2. AI worker request deserialization & metadata derivation
    from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
    worker_request = InstanceSegmentationTrainingRequest.model_validate_json(serialized_json)

    assert worker_request.dataset_id == test_client["dataset_id"]
    assert worker_request.selected_label_ids == [child_label.id]
    assert worker_request.enable_hierarchy is True

    # 3. AI Worker metadata & temporary MLflow publication
    from iquana_toolbox.mlflow import MLFlowModelRegistry
    from iquana_toolbox.schemas.model_info import ModelInfo

    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow_e2e.db'}"
    registry = MLFlowModelRegistry(mlflow_uri)

    segmentation_mode = "hierarchical" if worker_request.enable_hierarchy else "flat"
    target_encoding = "exclusive_hierarchy_v1" if worker_request.enable_hierarchy else "standard"

    info = ModelInfo(
        name="E2E Proof Model",
        registry_key=f"e2e_model_ds{worker_request.dataset_id}",
        task="instance-segmentation",
        model_role="trained",
        dataset_id=worker_request.dataset_id,
        trained_by=worker_request.user_id,
        selected_label_ids=worker_request.selected_label_ids,
        description="E2E test model publication",
        usage_tip="Test usage tip",
    )

    import mlflow.pyfunc

    class DummyModel(mlflow.pyfunc.PythonModel):
        model_info = info
        def predict(self, req, params=None):
            return []
        def get_artifacts(self, path):
            return {}

    dummy_model = DummyModel()
    dummy_model.model_info.tags["segmentation_mode"] = segmentation_mode
    dummy_model.model_info.tags["target_encoding"] = target_encoding
    dummy_model.model_info.tags["dataset_id"] = str(worker_request.dataset_id)
    dummy_model.model_info.tags["task_instance_segmentation"] = "true"
    dummy_model.model_info.tags["model_role"] = "trained"
    dummy_model.model_info.tags["status"] = "ready"

    pub_result = registry.register_model(dummy_model, assign_alias="active")
    assert pub_result.registry_key == f"e2e_model_ds{worker_request.dataset_id}"

    # 4. Backend dataset-scoped discovery
    monkeypatch.setattr("app.services.model_registry.MLFLOW_URL", mlflow_uri)
    monkeypatch.setattr("app.services.model_registry.MODEL_REGISTRY", MLFlowModelRegistry(mlflow_uri))
    from app.services.model_registry import list_available_models

    discovery = list_available_models(
        task="instance-segmentation",
        model_role="trained",
        dataset_id=worker_request.dataset_id,
    )

    assert discovery["success"] is True
    assert len(discovery["result"]) == 1
    discovered_model = discovery["result"][0]
    assert discovered_model["name"] == "E2E Proof Model"
    assert discovered_model["registry_key"] == f"e2e_model_ds{worker_request.dataset_id}"


def test_read_training_snapshot_fails_closed_when_dataset_id_missing(test_client):
    """Verify that jobs missing dataset_id metadata fail closed with 404."""
    async def mock_get_state(task_id):
        return {"task_id": task_id, "state": "RUNNING"}

    with patch("app.routes.services.instance_seg_router.service.get_training_task_state", side_effect=mock_get_state):
        response = test_client["client"].get("/instance_segmentation/training/missing-ds-task")
        assert response.status_code == 404
        assert "missing dataset metadata" in response.json()["detail"]


def test_training_status_requires_ai_train_permission(test_client):
    """Verify that training status checks require AI_TRAIN permission."""
    async def mock_get_state(task_id):
        return {"task_id": task_id, "dataset_id": test_client["dataset_id"], "state": "RUNNING"}

    with patch("app.routes.services.instance_seg_router.service.get_training_task_state", side_effect=mock_get_state):
        with patch("app.routes.services.instance_seg_router.ensure_permission") as mock_perm:
            response = test_client["client"].get("/instance_segmentation/training/task-123")
            assert response.status_code == 200
            mock_perm.assert_called_once()
            from app.services.permissions import Permission
            assert mock_perm.call_args[0][2] == Permission.AI_TRAIN
