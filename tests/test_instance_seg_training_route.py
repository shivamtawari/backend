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
    assert response.json()["detail"]["error_code"] == "empty_export"
    
    mock_start_training.assert_not_called()

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
    
    # Check artifact writing
    assert req.annotation_file_url
    assert os.path.exists(req.annotation_file_url)
    with open(req.annotation_file_url) as f:
        payload = json.load(f)
        assert "images" in payload
        assert "annotations" in payload
        assert len(payload["annotations"]) == 1
