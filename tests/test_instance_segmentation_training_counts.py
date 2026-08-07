"""Focused endpoint checks for instance-segmentation training annotation counts."""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.routes.services.instance_seg_router import router as instance_segmentation_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole
from app.services.auth import get_current_user


@pytest.fixture
def ctx(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'instance-seg-counts.db'}",
        connect_args={"check_same_thread": False},
    )
    database.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    owner = Users(username="owner", hashed_password="x")
    stranger = Users(username="stranger", hashed_password="x")
    db.add_all([owner, stranger])
    dataset = Datasets(name="training-counts", description="", dataset_type="image",
                       folder_path="/tmp/training-counts", created_by=owner.username)
    db.add(dataset)
    db.flush()
    db.add(DatasetMembers(dataset_id=dataset.id, username=owner.username,
                          role=DatasetRole.OWNER.value,
                          extra_permissions=[], denied_permissions=[]))

    cell = Labels(dataset_id=dataset.id, name="cell", value=1)
    nucleus = Labels(dataset_id=dataset.id, name="nucleus", value=2)
    db.add_all([cell, nucleus])
    db.flush()

    finished_image = Images(
        dataset_id=dataset.id, file_name="finished.png", file_path="/tmp/finished.png",
        thumbnail_file_path="/tmp/finished-thumb.png", width=10, height=10,
        color_mode="RGB",
    )
    incomplete_image = Images(
        dataset_id=dataset.id, file_name="incomplete.png", file_path="/tmp/incomplete.png",
        thumbnail_file_path="/tmp/incomplete-thumb.png", width=10, height=10,
        color_mode="RGB",
    )
    db.add_all([finished_image, incomplete_image])
    db.flush()
    finished_mask = Masks(image_id=finished_image.id, fully_annotated=True,
                          file_path="/tmp/finished-mask.png")
    incomplete_mask = Masks(image_id=incomplete_image.id, fully_annotated=False,
                            file_path="/tmp/incomplete-mask.png")
    db.add_all([finished_mask, incomplete_mask])
    db.flush()

    def add_contour(mask_id, label_id, reviewed):
        contour = Contours(
            mask_id=mask_id, label_id=label_id, added_by="manual",
            author_username=owner.username, confidence_score=1.0, area=1.0,
            perimeter=1.0, circularity=1.0, diameter=1.0,
            x=[0.1, 0.2], y=[0.1, 0.2],
        )
        if reviewed:
            contour.reviewed_by.append(owner)
        db.add(contour)
        return contour

    add_contour(finished_mask.id, cell.id, reviewed=True)
    add_contour(finished_mask.id, nucleus.id, reviewed=True)
    add_contour(finished_mask.id, cell.id, reviewed=False)
    add_contour(incomplete_mask.id, cell.id, reviewed=True)
    db.commit()

    app = FastAPI()
    app.include_router(instance_segmentation_router)
    current_user = {"username": owner.username}

    def _session_override():
        request_db = SessionLocal()
        try:
            yield request_db
        finally:
            request_db.close()

    def _user_override(session=Depends(_session_override)):
        user = session.query(Users).filter_by(username=current_user["username"]).one()
        return AuthenticatedUser.from_query(user)

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override

    with TestClient(app) as client:
        yield {
            "client": client,
            "dataset_id": dataset.id,
            "labels": {"cell": cell.id, "nucleus": nucleus.id},
            "current_user": current_user,
        }

    db.close()
    engine.dispose()


def test_counts_include_reviewed_contours_on_fully_annotated_masks(ctx):
    response = ctx["client"].get(
        "/instance_segmentation/training/label-annotation-counts",
        params={"dataset_id": ctx["dataset_id"]},
    )

    assert response.status_code == 200
    assert response.json()["reviewed_annotation_counts"] == {
        str(ctx["labels"]["cell"]): 1,
        str(ctx["labels"]["nucleus"]): 1,
    }


def test_counts_exclude_reviewed_contours_on_incomplete_masks(ctx):
    response = ctx["client"].get(
        "/instance_segmentation/training/label-annotation-counts",
        params={"dataset_id": ctx["dataset_id"]},
    )

    assert response.status_code == 200
    counts = response.json()["reviewed_annotation_counts"]
    assert counts[str(ctx["labels"]["cell"])] == 1


def test_counts_require_dataset_ai_train_permission(ctx):
    ctx["current_user"]["username"] = "stranger"

    response = ctx["client"].get(
        "/instance_segmentation/training/label-annotation-counts",
        params={"dataset_id": ctx["dataset_id"]},
    )

    assert response.status_code == 403
    assert "ai.train" in response.json()["detail"]
