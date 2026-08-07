"""Focused tests for the instance segmentation patch/replace transactional service."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.contours  # noqa
import app.database.datasets  # noqa
import app.database.images  # noqa
import app.database.labels  # noqa
import app.database.masks  # noqa
import app.database.users  # noqa
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.labels import Labels
from app.database.users import Users
from iquana_toolbox.schemas.database.contours import Contour, QuantificationModel
from iquana_toolbox.schemas.model_info import InstanceSegmentationModelInfo
from app.services.instance_prediction_application import apply_instance_segmentation_predictions

@pytest.fixture
def db(tmp_path):
    """An in-memory database with required relations."""
    from sqlalchemy import event
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    u = Users(username="test_user", hashed_password="x")
    session.add(u)
    session.commit()
    
    ds = Datasets(id=1, name="ds", description="", dataset_type="image", created_by="test_user", folder_path="x")
    session.add(ds)
    session.commit()
    
    img = Images(id=1, dataset_id=1, file_name="x", file_path="x", thumbnail_file_path="x", width=100, height=100, color_mode="RGB")
    session.add(img)
    session.commit()
    
    mask = Masks(id=1, image_id=1, file_path="x")
    session.add(mask)
    session.commit()
    
    # Label 10: Parent, Label 20: Child, Label 30: Unrelated
    session.add(Labels(id=10, dataset_id=1, name="Parent", value=1))
    session.add(Labels(id=20, dataset_id=1, name="Child", value=2))
    session.add(Labels(id=30, dataset_id=1, name="Unrelated", value=3))
    
    session.commit()
    yield session
    session.close()

def _add_contour(db, contour_id, label_id, parent_id=None, mask_id=1, x=None, y=None):
    if x is None: x = [0.1, 0.2, 0.2]
    if y is None: y = [0.1, 0.1, 0.2]
    c = Contours(
        id=contour_id,
        mask_id=mask_id,
        label_id=label_id,
        parent_id=parent_id,
        added_by="test",
        author_username="test_user",
        confidence_score=1.0,
        x=x, y=y,
        area=10.0, perimeter=10.0, circularity=1.0, diameter=5.0
    )
    db.add(c)
    db.commit()
    return c

def _pred(label_id, id=None, parent_id=None, x=None, y=None):
    if x is None: x = [0.1, 0.2, 0.2]
    if y is None: y = [0.1, 0.1, 0.2]
    return Contour(
        id=id,
        label_id=label_id,
        confidence=0.9,
        x=x, y=y,
        parent_id=parent_id,
        quantification=QuantificationModel()
    )

def _model(label_ids):
    return InstanceSegmentationModelInfo(
        registry_key="test", name="test", description="x", usage_tip="x",
        task="instance_segmentation", label_ids=label_ids
    )

@pytest.mark.anyio
async def test_patch_suppresses_duplicates(db):
    _add_contour(db, 101, 10, x=[0.1, 0.2, 0.2], y=[0.1, 0.1, 0.2])
    
    preds = [_pred(10, x=[0.1, 0.2, 0.2], y=[0.1, 0.1, 0.2]), _pred(20)]
    stats = await apply_instance_segmentation_predictions(
        db, 1, 1, "test_user", preds, "patch", _model([10, 20])
    )
    
    assert stats["suppressed_count"] == 1
    assert stats["added_count"] == 1
    assert db.query(Contours).count() == 2

@pytest.mark.anyio
async def test_replace_deletes_scope_and_preserves_others(db):
    _add_contour(db, 101, 10)
    _add_contour(db, 102, 30)
    
    preds = [_pred(10)]
    stats = await apply_instance_segmentation_predictions(
        db, 1, 1, "test_user", preds, "replace", _model([10])
    )
    
    assert stats["replaced_count"] == 1
    assert stats["added_count"] == 1
    
    # 102 (label 30) should remain, new prediction (label 10) added, 101 deleted
    assert db.query(Contours).count() == 2
    labels = {c.label_id for c in db.query(Contours).all()}
    assert labels == {10, 30}

@pytest.mark.anyio
async def test_replace_preserves_unlabeled_children(db):
    # Null label id test
    parent = _add_contour(db, 101, 10)
    child = _add_contour(db, 102, None, parent_id=101)
    
    stats = await apply_instance_segmentation_predictions(
        db, 1, 1, "test_user", [], "replace", _model([10])
    )
    
    # Child should be preserved, parent deleted
    assert db.query(Contours).filter_by(id=101).count() == 0
    assert db.query(Contours).filter_by(id=102).count() == 1
    assert db.query(Contours).filter_by(id=102).one().parent_id is None

@pytest.mark.anyio
async def test_hierarchy_reconstruction(db):
    preds = [
        _pred(10, id=1),
        _pred(20, id=2, parent_id=1),
    ]
    stats = await apply_instance_segmentation_predictions(
        db, 1, 1, "test_user", preds, "replace", _model([10, 20])
    )
    assert stats["added_count"] == 2
    
    parent = db.query(Contours).filter_by(label_id=10).one()
    child = db.query(Contours).filter_by(label_id=20).one()
    assert child.parent_id == parent.id

@pytest.mark.anyio
async def test_invalid_mode(db):
    with pytest.raises(ValueError, match="Invalid apply_mode"):
        await apply_instance_segmentation_predictions(db, 1, 1, "u", [], "invalid", _model([]))

@pytest.mark.anyio
async def test_out_of_scope_prediction(db):
    preds = [_pred(99)]
    with pytest.raises(ValueError, match="Prediction label"):
        await apply_instance_segmentation_predictions(db, 1, 1, "u", preds, "replace", _model([10]))
