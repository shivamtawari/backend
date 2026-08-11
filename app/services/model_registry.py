"""Direct MLflow access for model discovery.

The backend reads available models straight from the shared MLflow registry
instead of HTTP-hopping through each AI service's ``/models`` endpoint. MLflow
is the source of truth, so this removes one network round-trip per request.
"""
from logging import getLogger

import mlflow
from iquana_toolbox.mlflow import MLFlowModelRegistry

from config import MLFLOW_URL

logger = getLogger(__name__)

# Client only; no connection is made until a query runs.
MODEL_REGISTRY = MLFlowModelRegistry(MLFLOW_URL)


def _search_registered_models_by_tags(tags: dict):
    """Return registered models matching ``tags`` (hits carry a ``.name``).

    We search MLflow directly instead of going through the toolbox's
    ``get_model_infos_via_tags``. That helper eagerly rebuilds a ``ModelInfo``
    from each registered model's tags, which fails when the tag set lacks the
    (now required) ``ModelInfo`` fields like ``registry_key`` / ``name`` /
    ``description`` / ``usage_tip`` -- tags only carry the filterable subset
    (task/status/...). We only need the names here; the full info is read from
    artifact metadata by ``_full_model_info``, which is the source of truth.
    """
    filter_string = " AND ".join(f"tags.{key} = '{value}'" for key, value in tags.items())
    return MODEL_REGISTRY.client.search_registered_models(filter_string=filter_string)


def _registry_key(model) -> str:
    """Pull the registry key from a tag-search hit (dict or ModelInfo)."""
    if isinstance(model, dict):
        return model["name"]
    return getattr(model, "registry_key", None) or model.name


def _models_for_task(task: str, model_role: str | None = None, dataset_id: int | None = None):
    """Registered, ready-to-serve models advertising ``task`` (optionally filtered by role and dataset_id)."""
    task_clean = task.strip().lower()
    task_boolean_tag = "task_" + task_clean.replace("-", "_")
    task_underscore = task_clean.replace("-", "_")
    task_hyphen = task_clean.replace("_", "-")

    by_name: dict[str, object] = {}
    for base_tags in (
        {task_boolean_tag: "true", "status": "ready"},
        {"task": task_hyphen, "status": "ready"},
        {"task": task_underscore, "status": "ready"},
    ):
        tags = dict(base_tags)
        if model_role:
            tags["model_role"] = model_role
        if dataset_id is not None:
            tags["dataset_id"] = str(dataset_id)
        for model in _search_registered_models_by_tags(tags):
            by_name[model.name] = model
    return list(by_name.values())



def _full_model_info(registry_key: str, default_alias: str = "active") -> dict:
    """Return a model's complete ``model_info`` from its artifact metadata."""
    for target in (f"models:/{registry_key}@{default_alias}", f"models:/{registry_key}@latest"):
        try:
            info = mlflow.models.get_model_info(target)
            if info.metadata:
                return info.metadata
        except Exception:
            continue
    logger.warning("Model '%s' has no artifact metadata; returning stub.", registry_key)
    return {"registry_key": registry_key, "name": registry_key}


def list_available_models(
    task: str,
    model_role: str | None = None,
    dataset_id: int | None = None,
) -> dict:
    """Return ready-to-serve models for ``task`` directly from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URL)
    matched = _models_for_task(task, model_role=model_role, dataset_id=dataset_id)
    models = [_full_model_info(_registry_key(m)) for m in matched]
    return {
        "success": True,
        "message": f"Retrieved {len(models)} available models.",
        "result": models,
    }
