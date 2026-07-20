import os
from contextlib import contextmanager
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "agentic-rag")

try:
    import mlflow
    _mlflow_available = True
except ImportError:
    _mlflow_available = False


def ensure_experiment(experiment_name: str = _EXPERIMENT_NAME):
    if not _mlflow_available:
        return
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning("MLflow experiment setup failed", extra={"error": str(e)})


@contextmanager
def start_run(run_name: Optional[str] = None, experiment_name: str = _EXPERIMENT_NAME):
    if not _mlflow_available:
        yield None
        return
    ensure_experiment(experiment_name)
    try:
        with mlflow.start_run(run_name=run_name) as run:
            logger.info("MLflow run started", extra={"run_id": run.info.run_id})
            yield run
    except Exception as e:
        logger.warning("MLflow run failed", extra={"error": str(e)})
        yield None


def log_params(params: dict):
    if not _mlflow_available:
        return
    try:
        mlflow.log_params(params)
    except Exception as e:
        logger.warning("MLflow log_params failed", extra={"error": str(e)})


def log_metrics(metrics: dict, step: Optional[int] = None):
    if not _mlflow_available:
        return
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        logger.warning("MLflow log_metrics failed", extra={"error": str(e)})


def log_artifact(local_path: str):
    if not _mlflow_available:
        return
    try:
        mlflow.log_artifact(local_path)
    except Exception as e:
        logger.warning("MLflow log_artifact failed", extra={"error": str(e)})
