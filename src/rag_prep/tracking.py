from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow

from rag_prep.utils import flatten_dict

LOGGER = logging.getLogger(__name__)


class MLflowTracker:
    def __init__(self, config: Any):
        self.config = config

    def log_run(self, counts: dict[str, int | float], export: Any) -> None:
        if not self.config.logging.mlflow_enabled:
            LOGGER.info("MLflow logging disabled")
            return

        tracking_uri = self._tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.logging.mlflow_experiment)

        params = flatten_dict(self.config.model_dump(mode="json"))
        with mlflow.start_run(
            run_name=self.config.run.name,
            nested=mlflow.active_run() is not None,
        ):
            mlflow.log_params({key: self._safe_param(value) for key, value in params.items()})
            for key, value in counts.items():
                mlflow.log_metric(key, value)
            mlflow.log_artifact(str(export.json_path))
            mlflow.log_artifact(str(export.jsonl_path))
            mlflow.log_artifact(str(export.manifest_path))

        LOGGER.info("Logged run to MLflow tracking URI %s", tracking_uri)

    def _tracking_uri(self) -> str:
        uri = self.config.logging.mlflow_tracking_uri
        path = Path(uri)
        if path.is_absolute() or path.drive:
            return path.resolve().as_uri()

        parsed = urlparse(uri)
        if parsed.scheme:
            return uri

        return (Path.cwd() / path).resolve().as_uri()

    @staticmethod
    def _safe_param(value: Any) -> str | int | float | bool:
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
        else:
            text = repr(value)
        return text[:500]
