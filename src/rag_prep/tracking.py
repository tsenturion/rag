"""Логирование запусков в MLflow для RAG-конвейера."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import mlflow

from rag_prep.utils import flatten_dict
from rag_prep.mlflow_uri import (
    ensure_mlflow_tracking_parent,
    resolve_mlflow_tracking_uri,
)

LOGGER = logging.getLogger(__name__)


class SupportsArtifactPaths(Protocol):
    """Контракт результата экспорта, предоставляющего пути MLflow-артефактов."""

    def artifact_paths(self) -> list[Path]:
        """Возвращает существующие или ожидаемые пути артефактов запуска."""
        ...


class MLflowTracker:
    """Обеспечивает централизованное логирование параметров, метрик и артефактов RAG-конвейера в MLflow с учётом конфигурации."""

    def __init__(self, config: Any):
        """Готовит экземпляр для логирования, сохраняя конфигурацию и обеспечивая доступ к настройкам MLflow."""
        self.config = config

    def log_run(
        self, counts: Mapping[str, int | float], export: SupportsArtifactPaths
    ) -> None:
        """Логирует параметры, метрики и артефакты запуска в MLflow, если логирование включено, обеспечивая воспроизводимость и мониторинг."""
        if not self.config.logging.mlflow_enabled:
            LOGGER.info("Логирование MLflow отключено")
            return

        tracking_uri = self._tracking_uri()
        ensure_mlflow_tracking_parent(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.logging.mlflow_experiment)

        params = flatten_dict(self.config.model_dump(mode="json"))
        with mlflow.start_run(
            run_name=self.config.run.name,
            nested=mlflow.active_run() is not None,
        ):
            mlflow.log_params(
                {key: self._safe_param(value) for key, value in params.items()}
            )
            for key, value in counts.items():
                mlflow.log_metric(key, value)
            for path in export.artifact_paths():
                mlflow.log_artifact(str(path))

        LOGGER.info("Запуск залогирован в MLflow tracking URI %s", tracking_uri)

    def _tracking_uri(self) -> str:
        """Определяет корректный URI для MLflow tracking, учитывая абсолютные и относительные пути, гарантируя доступность хранилища."""
        project_root = Path(__file__).resolve().parents[2]
        uri = resolve_mlflow_tracking_uri(
            self.config.logging.mlflow_tracking_uri,
            base_dir=project_root,
        )
        path = Path(uri)
        if path.is_absolute() or path.drive:
            return path.resolve().as_uri()
        parsed = urlparse(uri)
        if parsed.scheme:
            return uri
        return path.resolve().as_uri()

    @staticmethod
    def _safe_param(value: Any) -> str | int | float | bool:
        """Преобразует параметр в безопасный для MLflow формат, ограничивая длину и предотвращая ошибки при логировании."""
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
        else:
            text = repr(value)
        return text[:500]
