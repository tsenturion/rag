from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from rag_prep.config import VectorStorePipelineConfig
from rag_prep.models import (
    VectorSearchResult,
    VectorStoreExportResult,
    VectorStoreIndexResult,
    VectorStoreValidationResult,
)
from rag_prep.utils import json_dump

LOGGER = logging.getLogger(__name__)


class VectorStoreExportStage:
    """Сохраняет validation, smoke-тесты поиска и manifest для vector store."""

    def __init__(self, config: VectorStorePipelineConfig):
        self.config = config

    def run(
        self,
        *,
        index: VectorStoreIndexResult,
        validation: VectorStoreValidationResult,
        search_results: list[VectorSearchResult],
        run_id: str,
        counts: dict[str, int | float],
        diagnostics: dict[str, Any] | None = None,
    ) -> VectorStoreExportResult:
        output_dir = self.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / self.config.paths.manifest_filename
        validation_path = output_dir / self.config.paths.validation_filename
        search_results_path = output_dir / self.config.paths.search_results_filename

        json_dump(validation_path, validation.model_dump(mode="json"))
        json_dump(
            search_results_path,
            [result.model_dump(mode="json") for result in search_results],
        )
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"),
            "index": index.model_dump(mode="json"),
            "counts": counts,
            "diagnostics": diagnostics or {},
            "outputs": {
                "validation": str(validation_path),
                "search_results": str(search_results_path),
            },
        }
        json_dump(manifest_path, manifest)

        LOGGER.info(
            "Сохранены manifest vector store в %s и результаты поиска в %s",
            manifest_path,
            search_results_path,
        )
        return VectorStoreExportResult(
            manifest_path=manifest_path,
            validation_path=validation_path,
            search_results_path=search_results_path,
            run_id=run_id,
        )
