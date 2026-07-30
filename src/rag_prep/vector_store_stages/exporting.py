"""Экспорт воспроизводимых артефактов для индексации в Qdrant."""

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
from rag_prep.utils import (
    artifact_integrity,
    artifact_set_transaction,
    json_dump,
    verify_upstream_artifact,
)

LOGGER = logging.getLogger(__name__)


class VectorStoreExportStage:
    """Сохраняет validation, smoke-тесты поиска и manifest для vector store."""

    def __init__(self, config: VectorStorePipelineConfig):
        """Обеспечивает готовность этапа экспорта с конфигурацией, необходимой для сохранения результатов индексации и диагностики в файловой системе."""
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
        """Сохраняет результаты индексации, валидации и поиска в устойчивом виде с атомарной гарантией целостности файлов и ведёт журнал операций."""
        output_dir = self.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = output_dir / self.config.paths.manifest_filename
        validation_path = output_dir / self.config.paths.validation_filename
        search_results_path = output_dir / self.config.paths.search_results_filename

        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"),
            "index": index.model_dump(mode="json"),
            "counts": counts,
            "diagnostics": diagnostics or {},
            "upstream": verify_upstream_artifact(self.config.paths.input_jsonl),
            "outputs": {
                "validation": validation_path.relative_to(output_dir).as_posix(),
                "search_results": search_results_path.relative_to(
                    output_dir
                ).as_posix(),
            },
        }
        with artifact_set_transaction(
            [validation_path, search_results_path, manifest_path]
        ) as staged:
            json_dump(
                staged[validation_path.resolve()],
                validation.model_dump(mode="json"),
            )
            json_dump(
                staged[search_results_path.resolve()],
                [result.model_dump(mode="json") for result in search_results],
            )
            manifest["integrity"] = artifact_integrity(
                staged,
                {
                    "validation": validation_path,
                    "search_results": search_results_path,
                },
            )
            json_dump(staged[manifest_path.resolve()], manifest)

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
