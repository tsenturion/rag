"""Экспорт воспроизводимых артефактов для чанкинга документов."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_prep.config import ChunkingPipelineConfig
from rag_prep.models import ChunkingExportResult, PreparedChunk
from rag_prep.utils import (
    artifact_integrity,
    artifact_set_transaction,
    atomic_text_writer,
    json_dump,
    verify_upstream_artifact,
)

LOGGER = logging.getLogger(__name__)


class ChunkExportStage:
    """Сохраняет чанки, готовые к embeddings, в JSON и JSONL."""

    def __init__(self, config: ChunkingPipelineConfig):
        """Готовит экземпляр к экспорту чанков, гарантируя доступ к конфигурации пайплайна и корректную инициализацию зависимостей."""
        self.config = config

    def run(
        self,
        chunks: list[PreparedChunk],
        *,
        run_id: str,
        counts: dict[str, int | float],
        diagnostics: dict[str, Any] | None = None,
    ) -> ChunkingExportResult:
        """Гарантирует атомарное сохранение чанков, метаданных и манифеста в формате, пригодном для последующего использования и аудита."""
        output_dir = self.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / self.config.paths.json_filename
        jsonl_path = output_dir / self.config.paths.jsonl_filename
        manifest_path = output_dir / self.config.paths.manifest_filename

        payload = [chunk.model_dump(mode="json") for chunk in chunks]
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"),
            "counts": counts,
            "diagnostics": diagnostics or {},
            "upstream": verify_upstream_artifact(self.config.paths.input_jsonl),
            "outputs": {
                "json": json_path.relative_to(output_dir).as_posix(),
                "jsonl": jsonl_path.relative_to(output_dir).as_posix(),
            },
        }
        with artifact_set_transaction([json_path, jsonl_path, manifest_path]) as staged:
            json_dump(staged[json_path.resolve()], payload)
            self._write_jsonl(staged[jsonl_path.resolve()], payload)
            manifest["integrity"] = artifact_integrity(
                staged,
                {"json": json_path, "jsonl": jsonl_path},
            )
            json_dump(staged[manifest_path.resolve()], manifest)

        LOGGER.info("Сохранены chunks JSON в %s и JSONL в %s", json_path, jsonl_path)
        return ChunkingExportResult(
            json_path=json_path,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            chunks_count=len(chunks),
            run_id=run_id,
        )

    @staticmethod
    def _write_jsonl(path: Path, payload: list[dict[str, Any]]) -> None:
        """Гарантирует корректную запись списка чанков в JSONL-файл с поддержкой атомарности и кодировки UTF-8."""
        with atomic_text_writer(path) as file:
            for item in payload:
                file.write(json.dumps(item, ensure_ascii=False))
                file.write("\n")
