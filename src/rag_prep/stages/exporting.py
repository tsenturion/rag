"""Экспорт воспроизводимых артефактов для подготовки документов."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_prep.config import PipelineConfig
from rag_prep.models import ExportResult, PreparedDocument
from rag_prep.utils import (
    artifact_integrity,
    artifact_set_transaction,
    atomic_text_writer,
    json_dump,
)

LOGGER = logging.getLogger(__name__)


class ExportStage:
    """Сохраняет подготовленные документы в JSON и JSONL для следующих этапов."""

    def __init__(self, config: PipelineConfig):
        """Готовит экземпляр к экспорту данных, фиксируя параметры вывода и политику сериализации."""
        self.config = config

    def run(
        self,
        documents: list[PreparedDocument],
        *,
        run_id: str,
        counts: dict[str, int],
        diagnostics: dict[str, Any] | None = None,
    ) -> ExportResult:
        """Гарантирует атомарное сохранение подготовленных документов и метаданных в формате, пригодном для дальнейшей автоматизации и аудита."""
        output_dir = self.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / self.config.paths.json_filename
        jsonl_path = output_dir / self.config.paths.jsonl_filename
        manifest_path = output_dir / self.config.paths.manifest_filename

        payload = [document.model_dump(mode="json") for document in documents]
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"),
            "counts": counts,
            "diagnostics": diagnostics or {},
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

        LOGGER.info("Сохранены JSON в %s и JSONL в %s", json_path, jsonl_path)
        return ExportResult(
            json_path=json_path,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            documents_count=len(documents),
            duplicates_removed=counts.get("duplicates_removed", 0),
            run_id=run_id,
        )

    @staticmethod
    def _write_jsonl(path: Path, payload: list[dict[str, Any]]) -> None:
        """Гарантирует корректную запись коллекции документов в формате JSONL с поддержкой атомарности и Unicode."""
        with atomic_text_writer(path) as file:
            for item in payload:
                file.write(json.dumps(item, ensure_ascii=False))
                file.write("\n")
