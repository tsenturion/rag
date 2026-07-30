"""Экспорт воспроизводимых артефактов для расчёта embeddings."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_prep.config import EmbeddingPipelineConfig
from rag_prep.models import EmbeddedChunk, EmbeddingExportResult
from rag_prep.utils import (
    artifact_integrity,
    artifact_set_transaction,
    atomic_text_writer,
    json_dump,
    verify_upstream_artifact,
)

LOGGER = logging.getLogger(__name__)


class EmbeddingExportStage:
    """Сохраняет записи embeddings для дальнейшей индексации в vector store."""

    def __init__(self, config: EmbeddingPipelineConfig):
        """Гарантирует готовность экземпляра к экспорту embeddings с учётом всех параметров пайплайна."""
        self.config = config

    def run(
        self,
        embedded_chunks: list[EmbeddedChunk],
        *,
        run_id: str,
        counts: dict[str, int | float],
        diagnostics: dict[str, Any] | None = None,
    ) -> EmbeddingExportResult:
        """Гарантирует атомарное сохранение embeddings, метаданных и манифеста в формате, пригодном для дальнейшей автоматизации и аудита."""
        output_dir = self.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / self.config.paths.json_filename
        jsonl_path = output_dir / self.config.paths.jsonl_filename
        manifest_path = output_dir / self.config.paths.manifest_filename

        payload = [chunk.model_dump(mode="json") for chunk in embedded_chunks]
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self._safe_config(),
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

        LOGGER.info(
            "Сохранены embeddings JSON в %s и JSONL в %s", json_path, jsonl_path
        )
        return EmbeddingExportResult(
            json_path=json_path,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            embeddings_count=len(embedded_chunks),
            run_id=run_id,
        )

    def _safe_config(self) -> dict[str, Any]:
        """Обеспечивает публикацию конфигурации без утечки чувствительных данных, скрывая секретные ключи."""
        config = self.config.model_dump(mode="json")
        embedding = config.get("embedding", {})
        for key in ("api_key", "openai_api_key"):
            if key in embedding:
                embedding[key] = "<redacted>"
        return config

    @staticmethod
    def _write_jsonl(path: Path, payload: list[dict[str, Any]]) -> None:
        """Гарантирует корректную сериализацию и атомарную запись коллекции объектов в формате JSONL."""
        with atomic_text_writer(path) as file:
            for item in payload:
                file.write(json.dumps(item, ensure_ascii=False))
                file.write("\n")
