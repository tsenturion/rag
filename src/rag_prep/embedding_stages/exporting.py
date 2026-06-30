from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_prep.config import EmbeddingPipelineConfig
from rag_prep.models import EmbeddedChunk, EmbeddingExportResult
from rag_prep.utils import atomic_text_writer, json_dump

LOGGER = logging.getLogger(__name__)


class EmbeddingExportStage:
    """Сохраняет записи embeddings для дальнейшей индексации в vector store."""

    def __init__(self, config: EmbeddingPipelineConfig):
        self.config = config

    def run(
        self,
        embedded_chunks: list[EmbeddedChunk],
        *,
        run_id: str,
        counts: dict[str, int | float],
        diagnostics: dict[str, Any] | None = None,
    ) -> EmbeddingExportResult:
        output_dir = self.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / self.config.paths.json_filename
        jsonl_path = output_dir / self.config.paths.jsonl_filename
        manifest_path = output_dir / self.config.paths.manifest_filename

        payload = [chunk.model_dump(mode="json") for chunk in embedded_chunks]
        json_dump(json_path, payload)
        self._write_jsonl(jsonl_path, payload)

        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self._safe_config(),
            "counts": counts,
            "diagnostics": diagnostics or {},
            "outputs": {
                "json": str(json_path),
                "jsonl": str(jsonl_path),
            },
        }
        json_dump(manifest_path, manifest)

        LOGGER.info("Сохранены embeddings JSON в %s и JSONL в %s", json_path, jsonl_path)
        return EmbeddingExportResult(
            json_path=json_path,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            embeddings_count=len(embedded_chunks),
            run_id=run_id,
        )

    def _safe_config(self) -> dict[str, Any]:
        config = self.config.model_dump(mode="json")
        embedding = config.get("embedding", {})
        for key in ("api_key", "openai_api_key"):
            if key in embedding:
                embedding[key] = "<redacted>"
        return config

    @staticmethod
    def _write_jsonl(path: Path, payload: list[dict[str, Any]]) -> None:
        with atomic_text_writer(path) as file:
            for item in payload:
                file.write(json.dumps(item, ensure_ascii=False))
                file.write("\n")
