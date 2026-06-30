from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_prep.config import ChunkingPipelineConfig
from rag_prep.models import ChunkingExportResult, PreparedChunk
from rag_prep.utils import atomic_text_writer, json_dump

LOGGER = logging.getLogger(__name__)


class ChunkExportStage:
    """Сохраняет чанки, готовые к embeddings, в JSON и JSONL."""

    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config

    def run(
        self,
        chunks: list[PreparedChunk],
        *,
        run_id: str,
        counts: dict[str, int | float],
        diagnostics: dict[str, Any] | None = None,
    ) -> ChunkingExportResult:
        output_dir = self.config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / self.config.paths.json_filename
        jsonl_path = output_dir / self.config.paths.jsonl_filename
        manifest_path = output_dir / self.config.paths.manifest_filename

        payload = [chunk.model_dump(mode="json") for chunk in chunks]
        json_dump(json_path, payload)
        self._write_jsonl(jsonl_path, payload)

        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"),
            "counts": counts,
            "diagnostics": diagnostics or {},
            "outputs": {
                "json": str(json_path),
                "jsonl": str(jsonl_path),
            },
        }
        json_dump(manifest_path, manifest)

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
        with atomic_text_writer(path) as file:
            for item in payload:
                file.write(json.dumps(item, ensure_ascii=False))
                file.write("\n")

