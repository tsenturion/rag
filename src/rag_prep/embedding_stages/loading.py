from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_prep.models import PreparedChunk

LOGGER = logging.getLogger(__name__)


class ChunkLoadingStage:
    """Load prepared chunks from the chunking pipeline output."""

    def run(self, input_jsonl: Path) -> list[PreparedChunk]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Chunks file does not exist: {input_jsonl}")

        chunks: list[PreparedChunk] = []
        with input_jsonl.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(PreparedChunk.model_validate(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid chunk at {input_jsonl}:{line_number}") from exc

        LOGGER.info("Loaded %d chunks from %s", len(chunks), input_jsonl)
        return chunks
