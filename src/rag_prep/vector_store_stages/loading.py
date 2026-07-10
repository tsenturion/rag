from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_prep.models import EmbeddedChunk

LOGGER = logging.getLogger(__name__)


class EmbeddingLoadingStage:
    """Загружает записи embeddings, созданные пайплайном embeddings."""

    def run(self, input_jsonl: Path) -> list[EmbeddedChunk]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Файл embeddings не существует: {input_jsonl}")

        embedded_chunks: list[EmbeddedChunk] = []
        with input_jsonl.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    embedded_chunks.append(
                        EmbeddedChunk.model_validate(json.loads(line))
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Некорректная запись embedding в {input_jsonl}:{line_number}"
                    ) from exc

        LOGGER.info("Загружено embeddings: %d из %s", len(embedded_chunks), input_jsonl)
        return embedded_chunks
