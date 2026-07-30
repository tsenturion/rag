"""Загрузка входных данных для расчёта embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_prep.models import PreparedChunk
from rag_prep.utils import verify_upstream_artifact

LOGGER = logging.getLogger(__name__)


class ChunkLoadingStage:
    """Загружает подготовленные чанки из результата пайплайна чанкинга."""

    def run(self, input_jsonl: Path) -> list[PreparedChunk]:
        """Гарантирует воспроизводимую загрузку и валидацию чанков из JSONL-файла с явной ошибкой при повреждении данных."""
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Файл чанков не существует: {input_jsonl}")
        verify_upstream_artifact(input_jsonl)

        chunks: list[PreparedChunk] = []
        with input_jsonl.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(PreparedChunk.model_validate(json.loads(line)))
                except Exception as exc:
                    raise ValueError(
                        f"Некорректный чанк в {input_jsonl}:{line_number}"
                    ) from exc

        LOGGER.info("Загружено чанков: %d из %s", len(chunks), input_jsonl)
        return chunks
