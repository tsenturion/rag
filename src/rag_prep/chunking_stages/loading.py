from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_prep.models import PreparedDocument

LOGGER = logging.getLogger(__name__)


class PreparedDocumentLoadingStage:
    """Загружает подготовленные документы из предыдущего пайплайна."""

    def run(self, input_jsonl: Path) -> list[PreparedDocument]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Файл подготовленных документов не существует: {input_jsonl}")

        documents: list[PreparedDocument] = []
        with input_jsonl.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    documents.append(PreparedDocument.model_validate(json.loads(line)))
                except Exception as exc:
                    raise ValueError(
                        f"Некорректный подготовленный документ в {input_jsonl}:{line_number}"
                    ) from exc

        LOGGER.info("Загружено подготовленных документов: %d из %s", len(documents), input_jsonl)
        return documents

