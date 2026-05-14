from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_prep.models import PreparedDocument

LOGGER = logging.getLogger(__name__)


class PreparedDocumentLoadingStage:
    """Load prepared documents from the previous pipeline."""

    def run(self, input_jsonl: Path) -> list[PreparedDocument]:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Prepared documents file does not exist: {input_jsonl}")

        documents: list[PreparedDocument] = []
        with input_jsonl.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    documents.append(PreparedDocument.model_validate(json.loads(line)))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid prepared document at {input_jsonl}:{line_number}"
                    ) from exc

        LOGGER.info("Loaded %d prepared documents from %s", len(documents), input_jsonl)
        return documents

