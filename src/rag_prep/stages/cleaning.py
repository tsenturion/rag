from __future__ import annotations

import logging
import re

from rag_prep.config import CleaningConfig
from rag_prep.models import ProcessedElement, RawElement

LOGGER = logging.getLogger(__name__)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


class TextCleaningStage:
    """Remove parser noise and text artifacts without chunking text."""

    def __init__(self, config: CleaningConfig):
        self.config = config
        self.drop_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config.drop_patterns]

    def run(self, elements: list[RawElement]) -> list[ProcessedElement]:
        cleaned: list[ProcessedElement] = []
        for element in elements:
            text = self._clean_text(element.text)
            if len(text) < self.config.min_chars:
                continue
            if any(pattern.search(text) for pattern in self.drop_patterns):
                continue
            cleaned.append(
                ProcessedElement(
                    source_file=element.source_file,
                    element_index=element.element_index,
                    text=text,
                    element_type=element.element_type,
                    section=element.section,
                    metadata=element.metadata,
                )
            )
        LOGGER.info("Cleaned %d elements into %d elements", len(elements), len(cleaned))
        return cleaned

    def _clean_text(self, text: str) -> str:
        cleaned = text.replace("\ufeff", "")
        if self.config.remove_control_chars:
            cleaned = CONTROL_CHARS_RE.sub(" ", cleaned)
        if self.config.normalize_whitespace:
            cleaned = WHITESPACE_RE.sub(" ", cleaned)
            cleaned = BLANK_LINES_RE.sub("\n\n", cleaned)
        return cleaned.strip()

