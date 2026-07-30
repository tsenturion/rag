"""Очистка текста для подготовки документов."""

from __future__ import annotations

import logging
import re
from collections import Counter

from rag_prep.config import CleaningConfig
from rag_prep.models import ProcessedElement, RawElement

LOGGER = logging.getLogger(__name__)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


class TextCleaningStage:
    """Удаляет шум парсинга и текстовые артефакты без чанкинга."""

    def __init__(self, config: CleaningConfig):
        """Настраивает этап очистки текста, компилируя паттерны для удаления и выделения шаблонов, обеспечивая фильтрацию нежелательного контента."""
        self.config = config
        self.drop_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in config.drop_patterns
        ]
        self.boilerplate_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in config.boilerplate_patterns
        ]

    def run(self, elements: list[RawElement]) -> list[ProcessedElement]:
        """Фильтрует и очищает текстовые элементы, отбрасывая короткие и нежелательные по паттернам, добавляя метрики качества для последующей обработки."""
        cleaned: list[ProcessedElement] = []
        normalized_text_counts = Counter(
            self._repeat_key(element.text) for element in elements
        )
        for element in elements:
            text = self._clean_text(element.text)
            if len(text) < self.config.min_chars:
                continue
            if any(pattern.search(text) for pattern in self.drop_patterns):
                continue
            metadata = dict(element.metadata)
            metadata["quality"] = self._quality_signals(
                text=text,
                repeated_count=normalized_text_counts[self._repeat_key(element.text)],
            )
            cleaned.append(
                ProcessedElement(
                    source_file=element.source_file,
                    element_id=element.element_id,
                    element_index=element.element_index,
                    text=text,
                    element_type=element.element_type,
                    section=element.section,
                    section_path=element.section_path,
                    metadata=metadata,
                )
            )
        LOGGER.info("Очищено элементов: %d -> %d", len(elements), len(cleaned))
        return cleaned

    def _clean_text(self, text: str) -> str:
        """Гарантирует удаление управляющих символов и нормализацию пробелов для воспроизводимой предобработки текста перед анализом."""
        cleaned = text.replace("\ufeff", "")
        if self.config.remove_control_chars:
            cleaned = CONTROL_CHARS_RE.sub(" ", cleaned)
        if self.config.normalize_whitespace:
            cleaned = WHITESPACE_RE.sub(" ", cleaned)
            cleaned = BLANK_LINES_RE.sub("\n\n", cleaned)
        return cleaned.strip()

    def _quality_signals(self, text: str, repeated_count: int) -> dict[str, object]:
        """Вычисляет метрики осмысленности и вероятности мусора, чтобы фильтрация документов могла опираться на количественные признаки качества."""
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        alpha_chars = sum(1 for char in text if char.isalpha())
        alnum_chars = sum(1 for char in text if char.isalnum())
        printable_chars = sum(1 for char in text if char.isprintable())
        unique_tokens = {token.lower() for token in tokens}
        boilerplate_matches = [
            pattern.pattern
            for pattern in self.boilerplate_patterns
            if pattern.search(text)
        ]
        garbage_score = self._garbage_score(
            text, alpha_chars, alnum_chars, printable_chars
        )
        boilerplate_score = min(
            1.0,
            (0.35 if boilerplate_matches else 0.0)
            + (0.25 if repeated_count > 1 else 0.0)
            + (0.2 if len(tokens) <= 4 else 0.0),
        )
        meaningful_score = max(
            0.0, min(1.0, 1.0 - max(garbage_score, boilerplate_score))
        )
        return {
            "meaningful_score": round(meaningful_score, 3),
            "boilerplate_score": round(boilerplate_score, 3),
            "garbage_score": round(garbage_score, 3),
            "is_probable_boilerplate": boilerplate_score >= 0.5,
            "is_probable_garbage": garbage_score >= 0.5,
            "repeated_text_count": repeated_count,
            "unique_token_ratio": round(len(unique_tokens) / max(len(tokens), 1), 3),
            "matched_boilerplate_patterns": boilerplate_matches,
        }

    @staticmethod
    def _repeat_key(text: str) -> str:
        """Гарантирует идентичность ключа для повторяющихся текстов независимо от регистра и лишних пробелов."""
        return WHITESPACE_RE.sub(" ", text.strip().lower())

    @staticmethod
    def _garbage_score(
        text: str, alpha_chars: int, alnum_chars: int, printable_chars: int
    ) -> float:
        """Оценивает вероятность того, что текст является мусорным, по доле алфавитных, печатных и непечатных символов."""
        length = max(len(text), 1)
        non_printable_ratio = 1.0 - (printable_chars / length)
        alpha_ratio = alpha_chars / length
        alnum_ratio = alnum_chars / length
        score = 0.0
        if len(text) < 30:
            score += 0.15
        if alpha_ratio < 0.35:
            score += 0.35
        if alnum_ratio < 0.45:
            score += 0.25
        score += min(0.25, non_printable_ratio)
        return min(1.0, score)
