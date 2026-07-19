"""Нормализация текста для подготовки документов."""

from __future__ import annotations

import logging
import unicodedata

import spacy

from rag_prep.config import NormalizationConfig
from rag_prep.models import ProcessedElement

LOGGER = logging.getLogger(__name__)


class TextNormalizationStage:
    """Нормализует Unicode/регистр и собирает sentence statistics через spaCy."""

    def __init__(self, config: NormalizationConfig):
        """Готовит экземпляр к нормализации текста, обеспечивая корректную сегментацию предложений и настройку языка."""
        self.config = config
        self.nlp = spacy.blank(config.spacy_language)
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def run(self, elements: list[ProcessedElement]) -> list[ProcessedElement]:
        """Обеспечивает последовательную нормализацию текста и при необходимости собирает статистику предложений и токенов для последующего анализа качества данных."""
        texts = [self._normalize_text(element.text) for element in elements]
        if not self.config.collect_sentence_stats:
            normalized = [
                element.model_copy(update={"text": text})
                for element, text in zip(elements, texts)
            ]
            LOGGER.info("Нормализовано элементов: %d", len(normalized))
            return normalized

        normalized: list[ProcessedElement] = []
        for element, text, doc in zip(elements, texts, self.nlp.pipe(texts)):
            metadata = dict(element.metadata)
            metadata["sentence_count"] = sum(1 for _ in doc.sents)
            metadata["token_count"] = len(doc)
            normalized.append(
                element.model_copy(update={"text": text, "metadata": metadata})
            )

        LOGGER.info("Нормализовано элементов: %d", len(normalized))
        return normalized

    def _normalize_text(self, text: str) -> str:
        """Гарантирует единообразное представление текста с учётом юникод-формы и регистра для стабильной обработки на следующих этапах."""
        normalized = unicodedata.normalize(self.config.unicode_form, text)
        if self.config.lowercase:
            normalized = normalized.lower()
        return normalized
