from __future__ import annotations

import logging
import unicodedata

import spacy

from rag_prep.config import NormalizationConfig
from rag_prep.models import ProcessedElement

LOGGER = logging.getLogger(__name__)


class TextNormalizationStage:
    """Normalize Unicode/case and collect spaCy sentence statistics."""

    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.nlp = spacy.blank(config.spacy_language)
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def run(self, elements: list[ProcessedElement]) -> list[ProcessedElement]:
        normalized: list[ProcessedElement] = []
        for element in elements:
            text = unicodedata.normalize(self.config.unicode_form, element.text)
            if self.config.lowercase:
                text = text.lower()

            metadata = dict(element.metadata)
            if self.config.collect_sentence_stats:
                doc = self.nlp(text)
                metadata["sentence_count"] = sum(1 for _ in doc.sents)
                metadata["token_count"] = len(doc)

            normalized.append(element.model_copy(update={"text": text, "metadata": metadata}))

        LOGGER.info("Normalized %d elements", len(normalized))
        return normalized

