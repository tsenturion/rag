from __future__ import annotations

import logging
from typing import Any

from unstructured.partition.auto import partition

from rag_prep.config import ParserConfig
from rag_prep.models import RawElement, SourceFile

LOGGER = logging.getLogger(__name__)
SECTION_TYPES = {"Title", "Header"}


class UnstructuredParsingStage:
    """Parse files into semantic elements with Unstructured."""

    def __init__(self, config: ParserConfig, default_section: str):
        self.config = config
        self.default_section = default_section

    def run(self, sources: list[SourceFile]) -> list[RawElement]:
        elements: list[RawElement] = []
        for source in sources:
            elements.extend(self._parse_source(source))
        LOGGER.info("Parsed %d raw elements from %d files", len(elements), len(sources))
        return elements

    def _parse_source(self, source: SourceFile) -> list[RawElement]:
        parsed = partition(
            filename=str(source.path),
            strategy=self.config.strategy,
            encoding=self.config.encoding,
            languages=self.config.languages,
            pdf_infer_table_structure=self.config.pdf_infer_table_structure,
            skip_infer_table_types=self.config.skip_infer_table_types,
            metadata_filename=str(source.path),
        )

        section = self.default_section
        raw_elements: list[RawElement] = []
        for index, element in enumerate(parsed):
            text = str(element).strip()
            if not text:
                continue

            element_type = getattr(element, "category", element.__class__.__name__)
            if element_type in SECTION_TYPES:
                section = text

            raw_elements.append(
                RawElement(
                    source_file=source,
                    element_index=index,
                    text=text,
                    element_type=element_type,
                    section=section,
                    metadata=self._metadata_to_dict(getattr(element, "metadata", None)),
                )
            )
        return raw_elements

    @staticmethod
    def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
        if metadata is None:
            return {}
        if hasattr(metadata, "to_dict"):
            return metadata.to_dict()
        if isinstance(metadata, dict):
            return metadata
        return {}

