from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from llama_index.core import Document

from rag_prep.config import StructuringConfig
from rag_prep.models import DocumentMetadata, PreparedDocument, ProcessedElement
from rag_prep.utils import stable_id, text_sha256

LOGGER = logging.getLogger(__name__)


class LlamaIndexStructuringStage:
    """Create Pydantic and LlamaIndex documents for downstream RAG steps."""

    def __init__(self, config: StructuringConfig):
        self.config = config

    def run(self, elements: list[ProcessedElement], run_id: str) -> list[PreparedDocument]:
        documents = (
            self._group_by_section(elements, run_id)
            if self.config.group_by_section
            else self._one_document_per_element(elements, run_id)
        )
        LOGGER.info("Structured %d prepared documents", len(documents))
        return documents

    def to_llama_documents(self, documents: list[PreparedDocument]) -> list[Document]:
        llama_documents: list[Document] = []
        for document in documents:
            metadata = document.metadata.model_dump(mode="json")
            llama_documents.append(
                Document(
                    text=document.text,
                    metadata=metadata,
                    id_=document.metadata.id,
                )
            )
        return llama_documents

    def _group_by_section(
        self, elements: list[ProcessedElement], run_id: str
    ) -> list[PreparedDocument]:
        groups: dict[tuple[str, str], list[ProcessedElement]] = defaultdict(list)
        for element in elements:
            groups[(element.source_file.source, element.section or self.config.default_section)].append(
                element
            )

        documents: list[PreparedDocument] = []
        for (_, section), group in groups.items():
            documents.append(self._build_document(group=group, section=section, run_id=run_id))
        return documents

    def _one_document_per_element(
        self, elements: list[ProcessedElement], run_id: str
    ) -> list[PreparedDocument]:
        return [
            self._build_document(group=[element], section=element.section, run_id=run_id)
            for element in elements
        ]

    def _build_document(
        self, group: list[ProcessedElement], section: str, run_id: str
    ) -> PreparedDocument:
        first = group[0]
        text = "\n\n".join(element.text for element in group).strip()
        text_hash = text_sha256(text)
        page_number = self._first_page_number(group)
        sentence_count = sum(
            int(element.metadata.get("sentence_count", 0))
            for element in group
            if element.metadata.get("sentence_count") is not None
        )
        metadata = DocumentMetadata(
            id=stable_id(first.source_file.source, section, text_hash),
            source=first.source_file.source,
            section=section or self.config.default_section,
            file_name=first.source_file.file_name,
            file_type=first.source_file.file_type,
            source_hash=first.source_file.source_hash,
            text_hash=text_hash,
            element_start=min(element.element_index for element in group),
            element_end=max(element.element_index for element in group),
            element_types=sorted({element.element_type for element in group}),
            page_number=page_number,
            char_count=len(text),
            word_count=len(text.split()),
            sentence_count=sentence_count or None,
            pipeline_run_id=run_id,
            extra=self._merge_extra(group),
        )
        return PreparedDocument(text=text, metadata=metadata)

    @staticmethod
    def _first_page_number(group: list[ProcessedElement]) -> int | None:
        for element in group:
            page = element.metadata.get("page_number")
            if isinstance(page, int):
                return page
        return None

    @staticmethod
    def _merge_extra(group: list[ProcessedElement]) -> dict[str, Any]:
        keys = {"languages", "file_directory", "filename", "last_modified", "token_count"}
        extra: dict[str, Any] = {}
        for key in keys:
            values = [element.metadata.get(key) for element in group if element.metadata.get(key)]
            if values:
                extra[key] = values[0] if len(set(map(str, values))) == 1 else values
        return extra

