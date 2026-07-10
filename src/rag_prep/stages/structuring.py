from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Document

from rag_prep.config import StructuringConfig
from rag_prep.models import DocumentMetadata, PreparedDocument, ProcessedElement
from rag_prep.utils import stable_id, text_sha256

LOGGER = logging.getLogger(__name__)


class LlamaIndexStructuringStage:
    """Создаёт Pydantic и LlamaIndex документы для следующих этапов RAG."""

    def __init__(self, config: StructuringConfig):
        self.config = config

    def run(
        self, elements: list[ProcessedElement], run_id: str
    ) -> list[PreparedDocument]:
        documents = (
            self._group_by_section(elements, run_id)
            if self.config.group_by_section
            else self._one_document_per_element(elements, run_id)
        )
        LOGGER.info("Сформировано подготовленных документов: %d", len(documents))
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
        groups: list[tuple[str, list[ProcessedElement]]] = []
        current_key: tuple[str, str] | None = None
        current_group: list[ProcessedElement] = []

        for element in elements:
            section = element.section or self.config.default_section
            key = (element.source_file.source, section)
            if current_key is not None and key != current_key:
                groups.append((current_key[1], current_group))
                current_group = []
            current_key = key
            current_group.append(element)

        if current_key is not None:
            groups.append((current_key[1], current_group))

        documents: list[PreparedDocument] = []
        for section, group in groups:
            documents.append(
                self._build_document(group=group, section=section, run_id=run_id)
            )
        return documents

    def _one_document_per_element(
        self, elements: list[ProcessedElement], run_id: str
    ) -> list[PreparedDocument]:
        return [
            self._build_document(
                group=[element], section=element.section, run_id=run_id
            )
            for element in elements
        ]

    def _build_document(
        self, group: list[ProcessedElement], section: str, run_id: str
    ) -> PreparedDocument:
        first = group[0]
        text = "\n\n".join(element.text for element in group).strip()
        text_hash = text_sha256(text)
        page_number = self._first_page_number(group)
        element_start = min(element.element_index for element in group)
        element_end = max(element.element_index for element in group)
        origin_element_ids = [element.element_id for element in group]
        source_id = stable_id(first.source_file.source_hash)
        section_path = self._section_path(group, section)
        sentence_counts = [
            int(element.metadata["sentence_count"])
            for element in group
            if element.metadata.get("sentence_count") is not None
        ]
        sentence_count = sum(sentence_counts) if sentence_counts else None
        metadata = DocumentMetadata(
            id=stable_id(
                first.source_file.source_hash,
                section,
                element_start,
                element_end,
                text_hash,
            ),
            source=first.source_file.source,
            source_key=first.source_file.source_key,
            section=section or self.config.default_section,
            file_name=first.source_file.file_name,
            file_type=first.source_file.file_type,
            source_hash=first.source_file.source_hash,
            text_hash=text_hash,
            parent_ids=[source_id],
            origin_element_ids=origin_element_ids,
            lineage={
                "source_id": source_id,
                "source_key": first.source_file.source_key,
                "source_hash": first.source_file.source_hash,
                "origin_element_ids": origin_element_ids,
                "element_range": [element_start, element_end],
                "pipeline_stage": "prepared_document",
            },
            hierarchy={
                "section_path": section_path,
                "section_depth": len(section_path),
                "document_order": element_start,
            },
            element_start=element_start,
            element_end=element_end,
            element_types=sorted({element.element_type for element in group}),
            page_number=page_number,
            char_count=len(text),
            word_count=len(text.split()),
            sentence_count=sentence_count,
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
    def _section_path(group: list[ProcessedElement], section: str) -> list[str]:
        for element in group:
            if element.section_path:
                return element.section_path
        return [section] if section else []

    @staticmethod
    def _merge_extra(group: list[ProcessedElement]) -> dict[str, Any]:
        keys = {
            "languages",
            "file_directory",
            "filename",
            "last_modified",
            "token_count",
        }
        extra: dict[str, Any] = {}
        for key in keys:
            values = [
                element.metadata.get(key)
                for element in group
                if element.metadata.get(key) is not None
            ]
            if values:
                if key == "token_count":
                    extra[key] = sum(int(value) for value in values)
                else:
                    extra[key] = (
                        values[0] if len(set(map(str, values))) == 1 else values
                    )
        quality_values = [
            element.metadata.get("quality")
            for element in group
            if isinstance(element.metadata.get("quality"), dict)
        ]
        if quality_values:
            extra["quality"] = LlamaIndexStructuringStage._merge_quality(quality_values)
        return extra

    @staticmethod
    def _merge_quality(values: list[dict[str, Any]]) -> dict[str, Any]:
        numeric_keys = ["meaningful_score", "boilerplate_score", "garbage_score"]
        merged: dict[str, Any] = {}
        for key in numeric_keys:
            scores = [
                float(value[key]) for value in values if value.get(key) is not None
            ]
            if scores:
                merged[key] = round(sum(scores) / len(scores), 3)
        merged["is_probable_boilerplate"] = any(
            bool(value.get("is_probable_boilerplate")) for value in values
        )
        merged["is_probable_garbage"] = any(
            bool(value.get("is_probable_garbage")) for value in values
        )
        patterns = {
            pattern
            for value in values
            for pattern in value.get("matched_boilerplate_patterns", [])
        }
        merged["matched_boilerplate_patterns"] = sorted(patterns)
        return merged
