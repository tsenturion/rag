from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceFile(BaseModel):
    path: Path
    source: str
    file_name: str
    file_type: str
    source_hash: str
    size_bytes: int
    modified_at: datetime


class RawElement(BaseModel):
    source_file: SourceFile
    element_id: str
    element_index: int
    text: str
    element_type: str
    section: str
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseFailure(BaseModel):
    source: str
    file_name: str
    file_type: str
    error_type: str
    error_message: str


class ParseResult(BaseModel):
    elements: list[RawElement] = Field(default_factory=list)
    failures: list[ParseFailure] = Field(default_factory=list)


class ProcessedElement(BaseModel):
    source_file: SourceFile
    element_id: str
    element_index: int
    text: str
    element_type: str
    section: str
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    id: str
    source: str
    section: str
    file_name: str
    file_type: str
    source_hash: str
    text_hash: str
    parent_ids: list[str] = Field(default_factory=list)
    origin_element_ids: list[str] = Field(default_factory=list)
    lineage: dict[str, Any] = Field(default_factory=dict)
    hierarchy: dict[str, Any] = Field(default_factory=dict)
    element_start: int
    element_end: int
    element_types: list[str]
    page_number: int | None = None
    char_count: int
    word_count: int
    sentence_count: int | None = None
    pipeline_run_id: str
    parsed_at: datetime = Field(default_factory=utc_now)
    extra: dict[str, Any] = Field(default_factory=dict)


class PreparedDocument(BaseModel):
    text: str
    metadata: DocumentMetadata


class ExportResult(BaseModel):
    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    documents_count: int
    duplicates_removed: int
    run_id: str


class PipelineResult(BaseModel):
    run_id: str
    sources_count: int
    raw_elements_count: int
    parse_failed_sources_count: int = 0
    prepared_documents_count: int
    duplicates_removed: int
    export: ExportResult
