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


class ChunkMetadata(BaseModel):
    id: str
    document_id: str
    source: str
    section: str
    position: int
    chunk_start_char: int
    chunk_end_char: int
    chunk_token_count: int
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    tokenizer_model: str
    embedding_model: str
    semantic_block_ids: list[str] = Field(default_factory=list)
    semantic_block_start: int | None = None
    semantic_block_end: int | None = None
    offset_strategy: str = "semantic_block_span"
    parent_ids: list[str] = Field(default_factory=list)
    origin_element_ids: list[str] = Field(default_factory=list)
    lineage: dict[str, Any] = Field(default_factory=dict)
    hierarchy: dict[str, Any] = Field(default_factory=dict)
    source_hash: str
    document_text_hash: str
    text_hash: str
    file_name: str
    file_type: str
    quality: dict[str, Any] = Field(default_factory=dict)
    chunked_at: datetime = Field(default_factory=utc_now)


class PreparedChunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class EmbeddedChunkMetadata(ChunkMetadata):
    embedding_provider: str
    embedding_dimensions: int
    embedding_vector_hash: str
    embedding_norm: float
    embedding_run_id: str
    embedded_at: datetime = Field(default_factory=utc_now)


class EmbeddedChunk(BaseModel):
    text: str
    embedding: list[float]
    metadata: EmbeddedChunkMetadata


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


class ChunkingExportResult(BaseModel):
    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    chunks_count: int
    run_id: str


class ChunkingValidationResult(BaseModel):
    empty_chunks_count: int = 0
    undersized_chunks_count: int = 0
    oversized_chunks_count: int = 0
    estimated_offsets_count: int = 0
    missing_parent_count: int = 0
    missing_lineage_count: int = 0
    low_quality_chunks_count: int = 0

    @property
    def has_errors(self) -> bool:
        return any(
            [
                self.empty_chunks_count,
                self.undersized_chunks_count,
                self.oversized_chunks_count,
                self.estimated_offsets_count,
                self.missing_parent_count,
                self.missing_lineage_count,
                self.low_quality_chunks_count,
            ]
        )


class ChunkingPipelineResult(BaseModel):
    run_id: str
    documents_count: int
    chunks_count: int
    validation: ChunkingValidationResult
    export: ChunkingExportResult


class EmbeddingExportResult(BaseModel):
    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    embeddings_count: int
    run_id: str


class EmbeddingValidationResult(BaseModel):
    chunk_count_mismatch: int = 0
    missing_embeddings_count: int = 0
    dimension_mismatch_count: int = 0
    non_finite_values_count: int = 0
    duplicate_chunk_ids_count: int = 0
    missing_metadata_count: int = 0
    model_mismatch_count: int = 0
    token_limit_exceeded_count: int = 0

    @property
    def has_errors(self) -> bool:
        return any(
            (
                self.chunk_count_mismatch,
                self.missing_embeddings_count,
                self.dimension_mismatch_count,
                self.non_finite_values_count,
                self.duplicate_chunk_ids_count,
                self.missing_metadata_count,
                self.model_mismatch_count,
                self.token_limit_exceeded_count,
            )
        )


class EmbeddingPipelineResult(BaseModel):
    run_id: str
    chunks_count: int
    embeddings_count: int
    validation: EmbeddingValidationResult
    export: EmbeddingExportResult
