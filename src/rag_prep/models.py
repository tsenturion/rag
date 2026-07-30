"""Типизированные модели данных для RAG-конвейера."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Возвращает текущее время в UTC с информацией о часовом поясе."""
    return datetime.now(timezone.utc)


class SourceFile(BaseModel):
    """Гарантирует неизменяемое описание исходного файла с уникальным идентификатором, хешем и метаданными для отслеживания происхождения данных."""

    path: Path
    source: str
    source_key: str
    file_name: str
    file_type: str
    source_hash: str
    size_bytes: int
    modified_at: datetime


class RawElement(BaseModel):
    """Гарантирует уникальную идентификацию и воспроизводимость текстового элемента, извлечённого из исходного файла, с полной трассировкой метаданных."""

    source_file: SourceFile
    element_id: str
    element_index: int
    text: str
    element_type: str
    section: str
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseFailure(BaseModel):
    """Гарантирует воспроизводимое описание ошибки парсинга с деталями источника и типа сбоя для диагностики и аудита."""

    source: str
    file_name: str
    file_type: str
    error_type: str
    error_message: str


class ParseResult(BaseModel):
    """Гарантирует вызывающему коду полный отчёт о результатах парсинга источника, включая успешно извлечённые элементы и все ошибки разбора для последующего анализа."""

    elements: list[RawElement] = Field(default_factory=list)
    failures: list[ParseFailure] = Field(default_factory=list)


class ProcessedElement(BaseModel):
    """Обеспечивает однозначную идентификацию и структурированное представление фрагмента исходного документа с сохранением контекста и метаданных для дальнейшей обработки в пайплайне."""

    source_file: SourceFile
    element_id: str
    element_index: int
    text: str
    element_type: str
    section: str
    section_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Фиксирует полный контракт происхождения, структуры и идентификации документа, обеспечивая трассируемость и воспроизводимость на всех этапах обработки."""

    id: str
    source: str
    source_key: str | None = None
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
    """Гарантирует, что текст и метаданные документа согласованы и пригодны для дальнейшей нарезки или экспорта без потери контекста."""

    text: str
    metadata: DocumentMetadata


class ChunkMetadata(BaseModel):
    """Обеспечивает воспроизводимость нарезки и идентификацию чанка с полным описанием происхождения, параметров токенизации и качества."""

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
    chunking_run_id: str = "legacy"
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
    """Гарантирует, что текстовый чанк и его метаданные согласованы и пригодны для последующей семантической обработки или векторизации."""

    text: str
    metadata: ChunkMetadata


class EmbeddedChunkMetadata(ChunkMetadata):
    """Фиксирует параметры и происхождение векторного представления чанка, обеспечивая верифицируемость embedding-процедуры и совместимость с downstream-задачами."""

    embedding_provider: str
    embedding_dimensions: int
    embedding_vector_hash: str
    embedding_norm: float
    embedding_run_id: str
    embedded_at: datetime = Field(default_factory=utc_now)


class EmbeddedChunk(BaseModel):
    """Гарантирует согласованность текста, embedding-вектора и метаданных для поиска и хранения в векторных индексах."""

    text: str
    embedding: list[float]
    metadata: EmbeddedChunkMetadata


class ArtifactExportModel(BaseModel):
    """Определяет контракт на предоставление путей экспортируемых артефактов, гарантируя отсутствие артефактов при их отсутствии в конкретной реализации."""

    def artifact_paths(self) -> list[Path]:
        """Гарантирует отсутствие экспортируемых артефактов, если конкретная модель их не предоставляет."""
        return []


class ExportResult(ArtifactExportModel):
    """Гарантирует вызывающему коду воспроизводимый набор путей к экспортированным артефактам и статистику по результатам выгрузки."""

    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    documents_count: int
    duplicates_removed: int
    run_id: str

    def artifact_paths(self) -> list[Path]:
        """Гарантирует вызывающему коду полный список путей к артефактам экспорта для отслеживания и автоматизации загрузки."""
        return [self.json_path, self.jsonl_path, self.manifest_path]


class PipelineResult(BaseModel):
    """Гарантирует агрегированную статистику и ссылки на экспортированные данные по завершении всего RAG-конвейера для мониторинга и автоматизации."""

    run_id: str
    sources_count: int
    raw_elements_count: int
    parse_failed_sources_count: int = 0
    prepared_documents_count: int
    duplicates_removed: int
    export: ExportResult


class ChunkingExportResult(ArtifactExportModel):
    """Обеспечивает гарантии наличия путей к артефактам чанкинга и количества чанков для последующей обработки и аудита в RAG-конвейере."""

    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    chunks_count: int
    run_id: str

    def artifact_paths(self) -> list[Path]:
        """Гарантирует вызывающему коду полный список путей к артефактам, созданным при экспорте чанков, для автоматизации последующих этапов."""
        return [self.json_path, self.jsonl_path, self.manifest_path]


class ChunkingValidationResult(BaseModel):
    """Предоставляет сводку ошибок чанкинга, позволяя быстро определить качество разбиения и необходимость корректировок перед дальнейшей обработкой."""

    no_chunks_count: int = 0
    empty_chunks_count: int = 0
    undersized_chunks_count: int = 0
    oversized_chunks_count: int = 0
    estimated_offsets_count: int = 0
    missing_parent_count: int = 0
    missing_lineage_count: int = 0
    low_quality_chunks_count: int = 0

    @property
    def has_errors(self) -> bool:
        """Проверяет, что результат валидации содержит хотя бы одну ошибку, чтобы обеспечить корректную обработку невалидных данных."""
        return any(
            [
                self.no_chunks_count,
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
    """Агрегирует результаты выполнения чанкинга, включая идентификатор запуска, метрики и артефакты, обеспечивая целостность данных в RAG-конвейере."""

    run_id: str
    documents_count: int
    chunks_count: int
    validation: ChunkingValidationResult
    export: ChunkingExportResult


class EmbeddingExportResult(ArtifactExportModel):
    """Гарантирует доступность путей к экспортированным эмбеддингам и их количеству для интеграции и последующего использования в RAG-конвейере."""

    json_path: Path
    jsonl_path: Path
    manifest_path: Path
    embeddings_count: int
    run_id: str

    def artifact_paths(self) -> list[Path]:
        """Гарантирует вызывающему коду полный список путей к артефактам экспорта эмбеддингов для автоматизации загрузки и анализа."""
        return [self.json_path, self.jsonl_path, self.manifest_path]


class EmbeddingValidationResult(BaseModel):
    """Фиксирует и сигнализирует о несоответствиях и ошибках эмбеддингов, обеспечивая контроль качества перед индексированием и поиском."""

    empty_source_chunks_count: int = 0
    empty_embeddings_count: int = 0
    chunk_count_mismatch: int = 0
    missing_embeddings_count: int = 0
    missing_chunk_ids_count: int = 0
    unexpected_chunk_ids_count: int = 0
    source_chunk_duplicate_ids_count: int = 0
    dimension_mismatch_count: int = 0
    non_finite_values_count: int = 0
    duplicate_chunk_ids_count: int = 0
    text_mismatch_count: int = 0
    metadata_mismatch_count: int = 0
    missing_metadata_count: int = 0
    model_mismatch_count: int = 0
    provider_mismatch_count: int = 0
    declared_dimension_mismatch_count: int = 0
    token_limit_exceeded_count: int = 0

    @property
    def has_errors(self) -> bool:
        """Проверяет, что результат валидации эмбеддингов содержит хотя бы одну ошибку, чтобы обеспечить надёжную фильтрацию невалидных данных."""
        return any(
            (
                self.empty_source_chunks_count,
                self.empty_embeddings_count,
                self.chunk_count_mismatch,
                self.missing_embeddings_count,
                self.missing_chunk_ids_count,
                self.unexpected_chunk_ids_count,
                self.source_chunk_duplicate_ids_count,
                self.dimension_mismatch_count,
                self.non_finite_values_count,
                self.duplicate_chunk_ids_count,
                self.text_mismatch_count,
                self.metadata_mismatch_count,
                self.missing_metadata_count,
                self.model_mismatch_count,
                self.provider_mismatch_count,
                self.declared_dimension_mismatch_count,
                self.token_limit_exceeded_count,
            )
        )


class EmbeddingPipelineResult(BaseModel):
    """Объединяет результаты эмбеддинга с валидацией и экспортом, поддерживая целостность и воспроизводимость этапа эмбеддинга в RAG-конвейере."""

    run_id: str
    chunks_count: int
    embeddings_count: int
    validation: EmbeddingValidationResult
    export: EmbeddingExportResult


class VectorStoreIndexResult(BaseModel):
    """Документирует параметры и результаты индексирования в векторном хранилище, обеспечивая прозрачность и контроль над состоянием коллекции."""

    collection_name: str
    provider: str
    mode: str
    points_upserted: int
    stale_points_deleted: int = 0
    collection_points_count: int
    vector_size: int
    distance: str
    storage_path: Path | None = None
    url: str | None = None


class VectorStoreValidationResult(BaseModel):
    """Обеспечивает детальный контроль качества и целостности данных в векторном хранилище, выявляя критические несоответствия и ошибки."""

    embeddings_count: int = 0
    empty_embeddings_count: int = 0
    collection_points_count: int = 0
    count_mismatch: int = 0
    count_delta: int = 0
    extra_points_count: int = 0
    missing_points_count: int = 0
    missing_vector_count: int = 0
    collection_vector_size_mismatch_count: int = 0
    point_vector_size_mismatch_count: int = 0
    vector_size_mismatch_count: int = 0
    distance_mismatch_count: int = 0
    missing_payload_count: int = 0
    missing_text_count: int = 0
    missing_metadata_count: int = 0
    missing_required_metadata_count: int = 0
    sampled_points_count: int = 0
    verified_points_count: int = 0
    missing_expected_points_count: int = 0
    chunk_id_mismatch_count: int = 0
    text_mismatch_count: int = 0
    identity_metadata_mismatch_count: int = 0
    vector_content_mismatch_count: int = 0

    @property
    def has_errors(self) -> bool:
        """Проверяет, что результат валидации векторного хранилища содержит хотя бы одну ошибку, чтобы гарантировать корректную обработку нарушений инвариантов."""
        return any(
            (
                self.empty_embeddings_count != 0,
                self.count_mismatch != 0,
                self.missing_vector_count != 0,
                self.vector_size_mismatch_count != 0,
                self.distance_mismatch_count != 0,
                self.missing_payload_count != 0,
                self.missing_text_count != 0,
                self.missing_metadata_count != 0,
                self.missing_required_metadata_count != 0,
                self.missing_expected_points_count != 0,
                self.chunk_id_mismatch_count != 0,
                self.text_mismatch_count != 0,
                self.identity_metadata_mismatch_count != 0,
                self.vector_content_mismatch_count != 0,
            )
        )


class VectorSearchHit(BaseModel):
    """Инкапсулирует информацию о найденном векторном совпадении с метаданными, обеспечивая точность и контекст для ранжирования и анализа."""

    point_id: str
    chunk_id: str | None = None
    score: float
    text: str | None = None
    source: str | None = None
    section: str | None = None
    position: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """Представляет результаты поиска по векторному индексу с метриками совпадений и порогами, гарантируя информативность и фильтрацию релевантных ответов."""

    query_chunk_id: str
    query_text: str
    hits: list[VectorSearchHit] = Field(default_factory=list)
    self_match_at_1: bool = False
    self_match_returned: bool = False
    unfiltered_self_match_at_1: bool | None = None
    score_threshold: float | None = None


class VectorStoreExportResult(ArtifactExportModel):
    """Гарантирует вызывающему коду доступ к путям всех артефактов экспорта векторного хранилища для воспроизводимости и автоматизации этапов RAG-конвейера."""

    manifest_path: Path
    validation_path: Path
    search_results_path: Path
    run_id: str

    def artifact_paths(self) -> list[Path]:
        """Гарантирует вызывающему коду полный список путей к артефактам экспорта для автоматизации последующих этапов RAG-конвейера."""
        return [self.manifest_path, self.validation_path, self.search_results_path]


class VectorStorePipelineResult(BaseModel):
    """Обеспечивает целостное описание результата выполнения пайплайна векторного хранилища с гарантиями валидации и экспорта для последующего анализа."""

    run_id: str
    embeddings_count: int
    points_count: int
    search_results_count: int
    validation: VectorStoreValidationResult
    export: VectorStoreExportResult
