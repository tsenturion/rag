from __future__ import annotations

from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

from rag_prep.chunking_stages import (
    ChunkExportStage,
    ChunkSplittingStage,
    ChunkValidationStage,
    PreparedDocumentLoadingStage,
)
from rag_prep.config import (
    ChunkingPipelineConfig,
    EmbeddingPipelineConfig,
    PipelineConfig,
    VectorStorePipelineConfig,
    load_chunking_config,
    load_config,
    load_embedding_config,
    load_vector_store_config,
)
from rag_prep.embedding_stages import (
    ChunkLoadingStage,
    EmbeddingExportStage,
    EmbeddingValidationStage,
    build_embedding_counts,
    build_embedding_diagnostics,
    build_embedding_stage,
    ensure_embedding_runtime,
)
from rag_prep.models import (
    ChunkingExportResult,
    ChunkingPipelineResult,
    ChunkingValidationResult,
    EmbeddedChunk,
    EmbeddingExportResult,
    EmbeddingPipelineResult,
    EmbeddingValidationResult,
    ExportResult,
    ParseResult,
    PipelineResult,
    PreparedChunk,
    PreparedDocument,
    ProcessedElement,
    RawElement,
    SourceFile,
    VectorSearchResult,
    VectorStoreExportResult,
    VectorStoreIndexResult,
    VectorStorePipelineResult,
    VectorStoreValidationResult,
)
from rag_prep.stages import (
    DeduplicationStage,
    ExportStage,
    LlamaIndexLoadingStage,
    LlamaIndexStructuringStage,
    TextCleaningStage,
    TextNormalizationStage,
    UnstructuredParsingStage,
)
from rag_prep.tracking import MLflowTracker
from rag_prep.utils import new_run_id, setup_logging
from rag_prep.vector_store_stages import (
    EmbeddingLoadingStage,
    QdrantIndexingStage,
    QdrantSearchStage,
    QdrantValidationStage,
    VectorStoreExportStage,
    build_vector_store_counts,
    build_vector_store_diagnostics,
    qdrant_client_context,
)


@task(name="load_sources")
def load_sources_task(config: PipelineConfig) -> list[SourceFile]:
    sources = LlamaIndexLoadingStage(config.loader).run(config.paths.input_dir)
    get_run_logger().info("Загружено источников: %d", len(sources))
    return sources


@task(name="parse_documents")
def parse_documents_task(
    config: PipelineConfig, sources: list[SourceFile]
) -> ParseResult:
    result = UnstructuredParsingStage(
        config.parser, default_section=config.structuring.default_section
    ).run(sources)
    get_run_logger().info(
        "Распарсено raw elements: %d; файлов с ошибкой: %d",
        len(result.elements),
        len(result.failures),
    )
    return result


@task(name="clean_text")
def clean_text_task(
    config: PipelineConfig, elements: list[RawElement]
) -> list[ProcessedElement]:
    cleaned = TextCleaningStage(config.cleaning).run(elements)
    get_run_logger().info("Очищено элементов: %d", len(cleaned))
    return cleaned


@task(name="normalize_text")
def normalize_text_task(
    config: PipelineConfig, elements: list[ProcessedElement]
) -> list[ProcessedElement]:
    normalized = TextNormalizationStage(config.normalization).run(elements)
    get_run_logger().info("Нормализовано элементов: %d", len(normalized))
    return normalized


@task(name="deduplicate_text")
def deduplicate_text_task(
    config: PipelineConfig, elements: list[ProcessedElement]
) -> tuple[list[ProcessedElement], int, int, int]:
    result = DeduplicationStage(config.deduplication).run(elements)
    get_run_logger().info(
        "Удалено дублей: %d; exact=%d near=%d",
        result.duplicates_removed,
        result.exact_duplicates_removed,
        result.near_duplicates_removed,
    )
    return (
        result.elements,
        result.duplicates_removed,
        result.exact_duplicates_removed,
        result.near_duplicates_removed,
    )


@task(name="structure_documents")
def structure_documents_task(
    config: PipelineConfig, elements: list[ProcessedElement], run_id: str
) -> tuple[list[PreparedDocument], int]:
    stage = LlamaIndexStructuringStage(config.structuring)
    documents = stage.run(elements, run_id=run_id)
    llama_documents = stage.to_llama_documents(documents)
    get_run_logger().info(
        "Сформировано подготовленных документов: %d; LlamaIndex documents: %d",
        len(documents),
        len(llama_documents),
    )
    return documents, len(llama_documents)


@task(name="export_documents")
def export_documents_task(
    config: PipelineConfig,
    documents: list[PreparedDocument],
    run_id: str,
    counts: dict[str, int],
    diagnostics: dict[str, object],
) -> ExportResult:
    export = ExportStage(config).run(
        documents,
        run_id=run_id,
        counts=counts,
        diagnostics=diagnostics,
    )
    get_run_logger().info("Экспортировано документов: %d", export.documents_count)
    return export


@task(name="log_mlflow")
def log_mlflow_task(
    config: PipelineConfig, counts: dict[str, int], export: ExportResult
) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-data-preparation")
def rag_data_preparation_flow(
    config_path: str = "config/default.yaml",
) -> PipelineResult:
    import random

    config = load_config(Path(config_path))
    setup_logging(config.logging.level)
    random.seed(config.run.seed)
    run_id = new_run_id()
    logger = get_run_logger()
    logger.info("Старт запуска подготовки данных RAG: %s", run_id)

    sources = load_sources_task(config)
    parse_result = parse_documents_task(config, sources)
    raw = parse_result.elements
    cleaned = clean_text_task(config, raw)
    normalized = normalize_text_task(config, cleaned)
    (
        deduped,
        duplicates_removed,
        exact_duplicates_removed,
        near_duplicates_removed,
    ) = deduplicate_text_task(config, normalized)
    documents, llama_documents_count = structure_documents_task(config, deduped, run_id)

    counts = {
        "sources_count": len(sources),
        "raw_elements_count": len(raw),
        "parse_failed_sources_count": len(parse_result.failures),
        "cleaned_elements_count": len(cleaned),
        "normalized_elements_count": len(normalized),
        "deduplicated_elements_count": len(deduped),
        "duplicates_removed": duplicates_removed,
        "exact_duplicates_removed": exact_duplicates_removed,
        "near_duplicates_removed": near_duplicates_removed,
        "prepared_documents_count": len(documents),
        "llama_index_documents_count": llama_documents_count,
    }
    diagnostics = {
        "parse_failures": [
            failure.model_dump(mode="json") for failure in parse_result.failures
        ]
    }
    export = export_documents_task(config, documents, run_id, counts, diagnostics)
    log_mlflow_task(config, counts, export)

    logger.info("Запуск завершён: %s", run_id)
    return PipelineResult(
        run_id=run_id,
        sources_count=len(sources),
        raw_elements_count=len(raw),
        parse_failed_sources_count=len(parse_result.failures),
        prepared_documents_count=len(documents),
        duplicates_removed=duplicates_removed,
        export=export,
    )


@task(name="load_prepared_documents")
def load_prepared_documents_task(
    config: ChunkingPipelineConfig,
) -> list[PreparedDocument]:
    documents = PreparedDocumentLoadingStage().run(config.paths.input_jsonl)
    get_run_logger().info("Загружено подготовленных документов: %d", len(documents))
    return documents


@task(name="split_chunks")
def split_chunks_task(
    config: ChunkingPipelineConfig, documents: list[PreparedDocument]
) -> list[PreparedChunk]:
    chunks = ChunkSplittingStage(config.chunking).run(documents)
    get_run_logger().info("Создано чанков: %d", len(chunks))
    return chunks


@task(name="validate_chunks")
def validate_chunks_task(
    config: ChunkingPipelineConfig, chunks: list[PreparedChunk]
) -> ChunkingValidationResult:
    result = ChunkValidationStage(config.chunking).run(chunks)
    get_run_logger().info(
        (
            "Проверены чанки: empty=%d undersized=%d oversized=%d "
            "estimated_offsets=%d missing_parent=%d missing_lineage=%d low_quality=%d"
        ),
        result.empty_chunks_count,
        result.undersized_chunks_count,
        result.oversized_chunks_count,
        result.estimated_offsets_count,
        result.missing_parent_count,
        result.missing_lineage_count,
        result.low_quality_chunks_count,
    )
    return result


@task(name="export_chunks")
def export_chunks_task(
    config: ChunkingPipelineConfig,
    chunks: list[PreparedChunk],
    run_id: str,
    counts: dict[str, int | float],
    diagnostics: dict[str, object],
) -> ChunkingExportResult:
    export = ChunkExportStage(config).run(
        chunks,
        run_id=run_id,
        counts=counts,
        diagnostics=diagnostics,
    )
    get_run_logger().info("Экспортировано чанков: %d", export.chunks_count)
    return export


@task(name="log_chunking_mlflow")
def log_chunking_mlflow_task(
    config: ChunkingPipelineConfig,
    counts: dict[str, int | float],
    export: ChunkingExportResult,
) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-chunking")
def rag_chunking_flow(
    config_path: str,
) -> ChunkingPipelineResult:
    import random

    config = load_chunking_config(Path(config_path))
    setup_logging(config.logging.level)
    random.seed(config.run.seed)
    run_id = new_run_id()
    logger = get_run_logger()
    logger.info("Старт запуска чанкинга RAG: %s", run_id)

    documents = load_prepared_documents_task(config)
    chunks = split_chunks_task(config, documents)
    validation = validate_chunks_task(config, chunks)

    token_counts = [chunk.metadata.chunk_token_count for chunk in chunks]
    structure_scores = [
        float(chunk.metadata.quality["structure_score"])
        for chunk in chunks
        if chunk.metadata.quality.get("structure_score") is not None
    ]
    unique_block_ids = {
        block_id for chunk in chunks for block_id in chunk.metadata.semantic_block_ids
    }
    counts = {
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "semantic_blocks_count": len(unique_block_ids),
        "multi_block_chunks_count": sum(
            1 for chunk in chunks if len(chunk.metadata.semantic_block_ids) > 1
        ),
        "avg_chunk_tokens": round(sum(token_counts) / len(token_counts), 3)
        if token_counts
        else 0.0,
        "max_chunk_tokens": max(token_counts) if token_counts else 0,
        "min_chunk_tokens": min(token_counts) if token_counts else 0,
        "avg_chunk_structure_score": round(
            sum(structure_scores) / len(structure_scores), 3
        )
        if structure_scores
        else 0.0,
        "empty_chunks_count": validation.empty_chunks_count,
        "undersized_chunks_count": validation.undersized_chunks_count,
        "oversized_chunks_count": validation.oversized_chunks_count,
        "estimated_offsets_count": validation.estimated_offsets_count,
        "missing_parent_count": validation.missing_parent_count,
        "missing_lineage_count": validation.missing_lineage_count,
        "low_quality_chunks_count": validation.low_quality_chunks_count,
    }
    diagnostics = {"validation": validation.model_dump(mode="json")}
    export = export_chunks_task(config, chunks, run_id, counts, diagnostics)
    log_chunking_mlflow_task(config, counts, export)

    logger.info("Запуск чанкинга завершён: %s", run_id)
    return ChunkingPipelineResult(
        run_id=run_id,
        documents_count=len(documents),
        chunks_count=len(chunks),
        validation=validation,
        export=export,
    )


@task(name="load_chunks")
def load_chunks_task(config: EmbeddingPipelineConfig) -> list[PreparedChunk]:
    chunks = ChunkLoadingStage().run(config.paths.input_jsonl)
    get_run_logger().info("Загружено чанков: %d", len(chunks))
    return chunks


@task(name="calculate_embeddings")
def calculate_embeddings_task(
    config: EmbeddingPipelineConfig,
    chunks: list[PreparedChunk],
    run_id: str,
) -> list[EmbeddedChunk]:
    embedded_chunks = build_embedding_stage(config.embedding).run(chunks, run_id=run_id)
    get_run_logger().info("Посчитано embeddings: %d", len(embedded_chunks))
    return embedded_chunks


@task(name="validate_embeddings")
def validate_embeddings_task(
    config: EmbeddingPipelineConfig,
    chunks: list[PreparedChunk],
    embedded_chunks: list[EmbeddedChunk],
) -> EmbeddingValidationResult:
    result = EmbeddingValidationStage(config.embedding).run(chunks, embedded_chunks)
    get_run_logger().info(
        (
            "Проверены embeddings: count_mismatch=%d missing=%d "
            "missing_ids=%d unexpected_ids=%d source_duplicate_ids=%d "
            "dimension_mismatch=%d non_finite=%d duplicate_ids=%d "
            "text_mismatch=%d metadata_mismatch=%d missing_metadata=%d "
            "model_mismatch=%d token_limit_exceeded=%d"
        ),
        result.chunk_count_mismatch,
        result.missing_embeddings_count,
        result.missing_chunk_ids_count,
        result.unexpected_chunk_ids_count,
        result.source_chunk_duplicate_ids_count,
        result.dimension_mismatch_count,
        result.non_finite_values_count,
        result.duplicate_chunk_ids_count,
        result.text_mismatch_count,
        result.metadata_mismatch_count,
        result.missing_metadata_count,
        result.model_mismatch_count,
        result.token_limit_exceeded_count,
    )
    return result


@task(name="export_embeddings")
def export_embeddings_task(
    config: EmbeddingPipelineConfig,
    embedded_chunks: list[EmbeddedChunk],
    run_id: str,
    counts: dict[str, int | float],
    diagnostics: dict[str, object],
) -> EmbeddingExportResult:
    export = EmbeddingExportStage(config).run(
        embedded_chunks,
        run_id=run_id,
        counts=counts,
        diagnostics=diagnostics,
    )
    get_run_logger().info("Экспортировано embeddings: %d", export.embeddings_count)
    return export


@task(name="log_embedding_mlflow")
def log_embedding_mlflow_task(
    config: EmbeddingPipelineConfig,
    counts: dict[str, int | float],
    export: EmbeddingExportResult,
) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-embeddings")
def rag_embeddings_flow(
    config_path: str,
) -> EmbeddingPipelineResult:
    import random

    config = load_embedding_config(Path(config_path))
    setup_logging(config.logging.level)
    ensure_embedding_runtime(config.embedding)
    random.seed(config.run.seed)
    run_id = new_run_id()
    logger = get_run_logger()
    logger.info("Старт запуска embeddings RAG: %s", run_id)

    chunks = load_chunks_task(config)
    embedded_chunks = calculate_embeddings_task(config, chunks, run_id)
    validation = validate_embeddings_task(config, chunks, embedded_chunks)

    counts = build_embedding_counts(config, chunks, embedded_chunks, validation)
    diagnostics = build_embedding_diagnostics(validation)
    export = export_embeddings_task(
        config, embedded_chunks, run_id, counts, diagnostics
    )
    log_embedding_mlflow_task(config, counts, export)

    logger.info("Запуск embeddings завершён: %s", run_id)
    return EmbeddingPipelineResult(
        run_id=run_id,
        chunks_count=len(chunks),
        embeddings_count=len(embedded_chunks),
        validation=validation,
        export=export,
    )


@task(name="load_embedding_records")
def load_embedding_records_task(
    config: VectorStorePipelineConfig,
) -> list[EmbeddedChunk]:
    embedded_chunks = EmbeddingLoadingStage().run(config.paths.input_jsonl)
    get_run_logger().info("Загружено записей embeddings: %d", len(embedded_chunks))
    return embedded_chunks


@task(name="index_qdrant")
def index_qdrant_task(
    config: VectorStorePipelineConfig,
    embedded_chunks: list[EmbeddedChunk],
) -> VectorStoreIndexResult:
    index = QdrantIndexingStage(config.vector_store).run(embedded_chunks)
    get_run_logger().info(
        "Проиндексировано Qdrant points: %d в коллекции %s",
        index.points_upserted,
        index.collection_name,
    )
    return index


@task(name="validate_qdrant")
def validate_qdrant_task(
    config: VectorStorePipelineConfig,
    embedded_chunks: list[EmbeddedChunk],
) -> VectorStoreValidationResult:
    validation = QdrantValidationStage(config.vector_store).run(embedded_chunks)
    get_run_logger().info(
        (
            "Проверена коллекция Qdrant: count_mismatch=%d count_delta=%d "
            "extra_points=%d missing_points=%d missing_vectors=%d "
            "collection_vector_size_mismatch=%d point_vector_size_mismatch=%d "
            "vector_size_mismatch=%d distance_mismatch=%d missing_payload=%d "
            "missing_text=%d missing_metadata=%d missing_required_metadata=%d"
        ),
        validation.count_mismatch,
        validation.count_delta,
        validation.extra_points_count,
        validation.missing_points_count,
        validation.missing_vector_count,
        validation.collection_vector_size_mismatch_count,
        validation.point_vector_size_mismatch_count,
        validation.vector_size_mismatch_count,
        validation.distance_mismatch_count,
        validation.missing_payload_count,
        validation.missing_text_count,
        validation.missing_metadata_count,
        validation.missing_required_metadata_count,
    )
    return validation


@task(name="search_qdrant")
def search_qdrant_task(
    config: VectorStorePipelineConfig,
    embedded_chunks: list[EmbeddedChunk],
) -> list[VectorSearchResult]:
    search_results = QdrantSearchStage(config.vector_store).run(embedded_chunks)
    get_run_logger().info(
        "Выполнено Qdrant search smoke-тестов: %d", len(search_results)
    )
    return search_results


@task(name="index_validate_search_qdrant")
def index_validate_search_qdrant_task(
    config: VectorStorePipelineConfig,
    embedded_chunks: list[EmbeddedChunk],
) -> tuple[
    VectorStoreIndexResult, VectorStoreValidationResult, list[VectorSearchResult]
]:
    with qdrant_client_context(config.vector_store) as client:
        index = QdrantIndexingStage(config.vector_store).run(
            embedded_chunks,
            client=client,
        )
        validation = QdrantValidationStage(config.vector_store).run(
            embedded_chunks,
            client=client,
        )
        search_results = QdrantSearchStage(config.vector_store).run(
            embedded_chunks,
            client=client,
        )

    get_run_logger().info(
        (
            "Проиндексирована и проверена коллекция Qdrant %s: points=%d "
            "count_mismatch=%d count_delta=%d search_queries=%d"
        ),
        index.collection_name,
        index.collection_points_count,
        validation.count_mismatch,
        validation.count_delta,
        len(search_results),
    )
    return index, validation, search_results


@task(name="export_vector_store_report")
def export_vector_store_report_task(
    config: VectorStorePipelineConfig,
    index: VectorStoreIndexResult,
    validation: VectorStoreValidationResult,
    search_results: list[VectorSearchResult],
    run_id: str,
    counts: dict[str, int | float],
    diagnostics: dict[str, object],
) -> VectorStoreExportResult:
    export = VectorStoreExportStage(config).run(
        index=index,
        validation=validation,
        search_results=search_results,
        run_id=run_id,
        counts=counts,
        diagnostics=diagnostics,
    )
    get_run_logger().info("Экспортирован отчёт vector store")
    return export


@task(name="log_vector_store_mlflow")
def log_vector_store_mlflow_task(
    config: VectorStorePipelineConfig,
    counts: dict[str, int | float],
    export: VectorStoreExportResult,
) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-vector-store")
def rag_vector_store_flow(
    config_path: str,
) -> VectorStorePipelineResult:
    import random

    config = load_vector_store_config(Path(config_path))
    setup_logging(config.logging.level)
    random.seed(config.run.seed)
    run_id = new_run_id()
    logger = get_run_logger()
    logger.info("Старт запуска vector store RAG: %s", run_id)

    embedded_chunks = load_embedding_records_task(config)
    index, validation, search_results = index_validate_search_qdrant_task(
        config,
        embedded_chunks,
    )

    counts = build_vector_store_counts(
        config,
        embedded_chunks,
        index,
        validation,
        search_results,
    )
    diagnostics = build_vector_store_diagnostics(validation, search_results)
    export = export_vector_store_report_task(
        config,
        index,
        validation,
        search_results,
        run_id,
        counts,
        diagnostics,
    )
    log_vector_store_mlflow_task(config, counts, export)

    logger.info("Запуск vector store завершён: %s", run_id)
    return VectorStorePipelineResult(
        run_id=run_id,
        embeddings_count=len(embedded_chunks),
        points_count=index.collection_points_count,
        search_results_count=len(search_results),
        validation=validation,
        export=export,
    )
