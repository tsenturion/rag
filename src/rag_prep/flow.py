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
    load_chunking_config,
    load_config,
    load_embedding_config,
)
from rag_prep.embedding_stages import (
    ChunkLoadingStage,
    EmbeddingExportStage,
    EmbeddingValidationStage,
    OpenAIEmbeddingStage,
    build_embedding_counts,
    build_embedding_diagnostics,
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


@task(name="load_sources")
def load_sources_task(config: PipelineConfig) -> list[SourceFile]:
    sources = LlamaIndexLoadingStage(config.loader).run(config.paths.input_dir)
    get_run_logger().info("Loaded %d sources", len(sources))
    return sources


@task(name="parse_documents")
def parse_documents_task(config: PipelineConfig, sources: list[SourceFile]) -> ParseResult:
    result = UnstructuredParsingStage(
        config.parser, default_section=config.structuring.default_section
    ).run(sources)
    get_run_logger().info(
        "Parsed %d raw elements; %d files failed",
        len(result.elements),
        len(result.failures),
    )
    return result


@task(name="clean_text")
def clean_text_task(config: PipelineConfig, elements: list[RawElement]) -> list[ProcessedElement]:
    cleaned = TextCleaningStage(config.cleaning).run(elements)
    get_run_logger().info("Cleaned %d elements", len(cleaned))
    return cleaned


@task(name="normalize_text")
def normalize_text_task(
    config: PipelineConfig, elements: list[ProcessedElement]
) -> list[ProcessedElement]:
    normalized = TextNormalizationStage(config.normalization).run(elements)
    get_run_logger().info("Normalized %d elements", len(normalized))
    return normalized


@task(name="deduplicate_text")
def deduplicate_text_task(
    config: PipelineConfig, elements: list[ProcessedElement]
) -> tuple[list[ProcessedElement], int, int, int]:
    result = DeduplicationStage(config.deduplication).run(elements)
    get_run_logger().info(
        "Removed %d duplicates: exact=%d near=%d",
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
        "Structured %d prepared documents and %d LlamaIndex documents",
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
    get_run_logger().info("Exported %d documents", export.documents_count)
    return export


@task(name="log_mlflow")
def log_mlflow_task(config: PipelineConfig, counts: dict[str, int], export: ExportResult) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-data-preparation")
def rag_data_preparation_flow(config_path: str = "config/default.yaml") -> PipelineResult:
    import random

    config = load_config(Path(config_path))
    setup_logging(config.logging.level)
    random.seed(config.run.seed)
    run_id = new_run_id()
    logger = get_run_logger()
    logger.info("Starting RAG data preparation run %s", run_id)

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

    logger.info("Finished run %s", run_id)
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
def load_prepared_documents_task(config: ChunkingPipelineConfig) -> list[PreparedDocument]:
    documents = PreparedDocumentLoadingStage().run(config.paths.input_jsonl)
    get_run_logger().info("Loaded %d prepared documents", len(documents))
    return documents


@task(name="split_chunks")
def split_chunks_task(
    config: ChunkingPipelineConfig, documents: list[PreparedDocument]
) -> list[PreparedChunk]:
    chunks = ChunkSplittingStage(config.chunking).run(documents)
    get_run_logger().info("Created %d chunks", len(chunks))
    return chunks


@task(name="validate_chunks")
def validate_chunks_task(
    config: ChunkingPipelineConfig, chunks: list[PreparedChunk]
) -> ChunkingValidationResult:
    result = ChunkValidationStage(config.chunking).run(chunks)
    get_run_logger().info(
        (
            "Validated chunks: empty=%d undersized=%d oversized=%d "
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
    get_run_logger().info("Exported %d chunks", export.chunks_count)
    return export


@task(name="log_chunking_mlflow")
def log_chunking_mlflow_task(
    config: ChunkingPipelineConfig,
    counts: dict[str, int | float],
    export: ChunkingExportResult,
) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-chunking")
def rag_chunking_flow(config_path: str = "config/chunking.yaml") -> ChunkingPipelineResult:
    import random

    config = load_chunking_config(Path(config_path))
    setup_logging(config.logging.level)
    random.seed(config.run.seed)
    run_id = new_run_id()
    logger = get_run_logger()
    logger.info("Starting RAG chunking run %s", run_id)

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
        block_id
        for chunk in chunks
        for block_id in chunk.metadata.semantic_block_ids
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

    logger.info("Finished chunking run %s", run_id)
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
    get_run_logger().info("Loaded %d chunks", len(chunks))
    return chunks


@task(name="calculate_embeddings")
def calculate_embeddings_task(
    config: EmbeddingPipelineConfig,
    chunks: list[PreparedChunk],
    run_id: str,
) -> list[EmbeddedChunk]:
    embedded_chunks = OpenAIEmbeddingStage(config.embedding).run(chunks, run_id=run_id)
    get_run_logger().info("Calculated %d embeddings", len(embedded_chunks))
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
            "Validated embeddings: count_mismatch=%d missing=%d "
            "dimension_mismatch=%d non_finite=%d duplicate_ids=%d "
            "missing_metadata=%d model_mismatch=%d token_limit_exceeded=%d"
        ),
        result.chunk_count_mismatch,
        result.missing_embeddings_count,
        result.dimension_mismatch_count,
        result.non_finite_values_count,
        result.duplicate_chunk_ids_count,
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
    get_run_logger().info("Exported %d embeddings", export.embeddings_count)
    return export


@task(name="log_embedding_mlflow")
def log_embedding_mlflow_task(
    config: EmbeddingPipelineConfig,
    counts: dict[str, int | float],
    export: EmbeddingExportResult,
) -> None:
    MLflowTracker(config).log_run(counts, export)


@flow(name="rag-embeddings")
def rag_embeddings_flow(config_path: str = "config/embeddings.yaml") -> EmbeddingPipelineResult:
    import random

    config = load_embedding_config(Path(config_path))
    setup_logging(config.logging.level)
    OpenAIEmbeddingStage.ensure_api_key(config.embedding)
    random.seed(config.run.seed)
    run_id = new_run_id()
    logger = get_run_logger()
    logger.info("Starting RAG embeddings run %s", run_id)

    chunks = load_chunks_task(config)
    embedded_chunks = calculate_embeddings_task(config, chunks, run_id)
    validation = validate_embeddings_task(config, chunks, embedded_chunks)

    counts = build_embedding_counts(config, chunks, embedded_chunks, validation)
    diagnostics = build_embedding_diagnostics(validation)
    export = export_embeddings_task(config, embedded_chunks, run_id, counts, diagnostics)
    log_embedding_mlflow_task(config, counts, export)

    logger.info("Finished embeddings run %s", run_id)
    return EmbeddingPipelineResult(
        run_id=run_id,
        chunks_count=len(chunks),
        embeddings_count=len(embedded_chunks),
        validation=validation,
        export=export,
    )
