"""Расчёт метрик для расчёта embeddings."""

from __future__ import annotations

from rag_prep.config import EmbeddingPipelineConfig
from rag_prep.models import EmbeddedChunk, EmbeddingValidationResult, PreparedChunk


def build_embedding_counts(
    config: EmbeddingPipelineConfig,
    chunks: list[PreparedChunk],
    embedded_chunks: list[EmbeddedChunk],
    validation: EmbeddingValidationResult,
) -> dict[str, int | float]:
    """Предоставляет вызывающему коду агрегированные количественные характеристики пайплайна для мониторинга и аудита."""
    dimensions = [chunk.metadata.embedding_dimensions for chunk in embedded_chunks]
    norms = [chunk.metadata.embedding_norm for chunk in embedded_chunks]
    return {
        "chunks_count": len(chunks),
        "embeddings_count": len(embedded_chunks),
        "embedding_dimensions": dimensions[0] if dimensions else 0,
        "unique_embedding_dimensions_count": len(set(dimensions)),
        "batch_size": config.embedding.batch_size,
        "max_batch_tokens": config.embedding.max_batch_tokens,
        "avg_embedding_norm": round(sum(norms) / len(norms), 6) if norms else 0.0,
        "min_embedding_norm": round(min(norms), 6) if norms else 0.0,
        "max_embedding_norm": round(max(norms), 6) if norms else 0.0,
        "empty_source_chunks_count": validation.empty_source_chunks_count,
        "empty_embeddings_count": validation.empty_embeddings_count,
        "chunk_count_mismatch": validation.chunk_count_mismatch,
        "missing_embeddings_count": validation.missing_embeddings_count,
        "missing_chunk_ids_count": validation.missing_chunk_ids_count,
        "unexpected_chunk_ids_count": validation.unexpected_chunk_ids_count,
        "source_chunk_duplicate_ids_count": validation.source_chunk_duplicate_ids_count,
        "dimension_mismatch_count": validation.dimension_mismatch_count,
        "non_finite_values_count": validation.non_finite_values_count,
        "duplicate_chunk_ids_count": validation.duplicate_chunk_ids_count,
        "text_mismatch_count": validation.text_mismatch_count,
        "metadata_mismatch_count": validation.metadata_mismatch_count,
        "missing_metadata_count": validation.missing_metadata_count,
        "model_mismatch_count": validation.model_mismatch_count,
        "provider_mismatch_count": validation.provider_mismatch_count,
        "declared_dimension_mismatch_count": (
            validation.declared_dimension_mismatch_count
        ),
        "token_limit_exceeded_count": validation.token_limit_exceeded_count,
    }


def build_embedding_diagnostics(
    validation: EmbeddingValidationResult,
) -> dict[str, object]:
    """Гарантирует сериализацию результатов валидации embeddings для последующего анализа."""
    return {"validation": validation.model_dump(mode="json")}
