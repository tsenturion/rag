"""Расчёт метрик для индексации в Qdrant."""

from __future__ import annotations

from rag_prep.config import VectorStorePipelineConfig
from rag_prep.models import (
    EmbeddedChunk,
    VectorSearchResult,
    VectorStoreIndexResult,
    VectorStoreValidationResult,
)


def build_vector_store_counts(
    config: VectorStorePipelineConfig,
    embedded_chunks: list[EmbeddedChunk],
    index: VectorStoreIndexResult,
    validation: VectorStoreValidationResult,
    search_results: list[VectorSearchResult],
) -> dict[str, int | float]:
    """Гарантирует вызывающему коду агрегированные количественные метрики состояния индекса и поиска, необходимые для мониторинга и аудита загрузки в Qdrant."""
    self_matches = sum(1 for result in search_results if result.self_match_at_1)
    self_match_returned = sum(
        1 for result in search_results if result.self_match_returned
    )
    unfiltered_results = [
        result
        for result in search_results
        if result.unfiltered_self_match_at_1 is not None
    ]
    counts = {
        "embeddings_count": len(embedded_chunks),
        "points_upserted": index.points_upserted,
        "stale_points_deleted": index.stale_points_deleted,
        "collection_points_count": index.collection_points_count,
        "vector_size": config.vector_store.vector_size,
        "batch_size": config.vector_store.batch_size,
        "search_results_count": len(search_results),
        "search_hits_count": sum(len(result.hits) for result in search_results),
        "self_match_at_1_count": self_matches,
        "self_match_at_1_rate": round(self_matches / len(search_results), 6)
        if search_results
        else 0.0,
        "self_match_returned_count": self_match_returned,
        "self_match_returned_rate": round(self_match_returned / len(search_results), 6)
        if search_results
        else 0.0,
        "count_mismatch": validation.count_mismatch,
        "empty_embeddings_count": validation.empty_embeddings_count,
        "count_delta": validation.count_delta,
        "extra_points_count": validation.extra_points_count,
        "missing_points_count": validation.missing_points_count,
        "missing_vector_count": validation.missing_vector_count,
        "collection_vector_size_mismatch_count": (
            validation.collection_vector_size_mismatch_count
        ),
        "point_vector_size_mismatch_count": validation.point_vector_size_mismatch_count,
        "vector_size_mismatch_count": validation.vector_size_mismatch_count,
        "distance_mismatch_count": validation.distance_mismatch_count,
        "missing_payload_count": validation.missing_payload_count,
        "missing_text_count": validation.missing_text_count,
        "missing_metadata_count": validation.missing_metadata_count,
        "missing_required_metadata_count": validation.missing_required_metadata_count,
        "sampled_points_count": validation.sampled_points_count,
        "verified_points_count": validation.verified_points_count,
        "missing_expected_points_count": validation.missing_expected_points_count,
        "chunk_id_mismatch_count": validation.chunk_id_mismatch_count,
        "text_mismatch_count": validation.text_mismatch_count,
        "identity_metadata_mismatch_count": (
            validation.identity_metadata_mismatch_count
        ),
        "vector_content_mismatch_count": validation.vector_content_mismatch_count,
    }
    if unfiltered_results:
        unfiltered_self_matches = sum(
            1 for result in unfiltered_results if result.unfiltered_self_match_at_1
        )
        counts.update(
            {
                "unfiltered_self_match_at_1_count": unfiltered_self_matches,
                "unfiltered_self_match_at_1_rate": round(
                    unfiltered_self_matches / len(unfiltered_results),
                    6,
                ),
            }
        )
    return counts


def build_vector_store_diagnostics(
    validation: VectorStoreValidationResult,
    search_results: list[VectorSearchResult],
) -> dict[str, object]:
    """Гарантирует подробную диагностику ошибок и аномалий валидации и поиска, позволяя анализировать причины несоответствий при загрузке в Qdrant."""
    return {
        "validation": validation.model_dump(mode="json"),
        "search": {
            "queries_count": len(search_results),
            "self_match_at_1_failures": [
                result.query_chunk_id
                for result in search_results
                if not result.self_match_at_1
            ],
            "self_match_missing_from_results": [
                result.query_chunk_id
                for result in search_results
                if not result.self_match_returned
            ],
            "self_match_filtered_by_score_threshold": [
                result.query_chunk_id
                for result in search_results
                if result.unfiltered_self_match_at_1 and not result.self_match_returned
            ],
        },
    }
