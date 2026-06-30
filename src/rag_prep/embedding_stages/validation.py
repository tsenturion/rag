from __future__ import annotations

import logging
import math

from rag_prep.config import EmbeddingConfig
from rag_prep.models import EmbeddedChunk, EmbeddingValidationResult, PreparedChunk

LOGGER = logging.getLogger(__name__)


class EmbeddingValidationStage:
    """Валидирует embeddings перед индексацией в vector store."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def run(
        self,
        chunks: list[PreparedChunk],
        embedded_chunks: list[EmbeddedChunk],
    ) -> EmbeddingValidationResult:
        expected_dimensions = self.config.dimensions
        chunk_ids = [chunk.metadata.id for chunk in embedded_chunks]
        duplicate_chunk_ids_count = len(chunk_ids) - len(set(chunk_ids))

        result = EmbeddingValidationResult(
            chunk_count_mismatch=int(len(chunks) != len(embedded_chunks)),
            missing_embeddings_count=sum(
                1 for chunk in embedded_chunks if not chunk.embedding
            ),
            dimension_mismatch_count=sum(
                1
                for chunk in embedded_chunks
                if expected_dimensions is not None
                and len(chunk.embedding) != expected_dimensions
            ),
            non_finite_values_count=sum(
                1
                for chunk in embedded_chunks
                if any(not math.isfinite(value) for value in chunk.embedding)
            ),
            duplicate_chunk_ids_count=duplicate_chunk_ids_count,
            missing_metadata_count=sum(
                1
                for chunk in embedded_chunks
                if not chunk.metadata.id
                or not chunk.metadata.source
                or chunk.metadata.position is None
            ),
            model_mismatch_count=sum(
                1
                for chunk in embedded_chunks
                if chunk.metadata.embedding_model != self.config.model
            ),
            token_limit_exceeded_count=sum(
                1
                for chunk in chunks
                if chunk.metadata.chunk_token_count > self.config.max_input_tokens
            ),
        )

        LOGGER.info(
            (
                "Проверено embeddings: %d; count_mismatch=%d missing=%d "
                "dimension_mismatch=%d non_finite=%d duplicate_ids=%d "
                "missing_metadata=%d model_mismatch=%d token_limit_exceeded=%d"
            ),
            len(embedded_chunks),
            result.chunk_count_mismatch,
            result.missing_embeddings_count,
            result.dimension_mismatch_count,
            result.non_finite_values_count,
            result.duplicate_chunk_ids_count,
            result.missing_metadata_count,
            result.model_mismatch_count,
            result.token_limit_exceeded_count,
        )
        if self.config.fail_on_validation_error and result.has_errors:
            raise ValueError(f"Валидация embeddings завершилась ошибкой: {result.model_dump()}")
        return result
