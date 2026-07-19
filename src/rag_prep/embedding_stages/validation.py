"""Проверка выходных контрактов для расчёта embeddings."""

from __future__ import annotations

import logging
import math

from rag_prep.config import EmbeddingConfig
from rag_prep.models import EmbeddedChunk, EmbeddingValidationResult, PreparedChunk

LOGGER = logging.getLogger(__name__)


class EmbeddingValidationStage:
    """Валидирует embeddings перед индексацией в vector store."""

    def __init__(self, config: EmbeddingConfig):
        """Гарантирует готовность экземпляра к валидации embeddings с учётом всех параметров конфигурации."""
        self.config = config

    def run(
        self,
        chunks: list[PreparedChunk],
        embedded_chunks: list[EmbeddedChunk],
    ) -> EmbeddingValidationResult:
        """Сверяет embeddings с исходными чанками по ID, тексту и lineage."""
        expected_dimensions = self.config.dimensions
        source_chunk_ids = [chunk.metadata.id for chunk in chunks]
        embedded_chunk_ids = [chunk.metadata.id for chunk in embedded_chunks]
        source_ids = set(source_chunk_ids)
        embedded_ids = set(embedded_chunk_ids)
        # Наборы выявляют пропуски и неожиданные ID, а словари позволяют проверить,
        # что под общим ID не был подменён текст либо identity metadata.
        source_by_id = {chunk.metadata.id: chunk for chunk in chunks}
        embedded_by_id = {chunk.metadata.id: chunk for chunk in embedded_chunks}
        missing_chunk_ids = source_ids - embedded_ids
        unexpected_chunk_ids = embedded_ids - source_ids
        common_ids = source_ids & embedded_ids

        result = EmbeddingValidationResult(
            empty_source_chunks_count=int(not chunks),
            empty_embeddings_count=int(not embedded_chunks),
            chunk_count_mismatch=int(len(chunks) != len(embedded_chunks)),
            missing_embeddings_count=(
                len(missing_chunk_ids)
                + sum(1 for chunk in embedded_chunks if not chunk.embedding)
            ),
            missing_chunk_ids_count=len(missing_chunk_ids),
            unexpected_chunk_ids_count=len(unexpected_chunk_ids),
            source_chunk_duplicate_ids_count=(len(source_chunk_ids) - len(source_ids)),
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
            duplicate_chunk_ids_count=(len(embedded_chunk_ids) - len(embedded_ids)),
            text_mismatch_count=sum(
                1
                for chunk_id in common_ids
                if source_by_id[chunk_id].text != embedded_by_id[chunk_id].text
            ),
            metadata_mismatch_count=sum(
                1
                for chunk_id in common_ids
                if self._identity_metadata_mismatch(
                    source_by_id[chunk_id],
                    embedded_by_id[chunk_id],
                )
            ),
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
            provider_mismatch_count=sum(
                1
                for chunk in embedded_chunks
                if chunk.metadata.embedding_provider != self.config.provider
            ),
            declared_dimension_mismatch_count=sum(
                1
                for chunk in embedded_chunks
                if chunk.metadata.embedding_dimensions != len(chunk.embedding)
                or (
                    expected_dimensions is not None
                    and chunk.metadata.embedding_dimensions != expected_dimensions
                )
            ),
            token_limit_exceeded_count=sum(
                1
                for chunk in chunks
                if chunk.metadata.chunk_token_count > self.config.max_input_tokens
            ),
        )

        LOGGER.info(
            (
                "Проверено embeddings: %d; empty_source=%d empty_output=%d "
                "count_mismatch=%d missing=%d "
                "missing_ids=%d unexpected_ids=%d source_duplicate_ids=%d "
                "dimension_mismatch=%d non_finite=%d duplicate_ids=%d "
                "text_mismatch=%d metadata_mismatch=%d missing_metadata=%d "
                "model_mismatch=%d provider_mismatch=%d "
                "declared_dimension_mismatch=%d token_limit_exceeded=%d"
            ),
            len(embedded_chunks),
            result.empty_source_chunks_count,
            result.empty_embeddings_count,
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
            result.provider_mismatch_count,
            result.declared_dimension_mismatch_count,
            result.token_limit_exceeded_count,
        )
        if self.config.fail_on_validation_error and result.has_errors:
            raise ValueError(
                f"Валидация embeddings завершилась ошибкой: {result.model_dump()}"
            )
        return result

    @staticmethod
    def _identity_metadata_mismatch(
        source_chunk: PreparedChunk,
        embedded_chunk: EmbeddedChunk,
    ) -> bool:
        """Проверяет поля, которые embedding stage не вправе менять."""
        fields = (
            "document_id",
            "source",
            "section",
            "position",
            "source_hash",
            "document_text_hash",
            "text_hash",
        )
        return any(
            getattr(source_chunk.metadata, field)
            != getattr(embedded_chunk.metadata, field)
            for field in fields
        )
