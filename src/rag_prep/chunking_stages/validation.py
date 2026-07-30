"""Проверка выходных контрактов для чанкинга документов."""

from __future__ import annotations

import logging

from rag_prep.config import ChunkingConfig
from rag_prep.models import ChunkingValidationResult, PreparedChunk

LOGGER = logging.getLogger(__name__)


class ChunkValidationStage:
    """Валидирует чанки до расчёта embeddings в следующем пайплайне."""

    def __init__(self, config: ChunkingConfig):
        """Обеспечивает готовность экземпляра к валидации чанков с учётом заданной политики качества и ограничений."""
        self.config = config

    def run(self, chunks: list[PreparedChunk]) -> ChunkingValidationResult:
        """Гарантирует обнаружение и учёт всех нарушений политики чанкинга с возможностью аварийного завершения при ошибках."""
        result = ChunkingValidationResult(
            no_chunks_count=int(not chunks),
            empty_chunks_count=sum(1 for chunk in chunks if not chunk.text.strip()),
            undersized_chunks_count=sum(
                1
                for chunk in chunks
                if chunk.metadata.chunk_token_count < self.config.min_chunk_tokens
            ),
            oversized_chunks_count=sum(
                1
                for chunk in chunks
                if chunk.metadata.chunk_token_count > self.config.max_chunk_tokens
            ),
            estimated_offsets_count=sum(
                1 for chunk in chunks if "estimated" in chunk.metadata.offset_strategy
            ),
            missing_parent_count=sum(
                1 for chunk in chunks if not chunk.metadata.parent_ids
            ),
            missing_lineage_count=sum(
                1 for chunk in chunks if not chunk.metadata.lineage
            ),
            low_quality_chunks_count=sum(
                1
                for chunk in chunks
                if bool(chunk.metadata.quality.get("is_low_quality_chunk"))
            ),
        )

        LOGGER.info(
            (
                "Проверено чанков: %d; no_chunks=%d empty=%d undersized=%d oversized=%d "
                "estimated_offsets=%d missing_parent=%d missing_lineage=%d low_quality=%d"
            ),
            len(chunks),
            result.no_chunks_count,
            result.empty_chunks_count,
            result.undersized_chunks_count,
            result.oversized_chunks_count,
            result.estimated_offsets_count,
            result.missing_parent_count,
            result.missing_lineage_count,
            result.low_quality_chunks_count,
        )
        if result.no_chunks_count:
            raise ValueError("Чанкинг не сформировал ни одного чанка")
        if self.config.fail_on_validation_error and result.has_errors:
            raise ValueError(
                f"Валидация чанков завершилась ошибкой: {result.model_dump()}"
            )
        return result
