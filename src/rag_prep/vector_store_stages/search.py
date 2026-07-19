"""Поиск и формирование результатов для индексации в Qdrant."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from qdrant_client import QdrantClient

from rag_prep.config import VectorStoreConfig
from rag_prep.models import EmbeddedChunk, VectorSearchHit, VectorSearchResult
from rag_prep.vector_store_stages.client import qdrant_client_context

LOGGER = logging.getLogger(__name__)


class QdrantSearchStage:
    """Запускает smoke-проверки similarity search по проиндексированным vectors."""

    def __init__(self, config: VectorStoreConfig):
        """Обеспечивает готовность экземпляра к поисковым операциям по заданной конфигурации без владения внешними ресурсами."""
        self.config = config

    def run(
        self,
        embedded_chunks: list[EmbeddedChunk],
        *,
        client: QdrantClient | None = None,
    ) -> list[VectorSearchResult]:
        """Гарантирует корректное выполнение тестовых поисковых запросов по эмбеддингам с контролем числа запросов и управлением клиентом Qdrant."""
        if self.config.test_queries_count == 0 or not embedded_chunks:
            return []

        if client is None:
            with qdrant_client_context(self.config) as owned_client:
                return self.run(embedded_chunks, client=owned_client)

        queries = embedded_chunks[: self.config.test_queries_count]
        results = [self._search_one(client, query) for query in queries]
        LOGGER.info(
            "Выполнено тестовых Qdrant similarity search запросов: %d", len(results)
        )
        return results

    def _search_one(self, client, query: EmbeddedChunk) -> VectorSearchResult:
        """Гарантирует получение и анализ результатов поиска по одному эмбеддингу с учётом порога score и идентификации self-match."""
        response = client.query_points(
            collection_name=self.config.collection_name,
            query=query.embedding,
            limit=self.config.search_limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=self.config.score_threshold,
        )
        hits = [self._hit(point) for point in response.points]
        self_match_at_1 = bool(hits and hits[0].chunk_id == query.metadata.id)
        self_match_returned = any(hit.chunk_id == query.metadata.id for hit in hits)
        unfiltered_self_match_at_1 = None
        if self.config.score_threshold is not None:
            unfiltered = client.query_points(
                collection_name=self.config.collection_name,
                query=query.embedding,
                limit=1,
                with_payload=True,
                with_vectors=False,
                score_threshold=None,
            )
            unfiltered_hits = [self._hit(point) for point in unfiltered.points]
            unfiltered_self_match_at_1 = bool(
                unfiltered_hits and unfiltered_hits[0].chunk_id == query.metadata.id
            )
        return VectorSearchResult(
            query_chunk_id=query.metadata.id,
            query_text=query.text,
            hits=hits,
            self_match_at_1=self_match_at_1,
            self_match_returned=self_match_returned,
            unfiltered_self_match_at_1=unfiltered_self_match_at_1,
            score_threshold=self.config.score_threshold,
        )

    @staticmethod
    def _hit(point: Any) -> VectorSearchHit:
        """Гарантирует преобразование результата поиска Qdrant в структуру с валидированными полями для дальнейшей обработки пайплайном."""
        raw_payload = getattr(point, "payload", None)
        payload: dict[str, Any] = (
            dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        )
        raw_metadata = payload.get("metadata")
        metadata: dict[str, Any] = (
            dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        chunk_id = payload.get("chunk_id") or metadata.get("id")
        text = payload.get("text")
        source = payload.get("source") or metadata.get("source")
        section = payload.get("section") or metadata.get("section")
        position = payload.get("position")
        if position is None:
            position = metadata.get("position")
        return VectorSearchHit(
            point_id=str(point.id),
            chunk_id=str(chunk_id) if chunk_id is not None else None,
            score=float(point.score),
            text=text if isinstance(text, str) else None,
            source=str(source) if source is not None else None,
            section=str(section) if section is not None else None,
            position=position if isinstance(position, int) else None,
            metadata=metadata,
        )
