from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from rag_prep.config import VectorStoreConfig
from rag_prep.models import EmbeddedChunk, VectorStoreIndexResult
from rag_prep.vector_store_stages.client import (
    point_id_for_chunk,
    qdrant_client_context,
    qdrant_distance,
    qdrant_url,
)

LOGGER = logging.getLogger(__name__)


class QdrantIndexingStage:
    """Создаёт коллекцию Qdrant и загружает записи embeddings."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    def run(
        self,
        embedded_chunks: list[EmbeddedChunk],
        *,
        client: QdrantClient | None = None,
    ) -> VectorStoreIndexResult:
        if client is None:
            with qdrant_client_context(self.config) as owned_client:
                return self.run(embedded_chunks, client=owned_client)

        self._validate_embeddings(embedded_chunks)
        self._ensure_collection(client)
        points_upserted = 0
        for batch in self._batches(embedded_chunks, self.config.batch_size):
            points = [self._point(chunk) for chunk in batch]
            client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )
            points_upserted += len(points)
            LOGGER.info("Загружено Qdrant points: %d/%d", points_upserted, len(embedded_chunks))

        points_count = client.count(
            collection_name=self.config.collection_name,
            exact=True,
        ).count
        LOGGER.info(
            "Проиндексировано embeddings в коллекции Qdrant %s: %d",
            self.config.collection_name,
            points_upserted,
        )
        return VectorStoreIndexResult(
            collection_name=self.config.collection_name,
            provider=self.config.provider,
            mode=self.config.mode,
            points_upserted=points_upserted,
            collection_points_count=points_count,
            vector_size=self.config.vector_size,
            distance=self.config.distance,
            storage_path=self.config.local_storage_path
            if self.config.mode == "local"
            else None,
            url=qdrant_url(self.config),
        )

    def _ensure_collection(self, client: QdrantClient) -> None:
        vectors_config = qdrant_models.VectorParams(
            size=self.config.vector_size,
            distance=qdrant_distance(self.config.distance),
        )
        exists = client.collection_exists(self.config.collection_name)
        if exists and self.config.recreate_collection:
            LOGGER.warning(
                "Коллекция Qdrant %s будет удалена и создана заново, потому что recreate_collection=true",
                self.config.collection_name,
            )
            client.delete_collection(self.config.collection_name)
            exists = False
        if not exists:
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=vectors_config,
            )
            LOGGER.info("Создана коллекция Qdrant %s", self.config.collection_name)
            return
        LOGGER.info("Используется существующая коллекция Qdrant %s", self.config.collection_name)

    def _point(self, chunk: EmbeddedChunk) -> qdrant_models.PointStruct:
        metadata = chunk.metadata.model_dump(mode="json")
        payload = {
            "text": chunk.text,
            "chunk_id": chunk.metadata.id,
            "document_id": chunk.metadata.document_id,
            "source": chunk.metadata.source,
            "section": chunk.metadata.section,
            "position": chunk.metadata.position,
            "file_name": chunk.metadata.file_name,
            "file_type": chunk.metadata.file_type,
            "embedding_model": chunk.metadata.embedding_model,
            "embedding_provider": chunk.metadata.embedding_provider,
            "embedding_dimensions": chunk.metadata.embedding_dimensions,
            "metadata": metadata,
        }
        return qdrant_models.PointStruct(
            id=point_id_for_chunk(self.config.collection_name, chunk.metadata.id),
            vector=chunk.embedding,
            payload=payload,
        )

    def _validate_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        chunk_ids: set[str] = set()
        point_ids: set[str] = set()
        for chunk in embedded_chunks:
            if len(chunk.embedding) != self.config.vector_size:
                raise ValueError(
                    (
                        "Размерность embedding не совпадает с vector_store.vector_size: "
                        f"chunk_id={chunk.metadata.id} "
                        f"embedding_dim={len(chunk.embedding)} "
                        f"vector_size={self.config.vector_size}"
                    )
            )
            if chunk.metadata.id in chunk_ids:
                raise ValueError(f"Дублирующийся chunk id нельзя загрузить: {chunk.metadata.id}")
            chunk_ids.add(chunk.metadata.id)

            point_id = point_id_for_chunk(self.config.collection_name, chunk.metadata.id)
            if point_id in point_ids:
                raise ValueError(
                    (
                        "Сгенерирован дублирующийся Qdrant point id. "
                        f"chunk_id={chunk.metadata.id} point_id={point_id}"
                    )
                )
            point_ids.add(point_id)

    @staticmethod
    def _batches(
        embedded_chunks: list[EmbeddedChunk], batch_size: int
    ) -> Iterable[list[EmbeddedChunk]]:
        for start in range(0, len(embedded_chunks), batch_size):
            yield embedded_chunks[start : start + batch_size]
