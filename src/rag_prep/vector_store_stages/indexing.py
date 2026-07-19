"""Индексация в Qdrant для индексации в Qdrant."""

from __future__ import annotations

import logging
from collections.abc import Iterable

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
        """Инициализирует этап индексации с конфигурацией, обеспечивающей корректное взаимодействие с Qdrant и управление коллекциями."""
        self.config = config

    def run(
        self,
        embedded_chunks: list[EmbeddedChunk],
        *,
        client: QdrantClient | None = None,
    ) -> VectorStoreIndexResult:
        """Обеспечивает надёжную загрузку и индексацию эмбеддингов в Qdrant с проверкой данных и учётом параметров конфигурации."""
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
            LOGGER.info(
                "Загружено Qdrant points: %d/%d", points_upserted, len(embedded_chunks)
            )

        stale_points_deleted = 0
        if self.config.prune_stale_points:
            expected_point_ids = {
                point_id_for_chunk(self.config.collection_name, chunk.metadata.id)
                for chunk in embedded_chunks
            }
            stale_points_deleted = self._prune_stale_points(
                client,
                expected_point_ids=expected_point_ids,
            )

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
            stale_points_deleted=stale_points_deleted,
            collection_points_count=points_count,
            vector_size=self.config.vector_size,
            distance=self.config.distance,
            storage_path=self.config.local_storage_path
            if self.config.mode == "local"
            else None,
            url=qdrant_url(self.config),
        )

    def _prune_stale_points(
        self,
        client: QdrantClient,
        *,
        expected_point_ids: set[str],
    ) -> int:
        """Удаляет точки, не принадлежащие текущему snapshot embeddings.

        Сначала собирается полный список ID, а удаление начинается только после
        завершения scroll. Это исключает смещение cursor во время обхода и делает
        поведение одинаковым для embedded и HTTP Qdrant.
        """
        stored_ids: dict[str, str | int] = {}
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=self.config.collection_name,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for record in records:
                stored_ids[str(record.id)] = record.id
            if next_offset is None:
                break
            offset = next_offset

        stale_ids = [
            point_id
            for canonical_id, point_id in stored_ids.items()
            if canonical_id not in expected_point_ids
        ]
        for batch in self._id_batches(stale_ids, self.config.batch_size):
            client.delete(
                collection_name=self.config.collection_name,
                points_selector=qdrant_models.PointIdsList(points=batch),
                wait=True,
            )
        if stale_ids:
            LOGGER.warning(
                "Удалено устаревших Qdrant points из snapshot %s: %d",
                self.config.collection_name,
                len(stale_ids),
            )
        return len(stale_ids)

    def _ensure_collection(self, client: QdrantClient) -> None:
        """Гарантирует существование и корректность коллекции в Qdrant, создавая или пересоздавая её согласно настройкам конфигурации."""
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
            self._close_embedded_collection_before_delete(client)
            client.delete_collection(self.config.collection_name)
            exists = False
        if not exists:
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=vectors_config,
            )
            if self.config.recreate_collection:
                points_count = client.count(
                    collection_name=self.config.collection_name,
                    exact=True,
                ).count
                if points_count != 0:
                    raise RuntimeError(
                        "Qdrant не очистил пересозданную коллекцию "
                        f"{self.config.collection_name}: осталось точек {points_count}"
                    )
            LOGGER.info("Создана коллекция Qdrant %s", self.config.collection_name)
            return
        LOGGER.info(
            "Используется существующая коллекция Qdrant %s", self.config.collection_name
        )

    def _close_embedded_collection_before_delete(self, client: QdrantClient) -> None:
        """Закрывает SQLite storage embedded Qdrant до удаления каталога на Windows.

        Qdrant Local 1.18 удаляет collection directory с ``ignore_errors=True`` и не
        закрывает storage заранее. На Windows открытый SQLite-файл остаётся на диске,
        поэтому следующая ``create_collection`` незаметно загружает старые points.
        Для HTTP-режима и SDK без такого внутреннего объекта метод ничего не делает.
        """
        if self.config.mode != "local":
            return
        local_backend = getattr(client, "_client", None)
        collections = getattr(local_backend, "collections", None)
        if not isinstance(collections, dict):
            return
        collection = collections.get(self.config.collection_name)
        close = getattr(collection, "close", None)
        if callable(close):
            close()

    def _point(self, chunk: EmbeddedChunk) -> qdrant_models.PointStruct:
        """Формирует единичную точку данных с полным метаданным для индексации в Qdrant, обеспечивая однозначную идентификацию и полноту информации."""
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
        """Проверяет уникальность и соответствие размерности эмбеддингов требованиям конфигурации, предотвращая некорректную загрузку в хранилище."""
        if not embedded_chunks:
            raise ValueError(
                "Нельзя создать векторный индекс из пустого списка embeddings."
            )
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
                raise ValueError(
                    f"Дублирующийся chunk id нельзя загрузить: {chunk.metadata.id}"
                )
            chunk_ids.add(chunk.metadata.id)

            point_id = point_id_for_chunk(
                self.config.collection_name, chunk.metadata.id
            )
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
        """Разбивает список эмбеддингов на управляемые порции для эффективной и безопасной пакетной загрузки в Qdrant."""
        for start in range(0, len(embedded_chunks), batch_size):
            yield embedded_chunks[start : start + batch_size]

    @staticmethod
    def _id_batches(
        point_ids: list[str | int], batch_size: int
    ) -> Iterable[list[str | int]]:
        """Ограничивает размер delete-запросов тем же batch budget, что и upsert."""
        for start in range(0, len(point_ids), batch_size):
            yield point_ids[start : start + batch_size]
