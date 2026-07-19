"""Проверка выходных контрактов для индексации в Qdrant."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from rag_prep.config import VectorStoreConfig
from rag_prep.models import EmbeddedChunk, VectorStoreValidationResult
from rag_prep.vector_store_stages.client import (
    point_id_for_chunk,
    qdrant_client_context,
    qdrant_distance,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_METADATA_KEYS = (
    # Эти поля составляют минимальный контракт retrieval и attribution.
    "id",
    "document_id",
    "source",
    "section",
    "position",
    "embedding_model",
    "embedding_dimensions",
)


class QdrantValidationStage:
    """Проверяет, что Qdrant содержит ожидаемые vectors и payload metadata."""

    def __init__(self, config: VectorStoreConfig):
        """Обеспечивает готовность экземпляра к операциям валидации с сохранением параметров конфигурации без захвата внешних ресурсов."""
        self.config = config

    def run(
        self,
        embedded_chunks: list[EmbeddedChunk],
        *,
        client: QdrantClient | None = None,
    ) -> VectorStoreValidationResult:
        """Сверяет схему коллекции, число точек, vectors и обязательный payload."""
        if client is None:
            with qdrant_client_context(self.config) as owned_client:
                return self.run(embedded_chunks, client=owned_client)

        info = client.get_collection(self.config.collection_name)
        collection_count = client.count(
            collection_name=self.config.collection_name,
            exact=True,
        ).count
        vector_params = self._vector_params(info)
        collection_size = getattr(vector_params, "size", None)
        collection_distance = getattr(vector_params, "distance", None)

        sampled = self._sample_points(client)
        expected_points = self._expected_points(client, embedded_chunks)
        vector_lengths = [self._vector_length(point.vector) for point in sampled]
        # Знак сохраняется отдельно: положительное значение означает старые/лишние
        # точки, отрицательное — недозаписанные embeddings.
        count_delta = collection_count - len(embedded_chunks)
        collection_vector_size_mismatch = int(
            collection_size != self.config.vector_size
        )
        point_vector_size_mismatch = sum(
            1
            for length in vector_lengths
            # Отсутствующий vector учитывается отдельной метрикой и не должен
            # одновременно завышать mismatch размерности.
            if length is not None and length != self.config.vector_size
        )
        missing_expected_points = 0
        chunk_id_mismatch = 0
        text_mismatch = 0
        identity_metadata_mismatch = 0
        vector_content_mismatch = 0
        for chunk in embedded_chunks:
            point_id = point_id_for_chunk(
                self.config.collection_name, chunk.metadata.id
            )
            point = expected_points.get(point_id)
            if point is None:
                missing_expected_points += 1
                continue
            payload = point.payload or {}
            chunk_id_mismatch += int(payload.get("chunk_id") != chunk.metadata.id)
            text_mismatch += int(payload.get("text") != chunk.text)
            identity_metadata_mismatch += int(
                self._identity_metadata_mismatch(payload, chunk)
            )
            stored_vector = self._vector_values(point.vector)
            vector_content_mismatch += int(
                stored_vector is None
                or not self._vectors_match(
                    stored_vector,
                    self._vector_for_storage(chunk.embedding),
                )
            )
        result = VectorStoreValidationResult(
            embeddings_count=len(embedded_chunks),
            empty_embeddings_count=int(not embedded_chunks),
            collection_points_count=collection_count,
            count_mismatch=abs(count_delta),
            count_delta=count_delta,
            extra_points_count=max(count_delta, 0),
            missing_points_count=max(-count_delta, 0),
            missing_vector_count=sum(1 for length in vector_lengths if length is None),
            collection_vector_size_mismatch_count=collection_vector_size_mismatch,
            point_vector_size_mismatch_count=point_vector_size_mismatch,
            vector_size_mismatch_count=(
                collection_vector_size_mismatch + point_vector_size_mismatch
            ),
            distance_mismatch_count=int(
                collection_distance != qdrant_distance(self.config.distance)
            ),
            missing_payload_count=sum(1 for point in sampled if not point.payload),
            missing_text_count=sum(
                1 for point in sampled if not (point.payload or {}).get("text")
            ),
            missing_metadata_count=sum(
                1
                for point in sampled
                if not isinstance((point.payload or {}).get("metadata"), dict)
            ),
            missing_required_metadata_count=sum(
                1 for point in sampled if self._missing_required_metadata(point.payload)
            ),
            sampled_points_count=len(sampled),
            verified_points_count=len(expected_points),
            missing_expected_points_count=missing_expected_points,
            chunk_id_mismatch_count=chunk_id_mismatch,
            text_mismatch_count=text_mismatch,
            identity_metadata_mismatch_count=identity_metadata_mismatch,
            vector_content_mismatch_count=vector_content_mismatch,
        )
        LOGGER.info(
            (
                "Проверена коллекция Qdrant %s: count_mismatch=%d count_delta=%d "
                "extra_points=%d missing_points=%d missing_vectors=%d "
                "collection_vector_size_mismatch=%d point_vector_size_mismatch=%d "
                "vector_size_mismatch=%d distance_mismatch=%d missing_payload=%d "
                "missing_text=%d missing_metadata=%d missing_required_metadata=%d "
                "sampled=%d verified=%d missing_expected=%d chunk_id_mismatch=%d "
                "text_mismatch=%d identity_metadata_mismatch=%d "
                "vector_content_mismatch=%d empty_embeddings=%d"
            ),
            self.config.collection_name,
            result.count_mismatch,
            result.count_delta,
            result.extra_points_count,
            result.missing_points_count,
            result.missing_vector_count,
            result.collection_vector_size_mismatch_count,
            result.point_vector_size_mismatch_count,
            result.vector_size_mismatch_count,
            result.distance_mismatch_count,
            result.missing_payload_count,
            result.missing_text_count,
            result.missing_metadata_count,
            result.missing_required_metadata_count,
            result.sampled_points_count,
            result.verified_points_count,
            result.missing_expected_points_count,
            result.chunk_id_mismatch_count,
            result.text_mismatch_count,
            result.identity_metadata_mismatch_count,
            result.vector_content_mismatch_count,
            result.empty_embeddings_count,
        )
        if self.config.fail_on_validation_error and result.has_errors:
            raise ValueError(
                f"Валидация vector store завершилась ошибкой: {result.model_dump()}"
            )
        return result

    def _expected_points(
        self,
        client: QdrantClient,
        embedded_chunks: list[EmbeddedChunk],
    ) -> dict[str, qdrant_models.Record]:
        """Загружает ожидаемые точки по стабильным ID, не полагаясь на случайный sample коллекции."""
        records_by_id: dict[str, qdrant_models.Record] = {}
        point_ids = [
            point_id_for_chunk(self.config.collection_name, chunk.metadata.id)
            for chunk in embedded_chunks
        ]
        for start in range(0, len(point_ids), 256):
            records = client.retrieve(
                collection_name=self.config.collection_name,
                ids=point_ids[start : start + 256],
                with_payload=True,
                with_vectors=True,
            )
            records_by_id.update({str(record.id): record for record in records})
        return records_by_id

    @staticmethod
    def _identity_metadata_mismatch(
        payload: dict[str, Any],
        chunk: EmbeddedChunk,
    ) -> bool:
        """Сверяет поля происхождения и модели в верхнем и полном payload точки."""
        expected = {
            "document_id": chunk.metadata.document_id,
            "source": chunk.metadata.source,
            "section": chunk.metadata.section,
            "position": chunk.metadata.position,
            "embedding_model": chunk.metadata.embedding_model,
            "embedding_provider": chunk.metadata.embedding_provider,
            "embedding_dimensions": chunk.metadata.embedding_dimensions,
        }
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return True
        return any(
            payload.get(key) != value or metadata.get(key) != value
            for key, value in expected.items()
        )

    def _sample_points(self, client) -> list[qdrant_models.Record]:
        """Последовательно scroll-ит bounded sample с защитой от зацикливания SDK."""
        points: list[qdrant_models.Record] = []
        offset = None
        seen_offsets: set[str] = set()
        # Даже патологический backend, возвращающий по одной точке, успеет выдать
        # весь sample; запас покрывает несколько пустых страниц с новым offset.
        max_iterations = self.config.validation_sample_size + 5
        iterations = 0
        empty_pages = 0
        while len(points) < self.config.validation_sample_size:
            iterations += 1
            if iterations > max_iterations:
                LOGGER.warning(
                    (
                        "Qdrant scroll validation остановлена после %d итераций; "
                        "проверено %d/%d запрошенных points"
                    ),
                    max_iterations,
                    len(points),
                    self.config.validation_sample_size,
                )
                break
            limit = min(self.config.validation_sample_size - len(points), 256)
            records, next_offset = client.scroll(
                collection_name=self.config.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if records:
                points.extend(records)
                empty_pages = 0
            else:
                empty_pages += 1
                if empty_pages >= 3:
                    LOGGER.warning(
                        (
                            "Stopped Qdrant scroll validation after %d consecutive "
                            "empty pages; sampled %d/%d requested points"
                        ),
                        empty_pages,
                        len(points),
                        self.config.validation_sample_size,
                    )
                    break
            if next_offset is None:
                break
            offset_key = repr(next_offset)
            # Повтор offset означает, что дальнейший scroll не продвинется.
            if offset_key in seen_offsets:
                LOGGER.warning(
                    "Stopped Qdrant scroll validation because offset repeated: %s",
                    next_offset,
                )
                break
            seen_offsets.add(offset_key)
            offset = next_offset
        return points

    @staticmethod
    def _vector_params(info: Any) -> Any:
        """Гарантирует извлечение параметров векторного пространства независимо от структуры конфигурации Qdrant."""
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            return next(iter(vectors.values()))
        return vectors

    @staticmethod
    def _vector_length(vector: Any) -> int | None:
        """Получает длину unnamed или первого named vector; None означает отсутствие."""
        if vector is None:
            return None
        if isinstance(vector, dict):
            first = next(iter(vector.values()), None)
            return len(first) if first is not None else None
        return len(vector)

    @staticmethod
    def _vector_values(vector: Any) -> list[float] | None:
        """Извлекает unnamed либо первый named vector для проверки содержимого."""
        if vector is None:
            return None
        if isinstance(vector, dict):
            vector = next(iter(vector.values()), None)
        if vector is None:
            return None
        return [float(value) for value in vector]

    @staticmethod
    def _vectors_match(
        actual: list[float],
        expected: list[float],
        *,
        absolute_tolerance: float = 1e-6,
    ) -> bool:
        """Сравнивает значения с допуском на float32-сериализацию Qdrant."""
        return len(actual) == len(expected) and all(
            abs(left - right) <= absolute_tolerance
            for left, right in zip(actual, expected, strict=True)
        )

    def _vector_for_storage(self, vector: list[float]) -> list[float]:
        """Повторяет L2-нормализацию Qdrant для cosine перед сравнением координат."""
        if self.config.distance.casefold() != "cosine":
            return vector
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _missing_required_metadata(payload: dict[str, Any] | None) -> bool:
        """Гарантирует определение отсутствия обязательных метаданных в payload для обеспечения целостности индекса."""
        if not payload:
            return True
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return True
        return any(metadata.get(key) is None for key in REQUIRED_METADATA_KEYS)
