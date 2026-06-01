from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from rag_prep.config import VectorStoreConfig
from rag_prep.models import EmbeddedChunk, VectorStoreValidationResult
from rag_prep.vector_store_stages.client import qdrant_client_context, qdrant_distance

LOGGER = logging.getLogger(__name__)

REQUIRED_METADATA_KEYS = (
    "id",
    "document_id",
    "source",
    "section",
    "position",
    "embedding_model",
    "embedding_dimensions",
)


class QdrantValidationStage:
    """Validate that Qdrant contains expected vectors and payload metadata."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    def run(
        self,
        embedded_chunks: list[EmbeddedChunk],
        *,
        client: QdrantClient | None = None,
    ) -> VectorStoreValidationResult:
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
        vector_lengths = [self._vector_length(point.vector) for point in sampled]
        count_delta = collection_count - len(embedded_chunks)
        collection_vector_size_mismatch = int(collection_size != self.config.vector_size)
        point_vector_size_mismatch = sum(
            1
            for length in vector_lengths
            if length is not None and length != self.config.vector_size
        )
        result = VectorStoreValidationResult(
            embeddings_count=len(embedded_chunks),
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
        )
        LOGGER.info(
            (
                "Validated Qdrant collection %s: count_mismatch=%d count_delta=%d "
                "extra_points=%d missing_points=%d missing_vectors=%d "
                "collection_vector_size_mismatch=%d point_vector_size_mismatch=%d "
                "vector_size_mismatch=%d distance_mismatch=%d missing_payload=%d "
                "missing_text=%d missing_metadata=%d missing_required_metadata=%d "
                "sampled=%d"
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
        )
        if self.config.fail_on_validation_error and result.has_errors:
            raise ValueError(f"Vector store validation failed: {result.model_dump()}")
        return result

    def _sample_points(self, client) -> list[qdrant_models.Record]:
        points: list[qdrant_models.Record] = []
        offset = None
        seen_offsets: set[str] = set()
        max_iterations = self.config.validation_sample_size + 5
        iterations = 0
        empty_pages = 0
        while len(points) < self.config.validation_sample_size:
            iterations += 1
            if iterations > max_iterations:
                LOGGER.warning(
                    (
                        "Stopped Qdrant scroll validation after %d iterations; "
                        "sampled %d/%d requested points"
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
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            return next(iter(vectors.values()))
        return vectors

    @staticmethod
    def _vector_length(vector: Any) -> int | None:
        if vector is None:
            return None
        if isinstance(vector, dict):
            first = next(iter(vector.values()), None)
            return len(first) if first is not None else None
        return len(vector)

    @staticmethod
    def _missing_required_metadata(payload: dict[str, Any] | None) -> bool:
        if not payload:
            return True
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return True
        return any(metadata.get(key) is None for key in REQUIRED_METADATA_KEYS)
