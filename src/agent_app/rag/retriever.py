from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from agent_app.rag.models import RagRetrievedChunk
from rag_prep.config import VectorStoreConfig


class QdrantKnowledgeRetriever:
    def __init__(self, config: VectorStoreConfig, client: QdrantClient):
        self.config = config
        self.client = client

    def search(
        self,
        vector: list[float],
        *,
        limit: int,
        source: str | None = None,
        section: str | None = None,
    ) -> list[RagRetrievedChunk]:
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=vector,
            query_filter=self._filter(source=source, section=section),
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=self.config.score_threshold,
        )
        chunks: list[RagRetrievedChunk] = []
        seen_ids: set[str] = set()
        for point in response.points:
            chunk = self._chunk(point)
            if not chunk.chunk_id or chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _filter(
        *,
        source: str | None,
        section: str | None,
    ) -> qdrant_models.Filter | None:
        conditions = []
        if source:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="source",
                    match=qdrant_models.MatchValue(value=source),
                )
            )
        if section:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="section",
                    match=qdrant_models.MatchValue(value=section),
                )
            )
        return qdrant_models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _chunk(point: Any) -> RagRetrievedChunk:
        payload = point.payload or {}
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_id = str(payload.get("chunk_id") or metadata.get("id") or "")
        return RagRetrievedChunk(
            point_id=str(point.id),
            chunk_id=chunk_id,
            document_id=payload.get("document_id") or metadata.get("document_id"),
            text=str(payload.get("text") or ""),
            source=payload.get("source") or metadata.get("source"),
            section=payload.get("section") or metadata.get("section"),
            position=(
                payload.get("position")
                if payload.get("position") is not None
                else metadata.get("position")
            ),
            score=float(point.score),
            metadata=metadata,
        )
