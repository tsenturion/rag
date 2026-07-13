from __future__ import annotations

import logging
import threading
from contextlib import AbstractContextManager
from typing import Any, Protocol

from qdrant_client import QdrantClient

from agent_app.config import AgentRagConfig
from agent_app.rag.context import RagContextBuilder
from agent_app.rag.models import RagReadiness, RagRetrievalResult
from agent_app.rag.retriever import QdrantKnowledgeRetriever
from rag_prep.embedding_stages.embedding import build_embedding_stage
from rag_prep.vector_store_stages.client import qdrant_client_context, qdrant_distance

LOGGER = logging.getLogger(__name__)


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class OnlineRagRuntime:
    def __init__(
        self,
        config: AgentRagConfig,
        *,
        embedder: QueryEmbedder | None = None,
        client: QdrantClient | None = None,
        auto_start: bool = True,
    ):
        if config.enabled and (
            config.tokenizer_model is None
            or config.embedding is None
            or config.vector_store is None
        ):
            raise ValueError(
                "OnlineRagRuntime требует явно заданные tokenizer_model, "
                "embedding и vector_store."
            )
        self.config = config
        self.embedding_config = config.embedding
        self.vector_store_config = config.vector_store
        self.embedder = embedder
        self.client = client
        self.retriever: QdrantKnowledgeRetriever | None = None
        self._client_context: AbstractContextManager | None = None
        self._owns_client = client is None
        self._error: str | None = None
        self._lock = threading.RLock()
        self.context_builder = (
            RagContextBuilder(
                max_tokens=config.max_context_tokens,
                excerpt_chars=config.excerpt_chars,
                tokenizer_model=config.tokenizer_model,
            )
            if config.tokenizer_model is not None
            else None
        )
        if auto_start and config.enabled:
            self.start()

    def start(self) -> RagReadiness:
        with self._lock:
            if not self.config.enabled:
                return self.readiness()
            if self.embedding_config is None or self.vector_store_config is None:
                self._error = "Конфигурация Online RAG не задана"
                return self.readiness()
            if self.retriever is not None and self._error is None:
                return self.readiness()
            self._error = None
            try:
                if self.embedder is None:
                    self.embedder = build_embedding_stage(self.embedding_config)
                if self.client is None:
                    self._client_context = qdrant_client_context(
                        self.vector_store_config
                    )
                    self.client = self._client_context.__enter__()
                self._validate_collection(self.client)
                self.retriever = QdrantKnowledgeRetriever(
                    self.vector_store_config,
                    self.client,
                )
            except Exception as exc:
                self._error = self._safe_error(exc)
                self.retriever = None
                LOGGER.exception("Online RAG не готов: %s", self._error)
            return self.readiness()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source: str | None = None,
        section: str | None = None,
    ) -> RagRetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            return self._error_result(query, "Пустой поисковый запрос")
        readiness = self.start()
        if (
            not readiness.ready
            or self.embedder is None
            or self.retriever is None
            or self.embedding_config is None
            or self.vector_store_config is None
            or self.context_builder is None
        ):
            return self._error_result(
                normalized_query, readiness.error or "RAG недоступен"
            )

        try:
            vector = self.embedder.embed_query(normalized_query)
            expected_size = self.vector_store_config.vector_size
            if len(vector) != expected_size:
                raise ValueError(
                    "Размер query embedding не совпадает с коллекцией: "
                    f"actual={len(vector)} expected={expected_size}"
                )
            chunks = self.retriever.search(
                vector,
                limit=top_k or self.config.top_k,
                source=source,
                section=section,
            )
            context, citations, context_tokens = self.context_builder.build(chunks)
            return RagRetrievalResult(
                status="ok" if citations else "empty",
                query=normalized_query,
                context=context,
                citations=citations,
                retrieved_count=len(chunks),
                used_count=len(citations),
                context_tokens=context_tokens,
                provider=self.embedding_config.provider,
                model=self.embedding_config.model,
                collection_name=self.vector_store_config.collection_name,
            )
        except Exception as exc:
            LOGGER.exception("Ошибка online retrieval")
            return self._error_result(normalized_query, self._safe_error(exc))

    def readiness(self) -> RagReadiness:
        return RagReadiness(
            enabled=self.config.enabled,
            ready=(
                not self.config.enabled
                or (self.retriever is not None and self._error is None)
            ),
            collection_name=(
                self.vector_store_config.collection_name
                if self.vector_store_config is not None
                else None
            ),
            embedding_provider=(
                self.embedding_config.provider
                if self.embedding_config is not None
                else None
            ),
            embedding_model=(
                self.embedding_config.model
                if self.embedding_config is not None
                else None
            ),
            vector_size=(
                self.vector_store_config.vector_size
                if self.vector_store_config is not None
                else None
            ),
            error=self._error,
        )

    def close(self) -> None:
        with self._lock:
            self.retriever = None
            if self._owns_client and self._client_context is not None:
                try:
                    self._client_context.__exit__(None, None, None)
                finally:
                    self._client_context = None
                    self.client = None

    def _validate_collection(self, client: QdrantClient) -> None:
        if self.embedding_config is None or self.vector_store_config is None:
            raise ValueError("Конфигурация Online RAG не задана")
        name = self.vector_store_config.collection_name
        if not client.collection_exists(name):
            raise RuntimeError(f"Коллекция Qdrant не найдена: {name}")
        info = client.get_collection(name)
        vectors: Any = info.config.params.vectors
        if isinstance(vectors, dict):
            vectors = next(iter(vectors.values()))
        size = getattr(vectors, "size", None)
        if size != self.vector_store_config.vector_size:
            raise ValueError(
                "Размер коллекции Qdrant не соответствует конфигу: "
                f"actual={size} expected={self.vector_store_config.vector_size}"
            )
        distance = getattr(vectors, "distance", None)
        expected_distance = qdrant_distance(self.vector_store_config.distance)
        if distance != expected_distance:
            raise ValueError(
                "Distance коллекции Qdrant не соответствует конфигу: "
                f"actual={distance} expected={expected_distance}"
            )
        embedding_dimensions = self.embedding_config.dimensions
        if embedding_dimensions is not None and embedding_dimensions != size:
            raise ValueError(
                "Размер embedding-модели не соответствует коллекции: "
                f"embedding={embedding_dimensions} collection={size}"
            )
        points_count = client.count(collection_name=name, exact=True).count
        if points_count == 0:
            raise RuntimeError(f"Коллекция Qdrant пуста: {name}")
        records, _ = client.scroll(
            collection_name=name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records or not records[0].payload:
            raise ValueError("В Qdrant отсутствует обязательный payload")
        payload = records[0].payload
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("В Qdrant отсутствует metadata подготовленного чанка")
        for key in ("text", "chunk_id", "document_id", "source", "section"):
            if payload.get(key) is None and metadata.get(key) is None:
                raise ValueError(
                    f"В Qdrant payload отсутствует обязательное поле: {key}"
                )
        actual_provider = payload.get("embedding_provider") or metadata.get(
            "embedding_provider"
        )
        actual_model = payload.get("embedding_model") or metadata.get("embedding_model")
        if actual_provider and actual_provider != self.embedding_config.provider:
            raise ValueError(
                "Embedding provider коллекции не соответствует конфигу: "
                f"actual={actual_provider} expected={self.embedding_config.provider}"
            )
        if actual_model and not self._embedding_models_match(
            str(actual_model),
            self.embedding_config.model,
            provider=self.embedding_config.provider,
        ):
            raise ValueError(
                "Embedding model коллекции не соответствует конфигу: "
                f"actual={actual_model} expected={self.embedding_config.model}"
            )

    @staticmethod
    def _embedding_models_match(
        actual: str,
        expected: str,
        *,
        provider: str,
    ) -> bool:
        if actual == expected:
            return True
        if provider != "local":
            return False
        actual_name = actual.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        expected_name = expected.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        return bool(actual_name) and actual_name.casefold() == expected_name.casefold()

    def _error_result(self, query: str, error: str) -> RagRetrievalResult:
        return RagRetrievalResult(
            status="unavailable",
            query=query,
            provider=(
                self.embedding_config.provider
                if self.embedding_config is not None
                else None
            ),
            model=(
                self.embedding_config.model
                if self.embedding_config is not None
                else None
            ),
            collection_name=(
                self.vector_store_config.collection_name
                if self.vector_store_config is not None
                else None
            ),
            error=error,
        )

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = str(exc).replace("\n", " ")
        return text[:500]
