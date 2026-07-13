from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import AgentRagConfig  # noqa: E402
from agent_app.rag.runtime import OnlineRagRuntime  # noqa: E402
from rag_prep.config import EmbeddingConfig, VectorStoreConfig  # noqa: E402


class FixedEmbedder:
    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class OnlineRagTest(unittest.TestCase):
    def test_local_model_path_is_portable_between_host_and_container(self) -> None:
        self.assertTrue(
            OnlineRagRuntime._embedding_models_match(
                r"C:\project\data\models\hf\multilingual-e5-small",
                "/app/data/models/hf/multilingual-e5-small",
                provider="local",
            )
        )
        self.assertFalse(
            OnlineRagRuntime._embedding_models_match(
                r"C:\models\multilingual-e5-small",
                "/app/models/another-model",
                provider="local",
            )
        )

    def test_empty_search_has_explicit_empty_status(self) -> None:
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="knowledge",
            vectors_config=qdrant_models.VectorParams(
                size=3,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        client.upsert(
            collection_name="knowledge",
            points=[
                qdrant_models.PointStruct(
                    id=str(uuid4()),
                    vector=[0.0, 1.0, 0.0],
                    payload={
                        "text": "Нерелевантная запись",
                        "chunk_id": "other-chunk",
                        "document_id": "other-document",
                        "source": "other.txt",
                        "section": "Прочее",
                        "position": 0,
                        "metadata": {
                            "id": "other-chunk",
                            "document_id": "other-document",
                            "source": "other.txt",
                            "section": "Прочее",
                            "position": 0,
                        },
                    },
                )
            ],
        )
        config = self._config()
        config.vector_store.score_threshold = 0.99
        runtime = OnlineRagRuntime(
            config,
            embedder=FixedEmbedder(),
            client=client,
        )
        try:
            result = runtime.retrieve("Неизвестная система")
        finally:
            client.close()

        self.assertEqual(result.status, "empty")
        self.assertEqual(result.retrieved_count, 0)
        self.assertEqual(result.citations, [])

    def test_query_embedding_search_context_and_citations(self) -> None:
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="knowledge",
            vectors_config=qdrant_models.VectorParams(
                size=3,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        client.upsert(
            collection_name="knowledge",
            points=[
                qdrant_models.PointStruct(
                    id=str(uuid4()),
                    vector=[1.0, 0.0, 0.0],
                    payload={
                        "text": "Для сброса пароля инженер проверяет личность пользователя.",
                        "chunk_id": "chunk-password",
                        "document_id": "document-support",
                        "source": "knowledge.txt",
                        "section": "Сброс пароля",
                        "position": 0,
                        "metadata": {
                            "id": "chunk-password",
                            "document_id": "document-support",
                            "source": "knowledge.txt",
                            "section": "Сброс пароля",
                            "position": 0,
                        },
                    },
                )
            ],
        )
        runtime = OnlineRagRuntime(
            self._config(),
            embedder=FixedEmbedder(),
            client=client,
        )
        try:
            result = runtime.retrieve("Как сбросить пароль?")
        finally:
            client.close()

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.retrieved_count, 1)
        self.assertEqual(result.used_count, 1)
        self.assertEqual(result.citations[0].chunk_id, "chunk-password")
        self.assertIn("[Источник 1]", result.context)
        self.assertIn("проверяет личность", result.context)

    def test_vector_dimension_mismatch_is_reported_before_search(self) -> None:
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="knowledge",
            vectors_config=qdrant_models.VectorParams(
                size=4,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        runtime = OnlineRagRuntime(
            self._config(),
            embedder=FixedEmbedder(),
            client=client,
        )
        try:
            readiness = runtime.readiness()
            result = runtime.retrieve("Запрос")
        finally:
            client.close()

        self.assertFalse(readiness.ready)
        self.assertEqual(result.status, "unavailable")
        self.assertIn("Размер коллекции", result.error)

    @staticmethod
    def _config() -> AgentRagConfig:
        return AgentRagConfig(
            enabled=True,
            top_k=3,
            max_context_tokens=200,
            tokenizer_model="cl100k_base",
            embedding=EmbeddingConfig(
                provider="openai",
                model="test-embedding",
                dimensions=3,
                api_key_env="TEST_OPENAI_API_KEY",
            ),
            vector_store=VectorStoreConfig(
                collection_name="knowledge",
                vector_size=3,
                local_storage_path=Path("unused-qdrant-storage"),
                test_queries_count=0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
