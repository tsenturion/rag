"""Регрессионные тесты для подсистемы qdrant_pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config import VectorStoreConfig  # noqa: E402
from rag_prep.config import ChunkingConfig  # noqa: E402
from rag_prep.chunking_stages.validation import ChunkValidationStage  # noqa: E402
from rag_prep.models import EmbeddedChunk, EmbeddedChunkMetadata  # noqa: E402
from rag_prep.vector_store_stages.indexing import QdrantIndexingStage  # noqa: E402
from rag_prep.vector_store_stages.search import QdrantSearchStage  # noqa: E402
from rag_prep.vector_store_stages.validation import QdrantValidationStage  # noqa: E402
from rag_prep.vector_store_stages.client import (  # noqa: E402
    point_id_for_chunk,
    qdrant_client_context,
)


class QdrantPipelineTest(unittest.TestCase):
    """Проверяет корректность работы подсистемы индексации, валидации и поиска в памяти с использованием Qdrant, гарантируя стабильность и точность поиска по векторам."""

    def test_index_validation_and_similarity_search_in_memory(self) -> None:
        """Проверяет, что индексация, валидация и поиск по векторным данным в памяти выполняются без ошибок, а результаты поиска корректно соответствуют исходным данным."""
        config = VectorStoreConfig(
            collection_name="test_chunks",
            vector_size=3,
            local_storage_path=Path("unused-qdrant-storage"),
            recreate_collection=True,
            batch_size=1,
            search_limit=2,
            test_queries_count=2,
            validation_sample_size=10,
        )
        chunks = [
            self._embedded_chunk("chunk-1", "Первый", [1.0, 0.0, 0.0], 0),
            self._embedded_chunk("chunk-2", "Второй", [0.0, 1.0, 0.0], 1),
        ]
        client = QdrantClient(":memory:")
        try:
            index = QdrantIndexingStage(config).run(chunks, client=client)
            validation = QdrantValidationStage(config).run(chunks, client=client)
            search = QdrantSearchStage(config).run(chunks, client=client)
        finally:
            client.close()

        self.assertEqual(index.points_upserted, 2)
        self.assertEqual(index.collection_points_count, 2)
        self.assertFalse(validation.has_errors)
        self.assertEqual(validation.sampled_points_count, 2)
        self.assertEqual(len(search), 2)
        self.assertTrue(all(result.self_match_at_1 for result in search))

    def test_search_hit_tolerates_missing_payload(self) -> None:
        """Проверяет, что поиск корректно обрабатывает результаты с отсутствующим полезным нагрузочным метаданными, не вызывая ошибок и возвращая ожидаемые значения."""
        hit = QdrantSearchStage._hit(
            SimpleNamespace(id="point-without-payload", score=0.5, payload=None)
        )

        self.assertEqual(hit.point_id, "point-without-payload")
        self.assertEqual(hit.score, 0.5)
        self.assertIsNone(hit.chunk_id)
        self.assertEqual(hit.metadata, {})

    def test_validation_rejects_collection_with_unrelated_points(self) -> None:
        """Проверяет, что одинаковое число посторонних точек не считается корректным индексом ожидаемых чанков."""
        config = VectorStoreConfig(
            collection_name="identity_check",
            vector_size=3,
            local_storage_path=Path("unused-qdrant-storage"),
            recreate_collection=True,
            validation_sample_size=10,
            fail_on_validation_error=False,
        )
        expected = [
            self._embedded_chunk("expected-1", "Ожидаемый первый", [1.0, 0.0, 0.0], 0),
            self._embedded_chunk("expected-2", "Ожидаемый второй", [0.0, 1.0, 0.0], 1),
        ]
        unrelated = [
            self._embedded_chunk("foreign-1", "Посторонний первый", [1.0, 0.0, 0.0], 0),
            self._embedded_chunk("foreign-2", "Посторонний второй", [0.0, 1.0, 0.0], 1),
        ]
        client = QdrantClient(":memory:")
        try:
            QdrantIndexingStage(config).run(unrelated, client=client)
            validation = QdrantValidationStage(config).run(expected, client=client)
        finally:
            client.close()

        self.assertTrue(validation.has_errors)
        self.assertEqual(validation.count_mismatch, 0)
        self.assertEqual(validation.missing_expected_points_count, 2)
        self.assertEqual(validation.verified_points_count, 0)

    def test_empty_pipeline_results_are_rejected(self) -> None:
        """Проверяет запрет пустого индекса и зелёной валидации пустого результата чанкинга."""
        chunk_config = ChunkingConfig(
            tokenizer_model="test-tokenizer",
            embedding_model="test-embedding",
            fail_on_validation_error=False,
        )
        chunk_validation = ChunkValidationStage(chunk_config).run([])
        vector_config = VectorStoreConfig(
            collection_name="empty_check",
            vector_size=3,
            local_storage_path=Path("unused-qdrant-storage"),
        )
        client = QdrantClient(":memory:")

        self.assertTrue(chunk_validation.has_errors)
        self.assertEqual(chunk_validation.no_chunks_count, 1)
        try:
            with self.assertRaisesRegex(ValueError, "пустого списка embeddings"):
                QdrantIndexingStage(vector_config).run([], client=client)
        finally:
            client.close()

    def test_validation_detects_replaced_vector_with_same_payload(self) -> None:
        """Проверяет координаты ожидаемой точки, а не только ID и размерность."""
        config = VectorStoreConfig(
            collection_name="vector_content_check",
            vector_size=3,
            local_storage_path=Path("unused-qdrant-storage"),
            recreate_collection=True,
            validation_sample_size=10,
            fail_on_validation_error=False,
        )
        chunks = [self._embedded_chunk("chunk-1", "Первый", [1.0, 0.0, 0.0], 0)]
        client = QdrantClient(":memory:")
        try:
            QdrantIndexingStage(config).run(chunks, client=client)
            point_id = point_id_for_chunk(config.collection_name, "chunk-1")
            stored = client.retrieve(
                collection_name=config.collection_name,
                ids=[point_id],
                with_payload=True,
            )[0]
            client.upsert(
                collection_name=config.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=[0.0, 0.0, 0.0],
                        payload=stored.payload,
                    )
                ],
                wait=True,
            )
            validation = QdrantValidationStage(config).run(chunks, client=client)
        finally:
            client.close()

        self.assertTrue(validation.has_errors)
        self.assertEqual(validation.vector_content_mismatch_count, 1)

    def test_embedded_recreate_removes_open_sqlite_collection_on_windows(self) -> None:
        """Проверяет, что повторное создание local collection не оставляет старые points."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = VectorStoreConfig(
                collection_name="recreate_windows",
                vector_size=3,
                local_storage_path=Path(temporary_dir) / "qdrant",
                recreate_collection=True,
                validation_sample_size=10,
            )
            first = [self._embedded_chunk("old", "Старый", [1.0, 0.0, 0.0], 0)]
            second = [self._embedded_chunk("new", "Новый", [0.0, 1.0, 0.0], 0)]
            with qdrant_client_context(config) as client:
                QdrantIndexingStage(config).run(first, client=client)
            with qdrant_client_context(config) as client:
                index = QdrantIndexingStage(config).run(second, client=client)
                validation = QdrantValidationStage(config).run(second, client=client)

        self.assertEqual(index.collection_points_count, 1)
        self.assertFalse(validation.has_errors)

    def test_cosine_validation_compares_qdrant_normalized_vector(self) -> None:
        """Считает L2-нормализацию Qdrant частью storage contract, а не подменой vector."""
        config = VectorStoreConfig(
            collection_name="cosine_normalization",
            vector_size=3,
            local_storage_path=Path("unused-qdrant-storage"),
            recreate_collection=True,
            validation_sample_size=10,
            fail_on_validation_error=False,
        )
        chunks = [self._embedded_chunk("scaled", "Масштаб", [2.0, 0.0, 0.0], 0)]
        client = QdrantClient(":memory:")
        try:
            QdrantIndexingStage(config).run(chunks, client=client)
            validation = QdrantValidationStage(config).run(chunks, client=client)
        finally:
            client.close()

        self.assertEqual(validation.vector_content_mismatch_count, 0)
        self.assertFalse(validation.has_errors)

    def test_snapshot_sync_prunes_points_absent_from_current_embeddings(self) -> None:
        """Проверяет удаление прежних deterministic IDs без recreate коллекции."""
        config = VectorStoreConfig(
            collection_name="snapshot_sync",
            vector_size=3,
            local_storage_path=Path("unused-qdrant-storage"),
            recreate_collection=False,
            prune_stale_points=True,
            validation_sample_size=10,
        )
        old_chunks = [self._embedded_chunk("old", "Старый", [1.0, 0.0, 0.0], 0)]
        current_chunks = [
            self._embedded_chunk("current", "Текущий", [0.0, 1.0, 0.0], 0)
        ]
        client = QdrantClient(":memory:")
        try:
            first = QdrantIndexingStage(config).run(old_chunks, client=client)
            second = QdrantIndexingStage(config).run(current_chunks, client=client)
            validation = QdrantValidationStage(config).run(
                current_chunks,
                client=client,
            )
        finally:
            client.close()

        self.assertEqual(first.stale_points_deleted, 0)
        self.assertEqual(second.stale_points_deleted, 1)
        self.assertEqual(second.collection_points_count, 1)
        self.assertFalse(validation.has_errors)

    @staticmethod
    def _embedded_chunk(
        chunk_id: str,
        text: str,
        vector: list[float],
        position: int,
    ) -> EmbeddedChunk:
        """Создаёт воспроизводимый объект встраиваемого фрагмента с метаданными для стабильного тестирования пайплайна Qdrant."""
        return EmbeddedChunk(
            text=text,
            embedding=vector,
            metadata=EmbeddedChunkMetadata(
                id=chunk_id,
                document_id="document-1",
                source="source.txt",
                section="Раздел",
                position=position,
                chunk_start_char=position * 10,
                chunk_end_char=position * 10 + len(text),
                chunk_token_count=2,
                chunk_size=100,
                chunk_overlap=10,
                chunking_strategy="sentence",
                tokenizer_model="test-tokenizer",
                embedding_model="test-embedding",
                source_hash="source-hash",
                document_text_hash="document-hash",
                text_hash=f"text-hash-{chunk_id}",
                file_name="source.txt",
                file_type="txt",
                embedding_provider="test",
                embedding_dimensions=3,
                embedding_vector_hash=f"vector-hash-{chunk_id}",
                embedding_norm=1.0,
                embedding_run_id="embedding-run",
            ),
        )


if __name__ == "__main__":
    unittest.main()
