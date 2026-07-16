from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config import VectorStoreConfig  # noqa: E402
from rag_prep.models import EmbeddedChunk, EmbeddedChunkMetadata  # noqa: E402
from rag_prep.vector_store_stages.indexing import QdrantIndexingStage  # noqa: E402
from rag_prep.vector_store_stages.search import QdrantSearchStage  # noqa: E402
from rag_prep.vector_store_stages.validation import QdrantValidationStage  # noqa: E402


class QdrantPipelineTest(unittest.TestCase):
    def test_index_validation_and_similarity_search_in_memory(self) -> None:
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
        hit = QdrantSearchStage._hit(
            SimpleNamespace(id="point-without-payload", score=0.5, payload=None)
        )

        self.assertEqual(hit.point_id, "point-without-payload")
        self.assertEqual(hit.score, 0.5)
        self.assertIsNone(hit.chunk_id)
        self.assertEqual(hit.metadata, {})

    @staticmethod
    def _embedded_chunk(
        chunk_id: str,
        text: str,
        vector: list[float],
        position: int,
    ) -> EmbeddedChunk:
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
