from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config import EmbeddingConfig  # noqa: E402
from rag_prep.embedding_stages.embedding import OpenAIEmbeddingStage  # noqa: E402
from rag_prep.models import ChunkMetadata, PreparedChunk  # noqa: E402


class EmbeddingStageTest(unittest.TestCase):
    def test_openai_stage_preserves_chunk_mapping_and_metadata(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            ]
        )
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
            api_key_env="TEST_OPENAI_API_KEY",
            batch_size=2,
            max_batch_tokens=100,
            max_input_tokens=100,
            max_retries=1,
            clear_no_proxy_for_openai=False,
        )
        chunks = [
            self._chunk("chunk-1", "Первый текст"),
            self._chunk("chunk-2", "Второй текст"),
        ]

        with (
            patch.dict(os.environ, {"TEST_OPENAI_API_KEY": "test-key"}),
            patch(
                "rag_prep.embedding_stages.embedding.OpenAI",
                return_value=client,
            ),
        ):
            embedded = OpenAIEmbeddingStage(config).run(chunks, run_id="embedding-run")

        self.assertEqual(
            [item.metadata.id for item in embedded], ["chunk-1", "chunk-2"]
        )
        self.assertEqual(embedded[0].embedding, [1.0, 0.0, 0.0])
        self.assertEqual(embedded[1].embedding, [0.0, 1.0, 0.0])
        self.assertEqual(embedded[0].metadata.embedding_dimensions, 3)
        self.assertEqual(embedded[0].metadata.embedding_run_id, "embedding-run")
        client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["Первый текст", "Второй текст"],
            encoding_format="float",
            dimensions=3,
        )

    def test_openai_request_omits_dimensions_when_not_configured(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0])]
        )
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
            api_key_env="TEST_OPENAI_API_KEY",
            max_retries=1,
            clear_no_proxy_for_openai=False,
        )

        with (
            patch.dict(os.environ, {"TEST_OPENAI_API_KEY": "test-key"}),
            patch(
                "rag_prep.embedding_stages.embedding.OpenAI",
                return_value=client,
            ),
        ):
            result = OpenAIEmbeddingStage(config)._embed_texts(["Текст"])

        self.assertEqual(result.vectors, [[1.0, 0.0, 0.0]])
        client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["Текст"],
            encoding_format="float",
        )

    @staticmethod
    def _chunk(chunk_id: str, text: str) -> PreparedChunk:
        return PreparedChunk(
            text=text,
            metadata=ChunkMetadata(
                id=chunk_id,
                document_id="document-1",
                source="source.txt",
                section="Раздел",
                position=0,
                chunk_start_char=0,
                chunk_end_char=len(text),
                chunk_token_count=3,
                chunk_size=100,
                chunk_overlap=10,
                chunking_strategy="sentence",
                tokenizer_model="text-embedding-3-small",
                embedding_model="text-embedding-3-small",
                source_hash="source-hash",
                document_text_hash="document-hash",
                text_hash=f"hash-{chunk_id}",
                file_name="source.txt",
                file_type="txt",
            ),
        )


if __name__ == "__main__":
    unittest.main()
