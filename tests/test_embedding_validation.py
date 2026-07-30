"""Регрессионные тесты для подсистемы embedding_validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config import EmbeddingConfig  # noqa: E402
from rag_prep.embedding_stages.validation import EmbeddingValidationStage  # noqa: E402
from rag_prep.models import (  # noqa: E402
    ChunkMetadata,
    EmbeddedChunk,
    EmbeddedChunkMetadata,
    PreparedChunk,
)


def prepared_chunk(chunk_id: str, text: str) -> PreparedChunk:
    """Генерирует подготовленный фрагмент с фиксированными метаданными для обеспечения стабильности тестов валидации эмбеддингов."""
    return PreparedChunk(
        text=text,
        metadata=ChunkMetadata(
            id=chunk_id,
            document_id=f"document-{chunk_id}",
            source="source.txt",
            section="section",
            position=0,
            chunk_start_char=0,
            chunk_end_char=len(text),
            chunk_token_count=3,
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
        ),
    )


def embedded_chunk(chunk: PreparedChunk) -> EmbeddedChunk:
    """Создаёт встроенный фрагмент с заданными эмбеддингами и метаданными для проверки корректности обработки эмбеддингов."""
    metadata = chunk.metadata.model_dump(mode="python")
    metadata.update(
        {
            "embedding_provider": "openai",
            "embedding_dimensions": 3,
            "embedding_vector_hash": "vector-hash",
            "embedding_norm": 1.0,
            "embedding_run_id": "run-id",
        }
    )
    return EmbeddedChunk(
        text=chunk.text,
        embedding=[1.0, 0.0, 0.0],
        metadata=EmbeddedChunkMetadata.model_validate(metadata),
    )


class EmbeddingValidationTest(unittest.TestCase):
    """Проверяет корректность валидации эмбеддингов, включая соответствие идентификаторов и метаданных, чтобы гарантировать целостность и согласованность данных."""

    def setUp(self) -> None:
        """Инициализирует конфигурацию и этап валидации эмбеддингов, обеспечивая готовность тестового окружения."""
        config = EmbeddingConfig(
            provider="openai",
            model="test-embedding",
            dimensions=3,
            api_key_env="TEST_OPENAI_API_KEY",
            fail_on_validation_error=False,
        )
        self.stage = EmbeddingValidationStage(config)

    def test_unrelated_embedding_ids_are_rejected(self) -> None:
        """Проверяет, что эмбеддинги с идентификаторами, не относящимися к исходным данным, отклоняются и учитываются как ошибки валидации."""
        source = [
            prepared_chunk("source-1", "Первый"),
            prepared_chunk("source-2", "Второй"),
        ]
        unrelated = [
            embedded_chunk(prepared_chunk("other-1", "Чужой первый")),
            embedded_chunk(prepared_chunk("other-2", "Чужой второй")),
        ]

        result = self.stage.run(source, unrelated)

        self.assertTrue(result.has_errors)
        self.assertEqual(result.missing_chunk_ids_count, 2)
        self.assertEqual(result.unexpected_chunk_ids_count, 2)
        self.assertEqual(result.missing_embeddings_count, 2)

    def test_empty_input_and_output_are_rejected(self) -> None:
        """Проверяет, что отсутствие исходных чанков и embeddings не маскируется нулевыми счётчиками."""
        result = self.stage.run([], [])

        self.assertTrue(result.has_errors)
        self.assertEqual(result.empty_source_chunks_count, 1)
        self.assertEqual(result.empty_embeddings_count, 1)

    def test_text_and_identity_metadata_mismatch_are_rejected(self) -> None:
        """Проверяет, что несоответствие текста и метаданных между исходным и эмбеддингом выявляется и приводит к ошибкам валидации."""
        source = prepared_chunk("source-1", "Исходный текст")
        embedded = embedded_chunk(source)
        embedded = embedded.model_copy(
            update={
                "text": "Подменённый текст",
                "metadata": embedded.metadata.model_copy(
                    update={"document_id": "other-document"}
                ),
            }
        )

        result = self.stage.run([source], [embedded])

        self.assertTrue(result.has_errors)
        self.assertEqual(result.text_mismatch_count, 1)
        self.assertEqual(result.metadata_mismatch_count, 1)

    def test_declared_provider_and_dimensions_must_match_vector(self) -> None:
        """Не допускает ложное происхождение вектора в downstream metadata."""
        source = prepared_chunk("source-1", "Исходный текст")
        embedded = embedded_chunk(source)
        embedded = embedded.model_copy(
            update={
                "metadata": embedded.metadata.model_copy(
                    update={
                        "embedding_provider": "gigachat",
                        "embedding_dimensions": 999,
                    }
                )
            }
        )

        result = self.stage.run([source], [embedded])

        self.assertTrue(result.has_errors)
        self.assertEqual(result.provider_mismatch_count, 1)
        self.assertEqual(result.declared_dimension_mismatch_count, 1)


if __name__ == "__main__":
    unittest.main()
