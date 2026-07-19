"""Проверки локального и GigaChat-провайдеров embeddings без сетевых вызовов."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rag_prep.config import EmbeddingConfig
from rag_prep.embedding_stages.embedding import (
    EmbeddingBatchResult,
    GigaChatEmbeddingStage,
    LocalEmbeddingStage,
    build_embedding_stage,
    ensure_embedding_runtime,
)
from rag_prep.models import ChunkMetadata, PreparedChunk, utc_now


def _chunk(chunk_id: str, text: str, position: int = 0) -> PreparedChunk:
    """Создаёт валидный чанк, чтобы проверять provider-контракт целиком."""
    return PreparedChunk(
        text=text,
        metadata=ChunkMetadata(
            id=chunk_id,
            document_id="document-1",
            source="source.txt",
            section="Раздел",
            position=position,
            chunk_start_char=0,
            chunk_end_char=len(text),
            chunk_token_count=3,
            chunk_size=100,
            chunk_overlap=10,
            chunking_strategy="sentence",
            tokenizer_model="embedding-model",
            embedding_model="embedding-model",
            source_hash="source-hash",
            document_text_hash="document-hash",
            text_hash=f"hash-{chunk_id}",
            file_name="source.txt",
            file_type="txt",
        ),
    )


class _Tokenizer:
    """Имитирует tokenizer и сохраняет тексты, переданные локальному encoder."""

    def __init__(self) -> None:
        """Создаёт журнал последнего batch токенизации."""
        self.last_texts: list[str] = []

    def encode(self, text: str, **_kwargs) -> list[int]:
        """Возвращает один token на слово для детерминированного batching."""
        return list(range(max(1, len(text.split()))))

    def __call__(self, texts: list[str], **_kwargs) -> dict[str, torch.Tensor]:
        """Создаёт batch с padding-mask для проверки mean pooling."""
        self.last_texts = texts
        return {
            "input_ids": torch.tensor([[1, 2], [3, 0]][: len(texts)]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]][: len(texts)]),
        }


class _Encoder:
    """Возвращает известные hidden states и фиксирует lifecycle модели."""

    def __init__(self) -> None:
        """Инициализирует признак inference-режима для тестового encoder."""
        self.evaluating = False

    def __call__(self, **_inputs):
        """Возвращает два двумерных представления с padding во второй строке."""
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [
                    [[1.0, 0.0], [3.0, 0.0]],
                    [[0.0, 2.0], [100.0, 100.0]],
                ]
            )[: len(_inputs["input_ids"])]
        )


def _local_stage(*, normalize: bool = False, pooling: str = "mean"):
    """Создаёт локальный stage без загрузки Hugging Face-модели."""
    stage = object.__new__(LocalEmbeddingStage)
    stage.config = EmbeddingConfig(
        provider="local",
        model="local-model",
        dimensions=2,
        api_key_env="UNUSED",
        batch_size=2,
        max_batch_tokens=20,
        max_input_tokens=10,
        normalize=normalize,
        pooling=pooling,
        passage_prefix="passage: ",
        query_prefix="query: ",
        local_device="cpu",
        local_dtype="fp32",
    )
    stage.device = "cpu"
    stage.tokenizer = _Tokenizer()
    stage.model = _Encoder()
    return stage


def test_local_embeddings_apply_prefix_pooling_and_metadata() -> None:
    """Проверяет E5-префикс, mean pooling, порядок и lineage metadata."""
    stage = _local_stage(normalize=False)

    result = stage.run(
        [_chunk("one", "первый текст"), _chunk("two", "второй", 1)],
        run_id="local-run",
    )

    assert stage.tokenizer.last_texts == ["passage: первый текст", "passage: второй"]
    assert result[0].embedding == [2.0, 0.0]
    assert result[1].embedding == [0.0, 2.0]
    assert result[1].metadata.embedding_provider == "local"
    assert result[1].metadata.embedding_run_id == "local-run"
    assert result[1].metadata.embedding_dimensions == 2


def test_local_query_normalization_and_batched_limits() -> None:
    """Проверяет query-prefix, L2-нормализацию и отказ для oversized-входа."""
    stage = _local_stage(normalize=True)

    vector = stage.embed_query("поиск")
    assert stage.tokenizer.last_texts == ["query: поиск"]
    assert pytest.approx(sum(value**2 for value in vector), rel=1e-6) == 1.0

    stage._token_count = lambda text: 11
    with pytest.raises(ValueError, match="token limit"):
        list(stage._batches([_chunk("large", "слишком большой")]))


def test_local_pooling_and_runtime_selection_helpers(monkeypatch) -> None:
    """Проверяет CLS/mean pooling, dtype и безопасные HF-переменные."""
    stage = _local_stage(pooling="cls")
    outputs = SimpleNamespace(
        last_hidden_state=torch.tensor([[[1.0, 2.0], [9.0, 9.0]]])
    )
    mask = torch.tensor([[1, 0]])
    assert stage._pool(outputs, mask).tolist() == [[1.0, 2.0]]
    stage.config = stage.config.model_copy(
        update={"hub_disable_xet": True, "hub_disable_symlink_warning": True}
    )
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    stage._configure_hf_hub_downloads()
    assert stage._select_device() == "cpu"
    assert stage._select_dtype("cpu") == torch.float32


def test_gigachat_batches_and_preserves_chunk_mapping() -> None:
    """Проверяет provider batch, timestamp и нормализованную метаинформацию."""
    stage = object.__new__(GigaChatEmbeddingStage)
    stage.config = EmbeddingConfig(
        provider="gigachat",
        model="Embeddings",
        dimensions=2,
        api_key_env="GIGACHAT_AUTH_KEY",
        batch_size=2,
        max_batch_tokens=100,
        max_input_tokens=100,
        normalize=True,
    )
    stage._embed_texts = lambda texts: EmbeddingBatchResult(
        vectors=[[3.0, 4.0] for _ in texts],
        embedded_at=utc_now(),
    )

    result = stage.run(
        [_chunk("one", "текст"), _chunk("two", "текст", 1)], run_id="giga"
    )

    assert [item.metadata.id for item in result] == ["one", "two"]
    assert result[0].embedding == pytest.approx([0.6, 0.8])
    assert result[0].metadata.embedding_provider == "gigachat"
    assert stage._token_count("1234567") == 3


def test_embedding_provider_factory_and_runtime_key_checks(monkeypatch) -> None:
    """Проверяет явную маршрутизацию provider и раннюю проверку секретов."""
    configs = {
        provider: EmbeddingConfig(
            provider=provider,
            model="model",
            dimensions=2,
            api_key_env=f"{provider.upper()}_KEY",
        )
        for provider in ("openai", "local", "gigachat")
    }
    sentinels = {name: object() for name in configs}
    with (
        patch(
            "rag_prep.embedding_stages.embedding.OpenAIEmbeddingStage",
            return_value=sentinels["openai"],
        ),
        patch(
            "rag_prep.embedding_stages.embedding.LocalEmbeddingStage",
            return_value=sentinels["local"],
        ),
        patch(
            "rag_prep.embedding_stages.embedding.GigaChatEmbeddingStage",
            return_value=sentinels["gigachat"],
        ),
    ):
        for provider, config in configs.items():
            assert build_embedding_stage(config) is sentinels[provider]

    monkeypatch.setenv("OPENAI_KEY", "OPENAI_KEY='secret'")
    monkeypatch.setenv("GIGACHAT_KEY", "secret")
    ensure_embedding_runtime(configs["openai"])
    ensure_embedding_runtime(configs["gigachat"])
    ensure_embedding_runtime(configs["local"])
