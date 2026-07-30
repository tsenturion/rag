"""Регрессионные тесты token budget локальной LLM."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from agent_app.config import AgentConfig
from agent_app.llm import LocalTransformersChatModel


class RecordingTokenizer:
    """Фиксирует параметры усечения и возвращает минимальные torch tensors."""

    truncation_side = "right"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        """Создаёт пустой журнал последнего вызова."""
        self.kwargs: dict[str, object] = {}

    def __call__(self, _text: str, **kwargs):
        """Возвращает вход длиной, которую разрешил вызывающий код."""
        self.kwargs = kwargs
        length = int(kwargs["max_length"])
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    @staticmethod
    def decode(_tokens, **_kwargs) -> str:
        """Возвращает детерминированный ответ без реальной модели."""
        return "готово"


class StubModel:
    """Имитирует causal LM с контекстным окном 64 токена."""

    config = SimpleNamespace(max_position_embeddings=64)

    @staticmethod
    def generate(**kwargs):
        """Добавляет один generated token к переданному prompt."""
        return torch.cat(
            [kwargs["input_ids"], torch.tensor([[2]], dtype=torch.long)],
            dim=1,
        )


def test_local_llm_caps_input_and_restores_tokenizer_state() -> None:
    """Резервирует output budget и не оставляет tokenizer в изменённом состоянии."""
    llm = LocalTransformersChatModel.__new__(LocalTransformersChatModel)
    llm.config = AgentConfig(
        provider="local",
        model="stub",
        max_input_tokens=128,
        max_new_tokens=20,
    )
    llm.device = "cpu"
    llm.tokenizer = RecordingTokenizer()
    llm.model = StubModel()
    llm.bound_tools = []
    llm._format_prompt = lambda _messages: "очень длинный prompt"

    response = llm.invoke("запрос")

    assert response.content == "готово"
    assert llm.tokenizer.kwargs["truncation"] is True
    assert llm.tokenizer.kwargs["max_length"] == 44
    assert llm.tokenizer.truncation_side == "right"
