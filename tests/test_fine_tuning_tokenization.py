"""Регрессионные тесты маскировки target-токенов fine-tuning."""

from __future__ import annotations

from llm_tuning.models import ChatMessage, FineTuningExample
from llm_tuning.tokenization import tokenize_example


class CharacterTokenizer:
    """Детерминированно кодирует символы для проверки границы prompt/answer."""

    chat_template = None

    def __call__(self, text: str, **_kwargs) -> dict[str, list[int]]:
        """Возвращает посимвольные IDs без внутреннего усечения."""
        ids = [ord(character) for character in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_long_prompt_keeps_real_answer_targets() -> None:
    """При переполнении удаляет начало prompt, но не размечает prompt как ответ."""
    example = FineTuningExample(
        id="long-prompt",
        messages=[
            ChatMessage(role="user", content="длинный контекст " * 30),
            ChatMessage(role="assistant", content="Итоговый ответ"),
        ],
    )

    item = tokenize_example(example, CharacterTokenizer(), max_seq_length=32)
    trained_ids = [
        token_id
        for token_id, label in zip(item["input_ids"], item["labels"], strict=True)
        if label != -100
    ]

    assert len(item["input_ids"]) == 32
    assert item["labels"][0] == -100
    assert trained_ids
    assert "Итоговый ответ" in "".join(chr(value) for value in trained_ids)
    assert all(
        label == token
        for label, token in zip(item["labels"], item["input_ids"], strict=True)
        if label != -100
    )
