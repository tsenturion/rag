"""Токенизация обучающих примеров для PEFT fine-tuning локальной LLM."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from llm_tuning.dataset import ChatFormatter
from llm_tuning.models import FineTuningExample


class SupervisedChatDataset(Dataset):
    """Инкапсулирует подготовку и хранение токенизированных примеров для обучения, обеспечивая доступ к данным в формате, пригодном для модели."""

    def __init__(
        self,
        examples: list[FineTuningExample],
        tokenizer: Any,
        *,
        max_seq_length: int,
    ):
        """Готовит набор токенизированных примеров для обучения, гарантируя, что все данные приведены к единому формату с учётом максимальной длины последовательности."""
        self.items = [
            tokenize_example(
                example,
                tokenizer,
                max_seq_length=max_seq_length,
            )
            for example in examples
        ]

    def __len__(self) -> int:
        """Возвращает количество доступных элементов."""
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """Возвращает элемент по индексу."""
        return self.items[index]


class DataCollatorForCausalLM:
    """Формирует батчи с правильным паддингом и масками, обеспечивая корректную обработку последовательностей разной длины при обучении causal LM."""

    def __init__(self, tokenizer: Any):
        """Инициализирует коллатор данных с токенизатором, обеспечивая корректное паддингование и подготовку батчей для обучения causal language model."""
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Формирует батчи с выравниванием по максимальной длине, гарантируя корректное паддингование input_ids, attention_mask и labels для обучения с учётом маскировки."""
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_example(
    example: FineTuningExample,
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Преобразует пример обучения в токенизированный формат с масками и метками, обеспечивая корректную подготовку данных для обучения модели с учётом разделения подсказки и ответа."""
    full_text = ChatFormatter.apply_chat_template(
        tokenizer,
        ChatFormatter.all_messages(example),
        add_generation_prompt=False,
    )
    prompt_text = ChatFormatter.apply_chat_template(
        tokenizer,
        ChatFormatter.prompt_messages(example),
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    input_ids = list(full["input_ids"])
    attention_mask = list(full["attention_mask"])
    prompt_length = min(len(prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    if not any(label != -100 for label in labels):
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
