"""Загрузка и проверка датасета для PEFT fine-tuning локальной LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_tuning.config import FineTuningPipelineConfig
from llm_tuning.models import (
    ChatMessage,
    DatasetStats,
    DatasetValidationResult,
    FineTuningExample,
)


class FineTuningDatasetLoader:
    """Обеспечивает загрузку, валидацию и статистику датасетов для PEFT fine-tuning локальной LLM, гарантируя корректность и полноту данных перед обучением."""

    def __init__(self, config: FineTuningPipelineConfig):
        """Готовит загрузчик с конфигурацией fine-tuning, обеспечивая доступ к путям и параметрам модели для последующих операций."""
        self.config = config

    def load_train(self) -> list[FineTuningExample]:
        """Загружает тренировочный датасет, гарантируя наличие и корректность данных для обучения модели."""
        return self._load_jsonl(self.config.paths.train_jsonl)

    def load_eval(self) -> list[FineTuningExample]:
        """Загружает валидационный датасет, обеспечивая корректность данных для оценки качества модели."""
        return self._load_jsonl(self.config.paths.eval_jsonl)

    def validate(self) -> DatasetValidationResult:
        """Проверяет целостность и уникальность данных в тренировочном и валидационном датасетах, выявляя пересечения и оценивая параметры для безопасного fine-tuning."""
        train = self.load_train()
        eval_examples = self.load_eval()
        train_ids = {example.id for example in train}
        eval_ids = {example.id for example in eval_examples}
        return DatasetValidationResult(
            train=self._stats(self.config.paths.train_jsonl, train),
            eval=self._stats(self.config.paths.eval_jsonl, eval_examples),
            train_eval_id_overlap_count=len(train_ids & eval_ids),
            max_seq_length=self.config.model.max_seq_length,
            estimated_train_tokens_max=self._estimated_tokens_max(train),
            estimated_eval_tokens_max=self._estimated_tokens_max(eval_examples),
        )

    @staticmethod
    def _load_jsonl(path: Path) -> list[FineTuningExample]:
        """Загружает и валидирует JSONL-файл с примерами, гарантируя непустой и корректный формат данных, выбрасывая ошибки при нарушениях."""
        if not path.exists():
            raise FileNotFoundError(f"Файл датасета не найден: {path}")
        examples = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                    examples.append(FineTuningExample.model_validate(payload))
                except Exception as exc:
                    raise ValueError(
                        f"Ошибка чтения {path} на строке {line_number}: {exc}"
                    ) from exc
        if not examples:
            raise ValueError(f"Файл датасета пуст: {path}")
        return examples

    @staticmethod
    def _stats(path: Path, examples: list[FineTuningExample]) -> DatasetStats:
        """Вычисляет статистику по датасету, включая длины и уникальность идентификаторов, для мониторинга качества и структуры данных."""
        prompt_lengths = [
            len(ChatFormatter.prompt_text(example)) for example in examples
        ]
        answer_lengths = [len(examples_answer(example)) for example in examples]
        ids = [example.id for example in examples]
        return DatasetStats(
            path=path,
            examples_count=len(examples),
            max_prompt_chars=max(prompt_lengths) if prompt_lengths else 0,
            max_answer_chars=max(answer_lengths) if answer_lengths else 0,
            avg_prompt_chars=round(sum(prompt_lengths) / len(prompt_lengths), 3)
            if prompt_lengths
            else 0.0,
            avg_answer_chars=round(sum(answer_lengths) / len(answer_lengths), 3)
            if answer_lengths
            else 0.0,
            ids_are_unique=len(ids) == len(set(ids)),
        )

    @staticmethod
    def _estimated_tokens_max(examples: list[FineTuningExample]) -> int:
        # Грубая оценка без загрузки tokenizer: для кириллицы 1 токен часто занимает 2-4 символа.
        # Здесь нужен только ранний sanity-check, точная обрезка делается tokenizer-ом.
        """Оценивает максимальное количество токенов в примерах для предварительной проверки соответствия ограничению модели без использования токенизатора."""
        if not examples:
            return 0
        return max(
            max(1, len(ChatFormatter.full_text(example)) // 3) for example in examples
        )


class ChatFormatter:
    """Форматирует chat-примеры в полный текст, prompt и ожидаемый ответ."""

    @staticmethod
    def full_text(example: FineTuningExample) -> str:
        """Гарантирует получение полного диалога в виде текста для обучения или анализа, сохраняя порядок и роли сообщений."""
        return "\n".join(
            f"{message.role}: {message.content}" for message in example.messages
        )

    @staticmethod
    def prompt_text(example: FineTuningExample) -> str:
        """Гарантирует получение текста диалога без последнего сообщения, чтобы использовать как входную подсказку для генерации ответа LLM."""
        return "\n".join(
            f"{message.role}: {message.content}" for message in example.messages[:-1]
        )

    @staticmethod
    def prompt_messages(example: FineTuningExample) -> list[dict[str, str]]:
        """Гарантирует сериализацию всех сообщений диалога, кроме последнего, в формате, совместимом с токенизаторами и LLM API."""
        return [message.model_dump() for message in example.messages[:-1]]

    @staticmethod
    def all_messages(example: FineTuningExample) -> list[dict[str, str]]:
        """Гарантирует сериализацию всех сообщений диалога для передачи в токенизатор или сохранения в датасете."""
        return [message.model_dump() for message in example.messages]

    @staticmethod
    def apply_chat_template(
        tokenizer: Any,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        """Гарантирует формирование текстовой подсказки по сообщениям в формате, совместимом с выбранным токенизатором и политикой генерации."""
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        lines = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                lines.append(f"Системная инструкция: {content}")
            elif role == "user":
                lines.append(f"Пользователь: {content}")
            elif role == "assistant":
                lines.append(f"Ассистент: {content}")
        if add_generation_prompt:
            lines.append("Ассистент:")
        return "\n".join(lines)


def examples_answer(example: FineTuningExample) -> str:
    """Извлекает итоговый ответ из примера для использования в обучении и оценке модели, обеспечивая единообразный формат вывода."""
    return example.messages[-1].content


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Преобразует сообщения чата в словарный формат для сериализации и совместимости с PEFT fine-tuning процессом."""
    return [message.model_dump() for message in messages]
