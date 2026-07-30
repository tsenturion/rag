"""Проверки недорогих частей fine-tuning без загрузки модели и запуска обучения."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_tuning.config import load_fine_tuning_config
from llm_tuning.dataset import (
    ChatFormatter,
    FineTuningDatasetLoader,
    messages_to_dicts,
)
from llm_tuning.device import build_device_report, select_dtype_name, torch_dtype
from llm_tuning.metrics import build_fine_tuning_counts


def test_dataset_loader_and_chat_formatters(tmp_path: Path) -> None:
    """Проверяет реальные JSONL, статистику и оба режима форматирования диалога."""
    config = load_fine_tuning_config("config/fine_tuning.yaml")
    loader = FineTuningDatasetLoader(config)
    validation = loader.validate()
    examples = loader.load_train()

    assert validation.train.examples_count == len(examples) > 0
    assert validation.eval.examples_count > 0
    assert validation.estimated_train_tokens_max > 0
    assert ChatFormatter.full_text(examples[0])
    assert ChatFormatter.prompt_text(examples[0])
    assert ChatFormatter.prompt_messages(examples[0])
    assert ChatFormatter.all_messages(examples[0]) == messages_to_dicts(
        examples[0].messages
    )

    templated = SimpleNamespace(
        chat_template="template",
        apply_chat_template=lambda messages, **kwargs: (
            f"templated:{len(messages)}:{kwargs['add_generation_prompt']}"
        ),
    )
    assert ChatFormatter.apply_chat_template(
        templated,
        ChatFormatter.prompt_messages(examples[0]),
        add_generation_prompt=True,
    ).startswith("templated:")
    fallback = SimpleNamespace(chat_template=None)
    assert "Ассистент:" in ChatFormatter.apply_chat_template(
        fallback,
        ChatFormatter.prompt_messages(examples[0]),
        add_generation_prompt=True,
    )

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="пуст"):
        loader._load_jsonl(empty)
    with pytest.raises(FileNotFoundError):
        loader._load_jsonl(tmp_path / "missing.jsonl")


def test_device_selection_and_complete_metric_summary() -> None:
    """Проверяет аппаратный отчёт, dtype mapping и все необязательные группы метрик."""
    config = load_fine_tuning_config("config/fine_tuning.yaml")
    device = build_device_report(config.model)
    assert device.selected_device in {"xpu", "cuda", "cpu"}
    assert select_dtype_name("auto", "cpu") == "fp32"
    assert select_dtype_name("auto", "xpu") == "bf16"
    assert select_dtype_name("fp16", "cpu") == "fp16"
    assert str(torch_dtype("bf16")).endswith("bfloat16")
    assert str(torch_dtype("fp16")).endswith("float16")
    assert str(torch_dtype("fp32")).endswith("float32")

    stats = SimpleNamespace(examples_count=2)
    dataset = SimpleNamespace(train=stats, eval=stats, train_eval_id_overlap_count=0)
    training = SimpleNamespace(
        metrics=SimpleNamespace(
            global_step=3,
            trainable_parameters=10,
            total_parameters=100,
            trainable_ratio=0.1,
            train_loss=0.4,
            eval_loss=0.5,
        )
    )
    baseline = SimpleNamespace(
        pass_rate=0.5, passed_count=1, failed_count=1, eval_loss=0.8
    )
    tuned = SimpleNamespace(
        pass_rate=1.0, passed_count=2, failed_count=0, eval_loss=0.3
    )
    comparison = SimpleNamespace(
        pass_rate_delta=0.5,
        improved_examples=["one"],
        regressed_examples=[],
        unchanged_failed_examples=[],
    )

    counts = build_fine_tuning_counts(
        dataset,
        device,
        training=training,
        baseline=baseline,
        tuned=tuned,
        comparison=comparison,
    )
    assert counts["global_step"] == 3
    assert counts["baseline_eval_loss"] == 0.8
    assert counts["tuned_eval_loss"] == 0.3
    assert counts["improved_examples_count"] == 1
