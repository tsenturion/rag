"""Проверки исполняемых этапов fine-tuning без загрузки весов модели."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from llm_tuning.config import load_fine_tuning_config
from llm_tuning.dataset import FineTuningDatasetLoader
from llm_tuning.evaluation import FineTuningEvaluationStage
from llm_tuning.generation import LocalGenerationStage
from llm_tuning.models import (
    ComparisonResult,
    DeviceReport,
    EvaluationReport,
    FineTuningExample,
    FineTuningExportResult,
    GeneratedAnswer,
    TrainingMetrics,
    TrainingResult,
    utc_now,
)
from llm_tuning.pipeline import FineTuningPipeline
from llm_tuning.training import FineTuningTrainingStage


def _config(tmp_path: Path):
    """Перенаправляет артефакты smoke-конфигурации во временный каталог."""
    config = load_fine_tuning_config("config/fine_tuning_smoke.yaml")
    paths = config.paths.model_copy(
        update={
            "output_dir": tmp_path / "runs",
            "adapter_output_dir": tmp_path / "adapters",
            "reports_dir": tmp_path / "reports",
        }
    )
    return config.model_copy(update={"paths": paths})


def _device() -> DeviceReport:
    """Возвращает переносимый CPU-отчёт для тестовых stages."""
    return DeviceReport(
        requested_device="auto",
        selected_device="cpu",
        torch_version=torch.__version__,
        xpu_available=False,
        cuda_available=False,
        selected_dtype="fp32",
    )


def _example(example_id: str = "example") -> FineTuningExample:
    """Создаёт пример с измеримыми required/forbidden критериями."""
    return FineTuningExample.model_validate(
        {
            "id": example_id,
            "messages": [
                {"role": "user", "content": "Как исправить инцидент?"},
                {"role": "assistant", "content": "Проверить журнал и сервис."},
            ],
            "required_keywords": ["журнал"],
            "forbidden_keywords": ["удалить данные"],
        }
    )


class _GenerationTokenizer:
    """Минимальный tokenizer для проверки реального generation-контракта stage."""

    chat_template = None
    pad_token_id = None
    eos_token_id = 9

    def __call__(self, _text: str, **_kwargs) -> dict[str, torch.Tensor]:
        """Кодирует prompt двумя токенами."""
        return {"input_ids": torch.tensor([[1, 2]])}

    @staticmethod
    def decode(tokens, **_kwargs) -> str:
        """Проверяет, что stage декодирует только generated suffix."""
        assert tokens.tolist() == [7, 8]
        return "  итоговый ответ  "


class _GenerationModel:
    """Фиксирует параметры generation и добавляет два токена ответа."""

    def __init__(self) -> None:
        """Создаёт журнал generation kwargs и состояния eval."""
        self.kwargs: dict[str, object] = {}
        self.eval_called = False

    def eval(self) -> None:
        """Фиксирует перевод модели в inference-режим."""
        self.eval_called = True

    def generate(self, **kwargs) -> torch.Tensor:
        """Возвращает prompt и известный generated suffix."""
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 7, 8]])


def test_local_generation_stage_uses_base_and_adapter_models(tmp_path: Path) -> None:
    """Проверяет prompt, generation options и выбор base/adapter загрузчика."""
    config = _config(tmp_path)
    stage = LocalGenerationStage(config)
    model = _GenerationModel()
    tokenizer = _GenerationTokenizer()
    stage.model_loader = SimpleNamespace(
        active_model_id="local-test-model",
        load_base_model=Mock(return_value=model),
        load_model_with_adapter=Mock(return_value=model),
        load_tokenizer=Mock(return_value=tokenizer),
    )

    with patch("llm_tuning.generation.build_device_report", return_value=_device()):
        base = stage.run(prompt="  запрос  ", max_new_tokens=5)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        tuned = stage.run(
            prompt="запрос",
            system_prompt=" системная инструкция ",
            adapter_path=adapter,
        )

    assert base.answer == "итоговый ответ"
    assert base.max_new_tokens == 5
    assert tuned.adapter_path == adapter
    assert model.eval_called
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["pad_token_id"] == tokenizer.eos_token_id
    stage.model_loader.load_base_model.assert_called_once()
    stage.model_loader.load_model_with_adapter.assert_called_once_with(adapter)
    with pytest.raises(ValueError, match="prompt"):
        stage.run(prompt="  ")
    with pytest.raises(FileNotFoundError):
        stage.run(prompt="запрос", adapter_path=tmp_path / "missing")


def test_evaluation_stage_builds_report_and_enforces_criteria(tmp_path: Path) -> None:
    """Проверяет полный evaluation lifecycle, pass rate и fail-fast политику."""
    config = _config(tmp_path)
    stage = FineTuningEvaluationStage(config)
    model = SimpleNamespace(eval=Mock())
    stage.model_loader = SimpleNamespace(
        active_model_id="local-test-model",
        load_base_model=Mock(return_value=model),
        load_model_with_adapter=Mock(return_value=model),
        load_tokenizer=Mock(return_value=object()),
    )
    stage.dataset_loader = SimpleNamespace(load_eval=Mock(return_value=[_example()]))
    stage._generate_one = Mock(return_value="Нужно проверить журнал")
    stage._compute_eval_loss = Mock(return_value=1.0)

    with patch("llm_tuning.evaluation.build_device_report", return_value=_device()):
        report = stage.run(run_id="evaluation-run")

    assert report.examples_count == 1
    assert report.passed_count == 1
    assert report.pass_rate == 1.0
    assert report.perplexity == pytest.approx(2.718282)
    assert report.answers[0].passed

    strict = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"fail_on_failed_criteria": True}
            )
        }
    )
    stage.config = strict
    stage._generate_one = Mock(return_value="удалить данные")
    with (
        patch("llm_tuning.evaluation.build_device_report", return_value=_device()),
        pytest.raises(ValueError, match="не прошла критерии"),
    ):
        stage.run(run_id="failed-run")


def test_training_stage_runs_trainer_and_saves_adapter(tmp_path: Path) -> None:
    """Проверяет связку dataset, PEFT model, Trainer, checkpoints и метрик."""
    config = _config(tmp_path)
    stage = FineTuningTrainingStage(config)
    tokenizer = SimpleNamespace(save_pretrained=Mock())
    model = object()
    stage.dataset_loader = SimpleNamespace(
        load_train=Mock(return_value=[_example("train")]),
        load_eval=Mock(return_value=[_example("eval")]),
    )
    stage.model_loader = SimpleNamespace(
        active_model_id="local-test-model",
        load_base_model=Mock(return_value=object()),
        load_tokenizer=Mock(return_value=tokenizer),
        prepare_peft_model=Mock(return_value=model),
    )

    trainer_instances = []

    class FakeTrainingArguments:
        """Сохраняет параметры, переданные training stage."""

        def __init__(self, **kwargs) -> None:
            """Фиксирует все параметры TrainingArguments без инициализации device."""
            self.kwargs = kwargs

    class FakeTrainer:
        """Имитирует train/evaluate/save без вычислений модели."""

        def __init__(self, **kwargs) -> None:
            """Сохраняет зависимости Trainer и создаёт воспроизводимое state."""
            self.kwargs = kwargs
            self.state = SimpleNamespace(
                global_step=2,
                log_history=[{"loss": 0.5, "opaque": Path("value")}],
            )
            self.saved_path = None
            trainer_instances.append(self)

        @staticmethod
        def train(**_kwargs):
            """Возвращает измеримые метрики завершённого обучающего шага."""
            return SimpleNamespace(metrics={"train_loss": 0.5, "train_runtime": 1.2})

        @staticmethod
        def evaluate():
            """Возвращает eval loss для итогового TrainingResult."""
            return {"eval_loss": 0.4}

        def save_model(self, path: str) -> None:
            """Фиксирует путь, по которому stage сохраняет PEFT adapter."""
            self.saved_path = path

    set_seed = Mock()
    fake_transformers = SimpleNamespace(
        Trainer=FakeTrainer,
        TrainingArguments=FakeTrainingArguments,
        set_seed=set_seed,
    )
    with (
        patch.dict(sys.modules, {"transformers": fake_transformers}),
        patch("llm_tuning.training.build_device_report", return_value=_device()),
        patch(
            "llm_tuning.training.trainable_parameter_metrics",
            return_value=TrainingMetrics(
                trainable_parameters=10,
                total_parameters=100,
                trainable_ratio=0.1,
            ),
        ),
        patch(
            "llm_tuning.training.SupervisedChatDataset",
            side_effect=lambda x, *_a, **_k: x,
        ),
        patch("llm_tuning.training.DataCollatorForCausalLM", return_value="collator"),
    ):
        result = stage.run(run_id="training-run")

    assert result.metrics.train_loss == 0.5
    assert result.metrics.eval_loss == 0.4
    assert result.metrics.global_step == 2
    assert result.adapter_path.exists()
    assert trainer_instances[0].saved_path == str(result.adapter_path)
    assert trainer_instances[0].kwargs["data_collator"] == "collator"
    assert result.log_history[0]["opaque"] == "value"
    set_seed.assert_called_once_with(config.run.seed)
    tokenizer.save_pretrained.assert_called_once_with(str(result.adapter_path))


def _evaluation_report(run_id: str, *, passed: bool) -> EvaluationReport:
    """Создаёт минимальный валидный отчёт для проверки pipeline orchestration."""
    answer = GeneratedAnswer(
        example_id="one",
        prompt="prompt",
        expected_answer="answer",
        generated_answer="answer" if passed else "bad",
        passed=passed,
    )
    return EvaluationReport(
        run_id=run_id,
        model_id="model",
        examples_count=1,
        passed_count=int(passed),
        failed_count=int(not passed),
        pass_rate=float(passed),
        answers=[answer],
    )


def test_fine_tuning_pipeline_orchestrates_all_public_modes(tmp_path: Path) -> None:
    """Проверяет baseline, train, adapter evaluation и report comparison как API."""
    config = _config(tmp_path)
    pipeline = object.__new__(FineTuningPipeline)
    pipeline.config = config
    validation = FineTuningDatasetLoader(config).validate()
    device = _device()
    pipeline.validate = Mock(return_value=(validation, device))
    baseline = _evaluation_report("pipeline-run", passed=False)
    tuned = _evaluation_report("pipeline-run", passed=True)
    adapter = tmp_path / "trained-adapter"
    adapter.mkdir()
    training = TrainingResult(
        run_id="pipeline-run",
        model_id="model",
        method="lora",
        adapter_path=adapter,
        trainer_output_dir=tmp_path / "trainer",
        metrics=TrainingMetrics(global_step=2),
        started_at=utc_now(),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    export = FineTuningExportResult(manifest_path=manifest, run_id="pipeline-run")
    pipeline.evaluator = SimpleNamespace(
        run=Mock(side_effect=[baseline, baseline, tuned, tuned])
    )
    pipeline.trainer = SimpleNamespace(run=Mock(return_value=training))
    comparison = ComparisonResult(
        run_id="pipeline-run",
        baseline_report_path=tmp_path / "baseline.json",
        tuned_report_path=tmp_path / "tuned.json",
        examples_count=1,
        baseline_pass_rate=0.0,
        tuned_pass_rate=1.0,
        pass_rate_delta=1.0,
        improved_examples=["one"],
        regressed_examples=[],
        unchanged_failed_examples=[],
        conclusion="Качество улучшилось",
    )
    pipeline.comparer = SimpleNamespace(compare=Mock(return_value=comparison))
    pipeline.exporter = SimpleNamespace(
        write_evaluation=Mock(
            side_effect=lambda _report, filename: tmp_path / filename
        ),
        write_comparison=Mock(return_value=tmp_path / "comparison.json"),
        write_manifest=Mock(return_value=export),
        load_evaluation_report=Mock(side_effect=[baseline, tuned]),
    )
    pipeline.tracker = SimpleNamespace(log_run=Mock())

    baseline_result = pipeline.run_baseline(run_id="pipeline-run")
    training_result = pipeline.run_training(run_id="pipeline-run")
    evaluation_result = pipeline.run_evaluation(
        adapter_path=adapter, run_id="pipeline-run"
    )
    comparison_result = pipeline.compare_reports(
        baseline_report_path=tmp_path / "baseline.json",
        tuned_report_path=tmp_path / "tuned.json",
        run_id="pipeline-run",
    )

    assert baseline_result.baseline is baseline
    assert training_result.training is training
    assert training_result.tuned is tuned
    assert evaluation_result.tuned is tuned
    assert comparison_result.comparison.pass_rate_delta == 1.0
    assert pipeline.tracker.log_run.call_count == 4
