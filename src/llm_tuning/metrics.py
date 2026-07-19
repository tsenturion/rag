"""Расчёт метрик для PEFT fine-tuning локальной LLM."""

from __future__ import annotations

from llm_tuning.models import (
    ComparisonResult,
    DatasetValidationResult,
    DeviceReport,
    EvaluationReport,
    TrainingResult,
)


def build_fine_tuning_counts(
    dataset_validation: DatasetValidationResult,
    device: DeviceReport,
    *,
    training: TrainingResult | None = None,
    baseline: EvaluationReport | None = None,
    tuned: EvaluationReport | None = None,
    comparison: ComparisonResult | None = None,
) -> dict[str, int | float]:
    """Формирует сводные метрики fine-tuning, объединяя данные валидации, обучения и оценки для комплексного анализа качества модели."""
    counts: dict[str, int | float] = {
        "train_examples_count": dataset_validation.train.examples_count,
        "eval_examples_count": dataset_validation.eval.examples_count,
        "train_eval_id_overlap_count": dataset_validation.train_eval_id_overlap_count,
        "xpu_available": int(device.xpu_available),
        "cuda_available": int(device.cuda_available),
    }
    if training is not None:
        counts.update(
            {
                "global_step": training.metrics.global_step,
                "trainable_parameters": training.metrics.trainable_parameters,
                "total_parameters": training.metrics.total_parameters,
                "trainable_ratio": training.metrics.trainable_ratio,
            }
        )
        if training.metrics.train_loss is not None:
            counts["train_loss"] = training.metrics.train_loss
        if training.metrics.eval_loss is not None:
            counts["training_eval_loss"] = training.metrics.eval_loss
    if baseline is not None:
        counts.update(
            {
                "baseline_pass_rate": baseline.pass_rate,
                "baseline_passed_count": baseline.passed_count,
                "baseline_failed_count": baseline.failed_count,
            }
        )
        if baseline.eval_loss is not None:
            counts["baseline_eval_loss"] = baseline.eval_loss
    if tuned is not None:
        counts.update(
            {
                "tuned_pass_rate": tuned.pass_rate,
                "tuned_passed_count": tuned.passed_count,
                "tuned_failed_count": tuned.failed_count,
            }
        )
        if tuned.eval_loss is not None:
            counts["tuned_eval_loss"] = tuned.eval_loss
    if comparison is not None:
        counts.update(
            {
                "pass_rate_delta": comparison.pass_rate_delta,
                "improved_examples_count": len(comparison.improved_examples),
                "regressed_examples_count": len(comparison.regressed_examples),
                "unchanged_failed_examples_count": len(
                    comparison.unchanged_failed_examples
                ),
            }
        )
    return counts
