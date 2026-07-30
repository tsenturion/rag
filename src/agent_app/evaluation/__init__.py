"""Публичный интерфейс для оценки качества агентной системы."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_app.evaluation.dataset import load_evaluation_suite
    from agent_app.evaluation.models import EvaluationReport, EvaluationSuite
    from agent_app.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationReport",
    "EvaluationRunner",
    "EvaluationSuite",
    "load_evaluation_suite",
]


def __getattr__(name: str):
    """Подключает модели, загрузчик или runner оценки только при использовании API."""
    if name == "load_evaluation_suite":
        from agent_app.evaluation.dataset import load_evaluation_suite

        return load_evaluation_suite
    if name in {"EvaluationReport", "EvaluationSuite"}:
        from agent_app.evaluation import models

        return getattr(models, name)
    if name == "EvaluationRunner":
        from agent_app.evaluation.runner import EvaluationRunner

        return EvaluationRunner
    raise AttributeError(name)
