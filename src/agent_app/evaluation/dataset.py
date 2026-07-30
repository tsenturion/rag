"""Загрузка и проверка датасета для оценки качества агентной системы."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_app.evaluation.models import EvaluationSuite
from agent_app.paths import resolve_project_file


def load_evaluation_suite(path: str | Path) -> EvaluationSuite:
    """Создаёт воспроизводимый набор тестов для оценки качества агента, гарантируя корректность структуры и существование файла."""
    suite_path = resolve_project_file(path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Набор evaluation не найден: {suite_path}")
    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Корень evaluation YAML должен быть mapping")
    return EvaluationSuite.model_validate(raw)
