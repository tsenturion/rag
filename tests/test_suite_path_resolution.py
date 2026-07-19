"""Регрессионные проверки загрузки YAML-наборов из произвольного каталога."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase

from agent_app.evaluation.dataset import load_evaluation_suite
from agent_app.multi_agent.runtime import load_comparison_suite


class SuitePathResolutionTest(TestCase):
    """Проверяет независимость загрузчиков сценариев от рабочего каталога процесса."""

    def test_comparison_suite_loads_from_project_root(self) -> None:
        """Загружает сравнительные сценарии по проектному пути из чужого CWD."""
        suite = self._load_from_foreign_cwd(
            load_comparison_suite,
            "config/multi_agent_scenarios.yaml",
        )

        self.assertGreater(len(suite.scenarios), 0)
        self.assertEqual(suite.scenarios[0].expected_terms, ["503", "500"])

    def test_evaluation_suite_loads_from_project_root(self) -> None:
        """Загружает evaluation-набор по проектному пути из чужого CWD."""
        suite = self._load_from_foreign_cwd(
            load_evaluation_suite,
            "config/evaluation/engineering_support_cases.yaml",
        )

        self.assertGreater(len(suite.cases), 0)
        self.assertEqual(suite.name, "engineering-support-quality")

    @staticmethod
    def _load_from_foreign_cwd(loader, path: str):
        """Временно меняет CWD и гарантированно восстанавливает его после загрузки."""
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_dir:
            try:
                os.chdir(temporary_dir)
                return loader(path)
            finally:
                os.chdir(original_cwd)
