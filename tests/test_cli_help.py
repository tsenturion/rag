"""Регрессионные проверки лёгкой и безопасной справки консольных команд."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import TestCase


class CliHelpTest(TestCase):
    """Проверяет, что запрос справки не запускает сервисы и тяжёлые runtime-компоненты."""

    def test_code_runner_help_does_not_start_uvicorn(self) -> None:
        """Запрашивает справку в отдельном процессе без открытия сетевого порта."""
        result = subprocess.run(
            [sys.executable, "-m", "code_runner.cli", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"--host", result.stdout)
        self.assertNotIn(b"Started server process", result.stderr)

    def test_cli_imports_do_not_load_runtime_modules(self) -> None:
        """Импортирует CLI и подтверждает отсутствие LangGraph, FastAPI и MLflow runner."""
        source = (
            "import sys; "
            "import rag_prep.cli, agent_app.cli, agent_app.service.cli, "
            "agent_app.evaluation.cli; "
            "forbidden = {'rag_prep.pipeline', 'agent_app.graph', "
            "'agent_app.service.app', 'agent_app.evaluation.runner'}; "
            "assert not (forbidden & set(sys.modules)), "
            "forbidden & set(sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
