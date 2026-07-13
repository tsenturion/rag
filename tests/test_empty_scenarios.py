from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import AgentAppConfig, AgentConfig, MemoryConfig  # noqa: E402
from agent_app.scenarios.models import ScenarioRunReport, ScenarioSuite  # noqa: E402
from agent_app.scenarios.runner import ScenarioRunner  # noqa: E402


class EmptyScenariosTest(unittest.TestCase):
    def test_empty_suite_is_rejected_by_validation(self) -> None:
        with self.assertRaises(ValidationError):
            ScenarioSuite.model_validate({"scenarios": []})

    def test_empty_successful_report_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ScenarioRunReport(
                config_path="config/scenarios.yaml",
                user_id="user",
                passed=True,
                results=[],
            )

    def test_runner_defensively_marks_constructed_empty_suite_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            suite = ScenarioSuite.model_construct(
                default_user_id="user",
                session_prefix="scenario",
                report_path=Path(temporary_dir) / "report.json",
                scenarios=[],
            )
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=Path(temporary_dir) / "memory.sqlite"),
            )

            report = ScenarioRunner(
                config,
                suite,
                config_path="config/scenarios.yaml",
            ).run_all()

            self.assertFalse(report.passed)
            self.assertEqual(report.results, [])


if __name__ == "__main__":
    unittest.main()
