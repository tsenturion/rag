from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.cli import main  # noqa: E402
from agent_app.scenarios.models import ScenarioRunReport  # noqa: E402


class AgentCliTest(unittest.TestCase):
    def test_failed_scenario_report_returns_nonzero_exit_code(self) -> None:
        report = ScenarioRunReport(
            config_path="config/agent_scenarios.yaml",
            user_id="test",
            passed=False,
            results=[],
        )
        scenario_runner = Mock()
        scenario_runner.run_all.return_value = report
        scenario_runner.write_report.return_value = Path("report.json")

        with (
            patch("agent_app.cli.ScenarioRunner", return_value=scenario_runner),
            patch.object(
                sys,
                "argv",
                [
                    "rag-agent",
                    "--config",
                    "config/agent_openai.yaml",
                    "--run-scenarios",
                ],
            ),
            patch("builtins.print"),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
