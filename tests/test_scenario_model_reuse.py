from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import load_agent_config  # noqa: E402
from agent_app.scenarios.models import ScenarioSuite  # noqa: E402
from agent_app.scenarios.runner import ScenarioRunner  # noqa: E402


class FakeChatModel:
    supports_tool_calling = False

    def invoke(self, _messages):
        return AIMessage(content="Готово")


class ScenarioModelReuseTest(unittest.TestCase):
    def test_scenario_suite_builds_llm_only_once(self) -> None:
        scenarios = [
            {
                "id": f"scenario_{index}",
                "test_case_id": f"TC-{index}",
                "title": f"Сценарий {index}",
                "type": "main",
                "goal": "Проверить переиспользование модели.",
                "user_request": "Ответить.",
                "expected_result": "Ответ получен.",
                "steps": [
                    {
                        "id": "answer",
                        "test_case_id": f"TC-{index}.1",
                        "title": "Ответить",
                        "user_request": "Ответь кратко.",
                        "expected_result": "Ответ получен.",
                    }
                ],
            }
            for index in range(2)
        ]
        suite = ScenarioSuite.model_validate({"scenarios": scenarios})

        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_agent_config(PROJECT_ROOT / "config" / "agent_local.yaml")
            config = config.model_copy(
                update={
                    "memory": config.memory.model_copy(
                        update={"sqlite_path": Path(temp_dir) / "memory.sqlite"}
                    )
                }
            )
            with patch(
                "agent_app.graph.build_llm",
                return_value=FakeChatModel(),
            ) as build_llm:
                report = ScenarioRunner(config, suite, config_path="test").run_all()

        self.assertTrue(report.passed)
        build_llm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
