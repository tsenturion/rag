from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import load_agent_config  # noqa: E402
from agent_app.graph import AgentRunner  # noqa: E402


class FlakyInput(BaseModel):
    value: str


class RetryAfterErrorModel:
    supports_tool_calling = True

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        results = [message for message in messages if isinstance(message, ToolMessage)]
        if len(results) >= 2:
            return AIMessage(content="Повторная попытка выполнена успешно.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "flaky",
                    "args": {"value": "test"},
                    "id": f"flaky_call_{len(results) + 1}",
                    "type": "tool_call",
                }
            ],
        )


class AgentToolRetryTest(unittest.TestCase):
    def test_identical_call_is_retried_once_after_tool_error(self) -> None:
        invocations = 0

        def flaky(value: str) -> str:
            nonlocal invocations
            invocations += 1
            if invocations == 1:
                return json.dumps({"status": "error", "message": "temporary"})
            return json.dumps({"status": "ok", "value": value})

        tool = StructuredTool.from_function(
            name="flaky",
            description="Временно нестабильный tool.",
            func=flaky,
            args_schema=FlakyInput,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_agent_config(PROJECT_ROOT / "config" / "agent.yaml")
            config = config.model_copy(
                update={
                    "memory": config.memory.model_copy(
                        update={"sqlite_path": Path(temp_dir) / "memory.sqlite"}
                    )
                }
            )
            with (
                patch(
                    "agent_app.graph.build_llm",
                    return_value=RetryAfterErrorModel(),
                ),
                patch("agent_app.graph.build_tools", return_value=[tool]),
            ):
                response = AgentRunner(
                    config, user_id="user", session_id="session"
                ).ask("Вызови flaky и повтори после временной ошибки.")

        self.assertEqual(invocations, 2)
        self.assertEqual(response.tool_calls, ["flaky", "flaky"])
        self.assertFalse(response.trace.loop_guard_triggered)
        self.assertTrue(response.trace.tool_results[0].is_error)
        self.assertFalse(response.trace.tool_results[1].is_error)


if __name__ == "__main__":
    unittest.main()
