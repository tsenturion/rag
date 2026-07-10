from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import load_agent_config  # noqa: E402
from agent_app.graph import AgentRunner  # noqa: E402
from agent_app.memory.store import SQLiteMemoryStore  # noqa: E402
from agent_app.memory.summary import SummaryMemory  # noqa: E402


class SummaryModel:
    def invoke(self, _messages):
        return AIMessage(content="Краткое резюме")


class SummaryFailingModel:
    supports_tool_calling = False

    def invoke(self, messages):
        if messages and isinstance(messages[0], SystemMessage):
            if "Сожми историю" in str(messages[0].content):
                raise RuntimeError("summary backend unavailable")
        return AIMessage(content="Готовый ответ")


class SummaryMemoryTest(unittest.TestCase):
    def test_history_is_trimmed_on_complete_turn_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = SummaryMemory(
                SQLiteMemoryStore(Path(temp_dir) / "memory.sqlite"),
                user_id="user",
                session_id="session",
                max_chars=500,
            )
            kept = summary.summarize_if_needed(
                llm=SummaryModel(),
                messages=[
                    HumanMessage(content="Вопрос 1"),
                    AIMessage(content="Ответ 1"),
                    HumanMessage(content="Вопрос 2"),
                    AIMessage(content="Ответ 2"),
                ],
                max_history_messages=3,
            )

        self.assertEqual([message.type for message in kept], ["human", "ai"])

    def test_summary_failure_does_not_discard_ready_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_agent_config(PROJECT_ROOT / "config" / "agent.yaml")
            config = config.model_copy(
                update={
                    "agent": config.agent.model_copy(
                        update={"max_history_messages": 3}
                    ),
                    "memory": config.memory.model_copy(
                        update={"sqlite_path": Path(temp_dir) / "memory.sqlite"}
                    ),
                }
            )
            with patch(
                "agent_app.graph.build_llm",
                return_value=SummaryFailingModel(),
            ):
                runner = AgentRunner(config, user_id="user", session_id="session")
                runner.ask("Первый вопрос")
                with self.assertLogs("agent_app.graph", level="ERROR"):
                    response = runner.ask("Второй вопрос")

        self.assertEqual(response.answer, "Готовый ответ")
        self.assertEqual(
            [message.type for message in runner.short_term.snapshot()],
            ["human", "ai"],
        )


if __name__ == "__main__":
    unittest.main()
