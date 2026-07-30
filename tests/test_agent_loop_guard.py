"""Регрессионные тесты для подсистемы agent_loop_guard."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import load_agent_config  # noqa: E402
from agent_app.graph import AgentRunner  # noqa: E402


class RepeatingToolModel:
    """Моделирует инструмент с повторяющимися вызовами, гарантируя отслеживание количества вызовов для тестирования защиты от бесконечных циклов."""

    supports_tool_calling = True

    def __init__(self) -> None:
        """Готовит тестовую модель к отслеживанию количества вызовов для проверки защиты от зацикливания."""
        self.calls = 0

    def bind_tools(self, _tools: list[Any]) -> "RepeatingToolModel":
        """Проверяет, что связывание инструментов не влияет на поведение тестовой модели в сценариях регрессионного тестирования."""
        return self

    def invoke(self, _messages: list[Any]) -> AIMessage:
        """Проверяет, что при каждом вызове генерируется новый tool_call с уникальным идентификатором для тестирования защиты от зацикливания."""
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "current_datetime",
                    "args": {"timezone": "UTC"},
                    "id": f"loop_call_{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


class ChangingToolModel(RepeatingToolModel):
    """Моделирует инструмент с изменяющимися параметрами вызовов, гарантируя уникальность каждого вызова для проверки реакции защиты на изменяющиеся запросы."""

    def invoke(self, _messages: list[Any]) -> AIMessage:
        """Проверяет, что последовательные tool_call отличаются параметрами и идентификаторами для тестирования реакции guard на изменяющиеся вызовы."""
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": f"{self.calls}+1"},
                    "id": f"changing_loop_call_{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


class AgentLoopGuardTest(unittest.TestCase):
    """Проверяет подсистему защиты от зацикливания в агенте, гарантируя корректное обнаружение и остановку бесконечных циклов в различных конфигурациях."""

    def test_loop_guard_works_for_all_agent_provider_configs(self) -> None:
        """Проверяет, что при бесконечном повторении одного инструмента агент корректно срабатывает защита от циклов, прерывая выполнение до достижения лимита рекурсии и возвращая статус отмены без ошибок."""
        config_paths = [
            PROJECT_ROOT / "config" / "agent_openai.yaml",
            PROJECT_ROOT / "config" / "agent_local.yaml",
            PROJECT_ROOT / "config" / "agent_gigachat.yaml",
        ]

        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config = load_agent_config(config_path)
                    memory = config.memory.model_copy(
                        update={"sqlite_path": Path(temp_dir) / "memory.sqlite"}
                    )
                    config = config.model_copy(update={"memory": memory})

                    with patch(
                        "agent_app.graph.build_llm",
                        return_value=RepeatingToolModel(),
                    ):
                        runner = AgentRunner(
                            config,
                            user_id=f"loop_guard_{config.agent.provider}",
                            session_id="loop_guard_test",
                        )
                        response = runner.ask(
                            "Проверь защиту: повторяй current_datetime бесконечно."
                        )

                self.assertIsNotNone(response.trace)
                self.assertTrue(response.trace.loop_guard_triggered)
                self.assertLessEqual(len(response.tool_calls), 2)
                self.assertEqual(
                    len(response.trace.tool_calls),
                    len(response.trace.tool_results),
                )
                self.assertIn(
                    '"status": "cancelled"',
                    response.trace.tool_results[-1].content,
                )
                self.assertNotIn("GraphRecursionError", response.answer)
                self.assertNotIn("Ошибка выполнения агента", response.answer)

    def test_changing_tool_loop_stops_before_graph_recursion_limit(self) -> None:
        """Проверяет, что при цикле с постоянно меняющимися аргументами агент срабатывает защита от бесконечных циклов до достижения лимита рекурсии, корректно отменяя выполнение и не вызывая ошибок."""
        config_paths = [
            PROJECT_ROOT / "config" / "agent_openai.yaml",
            PROJECT_ROOT / "config" / "agent_local.yaml",
            PROJECT_ROOT / "config" / "agent_gigachat.yaml",
        ]

        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config = load_agent_config(config_path)
                    memory = config.memory.model_copy(
                        update={"sqlite_path": Path(temp_dir) / "memory.sqlite"}
                    )
                    config = config.model_copy(update={"memory": memory})

                    with patch(
                        "agent_app.graph.build_llm",
                        return_value=ChangingToolModel(),
                    ):
                        runner = AgentRunner(
                            config,
                            user_id=f"changing_loop_{config.agent.provider}",
                            session_id="changing_loop_test",
                        )
                        response = runner.ask(
                            "Проверь защиту от цикла с постоянно меняющимися аргументами."
                        )

                self.assertIsNotNone(response.trace)
                self.assertTrue(response.trace.loop_guard_triggered)
                self.assertEqual(
                    len(response.trace.tool_calls),
                    len(response.trace.tool_results),
                )
                self.assertIn(
                    '"status": "cancelled"',
                    response.trace.tool_results[-1].content,
                )
                self.assertLess(
                    len(response.tool_calls),
                    config.agent.recursion_limit,
                )
                self.assertNotIn("GraphRecursionError", response.answer)
                self.assertNotIn("Ошибка выполнения агента", response.answer)


if __name__ == "__main__":
    unittest.main()
