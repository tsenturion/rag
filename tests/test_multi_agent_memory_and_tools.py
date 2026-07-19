"""Регрессионные тесты для подсистемы multi_agent_memory_and_tools."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agent_app.config import (
    AgentAppConfig,
    AgentConfig,
    AgentToolsConfig,
    MemoryConfig,
    MultiAgentConfig,
)
from agent_app.multi_agent.runtime import MultiAgentRuntime


class HistoryLLM:
    """Обеспечивает имитацию поведения LLM с учётом истории сообщений для проверки взаимодействия агентов и различения ролей в сценариях multi-agent."""

    supports_tool_calling = False

    def invoke(self, messages):
        """Проверяет, что ответы LLM соответствуют ожидаемым сценариям взаимодействия агентов и корректно различают роли."""
        system = str(messages[0].content).casefold()
        prompt = "\n".join(str(message.content) for message in messages)
        if "критик" in system:
            return AIMessage(content="Проверка выполнена.")
        if "координатор" in system:
            answer = "ALPHA-731" if "ALPHA-731" in prompt else "неизвестно"
            return AIMessage(content=f"Кодовое имя: {answer}.")
        if "сожми историю" in system:
            return AIMessage(content="Сервис имеет кодовое имя ALPHA-731.")
        return AIMessage(content="Отчёт специалиста.")


class ToolCallingLLM:
    """Обеспечивает имитацию LLM с поддержкой вызова инструментов для проверки корректности интеграции и обработки инструментальных вызовов в многоагентной системе."""

    supports_tool_calling = True

    def __init__(self):
        """Гарантирует, что экземпляр готов к связыванию инструментов и не содержит сторонних состояний."""
        self.bound_tools = []

    def bind_tools(self, tools):
        """Гарантирует, что экземпляр будет использовать только явно переданные инструменты при последующих вызовах."""
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        """Проверяет, что LLM корректно инициирует вызовы инструментов и возвращает ожидаемые ответы в зависимости от роли."""
        system = str(messages[0].content).casefold()
        if "агент безопасного" in system:
            if any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content="API вернул температуру 17 C.")
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "probe_weather",
                        "args": {"city": "Екатеринбург"},
                        "id": "weather-call",
                        "type": "tool_call",
                    }
                ],
            )
        if "критик" in system:
            return AIMessage(content="Фактический результат tool подтверждён.")
        if "координатор" in system:
            return AIMessage(content="Температура 17 C по данным API.")
        return AIMessage(content="Отчёт.")


def _config(root: Path, *, tools: list[str] | None = None) -> AgentAppConfig:
    """Проверяет, что конфигурация тестового окружения полностью воспроизводима и не зависит от внешних переменных."""
    return AgentAppConfig(
        agent=AgentConfig(provider="local", model="test-model"),
        memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
        tools=AgentToolsConfig(
            enabled=tools or [],
            incident_sqlite_path=root / "incidents.sqlite",
        ),
        multi_agent=MultiAgentConfig(
            enabled=True,
            output_dir=root / "runs",
            checkpoint_path=root / "checkpoints.sqlite",
            max_history_messages=6,
            mlflow_enabled=False,
        ),
    )


class MultiAgentMemoryAndToolsTest(unittest.TestCase):
    """Проверяет корректность работы подсистемы памяти и инструментов в многоагентной среде, включая сохранение, изоляцию и управление историей диалогов."""

    def test_history_survives_runner_and_runtime_restart(self) -> None:
        """Проверяет, что история диалогов сохраняется и доступна после перезапуска раннера и runtime, обеспечивая непрерывность сессии."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _config(root)
            runtime = MultiAgentRuntime(config, llm=HistoryLLM())
            runtime.ask(
                user_id="alice",
                session_id="incident",
                message="Кодовое имя сервиса ALPHA-731.",
            )
            second = runtime.ask(
                user_id="alice",
                session_id="incident",
                message="Какое кодовое имя сервиса я называла?",
            )
            runtime.close()

            restarted = MultiAgentRuntime(config, llm=HistoryLLM())
            third = restarted.ask(
                user_id="alice",
                session_id="incident",
                message="Повтори кодовое имя ещё раз.",
            )
            history = restarted.session_history(
                user_id="alice",
                session_id="incident",
            )
            restarted.close()

        self.assertIn("ALPHA-731", second.response.answer)
        self.assertIn("ALPHA-731", third.response.answer)
        self.assertEqual(len(history), 6)

    def test_history_is_isolated_by_user(self) -> None:
        """Проверяет, что история диалогов изолирована по пользователям и данные одного пользователя не доступны другому."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runtime = MultiAgentRuntime(_config(root), llm=HistoryLLM())
            runtime.ask(
                user_id="alice",
                session_id="shared-name",
                message="Кодовое имя сервиса ALPHA-731.",
            )
            bob = runtime.ask(
                user_id="bob",
                session_id="shared-name",
                message="Какое кодовое имя сервиса я называл?",
            )
            runtime.close()

        self.assertNotIn("ALPHA-731", bob.response.answer)

    def test_history_overflow_is_summarized_and_trimmed(self) -> None:
        """Проверяет, что при превышении лимита сообщений история автоматически суммируется и обрезается для оптимизации использования памяти."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _config(root)
            config = config.model_copy(
                update={
                    "multi_agent": config.multi_agent.model_copy(
                        update={"max_history_messages": 2}
                    )
                }
            )
            runtime = MultiAgentRuntime(config, llm=HistoryLLM())
            runtime.ask(
                user_id="alice",
                session_id="summary",
                message="Кодовое имя сервиса ALPHA-731.",
            )
            result = runtime.ask(
                user_id="alice",
                session_id="summary",
                message="Запомни контекст этого диалога.",
            )
            history = runtime.session_history(
                user_id="alice",
                session_id="summary",
            )
            runtime.close()

        self.assertTrue(result.response.summary_used)
        self.assertEqual(len(history), 2)

    def test_tool_agent_executes_allowlisted_external_tool(self) -> None:
        """Проверяет, что агент корректно вызывает разрешённый внешний инструмент и обрабатывает его результаты, обеспечивая безопасность и трассируемость вызовов."""
        calls: list[str] = []

        def probe_weather(city: str) -> str:
            """Проверяет, что инструмент возвращает фиксированный результат и регистрирует факт вызова для трассировки."""
            calls.append(city)
            return '{"status":"ok","city":"Екатеринбург","temperature":17}'

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _config(root, tools=["probe_weather"])
            config = config.model_copy(
                update={
                    "multi_agent": config.multi_agent.model_copy(
                        update={
                            "role_tool_allowlists": {"tool_agent": ["probe_weather"]}
                        }
                    )
                }
            )
            tool = StructuredTool.from_function(
                name="probe_weather",
                description="Получить погоду из тестового внешнего API.",
                func=probe_weather,
            )
            runtime = MultiAgentRuntime(
                config,
                llm=ToolCallingLLM(),
                external_tools=[tool],
            )
            result = runtime.ask(
                user_id="engineer",
                session_id="weather",
                message="Вызови probe_weather для Екатеринбурга.",
            )
            runtime.close()

        self.assertEqual(calls, ["Екатеринбург"])
        self.assertEqual(result.response.selected_agents, ["tool_agent"])
        self.assertEqual(result.response.task_results[0].tool_calls, ["probe_weather"])
        self.assertEqual(result.response.task_results[0].state, "completed")

    def test_direct_role_execution_does_not_restart_supervisor(self) -> None:
        """Оркестратор получает результат только назначенного specialist."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runtime = MultiAgentRuntime(_config(root), llm=HistoryLLM())
            result = runtime.ask_role(
                user_id="engineer",
                session_id="orchestration-step",
                message="Проведи диагностику HTTP 503",
                role="diagnostics_agent",
            )
            runtime.close()

        self.assertEqual(result.response.selected_agents, ["diagnostics_agent"])
        self.assertEqual(len(result.response.tasks), 1)
        self.assertEqual(
            result.response.task_results[0].agent_name, "diagnostics_agent"
        )


if __name__ == "__main__":
    unittest.main()
