from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent_app.config import (
    AgentAppConfig,
    AgentConfig,
    AgentToolsConfig,
    MemoryConfig,
    MultiAgentConfig,
    MultiAgentLLMProfileConfig,
)
from agent_app.multi_agent.models import (
    ComparisonScenario,
    ComparisonScenarioSuite,
)
from agent_app.multi_agent.graph import MultiAgentRunner
from agent_app.multi_agent.runtime import MultiAgentRuntime


class StubLLM:
    supports_tool_calling = False

    def invoke(self, messages):
        system = str(getattr(messages[0], "content", "")).casefold()
        if "критик" in system:
            return AIMessage(content="Противоречий и неподтверждённых фактов нет.")
        if "координатор" in system:
            return AIMessage(
                content=(
                    "HTTP 503 означает временную недоступность, а HTTP 500 - "
                    "внутреннюю ошибку. Для timeout проверьте зависимости."
                )
            )
        return AIMessage(content="Проверить timeout budget и состояние зависимостей.")


class RecordingLLM(StubLLM):
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return super().invoke(messages)


class FakeCitationLLM(StubLLM):
    def invoke(self, messages):
        system = str(getattr(messages[0], "content", "")).casefold()
        if "координатор" in system:
            return AIMessage(content="Результат вычисления равен 30 [Источник 1].")
        return super().invoke(messages)


def _config(root: Path) -> AgentAppConfig:
    return AgentAppConfig(
        agent=AgentConfig(provider="local", model="test-model"),
        memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
        tools=AgentToolsConfig(
            enabled=[
                "analyze_log_fragment",
                "build_diagnostic_checklist",
                "search_memory",
                "list_incidents",
            ],
            incident_sqlite_path=root / "incidents.sqlite",
        ),
        multi_agent=MultiAgentConfig(
            enabled=True,
            execution_mode="parallel",
            output_dir=root / "runs",
            checkpoint_path=root / "checkpoints.sqlite",
            mlflow_enabled=False,
        ),
    )


class MultiAgentRuntimeTest(unittest.TestCase):
    def test_answer_drops_citation_marker_without_rag_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runner = MultiAgentRunner(
                _config(root),
                user_id="engineer",
                session_id="no-citations",
                llm=FakeCitationLLM(),
            )
            try:
                result = runner.run("Посчитай значение без обращения к базе знаний.")
            finally:
                runner.close()

        self.assertEqual(result.response.citations, [])
        self.assertNotIn("[Источник", result.response.answer)

    def test_graph_uses_different_llms_for_incident_critic_and_coordinator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _config(root)
            config = config.model_copy(
                update={
                    "agent": AgentConfig(provider="openai", model="default-model"),
                    "multi_agent": config.multi_agent.model_copy(
                        update={
                            "execution_mode": "parallel",
                            "llm_profiles": {
                                "coord": MultiAgentLLMProfileConfig(
                                    provider="openai",
                                    model="coord-model",
                                ),
                                "review": MultiAgentLLMProfileConfig(
                                    provider="gigachat",
                                    model="review-model",
                                ),
                                "incidents": MultiAgentLLMProfileConfig(
                                    provider="local",
                                    model="incident-model",
                                ),
                            },
                            "role_llm_profiles": {
                                "coordinator": "coord",
                                "critic_agent": "review",
                                "incident_agent": "incidents",
                            },
                        }
                    ),
                }
            )
            default = RecordingLLM()
            coordinator = RecordingLLM()
            critic = RecordingLLM()
            incident = RecordingLLM()
            runner = MultiAgentRunner(
                config,
                user_id="engineer",
                session_id="mixed-routing",
                llm=default,
                role_llms={
                    "coordinator": coordinator,
                    "critic_agent": critic,
                    "incident_agent": incident,
                },
            )
            try:
                result = runner.run("Проверь память и текущий инцидент.")
            finally:
                runner.close()

        routes = {route.role: route for route in result.response.llm_routes}
        self.assertGreaterEqual(incident.calls, 1)
        self.assertGreaterEqual(critic.calls, 1)
        self.assertGreaterEqual(coordinator.calls, 1)
        self.assertEqual(routes["incident_agent"].provider, "local")
        self.assertEqual(routes["critic_agent"].provider, "gigachat")
        self.assertEqual(routes["coordinator"].provider, "openai")
        self.assertEqual(result.response.execution_mode, "sequential")

    def test_failed_specialist_is_retried_within_limits(self) -> None:
        calls = 0

        def flaky_log_analyzer(
            log_text: str,
            component: str | None = None,
        ) -> dict[str, object]:
            """Первый вызов имитирует временный сбой диагностического tool."""
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("Временная ошибка tool")
            return {"component": component, "log_text": log_text, "findings": []}

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = _config(root)
            config = config.model_copy(
                update={
                    "multi_agent": config.multi_agent.model_copy(
                        update={
                            "max_tasks": 1,
                            "max_delegations": 2,
                            "max_rounds": 2,
                        }
                    )
                }
            )
            runner = MultiAgentRunner(
                config,
                user_id="engineer",
                session_id="retry",
                llm=StubLLM(),
            )
            replacement = StructuredTool.from_function(
                func=flaky_log_analyzer,
                name="analyze_log_fragment",
                description="Тестовый анализатор логов.",
            )
            runner.tools = [
                replacement if tool.name == replacement.name else tool
                for tool in runner.tools
            ]

            result = runner.run("В логах timeout. Выполни диагностику.")
            runner.close()

        delegated = [
            event for event in result.response.lifecycle if event.state == "delegated"
        ]
        self.assertEqual(calls, 2)
        self.assertEqual(len(delegated), 2)
        self.assertEqual(result.response.tasks[0].state, "completed")
        self.assertFalse(result.response.degraded)

    def test_runtime_routes_task_and_exports_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runtime = MultiAgentRuntime(_config(root), llm=StubLLM())

            result = runtime.ask(
                user_id="engineer",
                session_id="incident-1",
                message="В логах timeout. Выполни диагностику.",
            )

            run_dir = Path(result.run_dir)
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            messages_exists = (run_dir / "messages.jsonl").exists()
            trace_exists = (run_dir / "trace.jsonl").exists()
            runtime.close()

        self.assertEqual(result.response.selected_agents, ["diagnostics_agent"])
        self.assertEqual(result.response.execution_mode, "sequential")
        self.assertEqual(result.response.lifecycle[-1].state, "completed")
        self.assertEqual(result.response.tasks[0].state, "completed")
        self.assertGreater(result.response.usage.llm_calls, 0)
        self.assertEqual(manifest["tasks_count"], 1)
        self.assertTrue(messages_exists)
        self.assertTrue(trace_exists)

    def test_comparison_uses_same_scenario_for_both_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runtime = MultiAgentRuntime(_config(root), llm=StubLLM())
            suite = ComparisonScenarioSuite(
                scenarios=[
                    ComparisonScenario(
                        id="http-status",
                        title="HTTP status",
                        request="Объясни различие HTTP 503 и HTTP 500.",
                        expected_terms=["503", "500"],
                        max_agents=1,
                    )
                ]
            )

            report = runtime.compare(suite, user_id="engineer")
            comparison_exists = Path(report.run_dir, "comparison.json").exists()
            runtime.close()

        self.assertEqual(len(report.cases), 1)
        self.assertEqual(report.cases[0].request, suite.scenarios[0].request)
        self.assertEqual(report.cases[0].multi.selected_agents, [])
        self.assertIsNotNone(report.run_dir)
        self.assertTrue(comparison_exists)


if __name__ == "__main__":
    unittest.main()
