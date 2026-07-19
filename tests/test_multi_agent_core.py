"""Регрессионные тесты для подсистемы multi_agent_core."""

from __future__ import annotations

import asyncio
import unittest

from agent_app.config import (
    AgentAppConfig,
    AgentConfig,
    MultiAgentConfig,
    MultiAgentLLMProfileConfig,
)
from agent_app.multi_agent.bus import AsyncMessageBus
from agent_app.multi_agent.decomposition import TaskDecomposer
from agent_app.multi_agent.lifecycle import LifecycleTracker
from agent_app.multi_agent.llm_routing import MultiAgentLLMRegistry
from agent_app.multi_agent.models import (
    AgentEnvelope,
    AgentRunState,
    MessageDeliveryState,
    MessageKind,
)
from agent_app.multi_agent.protocols.acp import ACPMessage, ACPProtocolAdapter
from agent_app.multi_agent.protocols.mcp import build_mcp_server
from agent_app.multi_agent.protocols.simulation import run_protocol_simulation
from agent_app.multi_agent.roles import default_role_definitions
from agent_app.multi_agent.usage import LLMCallTracker


class CountingLLM:
    """Обеспечивает тестовую реализацию LLM, которая отслеживает количество вызовов и возвращает предсказуемый ответ, позволяя проверять взаимодействия без внешних зависимостей."""

    def __init__(self):
        """Гарантирует, что экземпляр готов отслеживать количество вызовов и не требует внешних ресурсов."""
        self.calls = 0

    def invoke(self, messages):
        """Гарантирует инкремент счётчика вызовов и возвращает предсказуемый ответ для тестирования взаимодействия."""
        self.calls += 1
        return "Ответ"


class MultiAgentCoreTest(unittest.TestCase):
    """Проверяет корректность жизненного цикла агентов, управление бюджетом токенов и маршрутизацию вызовов LLM по профилям ролей в подсистеме multi-agent."""

    def test_lifecycle_rejects_invalid_transition(self) -> None:
        """Проверяет, что система корректно отклоняет недопустимые переходы состояний жизненного цикла агента, обеспечивая целостность и предсказуемость состояний."""
        tracker = LifecycleTracker()
        tracker.transition(AgentRunState.DECOMPOSED)

        with self.assertRaisesRegex(ValueError, "Недопустимый lifecycle-переход"):
            tracker.transition(AgentRunState.COMPLETED)

    def test_lifecycle_allows_bounded_retry_transition(self) -> None:
        """Проверяет, что жизненный цикл агента допускает ограниченное количество повторных переходов между состояниями без нарушения логики работы."""
        tracker = LifecycleTracker()
        tracker.transition(AgentRunState.DECOMPOSED)
        tracker.transition(AgentRunState.DELEGATED)
        tracker.transition(AgentRunState.RUNNING)
        tracker.transition(AgentRunState.REVIEWING)
        tracker.transition(AgentRunState.DELEGATED)

        self.assertEqual(tracker.state, AgentRunState.DELEGATED)

    def test_token_budget_blocks_call_before_provider(self) -> None:
        """Проверяет, что вызов LLM блокируется при исчерпании токен-бюджета до передачи запроса провайдеру, предотвращая лишние расходы."""
        llm = CountingLLM()
        tracker = LLMCallTracker(
            llm,
            model="gpt-4o-mini",
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            token_budget=1,
            max_output_tokens=1,
        )

        with self.assertRaisesRegex(RuntimeError, "Token budget исчерпан"):
            tracker.invoke(["Слишком длинный запрос"], "coordinator")

        self.assertEqual(llm.calls, 0)

    def test_llm_calls_are_routed_to_role_profiles(self) -> None:
        """Проверяет корректную маршрутизацию вызовов LLM к профилям ролей, обеспечивая использование соответствующих моделей и провайдеров для каждого агента."""
        default = CountingLLM()
        coordinator = CountingLLM()
        critic = CountingLLM()
        incident = CountingLLM()
        config = AgentAppConfig(
            agent=AgentConfig(provider="openai", model="default-model"),
            multi_agent=MultiAgentConfig(
                enabled=True,
                llm_profiles={
                    "openai_coordination": MultiAgentLLMProfileConfig(
                        provider="openai",
                        model="coord-model",
                    ),
                    "gigachat_review": MultiAgentLLMProfileConfig(
                        provider="gigachat",
                        model="review-model",
                    ),
                    "local_incidents": MultiAgentLLMProfileConfig(
                        provider="local",
                        model="incident-model",
                    ),
                },
                role_llm_profiles={
                    "coordinator": "openai_coordination",
                    "critic_agent": "gigachat_review",
                    "incident_agent": "local_incidents",
                },
                mlflow_enabled=False,
            ),
        )
        registry = MultiAgentLLMRegistry(
            config,
            default_llm=default,
            role_llms={
                "coordinator": coordinator,
                "critic_agent": critic,
                "incident_agent": incident,
            },
        )
        tracker = LLMCallTracker(
            default,
            model=config.agent.model,
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            route_resolver=registry.route,
        )
        try:
            tracker.invoke(["Координация"], "coordinator")
            tracker.invoke(["Проверка"], "critic_agent")
            tracker.invoke(["Инцидент"], "incident_agent")
            tracker.invoke(["Диагностика"], "diagnostics_agent")
            routes = {route.role: route for route in registry.route_info()}
        finally:
            registry.close()

        self.assertEqual(coordinator.calls, 1)
        self.assertEqual(critic.calls, 1)
        self.assertEqual(incident.calls, 1)
        self.assertEqual(default.calls, 1)
        self.assertEqual(routes["coordinator"].provider, "openai")
        self.assertEqual(routes["critic_agent"].provider, "gigachat")
        self.assertEqual(routes["incident_agent"].provider, "local")

    def test_decomposer_assigns_multiple_specialists(self) -> None:
        """Проверяет, что декомпозер задач распределяет подзадачи между несколькими специализированными агентами для параллельной обработки."""
        decomposer = TaskDecomposer(
            default_role_definitions(),
            max_tasks=3,
        )

        tasks = decomposer.decompose(
            "Найди runbook: в логах timeout, затем проверь текущий инцидент."
        )

        self.assertEqual(
            {task.assigned_to for task in tasks},
            {"knowledge_agent", "diagnostics_agent", "incident_agent"},
        )

    def test_sla_policy_question_always_includes_knowledge_agent(self) -> None:
        """Проверяет, что при декомпозиции вопросов SLA всегда назначается агент знаний для обеспечения полноты и точности ответов."""
        decomposer = TaskDecomposer(default_role_definitions(), max_tasks=3)

        tasks = decomposer.decompose(
            "Каков срок первой реакции на критический инцидент и кого уведомить?"
        )

        self.assertIn("knowledge_agent", {task.assigned_to for task in tasks})

    def test_message_bus_supports_deduplication_and_pubsub(self) -> None:
        """Проверяет, что асинхронная шина сообщений обеспечивает дедупликацию, корректную маршрутизацию запросов, ответов и событий между агентами."""

        async def exercise():
            """Проверяет, что асинхронная шина сообщений корректно маршрутизирует запросы, ответы и события между агентами."""
            bus = AsyncMessageBus()
            events: list[str] = []

            async def handler(envelope: AgentEnvelope) -> AgentEnvelope:
                """Гарантирует формирование корректного ответа-агента с сохранением причинно-следственной связи сообщений."""
                return AgentEnvelope(
                    correlation_id=envelope.correlation_id,
                    causation_id=envelope.message_id,
                    sender="worker",
                    recipient=envelope.sender,
                    kind=MessageKind.RESPONSE,
                    payload={"result": "ok"},
                )

            async def subscriber(envelope: AgentEnvelope) -> None:
                """Гарантирует регистрацию события по публикации сообщения в шине для последующей проверки."""
                events.append(str(envelope.payload["status"]))

            bus.register_agent("worker", handler)
            bus.subscribe("done", subscriber)
            request = AgentEnvelope(
                message_id="fixed-id",
                correlation_id="run",
                sender="coordinator",
                recipient="worker",
                kind=MessageKind.REQUEST,
            )
            first = await bus.request(request)
            second = await bus.request(request)
            count = await bus.publish(
                AgentEnvelope(
                    correlation_id="run",
                    sender="coordinator",
                    topic="done",
                    kind=MessageKind.EVENT,
                    payload={"status": "completed"},
                )
            )
            return bus, events, count, first, second

        bus, events, count, first, second = asyncio.run(exercise())

        self.assertEqual(first.payload, second.payload)
        self.assertEqual(events, ["completed"])
        self.assertEqual(count, 1)
        self.assertTrue(
            any(
                item.delivery_state == MessageDeliveryState.DUPLICATE
                for item in bus.journal()
            )
        )

    def test_message_timeout_goes_to_dead_letter(self) -> None:
        """Проверяет, что при превышении таймаута обработки сообщения запрос корректно перемещается в очередь мёртвых писем для последующего анализа."""

        async def exercise():
            """Проверяет, что запрос к агенту с задержкой приводит к ожидаемому исключению по таймауту."""
            bus = AsyncMessageBus()

            async def slow_handler(envelope: AgentEnvelope) -> AgentEnvelope:
                """Гарантирует задержку обработки сообщения для проверки реакции системы на превышение таймаута."""
                await asyncio.sleep(0.05)
                return envelope

            bus.register_agent("slow", slow_handler)
            request = AgentEnvelope(
                correlation_id="run",
                sender="coordinator",
                recipient="slow",
                kind=MessageKind.REQUEST,
                ttl_seconds=0.01,
            )
            with self.assertRaises(TimeoutError):
                await bus.request(request)
            return bus

        bus = asyncio.run(exercise())
        self.assertEqual(len(bus.dead_letters()), 1)

    def test_mcp_and_acp_adapters_are_available(self) -> None:
        """Проверяет доступность адаптеров MCP и ACP для преобразования и взаимодействия между протоколами обмена сообщениями."""
        server = build_mcp_server()
        tools = asyncio.run(server.list_tools())
        names = {tool.name for tool in tools}
        acp = ACPMessage(
            agent_name="legacy",
            role="user",
            parts=[{"content": "Проверь timeout"}],
        )

        migrated = ACPProtocolAdapter.to_a2a_message(acp)

        self.assertEqual(
            names,
            {"analyze_log_fragment", "build_diagnostic_checklist"},
        )
        self.assertEqual(migrated["metadata"]["migratedFrom"], "ACP")

    def test_protocol_simulation_has_no_dead_letters(self) -> None:
        """Проверяет, что при симуляции протокола отсутствуют сообщения в очереди мёртвых писем, гарантируя корректность обработки."""
        report = run_protocol_simulation()

        self.assertEqual(report["dead_letters"], 0)
        self.assertEqual(report["published_subscribers"], 1)
        self.assertEqual(report["a2a_migration"]["metadata"]["migratedFrom"], "ACP")


if __name__ == "__main__":
    unittest.main()
