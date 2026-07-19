"""Граф состояний LangGraph для мультиагентной системы."""

from __future__ import annotations

import asyncio
import logging
import re
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent_app.config import AgentAppConfig
from agent_app.currency import CBRCurrencyConverter
from agent_app.memory import SQLiteMemoryStore, SummaryMemory
from agent_app.multi_agent.bus import AsyncMessageBus
from agent_app.multi_agent.decomposition import TaskDecomposer
from agent_app.multi_agent.evaluation import assess_multi_response
from agent_app.multi_agent.exporting import MultiAgentExporter
from agent_app.multi_agent.lifecycle import LifecycleTracker
from agent_app.multi_agent.llm_routing import MultiAgentLLMRegistry
from agent_app.multi_agent.models import (
    AgentEnvelope,
    AgentRunState,
    AgentTask,
    AgentTaskResult,
    MessageKind,
    MultiAgentGraphState,
    MultiAgentResponse,
    MultiAgentRunResult,
    TaskExecutionState,
)
from agent_app.multi_agent.persistence import MultiAgentCheckpointStore
from agent_app.multi_agent.roles import (
    SpecialistAgent,
    compact_results,
    default_role_definitions,
    result_from_envelope,
)
from agent_app.multi_agent.tracking import MultiAgentTracker
from agent_app.multi_agent.usage import LLMCallTracker
from agent_app.guardrails import GuardrailPipeline
from agent_app.rag.models import RagCitation
from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.support.incidents import IncidentStore
from agent_app.tools import build_tools

LOGGER = logging.getLogger(__name__)


class MultiAgentRunner:
    """Supervisor-граф с изолированными ролями и типизированным обменом."""

    def __init__(
        self,
        config: AgentAppConfig,
        *,
        user_id: str,
        session_id: str,
        llm: Any | None = None,
        rag_runtime: OnlineRagRuntime | None = None,
        incident_store: IncidentStore | None = None,
        external_tools: list[BaseTool] | None = None,
        llm_registry: MultiAgentLLMRegistry | None = None,
        role_llms: dict[str, Any] | None = None,
        checkpointer: Any | None = None,
        currency_converter: CBRCurrencyConverter | None = None,
    ):
        """Инициализирует мультиагентный раннер с необходимыми ресурсами и зависимостями, гарантируя готовность к выполнению задач в соответствии с конфигурацией."""
        if not config.multi_agent.enabled:
            raise ValueError("Multi-agent режим отключён в конфигурации")
        self.config = config
        self.user_id = user_id
        self.session_id = session_id
        self._owns_llm_registry = llm_registry is None
        self.llm_registry = llm_registry or MultiAgentLLMRegistry(
            config,
            default_llm=llm,
            role_llms=role_llms,
        )
        self.llm = self.llm_registry.default_llm
        self.currency_converter = currency_converter or CBRCurrencyConverter(
            config.currency_conversion
        )
        self._owns_rag = rag_runtime is None and config.rag.enabled
        self.rag_runtime = (
            rag_runtime
            if rag_runtime is not None
            else OnlineRagRuntime(config.rag)
            if config.rag.enabled
            else None
        )
        self.incident_store = incident_store or IncidentStore(
            config.tools.incident_sqlite_path
        )
        self.memory_store = SQLiteMemoryStore(config.memory.sqlite_path)
        self.summary = SummaryMemory(
            self.memory_store,
            user_id=user_id,
            session_id=session_id,
            max_chars=config.agent.max_summary_chars,
        )
        self.checkpointer = checkpointer or InMemorySaver()
        self.tools = build_tools(
            config,
            self.memory_store,
            user_id=user_id,
            session_id=session_id,
            rag_runtime=self.rag_runtime,
            incident_store=self.incident_store,
            external_tools=external_tools,
        )
        self.exporter = MultiAgentExporter(config.multi_agent.output_dir)
        self.tracker = MultiAgentTracker(config.multi_agent)

    def run(self, request: str) -> MultiAgentRunResult:
        """Запускает мультиагентный процесс обработки запроса, обеспечивая контроль жизненного цикла, трекинг использования и координацию агентов с гарантией корректного выполнения и учёта ресурсов."""
        normalized = request.strip()
        if not normalized:
            raise ValueError("Запрос multi-agent системе не может быть пустым")
        started = perf_counter()
        run_id = str(uuid4())
        lifecycle = LifecycleTracker(
            details={"run_id": run_id, "user_id": self.user_id}
        )
        bus = AsyncMessageBus()
        usage_tracker = LLMCallTracker(
            self.llm,
            model=self.config.agent.model,
            input_cost_per_million=(
                self.config.multi_agent.cost.input_cost_per_million
            ),
            output_cost_per_million=(
                self.config.multi_agent.cost.output_cost_per_million
            ),
            cost_currency=self.config.multi_agent.cost.currency,
            currency_converter=self.currency_converter,
            serialize_calls=self.config.agent.provider == "local",
            token_budget=self.config.multi_agent.token_budget,
            max_output_tokens=self.config.agent.max_new_tokens,
            route_resolver=self.llm_registry.route,
        )
        definitions = self._definitions()
        guardrails = GuardrailPipeline(self.config.guardrails)
        specialists = {
            definition.name: SpecialistAgent(
                definition,
                tools=self.tools,
                rag_runtime=self.rag_runtime,
                llm_invoke=usage_tracker.invoke,
                llm_invoke_response=usage_tracker.invoke_response,
                tool_output_guardrail=(
                    lambda value: guardrails.inspect_tool_output(value).text
                ),
                tool_max_iterations=self.config.multi_agent.tool_max_iterations,
                tool_output_max_chars=(self.config.multi_agent.tool_output_max_chars),
            )
            for definition in definitions
        }
        for name, specialist in specialists.items():
            bus.register_agent(name, specialist.handle)
        decomposer = TaskDecomposer(
            definitions,
            max_tasks=self.config.multi_agent.max_tasks,
            mode=self.config.multi_agent.planner_mode,
            llm_invoke=usage_tracker.invoke,
            available_tool_names={tool.name for tool in self.tools},
        )

        graph = self._build_graph(
            lifecycle=lifecycle,
            bus=bus,
            decomposer=decomposer,
            usage_tracker=usage_tracker,
        )
        degraded = False
        thread_config = MultiAgentCheckpointStore.runnable_config(
            self.user_id,
            self.session_id,
        )
        try:
            state = graph.invoke(
                {
                    "run_id": run_id,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "request": normalized,
                    "history": [HumanMessage(content=normalized)],
                    "tasks": [],
                    "task_results": [],
                    "review": "",
                    "answer": "",
                    "citations": [],
                    "round_number": 0,
                    "delegations": 0,
                    "degraded": False,
                },
                config=thread_config,
            )
        except Exception as exc:
            LOGGER.exception("Multi-agent граф завершился с ошибкой")
            lifecycle.fail(str(exc))
            state = {
                "run_id": run_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "request": normalized,
                "history": [HumanMessage(content=normalized)],
                "tasks": [],
                "task_results": [],
                "review": "Проверка не выполнена из-за ошибки.",
                "answer": f"Не удалось завершить мультиагентный запуск: {str(exc)[:300]}",
                "citations": [],
                "round_number": 0,
                "delegations": 0,
                "degraded": True,
                "error": str(exc)[:500],
            }
            degraded = True

        state["history"] = self._compact_history(
            graph,
            state.get("history", []),
            usage_tracker=usage_tracker,
            thread_config=thread_config,
        )

        usage = usage_tracker.snapshot().model_copy(
            update={
                "tool_calls": sum(
                    len(result.tool_calls) for result in state["task_results"]
                ),
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
        )
        response = MultiAgentResponse(
            run_id=run_id,
            answer=state["answer"],
            user_id=self.user_id,
            session_id=self.session_id,
            selected_agents=list(
                dict.fromkeys(
                    task.assigned_to
                    for task in state["tasks"]
                    if task.assigned_to is not None
                )
            ),
            tasks=state["tasks"],
            task_results=state["task_results"],
            citations=state["citations"],
            review=state["review"],
            history_messages_used=len(state["history"]),
            summary_used=bool(self.summary.get()),
            llm_routes=self.llm_registry.route_info(),
            lifecycle=lifecycle.snapshot(),
            usage=usage,
            execution_mode=self._execution_mode(),
            degraded=degraded or state.get("degraded", False),
        )
        response = response.model_copy(
            update={"quality": assess_multi_response(response)}
        )
        self._publish_completion(bus, response)
        result = MultiAgentRunResult(
            response=response,
            messages=bus.journal(),
            dead_letters=bus.dead_letters(),
        )
        run_dir = self.exporter.export_run(result)
        response = response.model_copy()
        result = result.model_copy(
            update={"response": response, "run_dir": str(run_dir)}
        )
        self.tracker.log_run(result)
        return result

    def run_role(self, role: str, instruction: str) -> MultiAgentRunResult:
        """Выполняет ровно одну назначенную роль без повторной supervisor-декомпозиции."""
        normalized = instruction.strip()
        if not normalized:
            raise ValueError("Инструкция роли не может быть пустой")
        # route() одновременно валидирует имя роли и не позволяет оркестратору
        # обратиться к неописанному LLM-профилю.
        self.llm_registry.route(role)
        started = perf_counter()
        run_id = str(uuid4())
        lifecycle = LifecycleTracker(
            details={"run_id": run_id, "user_id": self.user_id, "direct_role": role}
        )
        lifecycle.transition(AgentRunState.DECOMPOSED, details={"tasks": 1})
        lifecycle.transition(AgentRunState.DELEGATED, details={"agents": [role]})
        lifecycle.transition(AgentRunState.RUNNING)
        usage_tracker = LLMCallTracker(
            self.llm,
            model=self.config.agent.model,
            input_cost_per_million=(
                self.config.multi_agent.cost.input_cost_per_million
            ),
            output_cost_per_million=(
                self.config.multi_agent.cost.output_cost_per_million
            ),
            cost_currency=self.config.multi_agent.cost.currency,
            currency_converter=self.currency_converter,
            serialize_calls=self.config.agent.provider == "local",
            token_budget=self.config.multi_agent.token_budget,
            max_output_tokens=self.config.agent.max_new_tokens,
            route_resolver=self.llm_registry.route,
        )
        guardrails = GuardrailPipeline(self.config.guardrails)
        definitions = {item.name: item for item in self._definitions()}
        definition = definitions.get(role)
        capability = (
            definition.capabilities[0].name
            if definition is not None
            else "role_analysis"
        )
        task = AgentTask(
            capability=capability,
            title=f"Прямое задание роли {role}",
            instruction=guardrails.inspect_context(normalized).text,
            assigned_to=role,
        )
        if definition is not None:
            specialist = SpecialistAgent(
                definition,
                tools=self.tools,
                rag_runtime=self.rag_runtime,
                llm_invoke=usage_tracker.invoke,
                llm_invoke_response=usage_tracker.invoke_response,
                tool_output_guardrail=(
                    lambda value: guardrails.inspect_tool_output(value).text
                ),
                tool_max_iterations=self.config.multi_agent.tool_max_iterations,
                tool_output_max_chars=self.config.multi_agent.tool_output_max_chars,
            )
            task_result = specialist.execute(task)
        else:
            try:
                content = usage_tracker.invoke(
                    [
                        SystemMessage(
                            content=(
                                f"Ты исполняешь только роль {role}. Выполни назначенный "
                                "шаг и не делегируй его другим агентам."
                            )
                        ),
                        HumanMessage(content=task.instruction),
                    ],
                    role,
                )
                task_result = AgentTaskResult(
                    task_id=task.id,
                    agent_name=role,
                    capability=capability,
                    state=TaskExecutionState.COMPLETED,
                    content=guardrails.inspect_output(content).text,
                )
            except Exception as exc:
                LOGGER.exception("Прямой вызов роли %s завершился ошибкой", role)
                task_result = self._failed_task(task, str(exc))

        completed = task_result.state == TaskExecutionState.COMPLETED
        if completed:
            lifecycle.transition(AgentRunState.REVIEWING)
            lifecycle.transition(AgentRunState.COMPLETED)
        else:
            lifecycle.fail(task_result.error or "Роль не выполнила задание")
        task = task.model_copy(update={"state": task_result.state})
        usage = usage_tracker.snapshot().model_copy(
            update={
                "tool_calls": len(task_result.tool_calls),
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
        )
        response = MultiAgentResponse(
            run_id=run_id,
            answer=task_result.content,
            user_id=self.user_id,
            session_id=self.session_id,
            selected_agents=[role],
            tasks=[task],
            task_results=[task_result],
            citations=task_result.citations,
            review="Прямое исполнение назначенной роли без supervisor-графа.",
            llm_routes=self.llm_registry.route_info(),
            lifecycle=lifecycle.snapshot(),
            usage=usage,
            execution_mode="sequential",
            degraded=not completed,
        )
        response = response.model_copy(
            update={"quality": assess_multi_response(response)}
        )
        result = MultiAgentRunResult(response=response)
        run_dir = self.exporter.export_run(result)
        result = result.model_copy(update={"run_dir": str(run_dir)})
        self.tracker.log_run(result)
        return result

    def close(self) -> None:
        """Освобождает владение ресурсами LLM и RAG, гарантируя корректное завершение работы и предотвращение утечек."""
        if self._owns_llm_registry:
            self.llm_registry.close()
        if self._owns_rag and self.rag_runtime is not None:
            self.rag_runtime.close()

    def _build_graph(
        self,
        *,
        lifecycle: LifecycleTracker,
        bus: AsyncMessageBus,
        decomposer: TaskDecomposer,
        usage_tracker: LLMCallTracker,
    ):
        """Строит и управляет графом задач мультиагентной системы, обеспечивая последовательное разложение, делегирование и агрегацию результатов с учётом ограничений и состояния."""

        def decompose(state: MultiAgentGraphState) -> dict[str, object]:
            """Разбивает исходный запрос пользователя на отдельные задачи и переводит граф в состояние декомпозиции, гарантируя корректную инициализацию мультиагентного процесса."""
            tasks = decomposer.decompose(state["request"])
            lifecycle.transition(
                AgentRunState.DECOMPOSED,
                details={"tasks": len(tasks)},
            )
            return {"tasks": tasks}

        def dispatch(state: MultiAgentGraphState) -> dict[str, object]:
            """Организует распределение задач между агентами с учётом ограничений по делегациям и обновляет состояние системы, гарантируя учёт результатов выполнения и индикацию деградации."""
            remaining_delegations = (
                self.config.multi_agent.max_delegations - state["delegations"]
            )
            candidates = (
                state["tasks"]
                if state["round_number"] == 0
                else [
                    task
                    for task in state["tasks"]
                    if task.state
                    in {TaskExecutionState.FAILED, TaskExecutionState.TIMED_OUT}
                ]
            )
            tasks = candidates[:remaining_delegations]
            lifecycle.transition(
                AgentRunState.DELEGATED,
                details={
                    "round": state["round_number"] + 1,
                    "agents": [task.assigned_to for task in tasks],
                },
            )
            lifecycle.transition(AgentRunState.RUNNING)
            results = asyncio.run(self._dispatch(tasks, bus, state["run_id"]))
            result_by_task = {
                result.task_id: result for result in state["task_results"]
            }
            result_by_task.update({result.task_id: result for result in results})
            merged_results = [
                result_by_task[task.id]
                for task in state["tasks"]
                if task.id in result_by_task
            ]
            state_by_task = {result.task_id: result.state for result in merged_results}
            updated_tasks = [
                task.model_copy(
                    update={"state": state_by_task.get(task.id, task.state)}
                )
                for task in state["tasks"]
            ]
            degraded = any(
                result.state != TaskExecutionState.COMPLETED
                for result in merged_results
            )
            return {
                "tasks": updated_tasks,
                "task_results": merged_results,
                "round_number": state["round_number"] + 1,
                "delegations": state["delegations"] + len(tasks),
                "degraded": degraded,
            }

        def review(state: MultiAgentGraphState) -> dict[str, object]:
            """Проводит критическую оценку результатов агентов с учётом лимита токенов, обеспечивая выявление ошибок и противоречий для повышения качества итогового решения."""
            lifecycle.transition(AgentRunState.REVIEWING)
            if not state["task_results"]:
                return {"review": "Специалисты не требовались для простого запроса."}
            if (
                usage_tracker.snapshot().total_tokens
                >= self.config.multi_agent.token_budget
            ):
                return {
                    "review": "Token budget исчерпан; выполнена структурная проверка без LLM.",
                    "degraded": True,
                }
            prompt = (
                "Проверь отчёты специалистов. Укажи противоречия, неподтверждённые "
                "утверждения, ошибки tools и недостающие проверки. Не раскрывай скрытые "
                "рассуждения; верни краткое заключение для координатора.\n\n"
                + compact_results(state["task_results"])
            )
            try:
                text = usage_tracker.invoke(
                    [
                        SystemMessage(content="Ты критик мультиагентной системы."),
                        HumanMessage(content=prompt),
                    ],
                    "critic_agent",
                )
                return {"review": text.strip()}
            except Exception as exc:
                LOGGER.exception("Критик не завершил проверку")
                return {
                    "review": f"Критик недоступен: {str(exc)[:200]}",
                    "degraded": True,
                }

        def next_after_review(state: MultiAgentGraphState) -> str:
            """Определяет следующий шаг мультиагентного процесса, обеспечивая повторные попытки при сбоях в рамках заданных ограничений по раундам и делегациям."""
            has_failed_tasks = any(
                result.state
                in {TaskExecutionState.FAILED, TaskExecutionState.TIMED_OUT}
                for result in state["task_results"]
            )
            can_retry = (
                state["round_number"] < self.config.multi_agent.max_rounds
                and state["delegations"] < self.config.multi_agent.max_delegations
            )
            return "retry" if has_failed_tasks and can_retry else "done"

        def synthesize(state: MultiAgentGraphState) -> dict[str, object]:
            """Формирует окончательный ответ на основе собранных данных и отзывов агентов, обеспечивая согласованность, полноту и корректность с учётом ссылок и возможной деградации."""
            citations = self._deduplicate_citations(state["task_results"])
            citation_guidance = (
                "Сохрани только существующие ссылки [Источник N]."
                if citations
                else "RAG-источников нет: не добавляй ссылки [Источник N] и список источников."
            )
            evidence = compact_results(state["task_results"])
            if not evidence:
                evidence = "Специализированные задания не потребовались."
            conversation = self._conversation_context(state["history"])
            summary = self.summary.get()
            long_term_memory = self._long_term_memory_context()
            prompt = (
                f"Запрос пользователя:\n{state['request']}\n\n"
                f"Резюме предыдущего диалога:\n{summary or 'нет'}\n\n"
                f"Последние сообщения сессии:\n{conversation or 'нет'}\n\n"
                f"Долговременная память пользователя:\n"
                f"{long_term_memory or 'нет'}\n\n"
                f"Отчёты специалистов:\n{evidence}\n\n"
                f"Проверка критика:\n{state['review']}\n\n"
                "Сформируй окончательный ответ на русском языке. Используй только "
                f"доступные данные. {citation_guidance} Отдели факты от гипотез "
                "и не упоминай внутренний протокол обмена."
            )
            try:
                answer = usage_tracker.invoke(
                    [
                        SystemMessage(
                            content="Ты координатор системы поддержки инженера."
                        ),
                        HumanMessage(content=prompt),
                    ],
                    "coordinator",
                ).strip()
            except Exception:
                LOGGER.exception("Координатор не сформировал ответ")
                answer = self._fallback_answer(state["task_results"])
                state["degraded"] = True
            if citations:
                answer = self._append_sources(answer, citations)
            else:
                answer = self._remove_unavailable_citations(answer)
            lifecycle.transition(
                AgentRunState.COMPLETED,
                details={"citations": len(citations)},
            )
            return {
                "answer": answer,
                "citations": citations,
                "history": [AIMessage(content=answer)],
                "degraded": state["degraded"],
            }

        workflow = StateGraph(MultiAgentGraphState)
        workflow.add_node("decompose", decompose)
        workflow.add_node("dispatch", dispatch)
        workflow.add_node("review", review)
        workflow.add_node("synthesize", synthesize)
        workflow.add_edge(START, "decompose")
        workflow.add_edge("decompose", "dispatch")
        workflow.add_edge("dispatch", "review")
        workflow.add_conditional_edges(
            "review",
            next_after_review,
            {"retry": "dispatch", "done": "synthesize"},
        )
        workflow.add_edge("synthesize", END)
        return workflow.compile(checkpointer=self.checkpointer)

    async def _dispatch(
        self,
        tasks: list[AgentTask],
        bus: AsyncMessageBus,
        run_id: str,
    ) -> list[AgentTaskResult]:
        """Обеспечивает асинхронное выполнение задач мультиагентной системы с обработкой ошибок и таймаутов, гарантируя возврат результатов для каждого задания."""

        async def execute(task: AgentTask) -> AgentTaskResult:
            """Асинхронно выполняет задачу агентом, обеспечивая обработку ошибок и таймаутов для надёжного получения результата или корректного отказа."""
            if not task.assigned_to:
                return self._failed_task(task, "Для capability не найден агент")
            envelope = AgentEnvelope(
                correlation_id=run_id,
                sender="coordinator",
                recipient=task.assigned_to,
                kind=MessageKind.REQUEST,
                payload={"task": task.model_dump(mode="json")},
                ttl_seconds=min(
                    self.config.multi_agent.message_ttl_seconds,
                    self.config.multi_agent.task_timeout_seconds,
                ),
            )
            try:
                response = await bus.request(envelope)
                return result_from_envelope(response)
            except TimeoutError as exc:
                return self._failed_task(task, str(exc), timed_out=True)
            except Exception as exc:
                return self._failed_task(task, str(exc))

        if self._execution_mode() == "parallel":
            return list(await asyncio.gather(*(execute(task) for task in tasks)))
        results: list[AgentTaskResult] = []
        for task in tasks:
            results.append(await execute(task))
        return results

    def _definitions(self):
        """Формирует актуальный набор ролей агентов с учётом доступных инструментов и конфигурационных ограничений, обеспечивая согласованность разрешений."""
        definitions = default_role_definitions()
        overrides = self.config.multi_agent.role_tool_allowlists
        available = {tool.name for tool in self.tools}
        resolved = []
        for definition in definitions:
            allowlist = overrides.get(definition.name, definition.tool_allowlist)
            resolved.append(
                definition.model_copy(
                    update={
                        "tool_allowlist": [
                            name for name in allowlist if name in available
                        ]
                    }
                )
            )
        return resolved

    def _execution_mode(self):
        """Определяет режим выполнения мультиагентной системы, выбирая последовательный при локальных маршрутах и конфигурационный в остальных случаях, гарантируя согласованность поведения."""
        if self.llm_registry.has_local_routes:
            return "sequential"
        return self.config.multi_agent.execution_mode

    def _compact_history(
        self,
        graph: Any,
        history: list[BaseMessage],
        *,
        usage_tracker: LLMCallTracker,
        thread_config: RunnableConfig,
    ) -> list[BaseMessage]:
        """Гарантирует, что история сообщений не превышает заданный лимит, при необходимости сжимая её с помощью LLM и обновляя состояние графа, чтобы избежать переполнения памяти и деградации качества диалога."""
        limit = self.config.multi_agent.max_history_messages
        if len(history) <= limit:
            return history
        if self.config.multi_agent.summary_enabled:

            class TrackedSummaryLLM:
                """Передаёт вызов summary через общий счётчик usage роли coordinator."""

                def invoke(_self, messages: list[BaseMessage]) -> AIMessage:
                    """Вызывает coordinator LLM и оборачивает текст в `AIMessage`."""
                    return AIMessage(
                        content=usage_tracker.invoke(messages, "coordinator")
                    )

            try:
                kept = self.summary.summarize_if_needed(
                    llm=TrackedSummaryLLM(),
                    messages=history,
                    max_history_messages=limit,
                )
            except Exception:
                LOGGER.exception("Не удалось обновить multi-agent summary memory")
                kept = self.summary.recent_messages(history, limit)
        else:
            kept = self.summary.recent_messages(history, limit)
        try:
            graph.update_state(
                thread_config,
                {
                    "history": [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        *kept,
                    ]
                },
            )
        except Exception:
            LOGGER.exception("Не удалось сократить checkpoint history")
            return history
        return kept

    def _conversation_context(self, history: list[BaseMessage]) -> str:
        """Формирует человекочитаемый контекст последних сообщений для передачи в LLM, обеспечивая согласованность диалога и корректную роль участников."""
        recent = history[-self.config.multi_agent.max_history_messages :]
        labels = {"human": "Пользователь", "ai": "Ассистент", "tool": "Tool"}
        return "\n".join(
            f"{labels.get(message.type, message.type)}: {message.content}"
            for message in recent
        )

    def _long_term_memory_context(self) -> str:
        """Предоставляет вызывающему коду срез долговременной памяти пользователя для поддержки персонализации и контекстуальности ответов агентов."""
        records = self.memory_store.list_memories(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=self.config.memory.search_limit,
        )
        return "\n".join(f"- {record.key}: {record.value}" for record in records)

    @staticmethod
    def _failed_task(
        task: AgentTask,
        error: str,
        *,
        timed_out: bool = False,
    ) -> AgentTaskResult:
        """Гарантирует корректное оформление результата неуспешного задания с указанием причины сбоя и статуса для последующей обработки в системе."""
        return AgentTaskResult(
            task_id=task.id,
            agent_name=task.assigned_to or "unassigned",
            capability=task.capability,
            state=(
                TaskExecutionState.TIMED_OUT if timed_out else TaskExecutionState.FAILED
            ),
            content="Задание не завершено.",
            error=error[:500],
        )

    @staticmethod
    def _deduplicate_citations(
        results: list[AgentTaskResult],
    ) -> list[RagCitation]:
        """Обеспечивает уникальность ссылок на источники в итоговом ответе, исключая дублирование и сохраняя корректную атрибуцию информации."""
        citations: list[RagCitation] = []
        seen: set[str] = set()
        for result in results:
            for citation in result.citations:
                if citation.chunk_id in seen:
                    continue
                seen.add(citation.chunk_id)
                citations.append(citation)
        return citations

    @staticmethod
    def _append_sources(answer: str, citations: list[RagCitation]) -> str:
        """Гарантирует, что к ответу будут добавлены ссылки на использованные источники в стандартизированном виде, если они ещё не присутствуют."""
        if "Источники:" in answer:
            return answer
        lines = ["", "Источники:"]
        for citation in citations:
            label = citation.source or citation.chunk_id
            if citation.section:
                label += f", раздел: {citation.section}"
            lines.append(f"{citation.reference} {label}")
        return answer.rstrip() + "\n" + "\n".join(lines)

    @staticmethod
    def _remove_unavailable_citations(answer: str) -> str:
        """Удаляет невалидные или отсутствующие ссылки на источники из ответа, чтобы предотвратить ввод пользователя в заблуждение."""
        cleaned = re.sub(r"\s*\[Источник\s+\d+\]", "", answer)
        cleaned = re.sub(
            r"(?ms)\n+\s*Источники:\s*\n.*\Z",
            "",
            cleaned,
        )
        return cleaned.strip()

    @staticmethod
    def _fallback_answer(results: list[AgentTaskResult]) -> str:
        """Гарантирует возврат осмысленного ответа пользователю даже при частичном или полном сбое агентов, агрегируя доступные результаты."""
        completed = [result.content for result in results if result.content]
        if not completed:
            return "Не удалось получить результаты специалистов."
        return "\n\n".join(completed)

    @staticmethod
    def _publish_completion(bus: AsyncMessageBus, response: MultiAgentResponse) -> None:
        """Публикует событие завершения мультиагентного запуска в шину сообщений, обеспечивая доставку статуса и метаданных для внешних подписчиков."""
        envelope = AgentEnvelope(
            correlation_id=response.run_id,
            sender="coordinator",
            topic="multi_agent.completed",
            kind=MessageKind.EVENT,
            payload={
                "run_id": response.run_id,
                "degraded": response.degraded,
                "quality": response.quality.score if response.quality else None,
            },
        )
        asyncio.run(bus.publish(envelope))
