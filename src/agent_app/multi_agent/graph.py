from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from agent_app.config import AgentAppConfig
from agent_app.memory import SQLiteMemoryStore
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
from agent_app.multi_agent.roles import (
    SpecialistAgent,
    compact_results,
    default_role_definitions,
    result_from_envelope,
)
from agent_app.multi_agent.tracking import MultiAgentTracker
from agent_app.multi_agent.usage import LLMCallTracker
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
    ):
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
            serialize_calls=self.config.agent.provider == "local",
            token_budget=self.config.multi_agent.token_budget,
            max_output_tokens=self.config.agent.max_new_tokens,
            route_resolver=self.llm_registry.route,
        )
        definitions = self._definitions()
        specialists = {
            definition.name: SpecialistAgent(
                definition,
                tools=self.tools,
                rag_runtime=self.rag_runtime,
                llm_invoke=usage_tracker.invoke,
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
        )

        graph = self._build_graph(
            lifecycle=lifecycle,
            bus=bus,
            decomposer=decomposer,
            usage_tracker=usage_tracker,
        )
        degraded = False
        try:
            state = graph.invoke(
                {
                    "run_id": run_id,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "request": normalized,
                    "tasks": [],
                    "task_results": [],
                    "review": "",
                    "answer": "",
                    "citations": [],
                    "round_number": 0,
                    "delegations": 0,
                    "degraded": False,
                }
            )
        except Exception as exc:
            LOGGER.exception("Multi-agent граф завершился с ошибкой")
            lifecycle.fail(str(exc))
            state = {
                "run_id": run_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "request": normalized,
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

    def close(self) -> None:
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
        def decompose(state: MultiAgentGraphState) -> dict[str, object]:
            tasks = decomposer.decompose(state["request"])
            lifecycle.transition(
                AgentRunState.DECOMPOSED,
                details={"tasks": len(tasks)},
            )
            return {"tasks": tasks}

        def dispatch(state: MultiAgentGraphState) -> dict[str, object]:
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
            citations = self._deduplicate_citations(state["task_results"])
            evidence = compact_results(state["task_results"])
            if not evidence:
                evidence = "Специализированные задания не потребовались."
            prompt = (
                f"Запрос пользователя:\n{state['request']}\n\n"
                f"Отчёты специалистов:\n{evidence}\n\n"
                f"Проверка критика:\n{state['review']}\n\n"
                "Сформируй окончательный ответ на русском языке. Используй только "
                "доступные данные, сохрани ссылки [Источник N], отдели факты от "
                "гипотез и не упоминай внутренний протокол обмена."
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
            lifecycle.transition(
                AgentRunState.COMPLETED,
                details={"citations": len(citations)},
            )
            return {
                "answer": answer,
                "citations": citations,
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
        return workflow.compile()

    async def _dispatch(
        self,
        tasks: list[AgentTask],
        bus: AsyncMessageBus,
        run_id: str,
    ) -> list[AgentTaskResult]:
        async def execute(task: AgentTask) -> AgentTaskResult:
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
        definitions = default_role_definitions()
        overrides = self.config.multi_agent.role_tool_allowlists
        return [
            definition.model_copy(update={"tool_allowlist": overrides[definition.name]})
            if definition.name in overrides
            else definition
            for definition in definitions
        ]

    def _execution_mode(self):
        if self.llm_registry.has_local_routes:
            return "sequential"
        return self.config.multi_agent.execution_mode

    @staticmethod
    def _failed_task(
        task: AgentTask,
        error: str,
        *,
        timed_out: bool = False,
    ) -> AgentTaskResult:
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
    def _fallback_answer(results: list[AgentTaskResult]) -> str:
        completed = [result.content for result in results if result.content]
        if not completed:
            return "Не удалось получить результаты специалистов."
        return "\n\n".join(completed)

    @staticmethod
    def _publish_completion(bus: AsyncMessageBus, response: MultiAgentResponse) -> None:
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
