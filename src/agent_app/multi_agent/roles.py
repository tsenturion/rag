from __future__ import annotations

import json
import re
from collections.abc import Callable
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agent_app.multi_agent.models import (
    AgentCapability,
    AgentDefinition,
    AgentEnvelope,
    AgentTask,
    AgentTaskResult,
    MessageKind,
    TaskExecutionState,
    UsageMetrics,
)
from agent_app.rag.runtime import OnlineRagRuntime


def default_role_definitions() -> list[AgentDefinition]:
    return [
        AgentDefinition(
            name="knowledge_agent",
            title="Агент знаний",
            goal="Найти подтверждённые сведения и сохранить атрибуцию источников.",
            capabilities=[
                AgentCapability(
                    name="knowledge_retrieval",
                    description="Поиск документации и runbook в Qdrant.",
                )
            ],
            tool_allowlist=["search_knowledge_base", "find_runbook"],
        ),
        AgentDefinition(
            name="diagnostics_agent",
            title="Агент диагностики",
            goal="Выделить наблюдаемые симптомы и построить проверяемый чек-лист.",
            capabilities=[
                AgentCapability(
                    name="technical_diagnostics",
                    description="Анализ логов и формирование диагностического плана.",
                )
            ],
            tool_allowlist=["analyze_log_fragment", "build_diagnostic_checklist"],
        ),
        AgentDefinition(
            name="incident_agent",
            title="Агент контекста и инцидентов",
            goal="Работать только с памятью и инцидентами текущего пользователя.",
            capabilities=[
                AgentCapability(
                    name="incident_context",
                    description="Чтение памяти и управление инженерным инцидентом.",
                )
            ],
            tool_allowlist=[
                "search_memory",
                "list_memories",
                "list_incidents",
                "create_incident",
                "get_incident",
                "update_incident_status",
            ],
            memory_access="read_write",
        ),
    ]


class SpecialistAgent:
    def __init__(
        self,
        definition: AgentDefinition,
        *,
        tools: list[BaseTool],
        rag_runtime: OnlineRagRuntime | None,
        llm_invoke: Callable[[list[object], str], str],
    ):
        self.definition = definition
        self.tools = {
            tool.name: tool for tool in tools if tool.name in definition.tool_allowlist
        }
        self.rag_runtime = rag_runtime
        self.llm_invoke = llm_invoke

    async def handle(self, envelope: AgentEnvelope) -> AgentEnvelope:
        task = AgentTask.model_validate(envelope.payload["task"])
        result = self.execute(task)
        return AgentEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.message_id,
            sender=self.definition.name,
            recipient=envelope.sender,
            kind=MessageKind.RESPONSE,
            payload={"result": result.model_dump(mode="json")},
            ttl_seconds=envelope.ttl_seconds,
        )

    def execute(self, task: AgentTask) -> AgentTaskResult:
        started = perf_counter()
        started_at = task.created_at
        before = UsageMetrics()
        try:
            if task.capability == "knowledge_retrieval":
                content, calls, citations = self._knowledge(task)
            elif task.capability == "technical_diagnostics":
                content, calls, citations = self._diagnostics(task)
            elif task.capability == "incident_context":
                content, calls, citations = self._incident_context(task)
            else:
                raise ValueError(f"Неподдерживаемая capability: {task.capability}")
            state = TaskExecutionState.COMPLETED
            error = None
        except Exception as exc:
            content = "Специалист не смог завершить задание."
            calls = []
            citations = []
            state = TaskExecutionState.FAILED
            error = str(exc)[:500]
        usage = before.model_copy(
            update={
                "tool_calls": len(calls),
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
        )
        return AgentTaskResult(
            task_id=task.id,
            agent_name=self.definition.name,
            capability=task.capability,
            state=state,
            content=content,
            tool_calls=calls,
            citations=citations,
            usage=usage,
            error=error,
            started_at=started_at,
        )

    def _knowledge(self, task: AgentTask):
        if self.rag_runtime is None:
            raise RuntimeError("Online RAG не подключён агенту знаний")
        result = self.rag_runtime.retrieve(task.instruction)
        if result.status == "unavailable":
            raise RuntimeError(result.error or "RAG недоступен")
        evidence = result.context or "Релевантные фрагменты не найдены."
        content = self._summarize(
            task,
            evidence,
            instruction=(
                "Составь краткий отчёт только по найденным источникам. Сохрани "
                "ссылки [Источник N] и явно укажи, если сведений недостаточно."
            ),
        )
        return content, ["search_knowledge_base"], result.citations

    def _diagnostics(self, task: AgentTask):
        calls: list[str] = []
        outputs: list[str] = []
        if "analyze_log_fragment" in self.tools:
            outputs.append(
                str(
                    self.tools["analyze_log_fragment"].invoke(
                        {"log_text": task.instruction, "component": None}
                    )
                )
            )
            calls.append("analyze_log_fragment")
        if "build_diagnostic_checklist" in self.tools:
            outputs.append(
                str(
                    self.tools["build_diagnostic_checklist"].invoke(
                        {
                            "component": self._component(task.instruction),
                            "symptoms": task.instruction,
                        }
                    )
                )
            )
            calls.append("build_diagnostic_checklist")
        content = self._summarize(
            task,
            "\n".join(outputs),
            instruction=(
                "Сформируй проверяемую диагностику: наблюдение, гипотеза, "
                "проверка и критерий результата. Не выдумывай факты."
            ),
        )
        return content, calls, []

    def _incident_context(self, task: AgentTask):
        calls: list[str] = []
        outputs: list[str] = []
        lower = task.instruction.casefold()
        if "search_memory" in self.tools:
            outputs.append(
                str(
                    self.tools["search_memory"].invoke(
                        {"query": task.instruction, "limit": 5}
                    )
                )
            )
            calls.append("search_memory")
        if "list_incidents" in self.tools:
            outputs.append(
                str(
                    self.tools["list_incidents"].invoke(
                        {"current_session_only": True, "limit": 20}
                    )
                )
            )
            calls.append("list_incidents")
        if "созд" in lower and "инцидент" in lower and "create_incident" in self.tools:
            outputs.append(
                str(
                    self.tools["create_incident"].invoke(
                        {
                            "title": self._title(task.instruction),
                            "description": task.instruction,
                            "priority": "medium",
                            "component": self._component(task.instruction),
                        }
                    )
                )
            )
            calls.append("create_incident")
        content = self._summarize(
            task,
            "\n".join(outputs),
            instruction=(
                "Верни только релевантный контекст текущего пользователя и сессии. "
                "Не раскрывай внутренние идентификаторы без необходимости."
            ),
        )
        return content, calls, []

    def _summarize(self, task: AgentTask, evidence: str, *, instruction: str) -> str:
        if not self.definition.use_llm:
            return evidence
        return self.llm_invoke(
            [
                SystemMessage(
                    content=(
                        f"Ты {self.definition.title}. Цель: {self.definition.goal} "
                        f"{instruction}"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Задание: {task.instruction}\n\nДоступные данные:\n{evidence}"
                    )
                ),
            ],
            self.definition.name,
        ).strip()

    @staticmethod
    def _component(text: str) -> str:
        match = re.search(r"(?i)(?:компонент|сервис|service)\s*[:=]?\s*([\w.-]+)", text)
        return match.group(1) if match else "не указан"

    @staticmethod
    def _title(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return (normalized[:117] + "...") if len(normalized) > 120 else normalized


def result_from_envelope(envelope: AgentEnvelope) -> AgentTaskResult:
    payload = envelope.payload.get("result")
    if not isinstance(payload, dict):
        raise ValueError("Ответ агента не содержит result")
    return AgentTaskResult.model_validate(payload)


def compact_results(results: list[AgentTaskResult]) -> str:
    return "\n\n".join(
        json.dumps(
            {
                "agent": result.agent_name,
                "capability": result.capability,
                "state": result.state,
                "content": result.content,
                "error": result.error,
            },
            ensure_ascii=False,
        )
        for result in results
    )
