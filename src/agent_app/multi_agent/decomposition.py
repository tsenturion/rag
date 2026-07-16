from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from agent_app.multi_agent.models import AgentDefinition, AgentTask

PlannerMode = Literal["rules", "hybrid", "llm"]


class TaskDecomposer:
    def __init__(
        self,
        definitions: list[AgentDefinition],
        *,
        max_tasks: int,
        mode: PlannerMode = "rules",
        llm_invoke: Callable[[list[object], str], str] | None = None,
    ):
        self.definitions = definitions
        self.max_tasks = max_tasks
        self.mode = mode
        self.llm_invoke = llm_invoke
        self._capability_owner = {
            capability.name: definition.name
            for definition in definitions
            for capability in definition.capabilities
        }

    def decompose(self, request: str) -> list[AgentTask]:
        rules = self._rule_tasks(request)
        if self.mode == "rules" or (self.mode == "hybrid" and rules):
            return self._assign(rules)
        llm_tasks = self._llm_tasks(request)
        return self._assign(llm_tasks or rules)

    def _rule_tasks(self, request: str) -> list[AgentTask]:
        lower = request.casefold()
        tasks: list[AgentTask] = []
        knowledge_markers = (
            "инструк",
            "регламент",
            "документ",
            "runbook",
            "что делать",
            "как ",
            "база знаний",
            "заявк",
            "процедур",
        )
        diagnostic_markers = (
            "ошиб",
            "сбой",
            "лог",
            "traceback",
            "timeout",
            "таймаут",
            "oom",
            "диагност",
            "не работает",
        )
        incident_markers = (
            "инцидент",
            "памят",
            "сесс",
            "предыдущ",
            "контекст",
        )
        if any(marker in lower for marker in knowledge_markers):
            tasks.append(
                self._task(
                    "knowledge_retrieval",
                    "Найти подтверждённые сведения",
                    request,
                )
            )
        if any(marker in lower for marker in diagnostic_markers):
            tasks.append(
                self._task(
                    "technical_diagnostics",
                    "Проанализировать симптомы и подготовить диагностику",
                    request,
                )
            )
        if any(marker in lower for marker in incident_markers):
            tasks.append(
                self._task(
                    "incident_context",
                    "Проверить память и состояние инцидентов",
                    request,
                )
            )
        return tasks[: self.max_tasks]

    def _llm_tasks(self, request: str) -> list[AgentTask]:
        if self.llm_invoke is None:
            return []
        capabilities = sorted(self._capability_owner)
        content = self.llm_invoke(
            [
                SystemMessage(
                    content=(
                        "Разбей запрос на независимые задания. Верни только JSON-массив "
                        "объектов capability, title, instruction. Допустимые capability: "
                        + ", ".join(capabilities)
                    )
                ),
                HumanMessage(content=request),
            ],
            "planner",
        )
        payload = self._parse_json_array(content)
        tasks: list[AgentTask] = []
        for item in payload[: self.max_tasks]:
            capability = str(item.get("capability", ""))
            if capability not in self._capability_owner:
                continue
            tasks.append(
                self._task(
                    capability,
                    str(item.get("title") or capability),
                    str(item.get("instruction") or request),
                )
            )
        return tasks

    def _assign(self, tasks: list[AgentTask]) -> list[AgentTask]:
        return [
            task.model_copy(
                update={
                    "assigned_to": self._capability_owner.get(task.capability),
                    "position": position,
                }
            )
            for position, task in enumerate(tasks[: self.max_tasks])
            if task.capability in self._capability_owner
        ]

    @staticmethod
    def _task(capability: str, title: str, instruction: str) -> AgentTask:
        return AgentTask(
            capability=capability,
            title=title,
            instruction=instruction,
        )

    @staticmethod
    def _parse_json_array(content: str) -> list[dict[str, object]]:
        match = re.search(r"\[[\s\S]*\]", content)
        if not match:
            return []
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return (
            [item for item in value if isinstance(item, dict)]
            if isinstance(value, list)
            else []
        )
