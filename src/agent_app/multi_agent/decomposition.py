"""Декомпозиция пользовательской задачи для мультиагентной системы."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from agent_app.multi_agent.models import AgentDefinition, AgentTask

PlannerMode = Literal["rules", "hybrid", "llm"]


class TaskDecomposer:
    """Декомпозирует запрос правилами или LLM и назначает владельца capability."""

    def __init__(
        self,
        definitions: list[AgentDefinition],
        *,
        max_tasks: int,
        mode: PlannerMode = "rules",
        llm_invoke: Callable[[list[object], str], str] | None = None,
        available_tool_names: set[str] | None = None,
    ):
        """Обеспечивает готовность декомпозиции задач с учётом ограничений по количеству, режиму планирования и доступным инструментам."""
        self.definitions = definitions
        self.max_tasks = max_tasks
        self.mode = mode
        self.llm_invoke = llm_invoke
        self.available_tool_names = available_tool_names or set()
        self._capability_owner = {
            capability.name: definition.name
            for definition in definitions
            for capability in definition.capabilities
        }

    def decompose(self, request: str) -> list[AgentTask]:
        """Выбирает стратегию планирования и возвращает bounded набор задач."""
        rules = self._rule_tasks(request)
        # Hybrid предпочитает детерминированные правила и расходует LLM только
        # тогда, когда они не смогли выделить ни одной применимой задачи.
        if self.mode == "rules" or (self.mode == "hybrid" and rules):
            return self._assign(rules)
        llm_tasks = self._llm_tasks(request)
        return self._assign(llm_tasks or rules)

    def _rule_tasks(self, request: str) -> list[AgentTask]:
        """Выделяет высокосигнальные capabilities без обращения к модели."""
        lower = request.casefold()
        tasks: list[AgentTask] = []
        requested_tools = self._requested_tools(request)
        if requested_tools:
            tasks.append(
                self._task(
                    "tool_execution",
                    "Выполнить разрешённые инструменты",
                    request,
                    required_tools=requested_tools,
                )
            )
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
            "sla",
            "срок",
            "уведом",
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
        """Просит planner сформировать JSON и отбрасывает неизвестные capabilities."""
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
        # Модель не является источником полномочий: capability и tools проходят
        # allowlist, а число задач обрезается системным max_tasks.
        for item in payload[: self.max_tasks]:
            capability = str(item.get("capability", ""))
            if capability not in self._capability_owner:
                continue
            tasks.append(
                self._task(
                    capability,
                    str(item.get("title") or capability),
                    str(item.get("instruction") or request),
                    required_tools=self._valid_tool_names(item.get("tools")),
                )
            )
        return tasks

    def _assign(self, tasks: list[AgentTask]) -> list[AgentTask]:
        """Назначает задачу зарегистрированному владельцу capability и позиции."""
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
    def _task(
        capability: str,
        title: str,
        instruction: str,
        *,
        required_tools: list[str] | None = None,
    ) -> AgentTask:
        """Гарантирует создание корректного задания агента с явно заданными требованиями к инструментам."""
        return AgentTask(
            capability=capability,
            title=title,
            instruction=instruction,
            required_tools=required_tools or [],
        )

    def _requested_tools(self, request: str) -> list[str]:
        """Сопоставляет запрос только с tools, доступными текущему runtime."""
        lower = request.casefold()
        requested: list[str] = []

        def add(name: str, markers: tuple[str, ...]) -> None:
            """Добавляет инструмент в список запрошенных только при совпадении маркеров и наличии в доступных, предотвращая дублирование и ошибки выбора."""
            if name in self.available_tool_names and any(
                marker in lower for marker in markers
            ):
                requested.append(name)

        add("get_weather", ("погод", "температур", "weather"))
        add(
            "execute_python",
            ("```python", "python-код", "python код", "код-интерпрет", "выполни код"),
        )
        add("calculator", ("калькулятор", "посчитай", "вычисли"))
        add("current_datetime", ("текущее время", "текущая дата", "который час"))
        if "файл" in lower or "workspace" in lower:
            add(
                "write_workspace_file",
                ("запиши", "создай файл", "измени файл", "сохрани в файл"),
            )
            add(
                "read_workspace_file",
                ("прочитай", "открой", "покажи содержимое"),
            )
            if not {
                "write_workspace_file",
                "read_workspace_file",
            }.intersection(requested):
                add(
                    "list_workspace_files",
                    ("список", "перечисли", "покажи файлы", "workspace"),
                )
        for name in sorted(self.available_tool_names):
            if name.casefold() in lower and name not in requested:
                requested.append(name)
        return list(dict.fromkeys(requested))

    def _valid_tool_names(self, value: object) -> list[str]:
        """Фильтрует предложенные моделью tools по runtime allowlist."""
        if not isinstance(value, list):
            return []
        return [
            str(item)
            for item in value
            if isinstance(item, str) and item in self.available_tool_names
        ]

    @staticmethod
    def _parse_json_array(content: str) -> list[dict[str, object]]:
        """Извлекает JSON-массив из ответа, возвращая безопасный пустой fallback."""
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
