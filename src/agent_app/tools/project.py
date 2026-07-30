"""Инструменты управления проектами для инструментов агента."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.memory.store import SQLiteMemoryStore

TaskStatus = Literal["todo", "in_progress", "blocked", "done"]


class CreateProjectInput(BaseModel):
    """Содержит проверенные данные для создания проекта, включая обязательные поля и необязательный срок, обеспечивая корректность и полноту информации при инициализации проекта."""

    project_name: str = Field(description="Название проекта.")
    goal: str = Field(description="Цель проекта.")
    deadline: str | None = Field(
        default=None, description="Срок проекта в свободной форме."
    )


class CreateTaskInput(BaseModel):
    """Определяет валидированные параметры для создания задачи в проекте, включая статус, сроки и ответственного, что гарантирует согласованность данных задачи."""

    project_name: str = Field(description="Название проекта.")
    task_title: str = Field(description="Название задачи.")
    status: TaskStatus = Field(default="todo", description="Статус задачи.")
    due_date: str | None = Field(
        default=None, description="Срок задачи в свободной форме."
    )
    owner: str | None = Field(default=None, description="Ответственный.")


class UpdateTaskStatusInput(BaseModel):
    """Обеспечивает проверку данных для обновления статуса задачи, гарантируя корректное изменение состояния задачи в рамках проекта."""

    project_name: str = Field(description="Название проекта.")
    task_title: str = Field(description="Название задачи.")
    status: TaskStatus = Field(description="Новый статус задачи.")


class ListProjectTasksInput(BaseModel):
    """Определяет валидированные параметры для запроса списка задач конкретного проекта, обеспечивая точность и релевантность выборки."""

    project_name: str = Field(description="Название проекта.")


class SummarizeProjectInput(BaseModel):
    """Содержит проверенные данные для запроса сводки по проекту, гарантируя корректность идентификации проекта для анализа."""

    project_name: str = Field(description="Название проекта.")


def project_tools(
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    session_id: str,
) -> list[StructuredTool]:
    """Обеспечивает доступ к инструментам управления проектами и задачами с сохранением истории действий пользователя в памяти агента."""

    def create_project(
        project_name: str,
        goal: str,
        deadline: str | None = None,
    ) -> str:
        """Гарантирует создание и сохранение новой проектной записи с уникальным ключом и метаданными в памяти пользователя."""
        key = _project_key(project_name)
        value = _project_value(project_name=project_name, goal=goal, deadline=deadline)
        record = store.save(
            user_id=user_id,
            # Проекты живут дольше одного диалога и принадлежат пользователю.
            session_id=None,
            memory_type="note",
            key=key,
            value=value,
            tags=["project", _slug(project_name)],
            importance=5,
            source="tool",
            metadata={
                "project_name": project_name,
                "goal": goal,
                "deadline": deadline,
                "entity": "project",
            },
        )
        return _json({"status": "saved", "record": record.model_dump(mode="json")})

    def create_task(
        project_name: str,
        task_title: str,
        status: TaskStatus = "todo",
        due_date: str | None = None,
        owner: str | None = None,
    ) -> str:
        """Создаёт и сохраняет задачу проекта с гарантией уникальности по названию и статусу, обеспечивая её доступность для последующего управления и отчётности."""
        key = _task_key(project_name, task_title)
        value = _task_value(
            project_name=project_name,
            task_title=task_title,
            status=status,
            due_date=due_date,
            owner=owner,
        )
        record = store.save(
            user_id=user_id,
            session_id=None,
            memory_type="task",
            key=key,
            value=value,
            tags=["project_task", _slug(project_name), status],
            importance=4,
            source="tool",
            metadata={
                "project_name": project_name,
                "task_title": task_title,
                "status": status,
                "due_date": due_date,
                "owner": owner,
                "entity": "task",
            },
        )
        return _json({"status": "saved", "record": record.model_dump(mode="json")})

    def update_task_status(
        project_name: str,
        task_title: str,
        status: TaskStatus,
    ) -> str:
        """Обновляет статус существующей задачи проекта с проверкой наличия, гарантируя актуальность состояния задачи в системе."""
        key = _task_key(project_name, task_title)
        record = store.find_by_key(
            user_id=user_id,
            key=key,
            memory_type="task",
            session_id=None,
        )
        if record is None:
            return _json(
                {
                    "status": "not_found",
                    "project_name": project_name,
                    "task_title": task_title,
                }
            )
        metadata = dict(record.metadata)
        metadata["status"] = status
        value = _task_value(
            project_name=project_name,
            task_title=task_title,
            status=status,
            due_date=metadata.get("due_date"),
            owner=metadata.get("owner"),
        )
        updated = store.update(
            record.id,
            user_id=user_id,
            value=value,
            tags=["project_task", _slug(project_name), status],
            metadata=metadata,
        )
        return _json({"status": "updated", "record": updated.model_dump(mode="json")})

    def list_project_tasks(project_name: str) -> str:
        """Возвращает полный список задач проекта с фильтрацией по имени, обеспечивая консистентный обзор текущих задач."""
        tasks = [
            record
            for record in store.list_memories(
                user_id=user_id,
                # Не включаем session-scoped задачи из других инструментов.
                session_id=None,
                memory_type="task",
                limit=200,
            )
            if record.session_id is None
            and (
                record.metadata.get("project_name") == project_name
                or _slug(project_name) in record.tags
            )
        ]
        return _json(
            {
                "project_name": project_name,
                "count": len(tasks),
                "tasks": [record.model_dump(mode="json") for record in tasks],
            }
        )

    def summarize_project_state(project_name: str) -> str:
        """Подсчитывает и агрегирует состояние задач проекта, предоставляя сводную информацию для оценки прогресса и проблем."""
        tasks_payload = json.loads(list_project_tasks(project_name))
        tasks = tasks_payload.get("tasks", [])
        status_counts: dict[str, int] = {}
        for task in tasks:
            status = (task.get("metadata") or {}).get("status") or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
        project = store.find_by_key(
            user_id=user_id,
            key=_project_key(project_name),
            memory_type="note",
            session_id=None,
        )
        result = {
            "project_name": project_name,
            "project": project.model_dump(mode="json") if project else None,
            "tasks_count": len(tasks),
            "status_counts": status_counts,
            "tasks": tasks,
        }
        return _json(result)

    return [
        StructuredTool.from_function(
            name="create_project",
            description="Создаёт проект в долговременной памяти агента.",
            func=create_project,
            args_schema=CreateProjectInput,
        ),
        StructuredTool.from_function(
            name="create_task",
            description="Создаёт задачу проекта в долговременной памяти агента.",
            func=create_task,
            args_schema=CreateTaskInput,
        ),
        StructuredTool.from_function(
            name="update_task_status",
            description="Обновляет статус задачи проекта.",
            func=update_task_status,
            args_schema=UpdateTaskStatusInput,
        ),
        StructuredTool.from_function(
            name="list_project_tasks",
            description="Показывает задачи проекта из памяти.",
            func=list_project_tasks,
            args_schema=ListProjectTasksInput,
        ),
        StructuredTool.from_function(
            name="summarize_project_state",
            description="Собирает краткое состояние проекта из памяти и задач.",
            func=summarize_project_state,
            args_schema=SummarizeProjectInput,
        ),
    ]


def _project_key(project_name: str) -> str:
    """Гарантирует уникальность ключа проекта для идентификации записей в хранилище памяти."""
    return f"project:{project_name}"


def _task_key(project_name: str, task_title: str) -> str:
    """Гарантирует уникальность ключа задачи в рамках проекта для корректного хранения и поиска."""
    return f"task:{project_name}:{task_title}"


def _project_value(*, project_name: str, goal: str, deadline: str | None) -> str:
    """Формирует человекочитаемое описание проекта для хранения и отображения в истории пользователя."""
    parts = [f"Проект: {project_name}", f"Цель: {goal}"]
    if deadline:
        parts.append(f"Срок: {deadline}")
    return "; ".join(parts)


def _task_value(
    *,
    project_name: str,
    task_title: str,
    status: str,
    due_date: str | None,
    owner: str | None,
) -> str:
    """Формирует человекочитаемое описание задачи с учётом статуса, срока и ответственного для истории пользователя."""
    parts = [
        f"Проект: {project_name}",
        f"Задача: {task_title}",
        f"Статус: {status}",
    ]
    if due_date:
        parts.append(f"Срок: {due_date}")
    if owner:
        parts.append(f"Ответственный: {owner}")
    return "; ".join(parts)


def _slug(value: str) -> str:
    """Гарантирует получение идентификатора без пробелов и в нижнем регистре для тегирования и поиска."""
    return value.strip().lower().replace(" ", "_")


def _json(payload: object) -> str:
    """Гарантирует корректную сериализацию результата в JSON с поддержкой Unicode для передачи между подсистемами."""
    return json.dumps(payload, ensure_ascii=False)
