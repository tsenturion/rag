"""Регрессионные тесты устойчивого owner-scoped хранилища A2A-задач."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

from a2a.server.context import ServerCallContext
from a2a.types import ListTasksRequest, Task, TaskState, TaskStatus

from agent_app.multi_agent.protocols.a2a import (
    MultiAgentA2AHandler,
    PrincipalA2AUser,
)
from agent_app.multi_agent.protocols.a2a_store import A2ATaskStore
from agent_app.service.auth import Principal


def _task(task_id: str, *, context_id: str = "incident") -> Task:
    """Создаёт минимальную A2A-задачу с валидным состоянием для хранилища."""
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
    )


def _context(user_id: str) -> ServerCallContext:
    """Создаёт аутентифицированный A2A-контекст обычного инженера."""
    principal = Principal(subject=user_id, roles=["engineer"], auth_method="jwt")
    return ServerCallContext(
        user=PrincipalA2AUser(principal),
        state={"roles": ["engineer"]},
    )


def test_a2a_store_persists_tasks_and_expires_them() -> None:
    """Сохраняет задачу между экземплярами store и удаляет её после TTL."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "a2a.sqlite"
        store = A2ATaskStore(path, ttl_seconds=1, max_tasks=10)
        with patch(
            "agent_app.multi_agent.protocols.a2a_store.time.time",
            return_value=100.0,
        ):
            store.save(_task("task-1"), owner_id="alice")

        reopened = A2ATaskStore(path, ttl_seconds=1, max_tasks=10)
        with patch(
            "agent_app.multi_agent.protocols.a2a_store.time.time",
            return_value=100.5,
        ):
            loaded = reopened.get("task-1")
        with patch(
            "agent_app.multi_agent.protocols.a2a_store.time.time",
            return_value=101.1,
        ):
            expired = reopened.get("task-1")

    assert loaded is not None
    assert loaded[1] == "alice"
    assert expired is None


def test_a2a_list_uses_owner_scope_and_page_token() -> None:
    """Не смешивает владельцев и продолжает выдачу с позиции page token."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        store = A2ATaskStore(
            Path(temporary_dir) / "a2a.sqlite",
            ttl_seconds=60,
            max_tasks=10,
        )
        for timestamp, task_id, owner in (
            (100.0, "alice-1", "alice"),
            (101.0, "bob-1", "bob"),
            (102.0, "alice-2", "alice"),
            (103.0, "alice-3", "alice"),
        ):
            with patch(
                "agent_app.multi_agent.protocols.a2a_store.time.time",
                return_value=timestamp,
            ):
                store.save(_task(task_id), owner_id=owner)

        handler = MultiAgentA2AHandler(
            card=None, ask=lambda **_kwargs: None, store=store
        )
        with patch(
            "agent_app.multi_agent.protocols.a2a_store.time.time",
            return_value=104.0,
        ):
            first = asyncio.run(
                handler.on_list_tasks(ListTasksRequest(page_size=2), _context("alice"))
            )
            second = asyncio.run(
                handler.on_list_tasks(
                    ListTasksRequest(
                        page_size=2,
                        page_token=first.next_page_token,
                    ),
                    _context("alice"),
                )
            )

    first_ids = {task.id for task in first.tasks}
    second_ids = {task.id for task in second.tasks}
    assert first.total_size == 3
    assert first.page_size == 2
    assert first.next_page_token
    assert second.page_size == 1
    assert not second.next_page_token
    assert not first_ids.intersection(second_ids)
    assert first_ids | second_ids == {"alice-1", "alice-2", "alice-3"}
