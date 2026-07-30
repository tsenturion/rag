"""SQLite-хранилище A2A-задач с owner scope, TTL и ограничением размера."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from a2a.types import Task


class A2ATaskStore:
    """Сохраняет protobuf Task между запросами, процессами и перезапусками."""

    def __init__(self, path: Path, *, ttl_seconds: int, max_tasks: int):
        """Создаёт schema и фиксирует политику хранения A2A-задач."""
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_tasks = max_tasks
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS a2a_tasks (
                    task_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_a2a_owner_updated "
                "ON a2a_tasks(owner_id, updated_at DESC)"
            )

    def save(self, task: Task, *, owner_id: str) -> None:
        """Атомарно сохраняет task и удаляет просроченные/самые старые записи."""
        now = time.time()
        with self._connect() as connection:
            connection.execute("DELETE FROM a2a_tasks WHERE expires_at <= ?", (now,))
            connection.execute(
                """
                INSERT INTO a2a_tasks (
                    task_id, owner_id, context_id, payload, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    context_id = excluded.context_id,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    task.id,
                    owner_id,
                    task.context_id,
                    task.SerializeToString(),
                    now,
                    now + self.ttl_seconds,
                ),
            )
            connection.execute(
                """
                DELETE FROM a2a_tasks
                WHERE task_id IN (
                    SELECT task_id FROM a2a_tasks
                    ORDER BY updated_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_tasks,),
            )

    def get(self, task_id: str) -> tuple[Task, str] | None:
        """Возвращает актуальную задачу и owner либо удаляет истёкшую запись."""
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, owner_id, expires_at FROM a2a_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            if float(row["expires_at"]) <= now:
                connection.execute(
                    "DELETE FROM a2a_tasks WHERE task_id = ?", (task_id,)
                )
                return None
        task = Task()
        task.ParseFromString(row["payload"])
        return task, str(row["owner_id"])

    def list(self, *, owner_id: str | None = None) -> list[tuple[Task, str]]:
        """Возвращает задачи от новых к старым, при необходимости только owner."""
        now = time.time()
        sql = "SELECT payload, owner_id FROM a2a_tasks WHERE expires_at > ?"
        params: list[object] = [now]
        if owner_id is not None:
            sql += " AND owner_id = ?"
            params.append(owner_id)
        sql += " ORDER BY updated_at DESC"
        with self._connect() as connection:
            connection.execute("DELETE FROM a2a_tasks WHERE expires_at <= ?", (now,))
            rows = connection.execute(sql, params).fetchall()
        result: list[tuple[Task, str]] = []
        for row in rows:
            task = Task()
            task.ParseFromString(row["payload"])
            result.append((task, str(row["owner_id"])))
        return result

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Открывает транзакцию и гарантированно закрывает короткое WAL-соединение."""
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
