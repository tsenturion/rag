"""Хранилище A2A-задач с owner scope, TTL и ограничением размера."""

from __future__ import annotations

import time
import threading
from pathlib import Path

from a2a.types import Task

from agent_app.database import DatabaseRuntime


class A2ATaskStore:
    """Сохраняет protobuf Task между запросами, процессами и перезапусками."""

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: int,
        max_tasks: int,
        database: DatabaseRuntime | None = None,
    ):
        """Создаёт schema и фиксирует политику хранения A2A-задач."""
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_tasks = max_tasks
        self._owns_database = database is None
        self.database = database or DatabaseRuntime(backend="sqlite")
        self._initialized = False
        self._initialize_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Лениво создаёт schema, не подключаясь к БД при генерации OpenAPI."""
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            self._initialize_schema()
            self._initialized = True

    def _initialize_schema(self) -> None:
        """Создаёт таблицу и индекс A2A-задач в выбранном backend."""
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
        self._ensure_initialized()
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
        self._ensure_initialized()
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
        task.ParseFromString(bytes(row["payload"]))
        return task, str(row["owner_id"])

    def list(self, *, owner_id: str | None = None) -> list[tuple[Task, str]]:
        """Возвращает задачи от новых к старым, при необходимости только owner."""
        self._ensure_initialized()
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
            task.ParseFromString(bytes(row["payload"]))
            result.append((task, str(row["owner_id"])))
        return result

    def close(self) -> None:
        """Закрывает самостоятельно созданный database runtime."""
        if self._owns_database:
            self.database.close()

    def _connect(self):
        """Открывает транзакцию через общий слой persistence."""
        return self.database.connection(self.path)
