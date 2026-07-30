"""Хранилище инженерных инцидентов для инженерной поддержки."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_app.support.security import redact_secrets

IncidentStatus = Literal["open", "in_progress", "resolved", "closed"]
IncidentPriority = Literal["low", "medium", "high", "critical"]


def utc_now() -> datetime:
    """Возвращает текущее время в UTC с информацией о часовом поясе."""
    return datetime.now(timezone.utc)


class IncidentRecord(BaseModel):
    """Гарантирует целостность и валидацию данных инцидента для передачи между слоями поддержки."""

    id: str
    user_id: str
    session_id: str
    title: str
    description: str
    status: IncidentStatus = "open"
    priority: IncidentPriority = "medium"
    component: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class IncidentStore:
    """Отвечает за хранение и управление инцидентами, гарантируя создание, обновление и безопасный доступ с валидацией данных."""

    def __init__(self, path: Path):
        """Гарантирует готовность хранилища инцидентов к работе и создание необходимых директорий и таблиц."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(
        self,
        *,
        user_id: str,
        session_id: str,
        title: str,
        description: str,
        priority: IncidentPriority = "medium",
        component: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> IncidentRecord:
        """Гарантирует атомарное создание инцидента с валидацией и защитой от утечки секретов, либо выбрасывает ValueError при ошибке."""
        now = utc_now()
        record = IncidentRecord(
            id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            title=redact_secrets(title.strip()),
            description=redact_secrets(description.strip()),
            priority=priority,
            component=component.strip() if component else None,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        if not record.title or not record.description:
            raise ValueError("Для инцидента нужны непустые title и description")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO incidents (
                    id, user_id, session_id, title, description, status,
                    priority, component, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(record),
            )
        return record

    def get(self, incident_id: str, *, user_id: str) -> IncidentRecord | None:
        """Гарантирует возврат инцидента по идентификатору и пользователю или None, если запись не найдена."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM incidents WHERE id = ? AND user_id = ?",
                (incident_id, user_id),
            ).fetchone()
        return self._record(row) if row else None

    def update_status(
        self,
        incident_id: str,
        *,
        user_id: str,
        status: IncidentStatus,
    ) -> IncidentRecord | None:
        """Гарантирует обновление статуса инцидента пользователя или возвращает None, если запись не найдена."""
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE incidents SET status = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (status, now.isoformat(), incident_id, user_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get(incident_id, user_id=user_id)

    def list(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        status: IncidentStatus | None = None,
        limit: int = 20,
    ) -> list[IncidentRecord]:
        """Гарантирует получение списка инцидентов пользователя с фильтрацией по сессии, статусу и лимиту."""
        conditions = ["user_id = ?"]
        values: list[object] = [user_id]
        if session_id is not None:
            conditions.append("session_id = ?")
            values.append(session_id)
        if status is not None:
            conditions.append("status = ?")
            values.append(status)
        values.append(limit)
        query = (
            "SELECT * FROM incidents WHERE "
            + " AND ".join(conditions)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._record(row) for row in rows]

    def _initialize(self) -> None:
        """Гарантирует наличие таблицы и индексов для хранения инцидентов без потери существующих данных."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    component TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_user ON incidents(user_id)"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_incidents_user_session
                ON incidents(user_id, session_id)
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Гарантирует безопасное подключение к базе с поддержкой транзакций и автоматическим откатом при ошибках."""
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _values(record: IncidentRecord) -> tuple[object, ...]:
        """Гарантирует сериализацию инцидента в кортеж для корректной записи в базу данных."""
        return (
            record.id,
            record.user_id,
            record.session_id,
            record.title,
            record.description,
            record.status,
            record.priority,
            record.component,
            json.dumps(record.metadata, ensure_ascii=False),
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> IncidentRecord:
        """Гарантирует преобразование строки из БД в инвариантный IncidentRecord для корректной работы подсистемы поддержки."""
        return IncidentRecord(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            title=row["title"],
            description=row["description"],
            status=row["status"],
            priority=row["priority"],
            component=row["component"],
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
