"""Аудит событий безопасности для безопасности и ручного контроля."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from agent_app.guardrails.models import SecurityAuditEvent


class SecurityAuditStore:
    """Обеспечивает надежное и потокобезопасное хранение событий аудита безопасности с гарантией атомарности операций и доступности данных для последующего анализа."""

    def __init__(self, path: Path):
        """Гарантирует потокобезопасное и атомарное хранение аудита безопасности в SQLite с созданием структуры хранения при первом запуске."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS security_audit (
                    id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    principal_id TEXT,
                    role TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    request_id TEXT,
                    trace_id TEXT,
                    details_json TEXT NOT NULL
                )
                """
            )

    def append(self, event: SecurityAuditEvent) -> SecurityAuditEvent:
        """Гарантирует, что событие аудита будет записано в хранилище без потери данных и доступно для последующего анализа."""
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO security_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.occurred_at.isoformat(),
                    event.event_type,
                    event.action,
                    event.principal_id,
                    event.role,
                    event.user_id,
                    event.session_id,
                    event.request_id,
                    event.trace_id,
                    json.dumps(event.details, ensure_ascii=False, sort_keys=True),
                ),
            )
        return event

    def list(self, *, limit: int = 100) -> list[SecurityAuditEvent]:
        """Гарантирует получение последних событий аудита в порядке убывания времени без дублирования и с ограничением по количеству."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM security_audit ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def close(self) -> None:
        """Гарантирует освобождение ресурсов и завершение всех операций с базой аудита без утечек."""
        with self._lock:
            self._connection.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> SecurityAuditEvent:
        """Преобразует строку из базы в валидированное событие аудита с восстановлением структуры details."""
        payload = dict(row)
        payload["details"] = json.loads(payload.pop("details_json"))
        return SecurityAuditEvent.model_validate(payload)
