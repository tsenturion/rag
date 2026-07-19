"""Очередь ручной проверки для безопасности и ручного контроля."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from agent_app.guardrails.models import HumanReviewRecord, utc_now


class HumanReviewStore:
    """Обеспечивает надежное хранение истории ручных проверок с гарантией целостности данных и поддержки аудита в многопоточной среде."""

    def __init__(self, path: Path):
        """Гарантирует потокобезопасное и атомарное хранение истории ручных ревью с автоматическим созданием структуры БД на диске."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS human_reviews (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT,
                    trace_id TEXT,
                    prompt TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    reviewer_id TEXT,
                    comment TEXT
                )
                """
            )

    def create(self, record: HumanReviewRecord) -> HumanReviewRecord:
        """Гарантирует запись нового события ручного ревью в хранилище с защитой от гонок и сохранением полной информации для аудита."""
        fields = record.model_dump(mode="json")
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO human_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(
                    fields[name]
                    for name in (
                        "id",
                        "created_at",
                        "updated_at",
                        "status",
                        "user_id",
                        "session_id",
                        "request_id",
                        "trace_id",
                        "prompt",
                        "answer",
                        "reason",
                        "reviewer_id",
                        "comment",
                    )
                ),
            )
        return record

    def list(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[HumanReviewRecord]:
        """Гарантирует получение отсортированного списка ревью с возможностью фильтрации по статусу и ограничением на количество для контроля нагрузки."""
        sql = "SELECT * FROM human_reviews"
        params: list[object] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [HumanReviewRecord.model_validate(dict(row)) for row in rows]

    def decide(
        self, review_id: str, *, approved: bool, reviewer_id: str, comment: str | None
    ) -> HumanReviewRecord | None:
        """Гарантирует атомарное принятие решения по ревью только в статусе 'pending' с фиксацией результата и идентификатора ревьюера."""
        status = "approved" if approved else "rejected"
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE human_reviews
                SET status = ?, reviewer_id = ?, comment = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (status, reviewer_id, comment, utc_now().isoformat(), review_id),
            )
            if cursor.rowcount == 0:
                return None
            row = self._connection.execute(
                "SELECT * FROM human_reviews WHERE id = ?", (review_id,)
            ).fetchone()
        return HumanReviewRecord.model_validate(dict(row))

    def close(self) -> None:
        """Гарантирует корректное освобождение ресурсов и завершение работы с файловым хранилищем ревью."""
        with self._lock:
            self._connection.close()
