"""Постоянная очередь web-операций с атомарным claim, отменой и owner scope."""

from __future__ import annotations

import json
import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agent_app.database import DatabaseRuntime
from agent_app.service.web_config import WebConfig


class OperationRequest(BaseModel):
    """Клиент выбирает зарегистрированный профиль и идентификаторы входов."""

    model_config = ConfigDict(extra="forbid")
    profile: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")
    idempotency_key: str = Field(min_length=8, max_length=128)
    upstream_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{36}$")
    baseline_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{36}$")
    source_ids: list[str] = Field(default_factory=list, max_length=100)
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict, max_length=20
    )


class OperationRecord(BaseModel):
    """Публичная запись задания: параметры, состояние и безопасная диагностика."""

    id: str
    user_id: str
    profile: str
    status: Literal[
        "queued",
        "running",
        "cancel_requested",
        "cancelled",
        "completed",
        "failed",
        "interrupted",
    ]
    created_at: float
    updated_at: float
    request: OperationRequest
    error: str | None = None
    exit_code: int | None = None


class OperationPage(BaseModel):
    """Продолжение страницы привязано к монотонному порядку UUID внутри времени."""

    items: list[OperationRecord]
    next_cursor: str | None = None


class OperationStore:
    """SQL является источником истины для API и независимо запущенных workers."""

    def __init__(self, database: DatabaseRuntime, config: WebConfig):
        """Разделяет путь локального SQLite и общую PostgreSQL БД."""
        self.database, self.config = database, config
        self.path = config.sqlite_path

    def initialize(self) -> None:
        """Создаёт очередь и строку сериализации постановки/выполнения."""
        with self.database.connection(self.path) as conn:
            if self.database.is_postgresql:
                conn.execute("SELECT pg_advisory_xact_lock(?)", (6384202,))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_operations (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, profile TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, request_json TEXT NOT NULL, idempotency_key TEXT NOT NULL, error TEXT, exit_code INTEGER, UNIQUE(user_id, idempotency_key))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS web_operations_owner ON web_operations(user_id, created_at, id)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_worker_lock (id INTEGER PRIMARY KEY, token TEXT NOT NULL, expires_at REAL NOT NULL)"
            )
            conn.execute(
                "INSERT INTO web_worker_lock VALUES (1, '', 0) ON CONFLICT(id) DO NOTHING"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_sources (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_worker_status (id TEXT PRIMARY KEY, capabilities TEXT NOT NULL, updated_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_operation_definitions (id TEXT PRIMARY KEY REFERENCES web_operations(id) ON DELETE CASCADE, payload TEXT NOT NULL)"
            )

    def submit(
        self, owner: str, payload: OperationRequest, *, definition: dict | None = None
    ) -> OperationRecord:
        """Проверяет capacity и уникальность ключа под одной SQL-блокировкой."""
        now, job_id = time.time(), str(uuid4())
        with self.database.connection(self.path) as conn:
            # UPDATE захватывает строку в PG и write-lock в SQLite до commit.
            conn.execute("UPDATE web_worker_lock SET id = id WHERE id = 1")
            row = conn.execute(
                "SELECT * FROM web_operations WHERE user_id = ? AND idempotency_key = ?",
                (owner, payload.idempotency_key),
            ).fetchone()
            if row is not None:
                record = self._record(row)
                if record.request != payload:
                    raise HTTPException(
                        409,
                        "Ключ идемпотентности уже использован с другими параметрами.",
                    )
                return record
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM web_operations WHERE status IN ('queued', 'running', 'cancel_requested')"
            ).fetchone()["n"]
            if count >= self.config.max_pending_operations:
                raise HTTPException(
                    429, "Очередь операций заполнена.", headers={"Retry-After": "10"}
                )
            for source_id in payload.source_ids:
                if (
                    conn.execute(
                        "SELECT id FROM web_sources WHERE id = ? AND user_id = ?",
                        (source_id, owner),
                    ).fetchone()
                    is None
                ):
                    raise HTTPException(404, "Источник не найден.")
            conn.execute(
                "INSERT INTO web_operations VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, NULL, NULL)",
                (
                    job_id,
                    owner,
                    payload.profile,
                    now,
                    now,
                    payload.model_dump_json(),
                    payload.idempotency_key,
                ),
            )
            if definition is not None:
                conn.execute(
                    "INSERT INTO web_operation_definitions VALUES (?, ?)",
                    (job_id, json.dumps(definition)),
                )
        return self.get(job_id, owner)

    def existing(self, owner: str, payload: OperationRequest) -> OperationRecord | None:
        """Повторный HTTP-запрос возвращает прежнюю работу даже после отключения worker."""
        with self.database.connection(self.path) as conn:
            row = conn.execute(
                "SELECT * FROM web_operations WHERE user_id = ? AND idempotency_key = ?",
                (owner, payload.idempotency_key),
            ).fetchone()
        if row is None:
            return None
        record = self._record(row)
        if record.request != payload:
            raise HTTPException(409, "Ключ идемпотентности уже занят другим запросом.")
        return record

    def definition(self, job_id: str) -> dict:
        """Снимок настроек не меняется при редактировании каталога между submit и claim."""
        with self.database.connection(self.path) as conn:
            row = conn.execute(
                "SELECT payload FROM web_operation_definitions WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise ValueError("Отсутствует снимок конфигурации операции.")
        return json.loads(row["payload"])

    def get(self, job_id: str, owner: str | None = None) -> OperationRecord:
        """Чужой идентификатор возвращает 404 и не раскрывает существование записи."""
        with self.database.connection(self.path) as conn:
            sql, args = "SELECT * FROM web_operations WHERE id = ?", [job_id]
            if owner is not None:
                sql += " AND user_id = ?"
                args.append(owner)
            row = conn.execute(sql, args).fetchone()
        if row is None:
            raise HTTPException(404, "Операция не найдена.")
        return self._record(row)

    def list(
        self,
        owner: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
        status: str | None = None,
    ) -> OperationPage:
        """Выбирает страницу до последней виденной записи, исключая чужие операции."""
        sql, args = "SELECT * FROM web_operations WHERE user_id = ?", [owner]
        if cursor:
            previous = self.get(cursor, owner)
            sql += " AND (created_at < ? OR (created_at = ? AND id < ?))"
            args += [previous.created_at, previous.created_at, previous.id]
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit + 1)
        with self.database.connection(self.path) as conn:
            rows = conn.execute(sql, args).fetchall()
        items = [self._record(row) for row in rows[:limit]]
        return OperationPage(
            items=items, next_cursor=items[-1].id if len(rows) > limit else None
        )

    def cancel(self, job_id: str, owner: str | None) -> OperationRecord:
        """Завершённую запись не меняет; работающую отменяет только после остановки процесса."""
        self.get(job_id, owner)
        with self.database.connection(self.path) as conn:
            conn.execute(
                "UPDATE web_operations SET status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE 'cancel_requested' END, updated_at = ? WHERE id = ? AND status IN ('queued', 'running')",
                (time.time(), job_id),
            )
        return self.get(job_id, owner)

    def claim(self, token: str, profiles: list[str]) -> OperationRecord | None:
        """Один worker выполняет ресурсозатратную операцию; остальные не дублируют её."""
        if not profiles:
            return None
        now = time.time()
        with self.database.connection(self.path) as conn:
            row = conn.execute(
                "UPDATE web_worker_lock SET token = ?, expires_at = ? WHERE id = 1 AND expires_at < ? RETURNING id",
                (token, now + self.config.worker_lease_seconds, now),
            ).fetchone()
            if row is None:
                return None
            # Истёкший lease не является разрешением повторить платный вызов.
            conn.execute(
                "UPDATE web_operations SET status = 'interrupted', error = 'Worker прерван; повторный запуск требует нового задания.', updated_at = ? WHERE status IN ('running', 'cancel_requested')",
                (now,),
            )
            placeholders = ",".join("?" for _ in profiles)
            row = conn.execute(
                f"SELECT id FROM web_operations WHERE status = 'queued' AND profile IN ({placeholders}) ORDER BY created_at, id LIMIT 1",
                profiles,
            ).fetchone()
            if row is None:
                conn.execute(
                    "UPDATE web_worker_lock SET expires_at = 0 WHERE token = ?",
                    (token,),
                )
                return None
            record = conn.execute(
                "UPDATE web_operations SET status = 'running', updated_at = ? WHERE id = ? AND status = 'queued' RETURNING *",
                (now, row["id"]),
            ).fetchone()
            if record is None:
                # Отмена могла зафиксироваться между SELECT и CAS в PostgreSQL.
                # Пустой claim освобождает слот, но не возвращает отменённую работу.
                conn.execute(
                    "UPDATE web_worker_lock SET expires_at = 0 WHERE token = ?",
                    (token,),
                )
                return None
        return self._record(record)

    def renew(self, token: str) -> bool:
        """Потеря lease требует остановки дочернего процесса до новых действий."""
        with self.database.connection(self.path) as conn:
            return (
                conn.execute(
                    "UPDATE web_worker_lock SET expires_at = ? WHERE token = ? AND expires_at > ?",
                    (
                        time.time() + self.config.worker_lease_seconds,
                        token,
                        time.time(),
                    ),
                ).rowcount
                == 1
            )

    def finish(
        self,
        job_id: str,
        token: str,
        *,
        status: str,
        exit_code: int | None,
        error: str | None = None,
    ) -> None:
        """CAS сохраняет отмену, даже если subprocess одновременно вернул exit code 0."""
        with self.database.connection(self.path) as conn:
            conn.execute(
                "UPDATE web_operations SET status = CASE WHEN status = 'cancel_requested' THEN 'cancelled' ELSE ? END, updated_at = ?, exit_code = ?, error = ? WHERE id = ? AND status IN ('running', 'cancel_requested') AND EXISTS(SELECT 1 FROM web_worker_lock WHERE token = ? AND expires_at > ?)",
                (status, time.time(), exit_code, error, job_id, token, time.time()),
            )
            conn.execute(
                "UPDATE web_worker_lock SET expires_at = 0 WHERE token = ?", (token,)
            )

    def heartbeat(self, worker_id: str, capabilities: list[str]) -> None:
        """Объявляет профили конкретного worker, а не предполагаемые пакеты API-процесса."""
        with self.database.connection(self.path) as conn:
            conn.execute(
                "INSERT INTO web_worker_status VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET capabilities = excluded.capabilities, updated_at = excluded.updated_at",
                (worker_id, json.dumps(capabilities), time.time()),
            )
            conn.execute(
                "DELETE FROM web_worker_status WHERE updated_at < ?",
                (time.time() - 86400,),
            )

    def capabilities(self) -> set[str]:
        """Устаревший heartbeat не показывает недоступный worker как готовый."""
        with self.database.connection(self.path) as conn:
            rows = conn.execute(
                "SELECT capabilities FROM web_worker_status WHERE updated_at > ?",
                (time.time() - self.config.worker_lease_seconds,),
            ).fetchall()
        return {item for row in rows for item in json.loads(row["capabilities"])}

    @staticmethod
    def _record(row: Any) -> OperationRecord:
        """Внутренние поля БД не попадают в HTTP-контракт."""
        return OperationRecord(
            id=row["id"],
            user_id=row["user_id"],
            profile=row["profile"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            request=OperationRequest.model_validate_json(row["request_json"]),
            error=row["error"],
            exit_code=row["exit_code"],
        )
