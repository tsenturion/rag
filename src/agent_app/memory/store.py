"""Хранилище состояния для памяти агента."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from psycopg.errors import UniqueViolation

from agent_app.database import DatabaseRuntime
from agent_app.memory.policy import (
    clamp_importance,
    normalize_key,
    validate_memory_key,
    validate_memory_value,
)
from agent_app.models import (
    ConversationMessage,
    ConversationSession,
    MemoryRecord,
    MemorySearchResult,
    MemorySource,
    MemoryType,
    utc_now,
)


class MemoryStore:
    """Долговременная память поверх локального SQLite или общего PostgreSQL."""

    def __init__(self, path: Path, *, database: DatabaseRuntime | None = None):
        """Создаёт таблицы в выбранном backend и фиксирует владение соединениями."""
        self.path = path
        self._owns_database = database is None
        self.database = database or DatabaseRuntime(backend="sqlite")
        self._init_schema()

    def close(self) -> None:
        """Закрывает database runtime, только если store создал его самостоятельно."""
        if self._owns_database:
            self.database.close()

    def save(
        self,
        *,
        user_id: str,
        key: str,
        value: str,
        memory_type: MemoryType = "fact",
        session_id: str | None = None,
        tags: list[str] | None = None,
        importance: int = 3,
        source: MemorySource = "user",
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Гарантирует атомарное сохранение или обновление уникальной записи памяти пользователя с учётом ключа, типа и сессии."""
        now = utc_now()
        normalized_key = normalize_key(validate_memory_key(key))
        cleaned_value = validate_memory_value(value)
        existing = self.find_by_key(
            user_id=user_id,
            key=normalized_key,
            memory_type=memory_type,
            session_id=session_id,
        )
        if existing is not None:
            return self.update(
                existing.id,
                user_id=user_id,
                value=cleaned_value,
                tags=tags,
                importance=importance,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )

        record = MemoryRecord(
            id=str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            key=normalized_key,
            value=cleaned_value,
            tags=tags or [],
            importance=clamp_importance(importance),
            source=source,
            created_at=now,
            updated_at=now,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO memories (
                      id, user_id, session_id, memory_type, key, value, tags,
                      importance, source, created_at, updated_at, last_accessed_at,
                      access_count, ttl_seconds, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._to_row(record),
                )
        except (sqlite3.IntegrityError, UniqueViolation):
            # Между предварительным чтением и INSERT другой worker мог создать
            # тот же scoped key. После rollback читаем победившую запись и
            # применяем обычную политику обновления вместо возврата ошибки.
            concurrent = self.find_by_key(
                user_id=user_id,
                key=normalized_key,
                memory_type=memory_type,
                session_id=session_id,
            )
            if concurrent is None:
                raise
            return self.update(
                concurrent.id,
                user_id=user_id,
                value=cleaned_value,
                tags=tags,
                importance=importance,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
        return record

    def get(self, memory_id: str, *, user_id: str | None = None) -> MemoryRecord | None:
        """Гарантирует получение актуальной и неистёкшей записи памяти по идентификатору, увеличивая счётчик обращений и удаляя устаревшие данные."""
        query = "SELECT * FROM memories WHERE id = ?"
        params: list[Any] = [memory_id]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        record = self._from_row(row)
        if self._is_expired(record):
            self.delete(memory_id)
            return None
        self._mark_accessed(record.id)
        return record.model_copy(update={"access_count": record.access_count + 1})

    def find_by_key(
        self,
        *,
        user_id: str,
        key: str,
        memory_type: MemoryType | None = None,
        session_id: str | None = None,
    ) -> MemoryRecord | None:
        """Гарантирует поиск самой свежей и валидной записи памяти по ключу, типу и сессии, автоматически удаляя истёкшие записи."""
        normalized_key = normalize_key(key)
        query = "SELECT * FROM memories WHERE user_id = ? AND key = ?"
        params: list[Any] = [user_id, normalized_key]
        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type)
        if session_id is None:
            query += " AND session_id IS NULL"
        else:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        record = self._from_row(row)
        if self._is_expired(record):
            self.delete(record.id)
            return None
        return record

    def update(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        value: str | None = None,
        tags: list[str] | None = None,
        importance: int | None = None,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Гарантирует согласованное обновление содержимого, тегов, важности и метаданных существующей записи памяти с проверкой прав пользователя."""
        current = self.get(memory_id, user_id=user_id)
        if current is None:
            raise KeyError(f"Запись памяти не найдена: {memory_id}")
        updated = current.model_copy(
            update={
                "value": validate_memory_value(value)
                if value is not None
                else current.value,
                "tags": tags if tags is not None else current.tags,
                "importance": clamp_importance(importance)
                if importance is not None
                else current.importance,
                "ttl_seconds": ttl_seconds
                if ttl_seconds is not None
                else current.ttl_seconds,
                "metadata": metadata if metadata is not None else current.metadata,
                "updated_at": utc_now(),
            }
        )
        query = """
                UPDATE memories
                SET value = ?, tags = ?, importance = ?, updated_at = ?,
                    ttl_seconds = ?, metadata = ?
                WHERE id = ?
                """
        params: list[Any] = [
            updated.value,
            json.dumps(updated.tags, ensure_ascii=False),
            updated.importance,
            updated.updated_at.isoformat(),
            updated.ttl_seconds,
            json.dumps(updated.metadata, ensure_ascii=False),
            updated.id,
        ]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            if cursor.rowcount == 0:
                raise KeyError(f"Запись памяти не найдена: {memory_id}")
        return updated

    def delete(self, memory_id: str, *, user_id: str | None = None) -> bool:
        """Гарантирует удаление записи памяти по идентификатору с учётом пользователя и возвращает факт успешного удаления."""
        query = "DELETE FROM memories WHERE id = ?"
        params: list[Any] = [memory_id]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0

    def delete_by_key(
        self,
        *,
        user_id: str,
        key: str,
        memory_type: MemoryType | None = None,
        session_id: str | None = None,
    ) -> int:
        """Гарантирует массовое удаление всех записей памяти пользователя по ключу, типу и сессии, возвращая количество удалённых записей."""
        normalized_key = normalize_key(key)
        query = "DELETE FROM memories WHERE user_id = ? AND key = ?"
        params: list[Any] = [user_id, normalized_key]
        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type)
        if session_id is None:
            query += " AND session_id IS NULL"
        else:
            query += " AND session_id = ?"
            params.append(session_id)
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return int(cursor.rowcount)

    def search(
        self,
        *,
        user_id: str,
        query: str,
        memory_type: MemoryType | None = None,
        session_id: str | None = None,
        limit: int = 5,
    ) -> MemorySearchResult:
        """Гарантирует полнотекстовый поиск по ключу, значению и тегам памяти пользователя с сортировкой по важности и свежести, исключая устаревшие записи."""
        like = f"%{query.strip()}%"
        sql = """
            SELECT * FROM memories
            WHERE user_id = ?
              AND (key LIKE ? OR value LIKE ? OR tags LIKE ?)
        """
        params: list[Any] = [user_id, like, like, like]
        if session_id is not None:
            sql += " AND (session_id IS NULL OR session_id = ?)"
            params.append(session_id)
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        records = [
            record for row in rows if (record := self._valid_record(row)) is not None
        ]
        for record in records:
            self._mark_accessed(record.id)
        return MemorySearchResult(records=records, query=query, count=len(records))

    def list_memories(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Возвращает память пользователя; при заданной сессии исключает записи других диалогов."""
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]
        if session_id is not None:
            # Global-записи относятся к пользователю целиком, а session-записи
            # доступны только внутри породившего их диалога.
            sql += " AND (session_id IS NULL OR session_id = ?)"
            params.append(session_id)
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            record for row in rows if (record := self._valid_record(row)) is not None
        ]

    def clear_session(self, *, user_id: str, session_id: str) -> int:
        """Удаляет все записи памяти, связанные с указанной сессией пользователя, гарантируя отсутствие остаточных данных для этой пары идентификаторов."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            conn.execute(
                "DELETE FROM conversation_history WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            return int(cursor.rowcount)

    def clear_user(self, *, user_id: str) -> int:
        """Удаляет все записи памяти, связанные с пользователем, обеспечивая полное удаление пользовательских данных из хранилища."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            conn.execute(
                "DELETE FROM conversation_history WHERE user_id = ?", (user_id,)
            )
            return int(cursor.rowcount)

    def save_conversation_history(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Атомарно сохраняет ограниченный диалоговый буфер для восстановления runner."""
        if not messages:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM conversation_history WHERE user_id = ? AND session_id = ?",
                    (user_id, session_id),
                )
            return
        payload = json.dumps(messages, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_history (user_id, session_id, messages, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                  messages = excluded.messages,
                  updated_at = excluded.updated_at
                """,
                (user_id, session_id, payload, utc_now().isoformat()),
            )

    def load_conversation_history(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> list[dict[str, str]]:
        """Восстанавливает только допустимые роли и строковое содержимое диалога."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT messages FROM conversation_history
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        if row is None:
            return []
        return [
            message.model_dump(mode="python")
            for message in self._decode_conversation_messages(row["messages"])
        ]

    def get_conversation_session(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> ConversationSession | None:
        """Возвращает сохранённую историю одной сессии без создания runner."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, messages, updated_at
                FROM conversation_history
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        return self._conversation_session_from_row(user_id, row) if row else None

    def list_conversation_sessions(
        self,
        *,
        user_id: str,
        limit: int,
        before_updated_at: datetime | None = None,
        before_session_id: str | None = None,
    ) -> list[ConversationSession]:
        """Возвращает страницу диалогов пользователя в стабильном порядке."""
        sql = (
            "SELECT session_id, messages, updated_at "
            "FROM conversation_history WHERE user_id = ?"
        )
        params: list[Any] = [user_id]
        if before_updated_at is not None and before_session_id is not None:
            sql += " AND (updated_at < ? OR (updated_at = ? AND session_id > ?))"
            timestamp = before_updated_at.isoformat()
            params.extend([timestamp, timestamp, before_session_id])
        sql += " ORDER BY updated_at DESC, session_id ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._conversation_session_from_row(user_id, row) for row in rows]

    @staticmethod
    def _decode_conversation_messages(value: object) -> list[ConversationMessage]:
        """Отбрасывает повреждённые элементы сохранённого диалога."""
        try:
            payload = json.loads(value) if isinstance(value, str) else None
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        messages: list[ConversationMessage] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                messages.append(ConversationMessage.model_validate(item))
            except ValueError:
                continue
        return messages

    @classmethod
    def _conversation_session_from_row(
        cls,
        user_id: str,
        row: Mapping[str, Any],
    ) -> ConversationSession:
        """Преобразует строку СУБД в типизированную историю диалога."""
        return ConversationSession(
            user_id=user_id,
            session_id=row["session_id"],
            messages=cls._decode_conversation_messages(row["messages"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _valid_record(self, row: Mapping[str, Any]) -> MemoryRecord | None:
        """Гарантирует, что возвращаемая запись памяти не просрочена, автоматически удаляя устаревшие записи и предотвращая их использование."""
        record = self._from_row(row)
        if self._is_expired(record):
            self.delete(record.id)
            return None
        return record

    def _mark_accessed(self, memory_id: str) -> None:
        """Фиксирует факт обращения к записи памяти, увеличивая счётчик и обновляя временную метку для поддержки актуальности и статистики."""
        now = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (now, memory_id),
            )

    def _init_schema(self) -> None:
        """Создаёт и поддерживает структуру таблиц и индексов в базе данных, обеспечивая согласованность и уникальность ключей памяти."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  session_id TEXT,
                  memory_type TEXT NOT NULL,
                  key TEXT NOT NULL,
                  value TEXT NOT NULL,
                  tags TEXT NOT NULL,
                  importance INTEGER NOT NULL,
                  source TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  last_accessed_at TEXT,
                  access_count INTEGER NOT NULL DEFAULT 0,
                  ttl_seconds INTEGER,
                  metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_unique_scoped_key
                ON memories(user_id, COALESCE(session_id, ''), memory_type, key)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_history (
                  user_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  messages TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (user_id, session_id)
                )
                """
            )

    def _connect(self):
        """Возвращает транзакцию общего database runtime для текущего store."""
        return self.database.connection(self.path)

    @staticmethod
    def _to_row(record: MemoryRecord) -> tuple[Any, ...]:
        """Гарантирует сериализацию объекта памяти в кортеж для корректного хранения в базе данных согласно публичному контракту."""
        return (
            record.id,
            record.user_id,
            record.session_id,
            record.memory_type,
            record.key,
            record.value,
            json.dumps(record.tags, ensure_ascii=False),
            record.importance,
            record.source,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            record.last_accessed_at.isoformat() if record.last_accessed_at else None,
            record.access_count,
            record.ttl_seconds,
            json.dumps(record.metadata, ensure_ascii=False),
        )

    @staticmethod
    def _from_row(row: Mapping[str, Any]) -> MemoryRecord:
        """Восстанавливает объект памяти из строки базы данных, обеспечивая целостность и соответствие контракту MemoryRecord."""
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            memory_type=row["memory_type"],
            key=row["key"],
            value=row["value"],
            tags=json.loads(row["tags"] or "[]"),
            importance=row["importance"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"])
            if row["last_accessed_at"]
            else None,
            access_count=row["access_count"],
            ttl_seconds=row["ttl_seconds"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _is_expired(record: MemoryRecord) -> bool:
        """Гарантирует определение просроченности записи памяти на основе времени жизни и предотвращает использование устаревших данных."""
        if record.ttl_seconds is None:
            return False
        age = datetime.now(timezone.utc) - record.updated_at
        return age.total_seconds() > record.ttl_seconds
