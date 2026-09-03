"""Единый слой соединений SQLite и PostgreSQL для состояния агентного приложения."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, Sequence

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


DatabaseBackend = Literal["sqlite", "postgresql"]


def _postgres_sql(statement: str) -> str:
    """Адаптирует ограниченный SQL-диалект локальных store-классов к PostgreSQL.

    Store-классы используют переносимые запросы и qmark-параметры SQLite. Здесь
    централизованно меняются только различающиеся типы, placeholders и форма
    неограниченного LIMIT; произвольный SQL или имена таблиц не принимаются.
    """
    sql = re.sub(r"\bBLOB\b", "BYTEA", statement, flags=re.IGNORECASE)
    sql = re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql, flags=re.IGNORECASE)
    sql = sql.replace("?", "%s")
    sql = re.sub(
        r"LIMIT\s+-1\s+OFFSET\s+%s",
        "LIMIT ALL OFFSET %s",
        sql,
        flags=re.IGNORECASE,
    )
    return sql


class DatabaseConnection:
    """Предоставляет store-классам одинаковый execute-интерфейс двух СУБД."""

    def __init__(self, connection: Any, *, backend: DatabaseBackend):
        """Связывает адаптер с одним соединением и его SQL-диалектом."""
        self._connection = connection
        self._backend = backend

    def execute(self, statement: str, parameters: Sequence[Any] = ()) -> Any:
        """Выполняет параметризованный запрос в синтаксисе выбранного backend."""
        if self._backend == "postgresql":
            statement = _postgres_sql(statement)
        return self._connection.execute(statement, parameters)


class DatabaseRuntime:
    """Управляет SQLite-соединениями либо общим потокобезопасным PostgreSQL pool.

    SQLite открывается отдельно для каждой короткой транзакции и предназначен
    для локального однопроцессного запуска. PostgreSQL pool лениво создаётся при
    первом обращении, поэтому чтение конфигурации и генерация OpenAPI не требуют
    доступной базы данных.
    """

    def __init__(
        self,
        *,
        backend: DatabaseBackend,
        database_url: str | None = None,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        connect_timeout_seconds: float = 10.0,
    ):
        """Фиксирует параметры backend; PostgreSQL pool создаётся лениво."""
        self.backend = backend
        self.database_url = database_url
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.connect_timeout_seconds = connect_timeout_seconds
        self._pool: ConnectionPool[Any] | None = None
        self._pool_lock = threading.Lock()
        self._closed = False

    @classmethod
    def from_config(cls, config: Any) -> DatabaseRuntime:
        """Создаёт runtime из секции persistence, не раскрывая URL в конфиге."""
        database_url = None
        if config.backend == "postgresql":
            database_url = os.getenv(config.database_url_env)
            if not database_url:
                raise RuntimeError(
                    "Для persistence.backend=postgresql задайте переменную "
                    f"{config.database_url_env}."
                )
        return cls(
            backend=config.backend,
            database_url=database_url,
            pool_min_size=config.pool_min_size,
            pool_max_size=config.pool_max_size,
            connect_timeout_seconds=config.connect_timeout_seconds,
        )

    @property
    def is_postgresql(self) -> bool:
        """Показывает, использует ли runtime production backend PostgreSQL."""
        return self.backend == "postgresql"

    @property
    def postgres_pool(self) -> ConnectionPool[Any]:
        """Возвращает открытый pool, необходимый официальному PostgresSaver."""
        if not self.is_postgresql:
            raise RuntimeError("PostgreSQL pool недоступен в режиме SQLite.")
        return self._ensure_pool()

    @contextmanager
    def connection(self, sqlite_path: Path) -> Iterator[DatabaseConnection]:
        """Открывает транзакцию и гарантирует commit либо rollback.

        Путь нужен только SQLite. Для PostgreSQL все логические хранилища живут
        в одной базе, а разграничение выполняется именами таблиц и user_id.
        """
        if self._closed:
            raise RuntimeError("DatabaseRuntime уже закрыт.")
        if self.is_postgresql:
            with self._ensure_pool().connection() as raw_connection:
                with raw_connection.transaction():
                    yield DatabaseConnection(raw_connection, backend=self.backend)
            return

        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        raw_connection = sqlite3.connect(sqlite_path, timeout=30)
        raw_connection.row_factory = sqlite3.Row
        raw_connection.execute("PRAGMA journal_mode=WAL")
        raw_connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield DatabaseConnection(raw_connection, backend=self.backend)
            raw_connection.commit()
        except Exception:
            raw_connection.rollback()
            raise
        finally:
            raw_connection.close()

    def ping(self) -> dict[str, object]:
        """Проверяет реальное соединение и возвращает безопасный readiness status."""
        try:
            with self.connection(Path(":memory:")) as connection:
                connection.execute("SELECT 1 AS ready").fetchone()
        except Exception as exc:
            return {
                "ready": False,
                "backend": self.backend,
                "error": str(exc)[:500],
            }
        return {"ready": True, "backend": self.backend, "error": None}

    def close(self) -> None:
        """Закрывает общий PostgreSQL pool; SQLite не держит постоянных ресурсов."""
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def _ensure_pool(self) -> ConnectionPool[Any]:
        """Лениво и ровно один раз открывает pool с dict rows и autocommit."""
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("DatabaseRuntime уже закрыт.")
            if self._pool is None:
                if not self.database_url:
                    raise RuntimeError("Не задан PostgreSQL database URL.")
                self._pool = ConnectionPool(
                    conninfo=self.database_url,
                    min_size=self.pool_min_size,
                    max_size=self.pool_max_size,
                    timeout=self.connect_timeout_seconds,
                    open=False,
                    kwargs={"autocommit": True, "row_factory": dict_row},
                    name="rag-support",
                )
                self._pool.open(wait=True, timeout=self.connect_timeout_seconds)
            return self._pool
