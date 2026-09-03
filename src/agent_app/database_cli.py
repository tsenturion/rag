"""CLI проверки PostgreSQL и однократного переноса локального SQLite-состояния."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from agent_app.config import AgentAppConfig, load_agent_config
from agent_app.database import DatabaseRuntime
from agent_app.guardrails import HumanReviewStore, SecurityAuditStore
from agent_app.memory import MemoryStore
from agent_app.multi_agent.persistence import MultiAgentCheckpointStore
from agent_app.multi_agent.protocols.a2a_store import A2ATaskStore
from agent_app.support.incidents import IncidentStore


TABLE_SOURCES = (
    ("memory.sqlite_path", "memories", "id"),
    ("memory.sqlite_path", "conversation_history", "user_id, session_id"),
    ("tools.incident_sqlite_path", "incidents", "id"),
    ("guardrails.audit_sqlite_path", "security_audit", "id"),
    ("guardrails.review_sqlite_path", "human_reviews", "id"),
    ("multi_agent.protocols.a2a_task_store_path", "a2a_tasks", "task_id"),
)


def build_parser() -> argparse.ArgumentParser:
    """Описывает команды без запуска приложения и LLM."""
    parser = argparse.ArgumentParser(
        description="Проверка PostgreSQL и миграция состояния из SQLite."
    )
    parser.add_argument("--config", required=True, help="Путь к agent YAML.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Проверить соединение и таблицы.")
    migration = subparsers.add_parser(
        "migrate-sqlite",
        help="Перенести локальные SQLite-данные в PostgreSQL.",
    )
    migration.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Не переносить LangGraph checkpoints.",
    )
    return parser


def _config_value(config: AgentAppConfig, dotted_path: str) -> Path:
    """Читает только заранее объявленный путь SQLite из типизированного конфига."""
    value: Any = config
    for name in dotted_path.split("."):
        value = getattr(value, name)
    return Path(value)


def _initialize_schema(
    config: AgentAppConfig,
    database: DatabaseRuntime,
) -> MultiAgentCheckpointStore:
    """Создаёт все прикладные и LangGraph-таблицы до импорта."""
    MemoryStore(config.memory.sqlite_path, database=database)
    IncidentStore(config.tools.incident_sqlite_path, database=database)
    SecurityAuditStore(config.guardrails.audit_sqlite_path, database=database)
    HumanReviewStore(config.guardrails.review_sqlite_path, database=database)
    a2a_store = A2ATaskStore(
        config.multi_agent.protocols.a2a_task_store_path,
        ttl_seconds=config.multi_agent.protocols.a2a_task_ttl_seconds,
        max_tasks=config.multi_agent.protocols.a2a_max_tasks,
        database=database,
    )
    a2a_store.list()
    return MultiAgentCheckpointStore(
        config.multi_agent.checkpoint_path,
        database=database,
    )


def _copy_table(
    source_path: Path,
    *,
    table: str,
    conflict_columns: str,
    database: DatabaseRuntime,
) -> int:
    """Копирует одну известную таблицу с idempotent upsert по её ключу."""
    if not source_path.is_file():
        return 0
    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    try:
        exists = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if exists is None:
            return 0
        columns = [
            str(row["name"])
            for row in source.execute(f"PRAGMA table_info({table})").fetchall()
        ]
        rows = source.execute(f"SELECT * FROM {table}").fetchall()
    finally:
        source.close()
    if not rows:
        return 0

    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(
        f"{column} = excluded.{column}"
        for column in columns
        if column not in {item.strip() for item in conflict_columns.split(",")}
    )
    statement = (
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict_columns}) DO UPDATE SET {updates}"
    )
    with database.connection(Path(":memory:")) as target:
        for row in rows:
            target.execute(statement, tuple(row[column] for column in columns))
    return len(rows)


def _copy_checkpoints(
    source_path: Path,
    target: MultiAgentCheckpointStore,
) -> int:
    """Переносит историю LangGraph в порядке от старых checkpoints к новым."""
    if not source_path.is_file():
        return 0
    source_connection = sqlite3.connect(
        source_path,
        check_same_thread=False,
        timeout=30,
    )
    source = SqliteSaver(source_connection, serde=target.saver.serde)
    source.setup()
    try:
        checkpoints = list(source.list(None))
        for item in reversed(checkpoints):
            configurable = item.config["configurable"]
            parent_configurable = (
                item.parent_config.get("configurable", {})
                if item.parent_config is not None
                else {}
            )
            put_config: dict[str, Any] = {
                "configurable": {
                    "thread_id": configurable["thread_id"],
                    "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                }
            }
            parent_id = parent_configurable.get("checkpoint_id")
            if parent_id:
                put_config["configurable"]["checkpoint_id"] = parent_id
            saved_config = target.saver.put(
                put_config,
                item.checkpoint,
                item.metadata,
                item.checkpoint.get("channel_versions", {}),
            )
            grouped_writes: dict[str, list[tuple[str, Any]]] = defaultdict(list)
            for task_id, channel, value in item.pending_writes or []:
                grouped_writes[str(task_id)].append((channel, value))
            for task_id, writes in grouped_writes.items():
                target.saver.put_writes(saved_config, writes, task_id)
        return len(checkpoints)
    finally:
        source_connection.close()


def migrate_sqlite(config: AgentAppConfig, *, skip_checkpoints: bool) -> dict[str, Any]:
    """Выполняет повторяемую миграцию всех известных локальных state stores."""
    database = DatabaseRuntime.from_config(config.persistence)
    if not database.is_postgresql:
        database.close()
        raise ValueError("Для миграции нужен config с persistence.backend=postgresql.")
    checkpoint_store = _initialize_schema(config, database)
    counts: dict[str, int] = {}
    try:
        for path_name, table, conflict_columns in TABLE_SOURCES:
            counts[table] = _copy_table(
                _config_value(config, path_name),
                table=table,
                conflict_columns=conflict_columns,
                database=database,
            )
        counts["langgraph_checkpoints"] = (
            0
            if skip_checkpoints
            else _copy_checkpoints(config.multi_agent.checkpoint_path, checkpoint_store)
        )
        return {"backend": database.backend, "migrated": counts}
    finally:
        checkpoint_store.close()
        database.close()


def database_status(config: AgentAppConfig) -> dict[str, Any]:
    """Проверяет backend и число строк в прикладных таблицах."""
    database = DatabaseRuntime.from_config(config.persistence)
    checkpoint_store = _initialize_schema(config, database)
    counts: dict[str, int] = {}
    try:
        with database.connection(Path(":memory:")) as connection:
            for _path_name, table, _key in TABLE_SOURCES:
                row = connection.execute(
                    f"SELECT COUNT(*) AS row_count FROM {table}"
                ).fetchone()
                counts[table] = int(row["row_count"])
        return {**database.ping(), "tables": counts}
    finally:
        checkpoint_store.close()
        database.close()


def main() -> None:
    """Загружает конфиг, выполняет выбранную операцию и печатает JSON-отчёт."""
    args = build_parser().parse_args()
    config = load_agent_config(args.config)
    if args.command == "status":
        result = database_status(config)
    else:
        result = migrate_sqlite(config, skip_checkpoints=args.skip_checkpoints)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
