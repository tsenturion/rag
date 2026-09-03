"""Проверки persistence backend, общего runtime и MLflow DSN."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest
from pydantic import ValidationError

from agent_app.config import PersistenceConfig, load_agent_config
from agent_app.database import DatabaseRuntime, _postgres_sql
from agent_app.memory import MemoryStore
from agent_app.support.incidents import IncidentStore
from rag_prep.config import load_embedding_config


def test_local_and_docker_configs_select_explicit_backends() -> None:
    """Локальный профиль сохраняет SQLite, а Docker явно выбирает PostgreSQL."""
    local = load_agent_config("config/support_agent_openai.yaml")
    docker = load_agent_config("config/support_agent_docker_openai.yaml")

    assert local.persistence.backend == "sqlite"
    assert docker.persistence.backend == "postgresql"
    assert docker.persistence.database_url_env == "AGENT_DATABASE_URL"


def test_persistence_config_rejects_inverted_pool_bounds() -> None:
    """Некорректные границы pool отклоняются до запуска сервиса."""
    with pytest.raises(ValidationError, match="pool_max_size"):
        PersistenceConfig(pool_min_size=5, pool_max_size=2)


def test_postgresql_runtime_requires_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production backend не начинает работу без отдельной env-переменной DSN."""
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    config = PersistenceConfig(
        backend="postgresql",
        database_url_env="TEST_DATABASE_URL",
    )

    with pytest.raises(RuntimeError, match="TEST_DATABASE_URL"):
        DatabaseRuntime.from_config(config)


def test_shared_sqlite_runtime_serves_memory_and_incidents(tmp_path: Path) -> None:
    """Несколько store используют общий lifecycle, сохраняя локальные файлы отдельно."""
    database = DatabaseRuntime(backend="sqlite")
    memory = MemoryStore(tmp_path / "memory.sqlite", database=database)
    incidents = IncidentStore(tmp_path / "incidents.sqlite", database=database)
    try:
        record = memory.save(user_id="alice", key="city", value="Казань")
        incident = incidents.create(
            user_id="alice",
            session_id="s1",
            title="Ошибка API",
            description="Сервис отвечает 503",
        )

        assert memory.get(record.id, user_id="alice") is not None
        assert incidents.get(incident.id, user_id="alice") is not None
    finally:
        database.close()


def test_postgresql_sql_adapter_handles_types_and_unlimited_offset() -> None:
    """Адаптер меняет только известные различия SQL-диалектов store-классов."""
    statement = _postgres_sql(
        "SELECT payload FROM items WHERE id = ? LIMIT -1 OFFSET ? /* BLOB REAL */"
    )

    assert "id = %s" in statement
    assert "LIMIT ALL OFFSET %s" in statement
    assert "BYTEA DOUBLE PRECISION" in statement


def test_rag_pipeline_accepts_mlflow_backend_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Docker override направляет offline tracking в общий PostgreSQL."""
    tracking_uri = "postgresql+psycopg://rag:secret@postgres:5432/rag"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    config = load_embedding_config("config/embeddings_openai.yaml")

    assert config.logging.mlflow_tracking_uri == tracking_uri


def test_memory_save_resolves_concurrent_unique_key_race(tmp_path: Path) -> None:
    """Одновременное создание scoped key завершается одной обновляемой записью."""
    store = MemoryStore(tmp_path / "memory.sqlite")
    workers = 6
    barrier = threading.Barrier(workers)
    thread_state = threading.local()
    original_find = store.find_by_key

    def synchronized_find(**kwargs):
        """Синхронизирует первые чтения, чтобы воспроизвести гонку INSERT."""
        if not getattr(thread_state, "initial_read_done", False):
            thread_state.initial_read_done = True
            barrier.wait(timeout=5)
            return None
        return original_find(**kwargs)

    store.find_by_key = synchronized_find  # type: ignore[method-assign]
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(
                executor.map(
                    lambda index: store.save(
                        user_id="alice",
                        key="concurrent-key",
                        value=f"value-{index}",
                    ),
                    range(workers),
                )
            )
        stored = store.list_memories(user_id="alice", limit=20)
    finally:
        store.close()

    assert len(records) == workers
    assert len(stored) == 1
    assert stored[0].key == "concurrent-key"
