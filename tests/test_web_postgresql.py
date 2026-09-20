"""Интеграционные проверки браузерного состояния на настоящем PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
import os
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest
from fastapi import HTTPException

from agent_app.database import DatabaseRuntime
from agent_app.service.accounts import (
    AccountStore,
    LoginRequest,
    UserCreate,
    UserUpdate,
)
from agent_app.service.operation_store import OperationRequest, OperationStore
from agent_app.service.web_config import WebConfig


@pytest.fixture
def pg_web(tmp_path):
    """Изолирует таблицы случайной схемой, не затрагивая рабочие данные пользователя."""
    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Для интеграции задайте TEST_POSTGRES_URL.")
    namespace = "web_test_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
    database = DatabaseRuntime(
        backend="postgresql",
        database_url=make_conninfo(url, options=f"-c search_path={namespace}"),
    )
    config = WebConfig(
        sqlite_path=tmp_path / "unused.sqlite", data_dir=tmp_path / "web"
    )
    try:
        yield database, config
    finally:
        database.close()
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(namespace))
            )


def test_accounts_shared_between_pg_workers_and_revoked(pg_web):
    """Cookie переживает замену API worker, но не смену пароля в другом worker."""
    database, config = pg_web
    first, second = AccountStore(database, config), AccountStore(database, config)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda store: store.initialize(), (first, second)))
    first.create(
        UserCreate(
            username="alice", display_name="Алиса", password="integration-password-123"
        )
    )
    token, session = first.login(
        LoginRequest(username="alice", password="integration-password-123"),
        client_id="local",
    )
    assert second.session(token).user.username == "alice"
    assert second.session(token).csrf_token == session.csrf_token
    second.update("alice", UserUpdate(password="new-integration-password-123"))
    with pytest.raises(HTTPException) as expired:
        first.session(token)
    assert expired.value.status_code == 401


def test_pg_queue_claim_capacity_and_cancellation(pg_web):
    """Параллельные транзакции разных store не дублируют задание и не стирают отмену."""
    database, config = pg_web
    first, second = OperationStore(database, config), OperationStore(database, config)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda store: store.initialize(), (first, second)))
    payload = OperationRequest(profile="prepare", idempotency_key="integration-key")
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = list(
            executor.map(lambda store: store.submit("alice", payload), (first, second))
        )
    assert jobs[0].id == jobs[1].id
    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(
            executor.map(
                lambda pair: pair[0].claim(pair[1], ["prepare"]),
                ((first, "a"), (second, "b")),
            )
        )
    assert sum(job is not None for job in claimed) == 1
    token = "a" if claimed[0] else "b"
    second.cancel(jobs[0].id, "alice")
    first.finish(jobs[0].id, token, status="completed", exit_code=0)
    assert second.get(jobs[0].id).status == "cancelled"
    first.heartbeat("a", ["prepare"])
    assert "prepare" in second.capabilities()


def test_pg_capacity_prevents_concurrent_overflow(pg_web):
    """Лимит общей очереди соблюдается даже при разных ключах и одновременном submit."""
    database, config = pg_web
    config.max_pending_operations = 1
    store = OperationStore(database, config)
    store.initialize()

    def submit(index):
        """Каждый worker пытается занять единственное свободное место."""
        try:
            store.submit(
                "alice",
                OperationRequest(profile="prepare", idempotency_key=f"request-{index}"),
            )
            return 202
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(submit, range(4)))
    assert sorted(statuses) == [202, 429, 429, 429]
