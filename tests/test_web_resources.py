"""Регрессии browser-маршрутов памяти, инцидентов и исходников базы знаний."""

from __future__ import annotations

import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import (  # noqa: E402
    AgentAppConfig,
    AgentConfig,
    AgentSecurityConfig,
    AgentToolsConfig,
    MemoryConfig,
)
from agent_app.database import DatabaseRuntime  # noqa: E402
from agent_app.memory.store import MemoryStore  # noqa: E402
from agent_app.service.accounts import AccountStore, UserCreate  # noqa: E402
from agent_app.service.app import create_app  # noqa: E402
from agent_app.service.operation_store import OperationRequest, OperationStore  # noqa: E402
from agent_app.service.web_config import WebConfig  # noqa: E402
from agent_app.support.incidents import IncidentStore  # noqa: E402


class ResourceTestRuntime:
    """Подставляет настоящие локальные stores в приложение без зависимостей LLM."""

    def __init__(self, config: AgentAppConfig) -> None:
        """Инициализирует единый SQLite backend для stores и web-маршрутов."""
        self.config = config
        self.database = DatabaseRuntime(backend="sqlite")
        self.memory_store = MemoryStore(
            config.memory.sqlite_path, database=self.database
        )
        self.incident_store = IncidentStore(
            config.tools.incident_sqlite_path, database=self.database
        )
        self.orchestration_service = None

    def close(self) -> None:
        """Закрывает database runtime по окончании теста."""
        self.database.close()


def resource_config(
    root: Path, *, upload_max_bytes: int = 20 * 1024 * 1024
) -> AgentAppConfig:
    """Создаёт минимальную browser-конфигурацию с изолированными файлами состояния."""
    return AgentAppConfig(
        agent=AgentConfig(provider="local", model="test-model"),
        memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
        tools=AgentToolsConfig(incident_sqlite_path=root / "incidents.sqlite"),
        web=WebConfig(
            enabled=True,
            sqlite_path=root / "web.sqlite",
            data_dir=root / "web-data",
            operations_catalog=PROJECT_ROOT / "config" / "web_operations.yaml",
            cookie_secure=True,
            upload_max_bytes=upload_max_bytes,
        ),
        security=AgentSecurityConfig(
            require_api_key=False,
            rate_limit_enabled=False,
            enforce_user_scope=True,
        ),
    )


def add_user(
    config: AgentAppConfig, database: DatabaseRuntime, username: str, roles: list[str]
) -> None:
    """Сохраняет пользователя с настоящим Argon2id-хешем до browser-входа."""
    AccountStore(database, config.web).create(
        UserCreate(
            username=username,
            display_name=username.title(),
            password="correct-password-123",
            roles=roles,
        )
    )


def login(client: TestClient, username: str) -> dict[str, str]:
    """Выполняет вход и формирует CSRF-заголовок для последующих записей."""
    response = client.post(
        "/v1/auth/login",
        json={"username": username, "password": "correct-password-123"},
    )
    assert response.status_code == 200, response.text
    return {"X-CSRF-Token": response.json()["csrf_token"]}


class WebResourcesTest(unittest.TestCase):
    """Проверяет owner isolation и базовые списки browser-ресурсов."""

    def test_memory_incident_and_source_are_isolated_by_browser_owner(self) -> None:
        """Другой инженер не читает, не удаляет и не скачивает ресурсы владельца."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = resource_config(Path(temporary_dir))
            runtime = ResourceTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with (
                TestClient(app, base_url="https://testserver") as alice,
                TestClient(app, base_url="https://testserver") as bob,
            ):
                add_user(config, runtime.database, "alice", ["operator"])
                add_user(config, runtime.database, "bob", ["operator"])
                alice_headers = login(alice, "alice")
                bob_headers = login(bob, "bob")

                memory = alice.post(
                    "/v1/memories",
                    headers=alice_headers,
                    json={"key": "environment", "value": "production"},
                )
                self.assertEqual(memory.status_code, 201)
                memory_id = memory.json()["id"]
                self.assertEqual(
                    alice.get(f"/v1/memories/{memory_id}").status_code, 200
                )
                self.assertEqual(bob.get(f"/v1/memories/{memory_id}").status_code, 404)
                incident = alice.post(
                    "/v1/incidents",
                    headers=alice_headers,
                    json={
                        "session_id": "investigation-1",
                        "title": "Timeout",
                        "description": "Сервис отвечает с задержкой.",
                    },
                )
                self.assertEqual(incident.status_code, 201)
                incident_id = incident.json()["id"]
                source = alice.post(
                    "/v1/knowledge/sources?filename=evidence.txt",
                    headers=alice_headers,
                    content=b"diagnostic evidence",
                )
                self.assertEqual(source.status_code, 201)
                source_id = source.json()["id"]

                self.assertEqual(bob.get("/v1/memories").json()["items"], [])
                self.assertEqual(bob.get("/v1/incidents").json()["items"], [])
                self.assertEqual(bob.get("/v1/knowledge/sources").json(), [])
                self.assertEqual(
                    bob.delete(
                        f"/v1/memories/{memory_id}", headers=bob_headers
                    ).status_code,
                    404,
                )
                self.assertEqual(
                    bob.get(f"/v1/incidents/{incident_id}").status_code, 404
                )
                self.assertEqual(
                    bob.get(f"/v1/knowledge/sources/{source_id}").status_code, 404
                )
                self.assertEqual(
                    bob.delete(
                        f"/v1/knowledge/sources/{source_id}", headers=bob_headers
                    ).status_code,
                    404,
                )
                malformed = alice.post(
                    "/v1/knowledge/sources?filename=..%2Foutside.txt",
                    headers=alice_headers,
                    content=b"must not be stored",
                )
                self.assertEqual(malformed.status_code, 422)
                self.assertEqual(
                    [item["id"] for item in alice.get("/v1/knowledge/sources").json()],
                    [source_id],
                )

                downloaded = alice.get(f"/v1/knowledge/sources/{source_id}")
                self.assertEqual(downloaded.status_code, 200)
                self.assertEqual(downloaded.content, b"diagnostic evidence")
                deleted = alice.delete(
                    f"/v1/knowledge/sources/{source_id}", headers=alice_headers
                )
                self.assertEqual(deleted.status_code, 204)
                self.assertEqual(
                    alice.get(f"/v1/knowledge/sources/{source_id}").status_code, 404
                )
            runtime.close()

    def test_source_upload_enforces_csrf_size_hash_and_pending_delete_guard(
        self,
    ) -> None:
        """Источник не обходится без CSRF, проверяет лимит и не удаляется из pending-операции."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = resource_config(Path(temporary_dir), upload_max_bytes=1024)
            runtime = ResourceTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "alice", ["operator"])
                headers = login(client, "alice")
                missing_csrf = client.post(
                    "/v1/knowledge/sources?filename=notes.txt", content=b"notes"
                )
                self.assertEqual(missing_csrf.status_code, 403)
                oversized = client.post(
                    "/v1/knowledge/sources?filename=large.txt",
                    headers=headers,
                    content=b"x" * 1025,
                )
                self.assertEqual(oversized.status_code, 413)
                self.assertEqual(client.get("/v1/knowledge/sources").json(), [])

                body = b"regression source"
                uploaded = client.post(
                    "/v1/knowledge/sources?filename=notes.txt",
                    headers=headers,
                    content=body,
                )
                self.assertEqual(uploaded.status_code, 201)
                source = uploaded.json()
                self.assertEqual(source["size_bytes"], len(body))
                self.assertEqual(source["sha256"], sha256(body).hexdigest())

                pending = OperationStore(runtime.database, config.web)
                pending.initialize()
                pending.submit(
                    "alice",
                    OperationRequest(
                        profile="prepare",
                        idempotency_key=str(uuid4()),
                        source_ids=[source["id"]],
                    ),
                )
                deleting = client.delete(
                    f"/v1/knowledge/sources/{source['id']}", headers=headers
                )
                self.assertEqual(deleting.status_code, 409)
                self.assertEqual(
                    client.get(f"/v1/knowledge/sources/{source['id']}").content, body
                )
            runtime.close()

    def test_projects_filter_regular_memories_and_paginate_only_owner_records(
        self,
    ) -> None:
        """Список проектов исключает обычную память и не раскрывает проекты другого пользователя."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = resource_config(Path(temporary_dir))
            runtime = ResourceTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with (
                TestClient(app, base_url="https://testserver") as alice,
                TestClient(app, base_url="https://testserver") as bob,
            ):
                add_user(config, runtime.database, "alice", ["engineer"])
                add_user(config, runtime.database, "bob", ["engineer"])
                alice_headers = login(alice, "alice")
                bob_headers = login(bob, "bob")
                regular = alice.post(
                    "/v1/memories",
                    headers=alice_headers,
                    json={"key": "regular", "value": "not a project"},
                )
                self.assertEqual(regular.status_code, 201)
                for project_name in ("Alpha", "Beta"):
                    created = alice.post(
                        "/v1/projects",
                        headers=alice_headers,
                        json={"project_name": project_name, "goal": "Проверка"},
                    )
                    self.assertEqual(created.status_code, 200)
                foreign = bob.post(
                    "/v1/projects",
                    headers=bob_headers,
                    json={"project_name": "Secret", "goal": "Не показывать Alice"},
                )
                self.assertEqual(foreign.status_code, 200)

                first = alice.get("/v1/projects?limit=1")
                self.assertEqual(first.status_code, 200)
                self.assertEqual(len(first.json()["items"]), 1)
                cursor = first.json()["next_cursor"]
                self.assertIsNotNone(cursor)
                second = alice.get(f"/v1/projects?limit=10&cursor={cursor}")
                self.assertEqual(second.status_code, 200)
                records = [*first.json()["items"], *second.json()["items"]]
                self.assertEqual(len(records), 2)
                self.assertEqual(
                    {record["metadata"]["project_name"] for record in records},
                    {"Alpha", "Beta"},
                )
                self.assertTrue(
                    all(record["key"].startswith("project:") for record in records)
                )
            runtime.close()

    def test_projects_use_memory_tools_and_return_cursor_pages(self) -> None:
        """Проект и задачи создаются через tools, обновляются и листаются без потери записей."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = resource_config(Path(temporary_dir))
            runtime = ResourceTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "alice", ["engineer"])
                headers = login(client, "alice")
                project = client.post(
                    "/v1/projects",
                    headers=headers,
                    json={"project_name": "Web backend", "goal": "Защитить API"},
                )
                self.assertEqual(project.status_code, 200)
                self.assertEqual(project.json()["status"], "saved")
                for title in ("Добавить CORS", "Проверить изоляцию"):
                    created = client.post(
                        "/v1/projects/tasks",
                        headers=headers,
                        json={"project_name": "Web backend", "task_title": title},
                    )
                    self.assertEqual(created.status_code, 200)
                    self.assertEqual(created.json()["status"], "saved")
                updated = client.patch(
                    "/v1/projects/tasks",
                    headers=headers,
                    json={
                        "project_name": "Web backend",
                        "task_title": "Добавить CORS",
                        "status": "done",
                    },
                )
                self.assertEqual(updated.status_code, 200)
                self.assertEqual(updated.json()["status"], "updated")
                self.assertEqual(updated.json()["record"]["metadata"]["status"], "done")

                first = client.get("/v1/projects?limit=1")
                self.assertEqual(first.status_code, 200)
                self.assertEqual(len(first.json()["items"]), 1)
                cursor = first.json()["next_cursor"]
                self.assertIsNotNone(cursor)
                second = client.get(f"/v1/projects?limit=10&cursor={cursor}")
                self.assertEqual(second.status_code, 200)
                first_ids = {item["id"] for item in first.json()["items"]}
                second_ids = {item["id"] for item in second.json()["items"]}
                self.assertTrue(second_ids)
                self.assertFalse(first_ids.intersection(second_ids))
                listed = [*first.json()["items"], *second.json()["items"]]
                self.assertEqual(len({item["id"] for item in listed}), 3)
                self.assertIn(
                    "done",
                    [item["metadata"].get("status") for item in listed],
                )
            runtime.close()

    def test_memory_updates_and_invalid_incident_inputs_preserve_existing_state(
        self,
    ) -> None:
        """PATCH меняет память, а невалидные инциденты не создают и не портят записи."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = resource_config(Path(temporary_dir))
            runtime = ResourceTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "alice", ["engineer"])
                headers = login(client, "alice")
                memory = client.post(
                    "/v1/memories",
                    headers=headers,
                    json={"key": "environment", "value": "staging"},
                )
                memory_id = memory.json()["id"]
                patched = client.patch(
                    f"/v1/memories/{memory_id}",
                    headers=headers,
                    json={"value": "production", "importance": 5},
                )
                self.assertEqual(patched.status_code, 200)
                self.assertEqual(patched.json()["value"], "production")
                self.assertEqual(patched.json()["importance"], 5)

                invalid = client.post(
                    "/v1/incidents",
                    headers=headers,
                    json={"session_id": "s1", "title": " ", "description": " "},
                )
                self.assertEqual(invalid.status_code, 422)
                self.assertEqual(client.get("/v1/incidents").json()["items"], [])

                incident = client.post(
                    "/v1/incidents",
                    headers=headers,
                    json={
                        "session_id": "s1",
                        "title": "Timeout",
                        "description": "Нет ответа",
                    },
                )
                incident_id = incident.json()["id"]
                invalid_status = client.patch(
                    f"/v1/incidents/{incident_id}",
                    headers=headers,
                    json={"status": "invalid"},
                )
                self.assertEqual(invalid_status.status_code, 422)
                preserved = client.get(f"/v1/incidents/{incident_id}")
                self.assertEqual(preserved.status_code, 200)
                self.assertEqual(preserved.json()["status"], "open")
            runtime.close()

    def test_bootstrap_and_resource_lists_expose_only_viewer_permissions(self) -> None:
        """Bootstrap отражает browser-вход, а viewer читает списки без прав записи."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = resource_config(Path(temporary_dir))
            runtime = ResourceTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "viewer", ["viewer"])
                bootstrap = client.get("/v1/app/config")
                self.assertEqual(bootstrap.status_code, 200)
                self.assertTrue(
                    bootstrap.json()["authentication"]["browser_session_enabled"]
                )
                self.assertFalse(bootstrap.json()["authentication"]["api_key_enabled"])

                headers = login(client, "viewer")
                principal = client.get("/v1/auth/me")
                self.assertEqual(principal.status_code, 200)
                self.assertEqual(principal.json()["roles"], ["viewer"])
                self.assertIn("memory:read", principal.json()["permissions"])
                self.assertNotIn("memory:write", principal.json()["permissions"])
                integrations = client.get("/v1/integrations")
                self.assertEqual(integrations.status_code, 200)
                self.assertEqual(
                    integrations.json()["llm"],
                    {"provider": "local", "model": "test-model"},
                )
                self.assertEqual(
                    integrations.json()["configuration_mode"], "server_profiles"
                )
                for path in (
                    "/v1/memories",
                    "/v1/incidents",
                    "/v1/projects",
                    "/v1/knowledge/sources",
                    "/v1/operations",
                ):
                    with self.subTest(path=path):
                        self.assertEqual(client.get(path).status_code, 200)
                self.assertEqual(
                    client.get("/v1/orchestration/jobs").status_code,
                    503,
                )
                self.assertEqual(
                    client.post(
                        "/v1/memories",
                        headers=headers,
                        json={"key": "forbidden", "value": "write"},
                    ).status_code,
                    403,
                )
            runtime.close()


if __name__ == "__main__":
    unittest.main()
