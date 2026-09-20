"""Регрессии браузерных учётных записей, сессий и разграничения доступа."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import (  # noqa: E402
    AgentAppConfig,
    AgentConfig,
    AgentSecurityConfig,
    AgentServiceConfig,
    AgentToolsConfig,
    MemoryConfig,
)
from agent_app.database import DatabaseRuntime  # noqa: E402
from agent_app.memory.store import MemoryStore  # noqa: E402
from agent_app.service.accounts import (  # noqa: E402
    AccountStore,
    UserCreate,
    UserUpdate,
    digest,
)
from agent_app.service.app import create_app  # noqa: E402
from agent_app.service.web_config import WebConfig  # noqa: E402
from agent_app.support.incidents import IncidentStore  # noqa: E402


class WebTestRuntime:
    """Предоставляет маршрутам реальные SQLite-хранилища без запуска LLM runtime."""

    def __init__(self, config: AgentAppConfig) -> None:
        """Создаёт общий SQLite runtime и пользовательские persistent stores."""
        self.config = config
        self.database = DatabaseRuntime(backend="sqlite")
        self.memory_store = MemoryStore(
            config.memory.sqlite_path, database=self.database
        )
        self.incident_store = IncidentStore(
            config.tools.incident_sqlite_path, database=self.database
        )

    def close(self) -> None:
        """Освобождает общий runtime после завершения теста."""
        self.database.close()


def web_config(
    root: Path, *, login_attempts: int = 8, cors_origins: list[str] | None = None
) -> AgentAppConfig:
    """Собирает изолированную web-конфигурацию с защищённой cookie."""
    return AgentAppConfig(
        agent=AgentConfig(provider="local", model="test-model"),
        memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
        tools=AgentToolsConfig(incident_sqlite_path=root / "incidents.sqlite"),
        web=WebConfig(
            enabled=True,
            sqlite_path=root / "web.sqlite",
            data_dir=root / "web-data",
            operations_catalog=root / "operations.yaml",
            cookie_secure=True,
            login_attempts=login_attempts,
            login_window_seconds=60,
        ),
        security=AgentSecurityConfig(
            require_api_key=False,
            rate_limit_enabled=False,
        ),
        service=AgentServiceConfig(cors_origins=cors_origins or []),
    )


def add_user(
    config: AgentAppConfig, database: DatabaseRuntime, username: str, roles: list[str]
) -> None:
    """Создаёт тестовую учётную запись через настоящее хранилище аккаунтов."""
    AccountStore(database, config.web).create(
        UserCreate(
            username=username,
            display_name=username.title(),
            password="correct-password-123",
            roles=roles,
        )
    )


def login(client: TestClient, username: str) -> str:
    """Выполняет браузерный вход и возвращает выданный CSRF-токен."""
    response = client.post(
        "/v1/auth/login",
        json={"username": username, "password": "correct-password-123"},
    )
    assert response.status_code == 200, response.text
    return response.json()["csrf_token"]


class WebAccountsTest(unittest.TestCase):
    """Проверяет browser login, отзыв сессий, лимиты и admin-ограничения."""

    def test_login_session_restore_csrf_and_logout_revoke_cookie(self) -> None:
        """Cookie имеет secure HttpOnly атрибуты, восстанавливается и отзывается logout."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = web_config(Path(temporary_dir))
            runtime = WebTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "alice", ["engineer"])
                authenticated = client.post(
                    "/v1/auth/login",
                    json={"username": "alice", "password": "correct-password-123"},
                )
                self.assertEqual(authenticated.status_code, 200)
                csrf = authenticated.json()["csrf_token"]
                set_cookie = client.cookies.get(config.web.cookie_name)
                response = client.get("/v1/auth/session")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["csrf_token"], csrf)
                self.assertIn("HttpOnly", authenticated.headers["set-cookie"])
                self.assertIn("Secure", authenticated.headers["set-cookie"])
                self.assertIn("SameSite=lax", authenticated.headers["set-cookie"])

                with TestClient(app, base_url="https://testserver") as restored:
                    restored.cookies.set(config.web.cookie_name, set_cookie)
                    restored_session = restored.get("/v1/auth/session")
                    self.assertEqual(restored_session.status_code, 200)
                    self.assertEqual(restored_session.json()["csrf_token"], csrf)

                rejected = client.post("/v1/auth/logout")
                self.assertEqual(rejected.status_code, 403)
                logged_out = client.post(
                    "/v1/auth/logout", headers={"X-CSRF-Token": csrf}
                )
                self.assertEqual(logged_out.status_code, 204)
                self.assertEqual(client.get("/v1/auth/session").status_code, 401)
            runtime.close()

    def test_auth_me_rejects_unauthenticated_and_supplied_disabled_api_key(
        self,
    ) -> None:
        """Произвольный API-key не включает обход browser-аутентификации."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = web_config(Path(temporary_dir))
            runtime = WebTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                self.assertEqual(client.get("/v1/auth/me").status_code, 401)
                response = client.get(
                    "/v1/auth/me", headers={"X-API-Key": "not-a-login"}
                )
                self.assertEqual(response.status_code, 401)
                resource_response = client.get(
                    "/v1/memories", headers={"X-API-Key": "not-a-login"}
                )
                self.assertEqual(resource_response.status_code, 401)
            runtime.close()

    def test_cors_preflight_allows_csrf_credentials_and_rejects_foreign_login(
        self,
    ) -> None:
        """Разрешённый frontend получает CSRF preflight, а чужой origin не входит в аккаунт."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            frontend_origin = "https://frontend.example.test"
            config = web_config(Path(temporary_dir), cors_origins=[frontend_origin])
            runtime = WebTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                preflight = client.options(
                    "/v1/memories",
                    headers={
                        "Origin": frontend_origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type, x-csrf-token",
                    },
                )
                self.assertEqual(preflight.status_code, 200)
                self.assertEqual(
                    preflight.headers["access-control-allow-origin"], frontend_origin
                )
                self.assertEqual(
                    preflight.headers["access-control-allow-credentials"], "true"
                )
                self.assertIn(
                    "x-csrf-token",
                    preflight.headers["access-control-allow-headers"].lower(),
                )

                add_user(config, runtime.database, "alice", ["engineer"])
                denied = client.post(
                    "/v1/auth/login",
                    headers={"Origin": "https://foreign.example.test"},
                    json={"username": "alice", "password": "correct-password-123"},
                )
                self.assertEqual(denied.status_code, 403)
                self.assertIsNone(client.cookies.get(config.web.cookie_name))
            runtime.close()

    def test_expired_session_row_rejects_cookie_without_waiting(self) -> None:
        """Просроченная серверная запись немедленно отключает сохранённую browser cookie."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = web_config(Path(temporary_dir))
            runtime = WebTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "alice", ["engineer"])
                login(client, "alice")
                token = client.cookies.get(config.web.cookie_name)
                with runtime.database.connection(config.web.sqlite_path) as connection:
                    connection.execute(
                        "UPDATE web_sessions SET expires_at = 0 WHERE token_hash = ?",
                        (digest(token),),
                    )
                response = client.get("/v1/auth/session")
                self.assertEqual(response.status_code, 401)
            runtime.close()

    def test_login_rate_limit_and_role_change_revoke_existing_session(self) -> None:
        """Лимит входа срабатывает до проверки пароля, а изменение роли отзывает cookie."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = web_config(Path(temporary_dir), login_attempts=2)
            runtime = WebTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "alice", ["engineer"])
                csrf = login(client, "alice")
                self.assertTrue(csrf)
                first = client.post(
                    "/v1/auth/login",
                    json={"username": "unknown", "password": "incorrect-password-123"},
                )
                self.assertEqual(first.status_code, 401)
                limited = client.post(
                    "/v1/auth/login",
                    json={"username": "unknown", "password": "incorrect-password-123"},
                )
                self.assertEqual(limited.status_code, 429)
                self.assertIn("Retry-After", limited.headers)

                AccountStore(runtime.database, config.web).update(
                    "alice", UserUpdate(roles=["viewer"])
                )
                self.assertEqual(client.get("/v1/auth/me").status_code, 401)
            runtime.close()

    def test_admin_routes_require_admin_and_prevent_self_revocation(self) -> None:
        """Только admin управляет пользователями и не может снять собственную admin-роль."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = web_config(Path(temporary_dir))
            runtime = WebTestRuntime(config)
            app = create_app(runtime=runtime)  # type: ignore[arg-type]
            with TestClient(app, base_url="https://testserver") as client:
                add_user(config, runtime.database, "admin", ["admin"])
                add_user(config, runtime.database, "engineer", ["engineer"])
                engineer_csrf = login(client, "engineer")
                self.assertTrue(engineer_csrf)
                self.assertEqual(client.get("/v1/admin/users").status_code, 403)

                client.cookies.clear()
                csrf = login(client, "admin")
                created = client.post(
                    "/v1/admin/users",
                    headers={"X-CSRF-Token": csrf},
                    json={
                        "username": "viewer",
                        "display_name": "Viewer",
                        "password": "correct-password-123",
                        "roles": ["viewer"],
                    },
                )
                self.assertEqual(created.status_code, 201)
                users = client.get("/v1/admin/users")
                self.assertEqual(users.status_code, 200)
                self.assertIn("viewer", [item["username"] for item in users.json()])
                self.assertNotIn("password_hash", users.text)
                rejected = client.patch(
                    "/v1/admin/users/admin",
                    headers={"X-CSRF-Token": csrf},
                    json={"roles": ["viewer"]},
                )
                self.assertEqual(rejected.status_code, 409)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
