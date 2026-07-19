"""Регрессионные тесты для подсистемы support_service."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import (  # noqa: E402
    AgentConfig,
    AgentAppConfig,
    AgentSecurityConfig,
    AgentServiceConfig,
    AgentToolsConfig,
    MemoryConfig,
)
from agent_app.service.app import create_app  # noqa: E402
from agent_app.service.runtime import SupportApplicationRuntime  # noqa: E402


class SimpleModel:
    """Обеспечивает базовую модель для тестирования, гарантирующую стабильный ответ без вызова инструментов, чтобы проверить корректность работы сервиса."""

    supports_tool_calling = False

    def invoke(self, _messages):
        """Проверяет, что сервис возвращает корректный ответ без ошибок при типовом вызове модели."""
        return AIMessage(content="Сервисный ответ")


class BlockingMultiAgentRuntime:
    """Имитирует sync LLM-вызов, продолжающий работать после отмены async wrapper."""

    def __init__(self) -> None:
        """Создаёт события для наблюдения начала, завершения и закрытия runtime."""
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def ask(self, **_kwargs):
        """Удерживает execution lock до явного разрешения тестом."""
        self.started.set()
        self.release.wait(timeout=2)
        return object()

    def close(self) -> None:
        """Фиксирует момент освобождения общего multi-agent runtime."""
        self.closed.set()


class SupportServiceTest(unittest.TestCase):
    """Проверяет ключевые аспекты работы сервиса поддержки, включая маршруты API, безопасность, жизненный цикл сессий и обработку запросов."""

    def test_openapi_documents_routes_examples_and_api_key_security(self) -> None:
        """Проверяет, что документация OpenAPI доступна, содержит примеры, корректно описывает безопасность через API-ключ и возвращает успешные HTTP-статусы."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                tools=AgentToolsConfig(incident_sqlite_path=root / "incidents.sqlite"),
                security=AgentSecurityConfig(
                    require_api_key=True,
                    api_key_env="TEST_SUPPORT_API_KEY",
                ),
            )
            runtime = SupportApplicationRuntime(config, llm=SimpleModel())
            app = create_app(runtime=runtime)
            with TestClient(app) as client:
                docs = client.get("/docs")
                redoc = client.get("/redoc")
                openapi = client.get("/openapi.json")
            runtime.close()

        schema = openapi.json()
        self.assertEqual(docs.status_code, 200)
        self.assertEqual(redoc.status_code, 200)
        self.assertEqual(openapi.status_code, 200)
        self.assertEqual(schema["info"]["title"], "ИИ-агент поддержки инженера")
        self.assertIn("SupportApiKey", schema["components"]["securitySchemes"])
        self.assertEqual(
            schema["components"]["securitySchemes"]["SupportApiKey"]["in"],
            "header",
        )
        self.assertIn(
            {"SupportApiKey": []},
            schema["paths"]["/v1/chat"]["post"]["security"],
        )
        self.assertIn(
            "examples",
            schema["components"]["schemas"]["ChatRequest"],
        )
        self.assertIn(
            "text/event-stream",
            schema["paths"]["/v1/chat/stream"]["post"]["responses"]["200"]["content"],
        )

    def test_health_auth_chat_stream_metrics_and_session_lifecycle(self) -> None:
        """Проверяет, что сервис корректно отвечает на запросы здоровья, аутентификации, потокового чата, метрик и управляет жизненным циклом сессий с учётом безопасности."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                tools=AgentToolsConfig(incident_sqlite_path=root / "incidents.sqlite"),
                service=AgentServiceConfig(request_max_chars=1000),
                security=AgentSecurityConfig(
                    require_api_key=True,
                    api_key_env="TEST_SUPPORT_API_KEY",
                ),
            )
            runtime = SupportApplicationRuntime(config, llm=SimpleModel())
            app = create_app(runtime=runtime)
            with patch.dict(os.environ, {"TEST_SUPPORT_API_KEY": "service-secret"}):
                with TestClient(app) as client:
                    self.assertEqual(client.get("/health").status_code, 200)
                    self.assertEqual(client.get("/ready").status_code, 200)
                    payload = {
                        "message": "Ответь кратко",
                        "user_id": "engineer",
                        "session_id": "session-1",
                    }
                    self.assertEqual(
                        client.post("/v1/chat", json=payload).status_code, 401
                    )
                    headers = {"X-API-Key": "service-secret"}
                    chat = client.post("/v1/chat", json=payload, headers=headers)
                    stream = client.post(
                        "/v1/chat/stream", json=payload, headers=headers
                    )
                    session = client.get(
                        "/v1/sessions/session-1?user_id=engineer",
                        headers=headers,
                    )
                    metrics_without_key = client.get("/metrics")
                    metrics = client.get("/metrics", headers=headers)
                    deleted = client.delete(
                        "/v1/sessions/session-1?user_id=engineer",
                        headers=headers,
                    )

            runtime.close()

        self.assertEqual(chat.status_code, 200)
        self.assertEqual(chat.json()["answer"], "Сервисный ответ")
        self.assertTrue(chat.json()["request_id"])
        self.assertIn("event: started", stream.text)
        self.assertIn("event: result", stream.text)
        self.assertEqual(session.status_code, 200)
        self.assertEqual(metrics_without_key.status_code, 401)
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("support_agent_requests_total", metrics.text)
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["runner_removed"])

    def test_message_limit_returns_413(self) -> None:
        """Проверяет, что при превышении максимального размера сообщения сервис возвращает HTTP-статус 413, обеспечивая ограничение на размер входящих данных."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                tools=AgentToolsConfig(incident_sqlite_path=root / "incidents.sqlite"),
                service=AgentServiceConfig(request_max_chars=100),
            )
            runtime = SupportApplicationRuntime(config, llm=SimpleModel())
            with TestClient(create_app(runtime=runtime)) as client:
                response = client.post(
                    "/v1/chat",
                    json={
                        "message": "x" * 101,
                        "user_id": "engineer",
                        "session_id": "session",
                    },
                )
            runtime.close()

        self.assertEqual(response.status_code, 413)

    def test_close_waits_for_inflight_multi_agent_call(self) -> None:
        """Не закрывает LLM registry и SQLite, пока sync-вызов реально работает."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                tools=AgentToolsConfig(incident_sqlite_path=root / "incidents.sqlite"),
            )
            runtime = SupportApplicationRuntime(config, llm=SimpleModel())
            blocking = BlockingMultiAgentRuntime()
            runtime.multi_agent_runtime = blocking  # type: ignore[assignment]
            request = threading.Thread(
                target=runtime.ask_multi,
                kwargs={
                    "user_id": "engineer",
                    "session_id": "shutdown",
                    "message": "Диагностика",
                },
            )
            request.start()
            self.assertTrue(blocking.started.wait(timeout=1))
            shutdown = threading.Thread(target=runtime.close)
            shutdown.start()
            time.sleep(0.05)

            self.assertTrue(shutdown.is_alive())
            self.assertFalse(blocking.closed.is_set())

            blocking.release.set()
            request.join(timeout=2)
            shutdown.join(timeout=2)

        self.assertFalse(request.is_alive())
        self.assertFalse(shutdown.is_alive())
        self.assertTrue(blocking.closed.is_set())


if __name__ == "__main__":
    unittest.main()
