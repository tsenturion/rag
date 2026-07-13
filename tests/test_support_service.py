from __future__ import annotations

import os
import sys
import tempfile
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
    supports_tool_calling = False

    def invoke(self, _messages):
        return AIMessage(content="Сервисный ответ")


class SupportServiceTest(unittest.TestCase):
    def test_openapi_documents_routes_examples_and_api_key_security(self) -> None:
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
                    metrics = client.get("/metrics")
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
        self.assertIn("support_agent_requests_total", metrics.text)
        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["runner_removed"])

    def test_message_limit_returns_413(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
