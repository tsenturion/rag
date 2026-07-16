from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent_app.config import (
    AgentAppConfig,
    AgentConfig,
    AgentServiceConfig,
    AgentToolsConfig,
    MemoryConfig,
    MultiAgentConfig,
    MultiAgentProtocolConfig,
)
from agent_app.service.app import create_app
from agent_app.service.runtime import SupportApplicationRuntime


class ServiceStubLLM:
    supports_tool_calling = False

    def invoke(self, messages):
        system = str(getattr(messages[0], "content", "")).casefold()
        if "координатор" in system:
            return AIMessage(content="HTTP 503 временный, HTTP 500 внутренний.")
        if "критик" in system:
            return AIMessage(content="Проверка пройдена.")
        return AIMessage(content="Диагностический отчёт готов.")


def _config(root: Path) -> AgentAppConfig:
    return AgentAppConfig(
        agent=AgentConfig(provider="local", model="test-model"),
        memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
        tools=AgentToolsConfig(
            enabled=["analyze_log_fragment", "build_diagnostic_checklist"],
            incident_sqlite_path=root / "incidents.sqlite",
        ),
        service=AgentServiceConfig(host="127.0.0.1", port=8000),
        multi_agent=MultiAgentConfig(
            enabled=True,
            output_dir=root / "runs",
            checkpoint_path=root / "checkpoints.sqlite",
            mlflow_enabled=False,
            protocols=MultiAgentProtocolConfig(
                a2a_enabled=True,
                mcp_enabled=True,
            ),
        ),
    )


class MultiAgentServiceTest(unittest.TestCase):
    def test_swagger_chat_run_and_agent_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runtime = SupportApplicationRuntime(_config(root), llm=ServiceStubLLM())
            app = create_app(runtime=runtime)
            with TestClient(app, base_url="http://127.0.0.1:8000") as client:
                response = client.post(
                    "/v1/multi-agent/chat",
                    json={
                        "message": "Объясни HTTP 503 и HTTP 500.",
                        "user_id": "engineer",
                        "session_id": "incident",
                    },
                )
                card = client.get("/.well-known/agent-card.json")
                a2a = client.post(
                    "/a2a",
                    headers={"A2A-Version": "1.0"},
                    json={
                        "jsonrpc": "2.0",
                        "id": "request-1",
                        "method": "SendMessage",
                        "params": {
                            "message": {
                                "messageId": "message-1",
                                "contextId": "a2a-context",
                                "role": "ROLE_USER",
                                "parts": [{"text": "Объясни HTTP 503 и HTTP 500."}],
                                "metadata": {
                                    "userId": "engineer",
                                    "sessionId": "incident-a2a",
                                },
                            }
                        },
                    },
                )
                mcp = client.post(
                    "/mcp/",
                    headers={"Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": "initialize-1",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-11-25",
                            "capabilities": {},
                            "clientInfo": {"name": "smoke", "version": "1.0"},
                        },
                    },
                )
                openapi = client.get("/openapi.json")
                metrics = client.get("/metrics")
                run = client.get(f"/v1/multi-agent/runs/{response.json()['run_id']}")
                session = client.get(
                    "/v1/sessions/incident",
                    params={"user_id": "engineer"},
                )
                deleted = client.delete(
                    "/v1/sessions/incident",
                    params={"user_id": "engineer"},
                )
                cleared_session = client.get(
                    "/v1/sessions/incident",
                    params={"user_id": "engineer"},
                )
            runtime.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lifecycle"][-1]["state"], "completed")
        self.assertEqual(card.status_code, 200)
        self.assertEqual(card.json()["name"], "Инженерная мультиагентная система")
        self.assertEqual(a2a.status_code, 200)
        self.assertEqual(
            a2a.json()["result"]["task"]["status"]["state"],
            "TASK_STATE_COMPLETED",
        )
        self.assertEqual(mcp.status_code, 200)
        self.assertEqual(mcp.json()["result"]["serverInfo"]["name"], "Инженерные tools")
        self.assertIn("/v1/multi-agent/chat", openapi.json()["paths"])
        self.assertIn("/a2a", openapi.json()["paths"])
        self.assertIn("/metrics", openapi.json()["paths"])
        self.assertEqual(metrics.status_code, 200)
        self.assertTrue(metrics.headers["content-type"].startswith("text/plain"))
        self.assertIn("support_agent_requests_total", metrics.text)
        self.assertEqual(run.status_code, 200)
        self.assertGreaterEqual(len(session.json()["multi_agent_history"]), 2)
        self.assertTrue(deleted.json()["multi_agent_checkpoint_deleted"])
        self.assertEqual(cleared_session.json()["multi_agent_history"], [])


if __name__ == "__main__":
    unittest.main()
