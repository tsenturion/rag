"""Регрессионные тесты для подсистемы external_mcp."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from agent_app.config import (
    AgentToolsConfig,
    ExternalMCPServerConfig,
)
from agent_app.tools.mcp_external import ExternalMCPToolManager


class ExternalMCPTest(unittest.TestCase):
    """Проверяет интеграцию и корректность работы внешних MCP-серверов, включая запуск, вызов инструментов и валидацию конфигураций."""

    def test_stdio_server_is_discovered_and_called(self) -> None:
        """Проверяет, что внешний MCP-сервер, запущенный через stdio, корректно обнаруживается, вызывается и возвращает ожидаемые результаты."""
        with tempfile.TemporaryDirectory() as directory:
            server_path = Path(directory) / "external_server.py"
            server_path.write_text(
                """from mcp.server.fastmcp import FastMCP

server = FastMCP("Тестовый внешний сервер")

@server.tool()
def multiply(left: int, right: int) -> int:
    return left * right

server.run(transport="stdio")
""",
                encoding="utf-8",
            )
            server = ExternalMCPServerConfig(
                name="test_server",
                required=True,
                transport="stdio",
                command=sys.executable,
                args=[str(server_path)],
                tool_allowlist=["multiply"],
            )
            manager = ExternalMCPToolManager([server])
            try:
                tools = manager.start()
                self.assertEqual(
                    [tool.name for tool in tools], ["mcp_test_server_multiply"]
                )
                result = json.loads(
                    tools[0].invoke(
                        {
                            "left": 6,
                            "right": 7,
                        }
                    )
                )
                self.assertFalse(result["isError"])
                self.assertEqual(result["structuredContent"]["result"], 42)
                self.assertEqual(manager.status()["connected"], ["test_server"])
            finally:
                manager.close()

    def test_transport_specific_fields_are_validated(self) -> None:
        """Проверяет, что при создании конфигурации сервера с определённым типом транспорта обязательные специфичные поля валидируются и вызывается ошибка при их отсутствии."""
        with self.assertRaises(ValidationError):
            ExternalMCPServerConfig(
                name="invalid",
                transport="streamable_http",
                command="python",
            )

    def test_server_names_must_be_unique(self) -> None:
        """Проверяет, что конфигурация инструментов не допускает дублирование имён серверов, обеспечивая уникальность идентификаторов в системе."""
        server = ExternalMCPServerConfig(
            name="duplicate",
            transport="stdio",
            command="python",
            tool_allowlist=["*"],
        )
        with self.assertRaises(ValidationError):
            AgentToolsConfig(mcp_servers=[server, server])


if __name__ == "__main__":
    unittest.main()
