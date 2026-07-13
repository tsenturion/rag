from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import (  # noqa: E402
    AgentAppConfig,
    AgentConfig,
    MemoryConfig,
    WeatherConfig,
)
from agent_app.memory.store import SQLiteMemoryStore  # noqa: E402
from agent_app.tools.calculator import calculate, calculator_tool  # noqa: E402
from agent_app.tools.registry import build_tools  # noqa: E402
from agent_app.tools.travel import travel_tools  # noqa: E402
from agent_app.tools.weather import weather_tool  # noqa: E402


class ToolsTest(unittest.TestCase):
    def test_calculator_and_travel_tools_execute_structured_inputs(self) -> None:
        self.assertEqual(calculate("(128 * 47) + 4"), "6020")
        self.assertIn("ошибка калькулятора", calculate("__import__('os')"))
        self.assertEqual(calculator_tool().invoke({"expression": "7 ** 2"}), "49")

        budget_tool = {tool.name: tool for tool in travel_tools()}[
            "calculate_travel_budget"
        ]
        budget = json.loads(
            budget_tool.invoke(
                {
                    "city": "Казань",
                    "days": 2,
                    "hotel_per_night": 4000,
                    "meals_per_day": 1000,
                    "transport_total": 2000,
                    "extra_per_day": 500,
                }
            )
        )
        self.assertEqual(budget["total"], 9000.0)

    def test_weather_tool_reports_missing_key_without_network_request(self) -> None:
        config = WeatherConfig(api_key_env="MISSING_WEATHER_KEY")
        with patch.dict(os.environ, {}, clear=True):
            result = json.loads(weather_tool(config).invoke({"city": "Пермь"}))
        self.assertEqual(result["error"], "missing_api_key")

    def test_registry_contains_operational_and_memory_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=Path(temporary_dir) / "memory.sqlite"),
            )
            store = SQLiteMemoryStore(config.memory.sqlite_path)
            names = {
                tool.name
                for tool in build_tools(
                    config,
                    store,
                    user_id="user",
                    session_id="session",
                )
            }

        self.assertTrue(
            {
                "calculator",
                "current_datetime",
                "get_weather",
                "save_memory",
                "search_memory",
                "calculate_travel_budget",
            }.issubset(names)
        )


if __name__ == "__main__":
    unittest.main()
