from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_app.config import AgentAppConfig
from agent_app.memory.store import SQLiteMemoryStore
from agent_app.tools.calculator import calculator_tool
from agent_app.tools.datetime_tool import datetime_tool
from agent_app.tools.memory_tools import memory_tools
from agent_app.tools.weather import weather_tool


def build_tools(
    config: AgentAppConfig,
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    session_id: str,
) -> list[BaseTool]:
    return [
        calculator_tool(),
        datetime_tool(),
        weather_tool(config.weather),
        *memory_tools(
            store,
            user_id=user_id,
            session_id=session_id,
            default_search_limit=config.memory.search_limit,
        ),
    ]
