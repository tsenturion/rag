from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_app.config import AgentAppConfig
from agent_app.memory.store import SQLiteMemoryStore
from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.support.incidents import IncidentStore
from agent_app.tools.calculator import calculator_tool
from agent_app.tools.datetime_tool import datetime_tool
from agent_app.tools.memory_tools import memory_tools
from agent_app.tools.project import project_tools
from agent_app.tools.support import support_tools
from agent_app.tools.travel import travel_tools
from agent_app.tools.weather import weather_tool


def build_tools(
    config: AgentAppConfig,
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    session_id: str,
    rag_runtime: OnlineRagRuntime | None = None,
    incident_store: IncidentStore | None = None,
    external_tools: list[BaseTool] | None = None,
) -> list[BaseTool]:
    tools: list[BaseTool] = [
        calculator_tool(),
        datetime_tool(),
        weather_tool(config.weather),
        *travel_tools(),
        *project_tools(
            store,
            user_id=user_id,
            session_id=session_id,
        ),
        *memory_tools(
            store,
            user_id=user_id,
            session_id=session_id,
            default_search_limit=config.memory.search_limit,
        ),
        *(external_tools or []),
    ]
    support_tool_names = {
        "search_knowledge_base",
        "find_runbook",
        "analyze_log_fragment",
        "create_incident",
        "get_incident",
        "update_incident_status",
        "list_incidents",
        "build_diagnostic_checklist",
    }
    enabled = set(config.tools.enabled)
    if config.rag.enabled or enabled.intersection(support_tool_names):
        effective_incident_store = incident_store or IncidentStore(
            config.tools.incident_sqlite_path
        )
        tools.extend(
            support_tools(
                rag_runtime=rag_runtime,
                incident_store=effective_incident_store,
                user_id=user_id,
                session_id=session_id,
                max_log_chars=config.tools.max_log_chars,
            )
        )

    names = [tool.name for tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Совпали имена локальных и внешних tools: " + ", ".join(duplicates)
        )

    disabled = set(config.tools.disabled)
    if enabled:
        tools = [tool for tool in tools if tool.name in enabled]
    return [tool for tool in tools if tool.name not in disabled]
