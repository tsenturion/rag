"""Публичный интерфейс для мультиагентной системы."""

from agent_app.multi_agent.models import (
    MultiAgentComparisonReport,
    MultiAgentResponse,
    MultiAgentRunResult,
)
from agent_app.multi_agent.runtime import MultiAgentRuntime

__all__ = [
    "MultiAgentComparisonReport",
    "MultiAgentResponse",
    "MultiAgentRunResult",
    "MultiAgentRuntime",
]
