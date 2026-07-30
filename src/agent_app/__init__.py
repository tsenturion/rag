"""Модуль агента с tools и памятью."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_app.config import AgentAppConfig, load_agent_config
    from agent_app.graph import AgentRunner

__all__ = ["AgentAppConfig", "AgentRunner", "load_agent_config"]


def __getattr__(name: str):
    """Загружает публичные объекты пакета только в момент фактического обращения."""
    if name in {"AgentAppConfig", "load_agent_config"}:
        from agent_app import config

        return getattr(config, name)
    if name == "AgentRunner":
        from agent_app import graph

        return graph.AgentRunner
    raise AttributeError(name)
