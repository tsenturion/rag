from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from agent_app.config import AgentConfig


def build_llm(config: AgentConfig) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY не задан в .env или переменных окружения.")
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )
