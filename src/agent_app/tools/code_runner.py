from __future__ import annotations

import json
import os
from typing import Any

import httpx2
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.config import CodeRunnerConfig
from agent_app.support.security import redact_secrets


class ExecutePythonInput(BaseModel):
    code: str = Field(
        min_length=1,
        description=(
            "Самодостаточный Python-код. Результат нужно явно вывести через print()."
        ),
    )


def code_runner_tool(config: CodeRunnerConfig) -> StructuredTool | None:
    if not config.enabled:
        return None

    def execute_python(code: str) -> str:
        if len(code) > config.max_code_chars:
            return _json(
                {
                    "status": "too_large",
                    "actual": len(code),
                    "limit": config.max_code_chars,
                }
            )
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            return _json(
                {
                    "status": "unavailable",
                    "error": f"Не задана переменная {config.api_key_env}.",
                }
            )
        try:
            with httpx2.Client(timeout=config.timeout_seconds + 2) as client:
                response = client.post(
                    f"{config.base_url}/v1/execute",
                    headers={"X-Code-Runner-Key": api_key},
                    json={
                        "code": code,
                        "timeout_seconds": config.timeout_seconds,
                        "max_output_chars": config.max_output_chars,
                    },
                )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return _json(
                {
                    "status": "unavailable",
                    "error": redact_secrets(str(exc))[:500],
                }
            )
        return _json(payload)

    return StructuredTool.from_function(
        name="execute_python",
        description=(
            "Выполнить вычислительный Python-код в отдельном изолированном "
            "контейнере без доступа к сети и workspace агента. Импорты и опасные "
            "встроенные функции ограничены."
        ),
        func=execute_python,
        args_schema=ExecutePythonInput,
    )


def code_runner_status(config: CodeRunnerConfig) -> dict[str, Any]:
    if not config.enabled:
        return {"enabled": False, "ready": True, "url": config.base_url}
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        return {
            "enabled": True,
            "ready": False,
            "url": config.base_url,
            "error": f"Не задана переменная {config.api_key_env}.",
        }
    try:
        with httpx2.Client(timeout=min(config.timeout_seconds, 3.0)) as client:
            response = client.get(f"{config.base_url}/health")
        response.raise_for_status()
        payload = response.json()
        return {
            "enabled": True,
            "ready": payload.get("status") == "ok",
            "url": config.base_url,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "ready": False,
            "url": config.base_url,
            "error": redact_secrets(str(exc))[:500],
        }


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
