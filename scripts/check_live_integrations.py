"""Реализация компонентов для вспомогательных сценариев проекта."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import AgentAppConfig, load_agent_config  # noqa: E402
from agent_app.graph import AgentRunner  # noqa: E402
from agent_app.tools.weather import weather_tool  # noqa: E402

TARGETS = {
    "core": ("openai", "gigachat", "openweather"),
    "all": ("openai", "gigachat", "openweather", "huggingface"),
    "openai": ("openai",),
    "gigachat": ("gigachat",),
    "openweather": ("openweather",),
    "huggingface": ("huggingface",),
}
SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "GIGACHAT_AUTH_KEY",
    "OPENWEATHER_API_KEY",
    "HF_TOKEN",
)
TOKEN_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"hf_[A-Za-z0-9_-]+"),
)


def utc_now() -> str:
    """Возвращает текущее время в UTC с информацией о часовом поясе."""
    return datetime.now(timezone.utc).isoformat()


def require_secret(name: str) -> str:
    """Гарантирует наличие обязательного секрета окружения и аварийно завершает выполнение при его отсутствии."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Не задан обязательный Actions secret: {name}")
    return value


def sanitize_error(error: BaseException) -> str:
    """Гарантирует, что сообщения об ошибках не содержат секретных значений и токенов, защищая их от утечки в логи и отчёты."""
    message = str(error)
    for name in SECRET_ENV_NAMES:
        value = os.getenv(name)
        if value:
            message = message.replace(value, f"<{name}:redacted>")
    for pattern in TOKEN_PATTERNS:
        message = pattern.sub("<token:redacted>", message)
    return message[:1000]


def isolated_config(config_path: Path, temporary_dir: Path) -> AgentAppConfig:
    """Создаёт полностью изолированную конфигурацию агента с временными путями и отключёнными интеграциями для безопасного тестирования."""
    temporary_dir.mkdir(parents=True, exist_ok=True)
    config = load_agent_config(config_path)
    return config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={"sqlite_path": temporary_dir / "memory.sqlite"}
            ),
            "tools": config.tools.model_copy(
                update={"incident_sqlite_path": temporary_dir / "incidents.sqlite"}
            ),
            "file_tools": config.file_tools.model_copy(
                update={"workspace_path": temporary_dir / "workspace"}
            ),
            "guardrails": config.guardrails.model_copy(
                update={
                    "audit_sqlite_path": temporary_dir / "security_audit.sqlite",
                    "review_sqlite_path": temporary_dir / "human_reviews.sqlite",
                }
            ),
            "rag": config.rag.model_copy(update={"enabled": False}),
            "multi_agent": config.multi_agent.model_copy(update={"enabled": False}),
            "orchestration": config.orchestration.model_copy(update={"enabled": False}),
            "observability": config.observability.model_copy(update={"enabled": False}),
        }
    )


def check_agent_provider(
    *,
    provider: str,
    secret_name: str,
    config_path: Path,
    temporary_dir: Path,
) -> dict[str, Any]:
    """Проверяет, что агент с заданным провайдером и секретом успешно отвечает на запрос, гарантируя работоспособность внешнего API."""
    require_secret(secret_name)
    config = isolated_config(config_path, temporary_dir / provider)
    runner = AgentRunner(
        config,
        user_id=f"github-actions-{provider}",
        session_id="live-api-smoke",
    )
    try:
        response = runner.ask(
            "Это проверка доступности API. Ответь одним коротким предложением "
            "на русском языке и не вызывай инструменты."
        )
    finally:
        runner.close()

    answer = response.answer.strip()
    if not answer:
        raise RuntimeError(f"{provider}: API вернул пустой ответ")
    if answer.startswith("Ошибка выполнения агента:"):
        raise RuntimeError(f"{provider}: AgentRunner вернул ошибку провайдера")
    return {
        "provider": provider,
        "model": config.agent.model,
        "answer_chars": len(answer),
        "tool_calls": response.tool_calls,
    }


def check_openweather(config_path: Path) -> dict[str, Any]:
    """Проверяет, что интеграция с OpenWeatherMap возвращает валидные погодные данные без ошибок для заданного города."""
    require_secret("OPENWEATHER_API_KEY")
    config = load_agent_config(config_path)
    payload = json.loads(weather_tool(config.weather).invoke({"city": "Екатеринбург"}))
    if payload.get("error"):
        raise RuntimeError(
            "OpenWeatherMap: " + str(payload.get("message") or payload["error"])
        )
    if payload.get("temperature") is None:
        raise RuntimeError("OpenWeatherMap: в ответе отсутствует температура")
    return {
        "city": payload.get("city"),
        "country": payload.get("country"),
        "temperature_received": True,
        "units": payload.get("units"),
    }


def check_huggingface() -> dict[str, Any]:
    """Проверяет, что токен Hugging Face позволяет аутентифицировать пользователя через публичный API."""
    token = require_secret("HF_TOKEN")
    from huggingface_hub import whoami

    identity = whoami(token=token)
    if not isinstance(identity, dict) or not identity.get("name"):
        raise RuntimeError("Hugging Face Hub не подтвердил пользователя токена")
    return {"authenticated": True}


def run_check(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Выполняет одну live-проверку и фиксирует её длительность, результат и безопасное описание ошибки."""
    started = time.perf_counter()
    try:
        details = operation()
    except Exception as exc:
        return {
            "name": name,
            "passed": False,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": sanitize_error(exc),
        }
    return {
        "name": name,
        "passed": True,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    """Гарантирует корректное получение и валидацию параметров запуска для сценариев проверки внешних API."""
    parser = argparse.ArgumentParser(
        description="Проверяет реальные внешние API без вывода секретов."
    )
    parser.add_argument("--target", choices=TARGETS, default="core")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "live-api-report.json",
    )
    return parser.parse_args()


def main() -> int:
    """Запускает командный интерфейс и возвращает код завершения."""
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    with tempfile.TemporaryDirectory(prefix="rag-live-api-") as temporary:
        temporary_dir = Path(temporary)
        operations: dict[str, Callable[[], dict[str, Any]]] = {
            "openai": lambda: check_agent_provider(
                provider="openai",
                secret_name="OPENAI_API_KEY",
                config_path=PROJECT_ROOT / "config" / "agent_openai.yaml",
                temporary_dir=temporary_dir,
            ),
            "gigachat": lambda: check_agent_provider(
                provider="gigachat",
                secret_name="GIGACHAT_AUTH_KEY",
                config_path=PROJECT_ROOT / "config" / "agent_gigachat.yaml",
                temporary_dir=temporary_dir,
            ),
            "openweather": lambda: check_openweather(
                PROJECT_ROOT / "config" / "agent_openai.yaml"
            ),
            "huggingface": check_huggingface,
        }
        results = [run_check(name, operations[name]) for name in TARGETS[args.target]]

    report = {
        "target": args.target,
        "started_at": started_at,
        "finished_at": utc_now(),
        "passed": all(result["passed"] for result in results),
        "checks": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
