from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentConfig(BaseModel):
    model: str = "gpt-4.1-nano"
    temperature: float = 0.0
    max_history_messages: int = Field(default=12, ge=2)
    max_summary_chars: int = Field(default=2500, ge=200)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)


class MemoryConfig(BaseModel):
    sqlite_path: Path = Path("data/agent/memory.sqlite")
    default_user_id: str = "default"
    default_session_id: str = "default"
    search_limit: int = Field(default=5, ge=1)


class WeatherConfig(BaseModel):
    api_key_env: str = "OPENWEATHER_API_KEY"
    default_city: str = "Екатеринбург"
    default_units: Literal["standard", "metric", "imperial"] = "metric"
    language: str = "ru"
    timeout_seconds: float = Field(default=10.0, gt=0)


class AgentLoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AgentAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    logging: AgentLoggingConfig = Field(default_factory=AgentLoggingConfig)

    @field_validator("memory")
    @classmethod
    def validate_memory_defaults(cls, value: MemoryConfig) -> MemoryConfig:
        if not value.default_user_id.strip():
            raise ValueError("memory.default_user_id cannot be empty")
        if not value.default_session_id.strip():
            raise ValueError("memory.default_session_id cannot be empty")
        return value


def load_agent_config(path: str | Path = "config/agent.yaml") -> AgentAppConfig:
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    config = AgentAppConfig.model_validate(raw)
    sqlite_path = config.memory.sqlite_path
    if not sqlite_path.is_absolute():
        sqlite_path = base_dir / sqlite_path
    return config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={"sqlite_path": sqlite_path.resolve()}
            )
        }
    )


def _resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path).expanduser()
    if config_path.is_absolute() or config_path.exists():
        return config_path.resolve()

    project_root = Path(__file__).resolve().parents[2]
    project_config_path = project_root / config_path
    if project_config_path.exists():
        return project_config_path.resolve()

    return config_path.resolve()


def _config_base_dir(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent
