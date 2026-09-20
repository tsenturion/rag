"""Параметры браузерных сессий и управляемых длительных операций."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class WebConfig(BaseModel):
    """Отделяет браузерный вход и worker от конфигурации конкретного LLM."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    sqlite_path: Path = Path("data/agent/web.sqlite")
    cookie_name: str = Field(default="rag_session", pattern=r"^[A-Za-z0-9_]+$")
    cookie_secure: bool = True
    session_ttl_seconds: int = Field(default=28800, ge=300, le=2592000)
    login_attempts: int = Field(default=8, ge=1, le=100)
    login_window_seconds: int = Field(default=900, ge=60, le=86400)
    operations_catalog: Path = Path("config/web_operations.yaml")
    data_dir: Path = Path("data/web")
    upload_max_bytes: int = Field(default=20971520, ge=1024, le=104857600)
    operation_timeout_seconds: int = Field(default=3600, ge=10, le=86400)
    max_pending_operations: int = Field(default=100, ge=1, le=10000)
    worker_lease_seconds: int = Field(default=60, ge=15, le=300)
