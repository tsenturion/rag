from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_prep.config import EmbeddingConfig, VectorStoreConfig
from rag_prep.config_composition import apply_rag_profile, load_composed_yaml


class AgentConfig(BaseModel):
    provider: Literal["openai", "local", "gigachat"]
    model: str
    adapter_path: Path | None = None
    temperature: float = 0.0
    max_new_tokens: int = Field(default=220, ge=1)
    max_history_messages: int = Field(default=12, ge=2)
    max_summary_chars: int = Field(default=2500, ge=200)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    recursion_limit: int = Field(default=12, ge=4)
    tool_error_retries: int = Field(default=1, ge=0)
    local_device: Literal["auto", "xpu", "cuda", "cpu"] = "auto"
    local_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    gigachat_auth_key_env: str = "GIGACHAT_AUTH_KEY"
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_verify_ssl_certs: bool = False
    gigachat_profanity_check: bool | None = None


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


class AgentRagConfig(BaseModel):
    enabled: bool = False
    top_k: int = Field(default=5, ge=1, le=50)
    max_context_tokens: int = Field(default=1800, ge=128)
    excerpt_chars: int = Field(default=600, ge=80, le=4000)
    tokenizer_model: str | None = None
    require_citations: bool = True
    embedding: EmbeddingConfig | None = None
    vector_store: VectorStoreConfig | None = None

    @model_validator(mode="after")
    def require_runtime_config_when_enabled(self) -> AgentRagConfig:
        if not self.enabled:
            return self
        missing = [
            name
            for name, value in (
                ("tokenizer_model", self.tokenizer_model),
                ("embedding", self.embedding),
                ("vector_store", self.vector_store),
            )
            if value is None
        ]
        if missing:
            raise ValueError("Для включённого RAG явно задайте: " + ", ".join(missing))
        return self


class ExternalMCPServerConfig(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    enabled: bool = True
    required: bool = False
    transport: Literal["stdio", "streamable_http"]
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    env_from_host: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    header_env: dict[str, str] = Field(default_factory=dict)
    verify_ssl: bool = True
    terminate_on_close: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    tool_allowlist: list[str] = Field(default_factory=list)
    tool_prefix: str | None = None
    max_output_chars: int = Field(default=12000, ge=100, le=100000)

    @model_validator(mode="after")
    def validate_transport_parameters(self) -> ExternalMCPServerConfig:
        if self.transport == "streamable_http":
            if not self.url or not self.url.startswith(("http://", "https://")):
                raise ValueError(
                    "Для transport=streamable_http нужен абсолютный HTTP(S) URL"
                )
            if self.command is not None:
                raise ValueError("Поле command допустимо только для transport=stdio")
        else:
            if not self.command:
                raise ValueError("Для transport=stdio обязательно поле command")
            if self.url is not None:
                raise ValueError(
                    "Поле url допустимо только для transport=streamable_http"
                )
            if self.headers or self.header_env:
                raise ValueError(
                    "HTTP-заголовки допустимы только для transport=streamable_http"
                )
        if self.tool_prefix is not None and not self.tool_prefix.strip():
            raise ValueError("tool_prefix не может быть пустым")
        if self.enabled and not self.tool_allowlist:
            raise ValueError(
                "Для включённого внешнего MCP-сервера обязателен tool_allowlist"
            )
        if "*" in self.tool_allowlist and self.tool_allowlist != ["*"]:
            raise ValueError("Значение '*' в tool_allowlist используется отдельно")
        return self


class AgentToolsConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)
    incident_sqlite_path: Path = Path("data/agent/incidents.sqlite")
    max_log_chars: int = Field(default=12000, ge=500, le=100000)
    mcp_servers: list[ExternalMCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_mcp_server_names(self) -> AgentToolsConfig:
        names = [server.name for server in self.mcp_servers]
        if len(names) != len(set(names)):
            raise ValueError("Имена внешних MCP-серверов должны быть уникальными")
        return self


class AgentServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    request_max_chars: int = Field(default=20000, ge=100, le=1000000)
    session_cache_size: int = Field(default=256, ge=1, le=10000)
    cors_origins: list[str] = Field(default_factory=list)


class AgentSecurityConfig(BaseModel):
    require_api_key: bool = False
    api_key_env: str = "SUPPORT_SERVICE_API_KEY"


class MultiAgentCostConfig(BaseModel):
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)


class MultiAgentLLMProfileConfig(AgentConfig):
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)


class MultiAgentProtocolConfig(BaseModel):
    a2a_enabled: bool = True
    a2a_rpc_path: str = "/a2a"
    a2a_rest_path: str = "/a2a/v1"
    mcp_enabled: bool = True
    mcp_path: str = "/mcp"
    acp_legacy_enabled: bool = True

    @field_validator("a2a_rpc_path", "a2a_rest_path", "mcp_path")
    @classmethod
    def require_absolute_url_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("Путь протокола должен начинаться с /")
        return value.rstrip("/") or "/"


class MultiAgentConfig(BaseModel):
    enabled: bool = False
    planner_mode: Literal["rules", "hybrid", "llm"] = "rules"
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    max_tasks: int = Field(default=3, ge=1, le=10)
    max_delegations: int = Field(default=6, ge=1, le=50)
    max_rounds: int = Field(default=2, ge=1, le=10)
    task_timeout_seconds: float = Field(default=45.0, gt=0)
    message_ttl_seconds: float = Field(default=60.0, gt=0)
    token_budget: int = Field(default=12000, ge=256)
    output_dir: Path = Path("data/multi_agent/runs")
    role_tool_allowlists: dict[str, list[str]] = Field(default_factory=dict)
    llm_profiles: dict[str, MultiAgentLLMProfileConfig] = Field(default_factory=dict)
    role_llm_profiles: dict[str, str] = Field(default_factory=dict)
    cost: MultiAgentCostConfig = Field(default_factory=MultiAgentCostConfig)
    protocols: MultiAgentProtocolConfig = Field(
        default_factory=MultiAgentProtocolConfig
    )
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "rag-multi-agent"

    @model_validator(mode="after")
    def validate_budgets(self) -> MultiAgentConfig:
        if self.max_delegations < self.max_tasks:
            raise ValueError(
                "multi_agent.max_delegations не может быть меньше max_tasks"
            )
        if self.message_ttl_seconds < self.task_timeout_seconds:
            raise ValueError(
                "multi_agent.message_ttl_seconds не может быть меньше task_timeout_seconds"
            )
        allowed_roles = {
            "planner",
            "coordinator",
            "critic_agent",
            "knowledge_agent",
            "diagnostics_agent",
            "incident_agent",
        }
        unknown_roles = sorted(set(self.role_llm_profiles) - allowed_roles)
        if unknown_roles:
            raise ValueError(
                "Неизвестные роли в multi_agent.role_llm_profiles: "
                + ", ".join(unknown_roles)
            )
        missing_profiles = sorted(
            set(self.role_llm_profiles.values()) - set(self.llm_profiles)
        )
        if missing_profiles:
            raise ValueError(
                "Не найдены LLM-профили из role_llm_profiles: "
                + ", ".join(missing_profiles)
            )
        invalid_profile_names = sorted(
            name
            for name in self.llm_profiles
            if not name or not name.replace("_", "").replace("-", "").isalnum()
        )
        if invalid_profile_names:
            raise ValueError(
                "Имена LLM-профилей могут содержать буквы, цифры, '_' и '-': "
                + ", ".join(invalid_profile_names)
            )
        return self


class AgentAppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    rag: AgentRagConfig = Field(default_factory=AgentRagConfig)
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    service: AgentServiceConfig = Field(default_factory=AgentServiceConfig)
    security: AgentSecurityConfig = Field(default_factory=AgentSecurityConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    logging: AgentLoggingConfig = Field(default_factory=AgentLoggingConfig)

    @field_validator("memory")
    @classmethod
    def validate_memory_defaults(cls, value: MemoryConfig) -> MemoryConfig:
        if not value.default_user_id.strip():
            raise ValueError("memory.default_user_id не может быть пустым")
        if not value.default_session_id.strip():
            raise ValueError("memory.default_session_id не может быть пустым")
        return value


def load_agent_config(path: str | Path) -> AgentAppConfig:
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    raw = apply_rag_profile(
        load_composed_yaml(config_path),
        config_path=config_path,
        target="agent",
    )

    config = AgentAppConfig.model_validate(raw)
    sqlite_path = config.memory.sqlite_path
    if not sqlite_path.is_absolute():
        sqlite_path = base_dir / sqlite_path
    incident_sqlite_path = config.tools.incident_sqlite_path
    if not incident_sqlite_path.is_absolute():
        incident_sqlite_path = base_dir / incident_sqlite_path
    mcp_servers = []
    for server in config.tools.mcp_servers:
        cwd = server.cwd
        if cwd is not None and not cwd.is_absolute():
            cwd = base_dir / cwd
        mcp_servers.append(
            server.model_copy(
                update={"cwd": cwd.resolve() if cwd is not None else None}
            )
        )
    agent = _resolve_agent_paths(config.agent, base_dir=base_dir)

    multi_agent = config.multi_agent
    llm_profiles = {
        name: _resolve_agent_paths(profile, base_dir=base_dir)
        for name, profile in multi_agent.llm_profiles.items()
    }
    output_dir = multi_agent.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    tracking_uri = multi_agent.mlflow_tracking_uri
    if "://" not in tracking_uri:
        tracking_path = Path(tracking_uri).expanduser()
        if not tracking_path.is_absolute():
            tracking_path = base_dir / tracking_path
        tracking_uri = str(tracking_path.resolve())

    return config.model_copy(
        update={
            "agent": agent,
            "memory": config.memory.model_copy(
                update={"sqlite_path": sqlite_path.resolve()}
            ),
            "tools": config.tools.model_copy(
                update={
                    "incident_sqlite_path": incident_sqlite_path.resolve(),
                    "mcp_servers": mcp_servers,
                }
            ),
            "rag": _resolve_rag_config(config.rag, base_dir=base_dir),
            "multi_agent": multi_agent.model_copy(
                update={
                    "output_dir": output_dir.resolve(),
                    "mlflow_tracking_uri": tracking_uri,
                    "llm_profiles": llm_profiles,
                }
            ),
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


def _resolve_local_reference(value: str, *, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve()) if path.exists() else value
    candidate = base_dir / path
    if candidate.exists():
        return str(candidate.resolve())
    if value.startswith(".") or value.startswith("data/") or value.startswith("data\\"):
        return str(candidate.resolve())
    return value


def _resolve_agent_paths(
    config: AgentConfig,
    *,
    base_dir: Path,
) -> AgentConfig:
    if config.provider != "local":
        return config
    update: dict[str, Any] = {
        "model": _resolve_local_reference(config.model, base_dir=base_dir)
    }
    if config.adapter_path is not None:
        adapter_path = config.adapter_path
        if not adapter_path.is_absolute():
            adapter_path = base_dir / adapter_path
        update["adapter_path"] = adapter_path.resolve()
    return config.model_copy(update=update)


def _resolve_rag_config(config: AgentRagConfig, *, base_dir: Path) -> AgentRagConfig:
    if config.embedding is None or config.vector_store is None:
        return config
    embedding = config.embedding
    embedding_update: dict[str, Any] = {}
    if embedding.env_file is not None:
        env_file = embedding.env_file
        if not env_file.is_absolute():
            env_file = base_dir / env_file
        embedding_update["env_file"] = env_file.resolve()
    if embedding.provider == "local":
        embedding_update["model"] = _resolve_local_reference(
            embedding.model,
            base_dir=base_dir,
        )

    vector_store = config.vector_store
    local_storage_path = vector_store.local_storage_path
    if not local_storage_path.is_absolute():
        local_storage_path = base_dir / local_storage_path
    return config.model_copy(
        update={
            "embedding": embedding.model_copy(update=embedding_update),
            "vector_store": vector_store.model_copy(
                update={"local_storage_path": local_storage_path.resolve()}
            ),
        }
    )
