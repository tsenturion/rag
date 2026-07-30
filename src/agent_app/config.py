"""Конфигурационные модели и загрузка настроек для агентного приложения."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_prep.config import EmbeddingConfig, VectorStoreConfig
from rag_prep.config_composition import apply_rag_profile, load_composed_yaml
from rag_prep.mlflow_uri import resolve_mlflow_tracking_uri


class StrictConfigModel(BaseModel):
    """Запрещает неизвестные поля во всех вложенных секциях конфигурации."""

    model_config = ConfigDict(extra="forbid")


# Базовый single-agent runtime и его общие зависимости.
class AgentConfig(StrictConfigModel):
    """Провайдер, модель и пределы одного агентного LLM runtime."""

    provider: Literal["openai", "local", "gigachat"]
    model: str
    adapter_path: Path | None = None
    temperature: float = 0.0
    max_new_tokens: int = Field(default=220, ge=1)
    max_input_tokens: int = Field(default=4096, ge=128)
    max_history_messages: int = Field(default=12, ge=2)
    max_summary_chars: int = Field(default=2500, ge=200)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    # recursion_limit ограничивает весь LangGraph, а tool_error_retries разрешает
    # только повторы после подтверждённой ошибки конкретного tool.
    recursion_limit: int = Field(default=12, ge=4)
    tool_error_retries: int = Field(default=1, ge=0)
    local_device: Literal["auto", "xpu", "cuda", "cpu"] = "auto"
    local_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    gigachat_auth_key_env: str = "GIGACHAT_AUTH_KEY"
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_verify_ssl_certs: bool = True
    gigachat_profanity_check: bool | None = None


class MemoryConfig(StrictConfigModel):
    """Гарантирует воспроизводимую и валидируемую конфигурацию хранилища пользовательских данных агента."""

    sqlite_path: Path = Path("data/agent/memory.sqlite")
    default_user_id: str = "default"
    default_session_id: str = "default"
    search_limit: int = Field(default=5, ge=1)


class WeatherConfig(StrictConfigModel):
    """Гарантирует валидируемую конфигурацию доступа к погодному API с предсказуемыми параметрами."""

    api_key_env: str = "OPENWEATHER_API_KEY"
    default_city: str = "Екатеринбург"
    default_units: Literal["standard", "metric", "imperial"] = "metric"
    language: str = "ru"
    timeout_seconds: float = Field(default=10.0, gt=0)


class AgentLoggingConfig(StrictConfigModel):
    """Гарантирует согласованную настройку уровня и формата логирования для всех подсистем."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False


class AgentRagConfig(StrictConfigModel):
    """Retrieval-профиль агента с согласованными embedding и vector store."""

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
        """Проверяет, что при включённом RAG заданы все обязательные параметры, предотвращая запуск с некорректной конфигурацией."""
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


class ExternalMCPServerConfig(StrictConfigModel):
    """Безопасное подключение одного внешнего MCP-сервера и allowlist tools."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    enabled: bool = True
    required: bool = False
    transport: Literal["stdio", "streamable_http"]
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    # Секреты перечисляются по именам и читаются из host env; их значения не
    # должны попадать в YAML и экспортируемую конфигурацию.
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
        """Проверяет согласованность и полноту параметров транспорта внешнего MCP-сервера, исключая некорректные комбинации и пропуски."""
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


class AgentToolsConfig(StrictConfigModel):
    """Определяет набор инструментов агента и гарантирует уникальность имён внешних MCP-серверов для предотвращения конфликтов при интеграции."""

    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)
    incident_sqlite_path: Path = Path("data/agent/incidents.sqlite")
    max_log_chars: int = Field(default=12000, ge=500, le=100000)
    mcp_servers: list[ExternalMCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_mcp_server_names(self) -> AgentToolsConfig:
        """Проверяет, что имена всех внешних MCP-серверов уникальны, предотвращая конфликты при маршрутизации."""
        names = [server.name for server in self.mcp_servers]
        if len(names) != len(set(names)):
            raise ValueError("Имена внешних MCP-серверов должны быть уникальными")
        return self


class FileToolsConfig(StrictConfigModel):
    """Sandbox-пределы файловых tools внутри выделенной workspace."""

    enabled: bool = False
    workspace_path: Path = Path("data/agent/workspace")
    allow_write: bool = False
    allow_hidden_files: bool = False
    max_file_bytes: int = Field(default=1_000_000, ge=1_024, le=20_000_000)
    max_list_entries: int = Field(default=200, ge=1, le=2_000)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".log",
            ".py",
        ]
    )

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        """Гарантирует, что список расширений файлов приведён к единому формату и не содержит дубликатов или некорректных значений."""
        normalized = []
        for value in values:
            extension = value.strip().casefold()
            if not extension.startswith(".") or len(extension) < 2:
                raise ValueError("Расширение файла должно начинаться с точки")
            if extension not in normalized:
                normalized.append(extension)
        if not normalized:
            raise ValueError("Нужно разрешить хотя бы одно расширение файла")
        return normalized


class CodeRunnerConfig(StrictConfigModel):
    """Гарантирует корректную интеграцию с внешним сервисом исполнения кода, включая валидацию URL и контроль лимитов на размер и время выполнения."""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:8010"
    api_key_env: str = "CODE_RUNNER_API_KEY"
    timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    max_code_chars: int = Field(default=12_000, ge=100, le=100_000)
    max_output_chars: int = Field(default=12_000, ge=100, le=100_000)

    @field_validator("base_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        """Гарантирует, что base_url для запуска кода всегда является корректным HTTP(S) URL без завершающего слеша."""
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("code_runner.base_url должен быть HTTP(S) URL")
        return normalized


class AgentServiceConfig(StrictConfigModel):
    """Обеспечивает воспроизводимую и безопасную конфигурацию сетевого API агента с контролем числа воркеров, лимитов запросов и CORS."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    request_max_chars: int = Field(default=20000, ge=100, le=1000000)
    session_cache_size: int = Field(default=256, ge=1, le=10000)
    cors_origins: list[str] = Field(default_factory=list)
    public_base_url: str | None = None

    @field_validator("public_base_url")
    @classmethod
    def normalize_public_base_url(cls, value: str | None) -> str | None:
        """Нормализует внешний URL для A2A discovery за proxy или в Docker."""
        if value is None:
            return None
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("service.public_base_url должен быть HTTP(S) URL")
        return normalized


class AgentSecurityConfig(StrictConfigModel):
    """Аутентификация, RBAC и привязка запросов к user scope."""

    require_api_key: bool = False
    api_key_env: str = "SUPPORT_SERVICE_API_KEY"
    api_key_role: Literal["viewer", "engineer", "operator", "admin", "service"] = (
        "service"
    )
    jwt_enabled: bool = False
    jwt_secret_env: str = "SUPPORT_JWT_SECRET"
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_issuer: str = "rag-support"
    jwt_audience: str = "rag-support-api"
    enforce_user_scope: bool = True
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=100_000)
    rate_limit_burst: int = Field(default=10, ge=1, le=10_000)
    rate_limit_backend: Literal["memory", "redis"] = "memory"
    rate_limit_redis_url_env: str = "ORCHESTRATION_REDIS_URL"
    rate_limit_key_prefix: str = "rag-support:rate-limit"
    rate_limit_bucket_ttl_seconds: int = Field(default=3600, ge=60, le=604_800)


class GuardrailsConfig(StrictConfigModel):
    """Гарантирует включение и настройку механизмов защиты от prompt injection, аудита и модерации вывода для повышения безопасности."""

    enabled: bool = True
    block_prompt_injection: bool = True
    redact_sensitive_data: bool = True
    output_review_enabled: bool = True
    audit_sqlite_path: Path = Path("data/agent/security_audit.sqlite")
    review_sqlite_path: Path = Path("data/agent/human_reviews.sqlite")


class ObservabilityConfig(StrictConfigModel):
    """Параметры OTLP export и автоматического инструментирования."""

    enabled: bool = False
    service_name: str = "rag-support-agent"
    environment: str = "local"
    otlp_http_endpoint: str = "http://127.0.0.1:4318"
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    instrument_http_clients: bool = True
    instrument_celery: bool = True

    @field_validator("otlp_http_endpoint")
    @classmethod
    def validate_otlp_endpoint(cls, value: str) -> str:
        """Гарантирует, что otlp_http_endpoint для мониторинга всегда задан как HTTP(S) URL без завершающего слеша."""
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("observability.otlp_http_endpoint должен быть HTTP(S) URL")
        return normalized


class EvaluationConfig(StrictConfigModel):
    """Определяет параметры автоматической оценки качества агента и гарантирует корректную интеграцию с MLflow для отслеживания экспериментов."""

    output_dir: Path = Path("data/evaluation/runs")
    repeats: int = Field(default=2, ge=1, le=10)
    min_task_success_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    min_fact_f1: float = Field(default=0.70, ge=0.0, le=1.0)
    min_consistency: float = Field(default=0.70, ge=0.0, le=1.0)
    max_p95_latency_ms: float = Field(default=120000.0, gt=0)
    max_average_cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Максимальная средняя стоимость одного evaluation-запуска в RUB.",
    )
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment: str = "rag-agent-quality"


class CurrencyConversionConfig(StrictConfigModel):
    """Настраивает получение официальных курсов валют Банка России.

    Конвертация нужна для сопоставимого отображения расходов разных LLM:
    исходная сумма сохраняется в валюте тарифа, а рядом рассчитывается RUB.
    """

    enabled: bool = True
    cbr_daily_rates_url: str = "https://www.cbr.ru/scripts/XML_daily.asp"
    timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    cache_ttl_seconds: int = Field(default=43_200, ge=60, le=86_400)
    allow_stale_on_error: bool = True
    fail_on_error: bool = False

    @field_validator("cbr_daily_rates_url")
    @classmethod
    def require_official_cbr_endpoint(cls, value: str) -> str:
        """Не позволяет подменить источник курса неофициальным сервисом."""
        normalized = value.strip()
        parsed = urlparse(normalized)
        if parsed.scheme != "https" or parsed.hostname not in {"cbr.ru", "www.cbr.ru"}:
            raise ValueError(
                "currency_conversion.cbr_daily_rates_url должен использовать "
                "официальный HTTPS-домен cbr.ru"
            )
        return normalized


# Роли, протоколы и бюджеты мультиагентного runtime.
class MultiAgentCostConfig(StrictConfigModel):
    """Фиксирует контракт расчёта стоимости токенов для мультиагентных сценариев, обеспечивая прозрачность биллинга."""

    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    currency: str = "RUB"

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Нормализует валюту тарифа к трёхбуквенному ISO-коду."""
        return _normalize_currency_code(value)


class MultiAgentLLMProfileConfig(AgentConfig):
    """Гарантирует согласованность профиля LLM с политикой расчёта стоимости и параметрами мультиагентной подсистемы."""

    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    currency: str = "RUB"

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        """Нормализует валюту тарифа конкретного LLM-профиля."""
        return _normalize_currency_code(value)


class MultiAgentProtocolConfig(StrictConfigModel):
    """Определяет включённые протоколы взаимодействия между агентами и гарантирует корректность URL-путей для RPC и REST."""

    a2a_enabled: bool = True
    a2a_rpc_path: str = "/a2a"
    a2a_rest_path: str = "/a2a/v1"
    a2a_task_store_path: Path = Path("data/agent/a2a_tasks.sqlite")
    a2a_task_ttl_seconds: int = Field(default=86_400, ge=60, le=2_592_000)
    a2a_max_tasks: int = Field(default=2_000, ge=10, le=100_000)
    mcp_enabled: bool = True
    mcp_path: str = "/mcp"
    acp_legacy_enabled: bool = True

    @field_validator("a2a_rpc_path", "a2a_rest_path", "mcp_path")
    @classmethod
    def require_absolute_url_path(cls, value: str) -> str:
        """Гарантирует, что путь протокола всегда абсолютный и не содержит лишних завершающих слешей."""
        if not value.startswith("/"):
            raise ValueError("Путь протокола должен начинаться с /")
        return value.rstrip("/") or "/"


class MultiAgentConfig(StrictConfigModel):
    """Границы декомпозиции, маршрутизация ролей и коммуникационные протоколы."""

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
    checkpoint_path: Path = Path("data/agent/multi_agent_checkpoints.sqlite")
    max_history_messages: int = Field(default=12, ge=2, le=100)
    summary_enabled: bool = True
    tool_max_iterations: int = Field(default=4, ge=1, le=12)
    tool_output_max_chars: int = Field(default=12_000, ge=100, le=100_000)
    # Обе карты используют имя роли как ключ: первая ограничивает полномочия,
    # вторая независимо выбирает LLM-провайдера для этой роли.
    role_tool_allowlists: dict[str, list[str]] = Field(default_factory=dict)
    llm_profiles: dict[str, MultiAgentLLMProfileConfig] = Field(default_factory=dict)
    role_llm_profiles: dict[str, str] = Field(default_factory=dict)
    cost: MultiAgentCostConfig = Field(default_factory=MultiAgentCostConfig)
    protocols: MultiAgentProtocolConfig = Field(
        default_factory=MultiAgentProtocolConfig
    )
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment: str = "rag-multi-agent"

    @model_validator(mode="after")
    def validate_budgets(self) -> MultiAgentConfig:
        """Гарантирует согласованность лимитов и профилей ролей в multi_agent-конфигурации, предотвращая некорректные или неразрешимые настройки."""
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
            "tool_agent",
        }
        unknown_roles = sorted(set(self.role_llm_profiles) - allowed_roles)
        if unknown_roles:
            raise ValueError(
                "Неизвестные роли в multi_agent.role_llm_profiles: "
                + ", ".join(unknown_roles)
            )
        unknown_tool_roles = sorted(set(self.role_tool_allowlists) - allowed_roles)
        if unknown_tool_roles:
            raise ValueError(
                "Неизвестные роли в multi_agent.role_tool_allowlists: "
                + ", ".join(unknown_tool_roles)
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


class CamundaConfig(StrictConfigModel):
    """Гарантирует корректную интеграцию с Camunda BPMN-оркестратором, включая параметры процессов и тайминги воркеров."""

    enabled: bool = False
    process_id: str = "engineer-support-process"
    process_path: Path = Path("bpmn/engineer_support.bpmn")
    worker_timeout_seconds: int = Field(default=300, ge=10, le=3600)
    poll_request_timeout_seconds: int = Field(default=5, ge=1, le=25)
    poll_interval_seconds: float = Field(default=1.0, gt=0, le=30)
    job_type_validate: str = "validate-support-request"
    job_type_classify: str = "classify-support-risk"
    job_type_agent: str = "run-support-agent"
    job_type_verify: str = "verify-support-result"


class OrchestrationConfig(StrictConfigModel):
    """Очереди, leases, backpressure и retry распределённого выполнения."""

    enabled: bool = False
    backend: Literal["inline", "celery"] = "inline"
    broker_url_env: str = "ORCHESTRATION_BROKER_URL"
    result_backend_url_env: str = "ORCHESTRATION_RESULT_BACKEND_URL"
    state_store_url_env: str = "ORCHESTRATION_REDIS_URL"
    queue_high: str = "agent.high.quorum"
    queue_default: str = "agent.default.quorum"
    queue_low: str = "agent.low.quorum"
    queue_dead_letter: str = "agent.dead_letter.quorum"
    max_priority: int = Field(default=9, ge=2, le=10)
    max_pending_jobs: int = Field(default=500, ge=1, le=100_000)
    state_ttl_seconds: int = Field(default=86_400, ge=60)
    idempotency_ttl_seconds: int = Field(default=86_400, ge=60)
    event_limit: int = Field(default=500, ge=10, le=10_000)
    max_parallelism: int = Field(default=3, ge=1, le=32)
    worker_concurrency: int = Field(default=2, ge=1, le=64)
    worker_prefetch_multiplier: int = Field(default=1, ge=1, le=16)
    task_soft_time_limit_seconds: int = Field(default=300, ge=10, le=7200)
    task_time_limit_seconds: int = Field(default=330, ge=10, le=7500)
    max_retries: int = Field(default=3, ge=0, le=20)
    retry_backoff_seconds: int = Field(default=5, ge=1, le=600)
    retry_backoff_max_seconds: int = Field(default=120, ge=1, le=3600)
    retry_jitter: bool = True
    # Отдельные лимиты не дают одному внешнему провайдеру или локальному
    # accelerator исчерпать общую конкурентность workers.
    provider_concurrency_limits: dict[str, int] = Field(
        default_factory=lambda: {"openai": 8, "gigachat": 4, "local": 1}
    )
    slot_lease_seconds: int = Field(default=600, ge=30, le=7200)
    eager: bool = False
    camunda: CamundaConfig = Field(default_factory=CamundaConfig)

    @model_validator(mode="after")
    def validate_runtime_limits(self) -> OrchestrationConfig:
        """Гарантирует согласованность временных и параллельных лимитов оркестрации, предотвращая ошибочные или опасные режимы работы."""
        if self.task_soft_time_limit_seconds >= self.task_time_limit_seconds:
            raise ValueError(
                "orchestration.task_soft_time_limit_seconds должен быть меньше "
                "task_time_limit_seconds"
            )
        if self.retry_backoff_seconds > self.retry_backoff_max_seconds:
            raise ValueError(
                "orchestration.retry_backoff_seconds не может быть больше "
                "retry_backoff_max_seconds"
            )
        invalid_limits = {
            name: limit
            for name, limit in self.provider_concurrency_limits.items()
            if not name.strip() or limit < 1
        }
        if invalid_limits:
            raise ValueError(
                "Некорректные provider_concurrency_limits: "
                + ", ".join(f"{key}={value}" for key, value in invalid_limits.items())
            )
        if self.enabled and self.backend == "celery" and self.eager:
            raise ValueError("eager-режим допустим только для backend=inline")
        return self


class AgentAppConfig(StrictConfigModel):
    """Корневой строгий контракт всех подсистем агентного сервиса."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    rag: AgentRagConfig = Field(default_factory=AgentRagConfig)
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    file_tools: FileToolsConfig = Field(default_factory=FileToolsConfig)
    code_runner: CodeRunnerConfig = Field(default_factory=CodeRunnerConfig)
    service: AgentServiceConfig = Field(default_factory=AgentServiceConfig)
    security: AgentSecurityConfig = Field(default_factory=AgentSecurityConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    currency_conversion: CurrencyConversionConfig = Field(
        default_factory=CurrencyConversionConfig
    )
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    logging: AgentLoggingConfig = Field(default_factory=AgentLoggingConfig)

    @field_validator("memory")
    @classmethod
    def validate_memory_defaults(cls, value: MemoryConfig) -> MemoryConfig:
        """Гарантирует, что идентификаторы пользователя и сессии памяти всегда заданы и не пусты."""
        if not value.default_user_id.strip():
            raise ValueError("memory.default_user_id не может быть пустым")
        if not value.default_session_id.strip():
            raise ValueError("memory.default_session_id не может быть пустым")
        return value


def load_agent_config(path: str | Path) -> AgentAppConfig:
    """Создаёт полностью разрешённую и валидированную конфигурацию запуска агента, независимую от текущего рабочего каталога и переменных окружения."""
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
    checkpoint_path = multi_agent.checkpoint_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path
    a2a_task_store_path = multi_agent.protocols.a2a_task_store_path
    if not a2a_task_store_path.is_absolute():
        a2a_task_store_path = base_dir / a2a_task_store_path
    process_path = config.orchestration.camunda.process_path
    if not process_path.is_absolute():
        process_path = base_dir / process_path
    workspace_path = config.file_tools.workspace_path
    if not workspace_path.is_absolute():
        workspace_path = base_dir / workspace_path
    tracking_uri = resolve_mlflow_tracking_uri(
        multi_agent.mlflow_tracking_uri,
        base_dir=base_dir,
    )
    audit_sqlite_path = _resolve_path(config.guardrails.audit_sqlite_path, base_dir)
    review_sqlite_path = _resolve_path(config.guardrails.review_sqlite_path, base_dir)
    evaluation_output_dir = _resolve_path(config.evaluation.output_dir, base_dir)
    evaluation_tracking_uri = resolve_mlflow_tracking_uri(
        config.evaluation.mlflow_tracking_uri,
        base_dir=base_dir,
    )

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
            "file_tools": config.file_tools.model_copy(
                update={"workspace_path": workspace_path.resolve()}
            ),
            "multi_agent": multi_agent.model_copy(
                update={
                    "output_dir": output_dir.resolve(),
                    "checkpoint_path": checkpoint_path.resolve(),
                    "protocols": multi_agent.protocols.model_copy(
                        update={"a2a_task_store_path": a2a_task_store_path.resolve()}
                    ),
                    "mlflow_tracking_uri": tracking_uri,
                    "llm_profiles": llm_profiles,
                }
            ),
            "orchestration": config.orchestration.model_copy(
                update={
                    "camunda": config.orchestration.camunda.model_copy(
                        update={"process_path": process_path.resolve()}
                    )
                }
            ),
            "guardrails": config.guardrails.model_copy(
                update={
                    "audit_sqlite_path": audit_sqlite_path,
                    "review_sqlite_path": review_sqlite_path,
                }
            ),
            "evaluation": config.evaluation.model_copy(
                update={
                    "output_dir": evaluation_output_dir,
                    "mlflow_tracking_uri": evaluation_tracking_uri,
                }
            ),
        }
    )


def _resolve_config_path(path: str | Path) -> Path:
    """Гарантирует, что путь к конфигурационному файлу будет найден относительно домашней директории пользователя или корня проекта, обеспечивая воспроизводимость запуска."""
    config_path = Path(path).expanduser()
    if config_path.is_absolute() or config_path.exists():
        return config_path.resolve()

    project_root = Path(__file__).resolve().parents[2]
    project_config_path = project_root / config_path
    if project_config_path.exists():
        return project_config_path.resolve()

    return config_path.resolve()


def _config_base_dir(config_path: Path) -> Path:
    """Определяет корневую директорию проекта для поиска относительных путей, исключая вложенность в папку config."""
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """Гарантирует, что относительный путь будет преобразован в абсолютный относительно базовой директории, исключая неоднозначность файловых ссылок."""
    return (path if path.is_absolute() else base_dir / path).resolve()


def _resolve_local_reference(value: str, *, base_dir: Path) -> str:
    """Гарантирует, что строковое значение, представляющее путь, будет разрешено в абсолютный путь относительно базовой директории, если это возможно."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve()) if path.exists() else value
    candidate = base_dir / path
    if candidate.exists():
        return str(candidate.resolve())
    if value.startswith(".") or value.startswith("data/") or value.startswith("data\\"):
        return str(candidate.resolve())
    return value


def _normalize_currency_code(value: str) -> str:
    """Проверяет ISO-подобный код валюты, используемый в тарифах LLM."""
    normalized = value.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("Валюта тарифа должна быть трёхбуквенным кодом, например USD")
    return normalized


def _resolve_agent_paths(
    config: AgentConfig,
    *,
    base_dir: Path,
) -> AgentConfig:
    """Обеспечивает, что все пути к моделям и адаптерам локального провайдера приведены к абсолютным, предотвращая ошибки загрузки."""
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
    """Гарантирует, что все пути к моделям, env-файлам и хранилищам в RAG-конфиге приведены к абсолютным, исключая ошибки разрешения файлов."""
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
