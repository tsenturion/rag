"""Конфигурационные модели и загрузка настроек для PEFT fine-tuning локальной LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_prep.mlflow_uri import resolve_mlflow_tracking_uri


class RunConfig(BaseModel):
    """Гарантирует валидацию и воспроизводимость базовых параметров запуска обучения и оценки."""

    name: str = "local_llm_fine_tuning"
    seed: int = 42


class FineTuningPathConfig(BaseModel):
    """Гарантирует валидацию и согласованность путей к данным, результатам и отчётам для всех этапов обучения."""

    train_jsonl: Path = Path("data/fine_tuning/train.jsonl")
    eval_jsonl: Path = Path("data/fine_tuning/eval.jsonl")
    output_dir: Path = Path("data/fine_tuning/runs")
    adapter_output_dir: Path = Path("data/models/lora")
    reports_dir: Path = Path("data/fine_tuning/reports")
    baseline_report_filename: str = "baseline_report.json"
    tuned_report_filename: str = "tuned_report.json"
    comparison_report_filename: str = "comparison_report.json"
    manifest_filename: str = "manifest.json"


class LocalModelConfig(BaseModel):
    """Базовая локальная модель, tokenizer и параметры размещения на устройстве."""

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model_id: str | None = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_id: str | None = None
    trust_remote_code: bool = True
    max_seq_length: int = Field(default=1024, ge=128)
    device: Literal["auto", "xpu", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    low_cpu_mem_usage: bool = True
    hub_disable_xet: bool = True
    hub_disable_symlink_warning: bool = True


class PeFTConfig(BaseModel):
    """Параметры LoRA/QLoRA-адаптера и целевые линейные слои модели."""

    method: Literal["lora", "qlora"] = "lora"
    r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_rslora: bool = False
    # Эти поля учитываются только для method=qlora; LoRA оставляет базовые веса
    # в dtype, выбранном в LocalModelConfig.
    qlora_load_in_4bit: bool = True
    qlora_quant_type: Literal["nf4", "fp4"] = "nf4"
    qlora_double_quant: bool = True

    @field_validator("target_modules")
    @classmethod
    def target_modules_must_not_be_empty(cls, values: list[str]) -> list[str]:
        """Проверяет, что список целевых модулей для PEFT не пуст и не содержит пустых строк, предотвращая ошибочную конфигурацию."""
        normalized = [value.strip() for value in values if value.strip()]
        if not normalized:
            raise ValueError("target_modules не должен быть пустым")
        return normalized


class FineTuningTrainConfig(BaseModel):
    """Гиперпараметры Trainer, checkpoints и мониторинга обучения."""

    learning_rate: float = Field(default=2e-4, gt=0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    # Effective batch равен batch_size * accumulation * числу устройств; значение
    # по умолчанию подходит для ограниченной памяти одного XPU/GPU.
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    num_train_epochs: float = Field(default=2.0, gt=0)
    max_steps: int = -1
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=5, ge=1)
    eval_steps: int = Field(default=25, ge=1)
    save_steps: int = Field(default=25, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = Field(default=0, ge=0)
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    eval_strategy: Literal["no", "steps", "epoch"] = "steps"
    save_strategy: Literal["no", "steps", "epoch"] = "steps"
    resume_from_checkpoint: str | None = None
    report_to: list[str] = Field(default_factory=lambda: ["none"])
    run_baseline_before_train: bool = True
    run_evaluation_after_train: bool = True


class GenerationConfig(BaseModel):
    """Гарантирует, что параметры генерации текста для локального fine-tuning LLM валидированы и соответствуют ожидаемым ограничениям подсистемы."""

    max_new_tokens: int = Field(default=220, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False


class FineTuningEvaluationConfig(BaseModel):
    """Гарантирует воспроизводимую и контролируемую конфигурацию оценки качества fine-tuning с учётом ограничений на примеры и критерии."""

    max_examples: int | None = Field(default=None, ge=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    required_keywords_case_sensitive: bool = False
    fail_on_failed_criteria: bool = False


class LoggingConfig(BaseModel):
    """Гарантирует корректную настройку логирования и интеграцию с MLflow для отслеживания экспериментов fine-tuning."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment: str = "local-llm-fine-tuning"


class FineTuningPipelineConfig(BaseModel):
    """Строгий корневой контракт воспроизводимого PEFT-запуска."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=RunConfig)
    paths: FineTuningPathConfig = Field(default_factory=FineTuningPathConfig)
    model: LocalModelConfig = Field(default_factory=LocalModelConfig)
    peft: PeFTConfig = Field(default_factory=PeFTConfig)
    training: FineTuningTrainConfig = Field(default_factory=FineTuningTrainConfig)
    evaluation: FineTuningEvaluationConfig = Field(
        default_factory=FineTuningEvaluationConfig
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_fine_tuning_config(path: str | Path) -> FineTuningPipelineConfig:
    """Загружает YAML, разрешает пути от корня проекта и нормализует MLflow URI."""
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    config = FineTuningPipelineConfig.model_validate(raw)
    return _resolve_logging_tracking_uri(
        _resolve_paths(config, base_dir=base_dir),
        base_dir=base_dir,
    )


def _resolve_config_path(path: str | Path) -> Path:
    """Обеспечивает однозначное разрешение пути к конфигурационному файлу независимо от текущего рабочего каталога."""
    config_path = Path(path).expanduser()
    if config_path.is_absolute() or config_path.exists():
        return config_path.resolve()

    project_root = Path(__file__).resolve().parents[2]
    project_config_path = project_root / config_path
    if project_config_path.exists():
        return project_config_path.resolve()

    return config_path.resolve()


def _config_base_dir(config_path: Path) -> Path:
    """Гарантирует определение базовой директории проекта для корректного разрешения относительных путей в конфигурации."""
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_paths(
    config: FineTuningPipelineConfig,
    *,
    base_dir: Path,
) -> FineTuningPipelineConfig:
    """Гарантирует, что все пути к данным и результатам в конфигурации приведены к абсолютным и воспроизводимым значениям относительно базовой директории."""
    paths = config.paths
    resolved = {}
    for name in (
        "train_jsonl",
        "eval_jsonl",
        "output_dir",
        "adapter_output_dir",
        "reports_dir",
    ):
        value = getattr(paths, name)
        resolved[name] = (
            value.resolve() if value.is_absolute() else (base_dir / value).resolve()
        )
    model = config.model
    resolved_model = {
        "model_id": _resolve_model_reference(model.model_id, base_dir=base_dir),
        "tokenizer_id": _resolve_optional_model_reference(
            model.tokenizer_id,
            base_dir=base_dir,
        ),
        "fallback_model_id": _resolve_optional_model_reference(
            model.fallback_model_id,
            base_dir=base_dir,
        ),
    }
    resume_from_checkpoint = config.training.resume_from_checkpoint
    resolved_checkpoint = (
        str(
            (
                Path(resume_from_checkpoint).expanduser()
                if Path(resume_from_checkpoint).expanduser().is_absolute()
                else base_dir / Path(resume_from_checkpoint).expanduser()
            ).resolve()
        )
        if resume_from_checkpoint
        else None
    )
    return config.model_copy(
        update={
            "paths": paths.model_copy(update=resolved),
            "model": model.model_copy(update=resolved_model),
            "training": config.training.model_copy(
                update={"resume_from_checkpoint": resolved_checkpoint}
            ),
        }
    )


def _resolve_optional_model_reference(
    value: str | None,
    *,
    base_dir: Path,
) -> str | None:
    """Гарантирует корректное разрешение необязательных ссылок на модели или их отсутствие без ошибок."""
    if value is None:
        return None
    return _resolve_model_reference(value, base_dir=base_dir)


def _resolve_model_reference(value: str, *, base_dir: Path) -> str:
    """Различает переносимый Hugging Face ID и путь к локальной модели."""
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())

    candidate = base_dir / expanded
    # Несуществующий, но явно локальный путь тоже разрешается от base_dir, чтобы
    # последующая ошибка называла стабильный абсолютный путь, а не зависела от CWD.
    if candidate.exists() or _looks_like_local_path(value):
        return str(candidate.resolve())
    return value


def _looks_like_local_path(value: str) -> bool:
    """Гарантирует определение, является ли строка относительным локальным путём для корректной маршрутизации загрузки ресурсов."""
    normalized = value.replace("\\", "/")
    return normalized.startswith(("./", "../", "data/"))


def _resolve_logging_tracking_uri(
    config: FineTuningPipelineConfig,
    *,
    base_dir: Path,
) -> FineTuningPipelineConfig:
    """Гарантирует, что URI для отслеживания логов приведён к абсолютному и совместимому с MLflow виду относительно базовой директории."""
    logging_config = config.logging.model_copy(
        update={
            "mlflow_tracking_uri": resolve_mlflow_tracking_uri(
                config.logging.mlflow_tracking_uri,
                base_dir=base_dir,
            )
        }
    )
    return config.model_copy(update={"logging": logging_config})
