"""Типизированные модели данных для PEFT fine-tuning локальной LLM."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    """Возвращает текущее время в UTC с информацией о часовом поясе."""
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    """Обеспечивает корректность сообщений в диалоге, гарантируя отсутствие пустого или некорректного текста для обучения модели."""

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        """Гарантирует, что сообщение не содержит пустого или пробельного текста, предотвращая некорректные данные в обучающей выборке."""
        text = value.strip()
        if not text:
            raise ValueError("content не должен быть пустым")
        return text


class FineTuningExample(BaseModel):
    """Гарантирует, что обучающий пример содержит минимум одно user-сообщение и завершается assistant-ответом, обеспечивая пригодность для fine-tuning."""

    id: str
    messages: list[ChatMessage]
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("messages")
    @classmethod
    def messages_must_end_with_assistant(
        cls,
        values: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Гарантирует корректную структуру обучающего примера: минимум два сообщения, последнее — assistant, и хотя бы одно — user."""
        if len(values) < 2:
            raise ValueError(
                "пример должен содержать минимум user и assistant сообщения"
            )
        if values[-1].role != "assistant":
            raise ValueError(
                "последнее сообщение в обучающем примере должно быть assistant"
            )
        if not any(message.role == "user" for message in values):
            raise ValueError("пример должен содержать хотя бы одно user сообщение")
        return values


class DatasetStats(BaseModel):
    """Гарантирует воспроизводимую статистику датасета для оценки пригодности и качества данных для обучения и валидации."""

    path: Path
    examples_count: int
    max_prompt_chars: int
    max_answer_chars: int
    avg_prompt_chars: float
    avg_answer_chars: float
    ids_are_unique: bool


class DatasetValidationResult(BaseModel):
    """Гарантирует целостность и разделимость обучающей и валидационной выборок, а также корректность идентификаторов примеров."""

    train: DatasetStats
    eval: DatasetStats
    train_eval_id_overlap_count: int
    max_seq_length: int
    estimated_train_tokens_max: int
    estimated_eval_tokens_max: int

    @property
    def has_errors(self) -> bool:
        """Проверяет, что валидация датасета выявила дубли, пересечения или другие критические ошибки, влияющие на корректность обучения."""
        return (
            not self.train.ids_are_unique
            or not self.eval.ids_are_unique
            or self.train_eval_id_overlap_count > 0
        )


class DeviceReport(BaseModel):
    """Гарантирует прозрачное описание выбранного устройства и его возможностей для корректной инициализации вычислений."""

    requested_device: str
    selected_device: str
    torch_version: str
    xpu_available: bool
    cuda_available: bool
    selected_dtype: str
    notes: list[str] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    """Гарантирует воспроизводимую структуру результата генерации с проверкой прохождения по ключевым словам и соответствию ожиданиям."""

    example_id: str
    prompt: str
    expected_answer: str
    generated_answer: str
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    passed_required_keywords: bool = True
    passed_forbidden_keywords: bool = True
    passed: bool = True


class EvaluationReport(BaseModel):
    """Гарантирует полный отчёт о качестве модели на тестовой выборке с фиксированными метриками и детализацией по каждому примеру."""

    run_id: str
    model_id: str
    adapter_path: Path | None = None
    examples_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    eval_loss: float | None = None
    perplexity: float | None = None
    generated_at: datetime = Field(default_factory=utc_now)
    answers: list[GeneratedAnswer] = Field(default_factory=list)


class TrainingMetrics(BaseModel):
    """Гарантирует вызывающему коду прозрачный контракт по ключевым метрикам и параметрам процесса обучения для последующего анализа и аудита."""

    train_loss: float | None = None
    eval_loss: float | None = None
    train_runtime: float | None = None
    train_samples_per_second: float | None = None
    train_steps_per_second: float | None = None
    global_step: int = 0
    trainable_parameters: int = 0
    total_parameters: int = 0
    trainable_ratio: float = 0.0


class TrainingResult(BaseModel):
    """Фиксирует воспроизводимый результат запуска обучения, включая идентификаторы, метрики, артефакты и временные метки для трассировки и автоматизации."""

    run_id: str
    model_id: str
    method: str
    adapter_path: Path
    trainer_output_dir: Path
    metrics: TrainingMetrics
    log_history: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime = Field(default_factory=utc_now)


class ComparisonResult(BaseModel):
    """Обеспечивает однозначную интерпретацию эффекта обучения через сравнение метрик и примеров между базовой и дообученной моделью."""

    run_id: str
    baseline_report_path: Path
    tuned_report_path: Path
    examples_count: int
    baseline_pass_rate: float
    tuned_pass_rate: float
    pass_rate_delta: float
    improved_examples: list[str] = Field(default_factory=list)
    regressed_examples: list[str] = Field(default_factory=list)
    unchanged_failed_examples: list[str] = Field(default_factory=list)
    conclusion: str


class FineTuningExportResult(BaseModel):
    """Гарантирует целостность и доступность путей к экспортированным артефактам дообучения для последующего использования или публикации."""

    manifest_path: Path
    baseline_report_path: Path | None = None
    tuned_report_path: Path | None = None
    comparison_report_path: Path | None = None
    adapter_path: Path | None = None
    run_id: str

    def artifact_paths(self) -> list[Path]:
        """Возвращает только реально существующие пути к артефактам, гарантируя их доступность для последующей обработки."""
        paths = [
            self.manifest_path,
            self.baseline_report_path,
            self.tuned_report_path,
            self.comparison_report_path,
        ]
        return [path for path in paths if path is not None and path.exists()]


class FineTuningPipelineResult(BaseModel):
    """Объединяет результаты обучения, оценки, сравнения и экспорта в итог одного запуска fine-tuning."""

    run_id: str
    dataset_validation: DatasetValidationResult
    device: DeviceReport
    training: TrainingResult | None = None
    baseline: EvaluationReport | None = None
    tuned: EvaluationReport | None = None
    comparison: ComparisonResult | None = None
    export: FineTuningExportResult


class LocalGenerationResult(BaseModel):
    """Гарантирует воспроизводимость локальной генерации ответа LLM с фиксацией параметров, устройства и времени генерации."""

    model_id: str
    adapter_path: Path | None = None
    prompt: str
    answer: str
    device: DeviceReport
    max_new_tokens: int
    generated_at: datetime = Field(default_factory=utc_now)
