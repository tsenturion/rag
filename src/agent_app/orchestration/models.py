"""Типизированные модели данных для распределённой оркестрации."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Возвращает текущее время в UTC с информацией о часовом поясе."""
    return datetime.now(timezone.utc)


class OrchestrationPattern(StrEnum):
    """Определяет способы организации выполнения задач в оркестрации, обеспечивая гибкость и адаптивность процесса."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    QUORUM = "quorum"
    DYNAMIC = "dynamic"


class JobPriority(StrEnum):
    """Определяет уровни приоритетов заданий, обеспечивая корректное сопоставление с числовыми значениями для управления порядком обработки в брокере очередей."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

    @property
    def broker_value(self) -> int:
        """Гарантирует сопоставление приоритета задания с числовым значением для корректной работы брокера очередей."""
        return {self.LOW: 1, self.NORMAL: 5, self.HIGH: 9}[self]


class JobStatus(StrEnum):
    """Обозначает жизненный цикл задания, позволяя определить, когда задание завершено и ресурсы можно освободить."""

    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        """Гарантирует определение финального статуса задания для корректного завершения и освобождения ресурсов."""
        return self in {
            self.COMPLETED,
            self.FAILED,
            self.CANCELLED,
            self.EXPIRED,
        }


class StepStatus(StrEnum):
    """Отражает состояние отдельного шага в процессе оркестрации, влияя на логику выполнения и обработку ошибок."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class OrchestrationJob(BaseModel):
    """Определяет контракт задания оркестрации с валидацией и нормализацией, гарантируя корректность и полноту данных для управления процессом."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=100_000)
    pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL
    priority: JobPriority = JobPriority.NORMAL
    risk_level: Literal["low", "medium", "high"] = "medium"
    quorum_size: int = Field(default=2, ge=1, le=3)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=200)
    deadline_at: datetime | None = None
    max_plan_revisions: int = Field(default=2, ge=0, le=10)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def normalize_and_validate(self) -> OrchestrationJob:
        """Обеспечивает корректность и полноту ключевых полей задания, гарантируя валидный и стандартизированный формат для дальнейшей обработки в оркестрации."""
        self.user_id = self.user_id.strip()
        self.session_id = self.session_id.strip()
        self.message = self.message.strip()
        if not self.user_id or not self.session_id or not self.message:
            raise ValueError("user_id, session_id и message не могут быть пустыми")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            self.deadline_at = self.deadline_at.replace(tzinfo=timezone.utc)
        return self

    @property
    def expired(self) -> bool:
        """Определяет, истёк ли срок выполнения задания, позволяя своевременно прекращать или переназначать устаревшие задачи."""
        return self.deadline_at is not None and utc_now() >= self.deadline_at


class PlanStep(BaseModel):
    """Моделирует шаг плана с валидацией параметров, обеспечивая корректное описание и управление зависимостями в оркестрации."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    kind: Literal["validate", "decision", "agent", "aggregate"]
    prompt: str = ""
    assigned_role: str | None = None
    fallback_roles: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    condition: Literal["always", "low_or_medium_risk", "high_risk"] = "always"
    required: bool = True
    timeout_seconds: float = Field(default=60.0, gt=0, le=900)


class ExecutionPlan(BaseModel):
    """Гарантирует непротиворечивую и воспроизводимую структуру зависимостей шагов для корректного исполнения распределённого плана."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    pattern: OrchestrationPattern
    steps: list[PlanStep] = Field(min_length=1)
    reason: str = "Первичный план"
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_graph(self) -> ExecutionPlan:
        """Гарантирует отсутствие циклов и корректность зависимостей между шагами плана, обеспечивая его выполнимость и предсказуемость."""
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Идентификаторы шагов плана должны быть уникальными")
        known: set[str] = set()
        for step in self.steps:
            unknown = sorted(set(step.depends_on) - set(ids))
            if unknown:
                raise ValueError(
                    f"Шаг {step.id} зависит от неизвестных шагов: {', '.join(unknown)}"
                )
            if step.id in step.depends_on:
                raise ValueError(f"Шаг {step.id} не может зависеть от самого себя")
            forward = sorted(set(step.depends_on) - known)
            if forward:
                raise ValueError(
                    f"Шаг {step.id} имеет циклическую или прямую зависимость: "
                    + ", ".join(forward)
                )
            known.add(step.id)
        return self


class StepResult(BaseModel):
    """Фиксирует результат выполнения шага с инвариантом однозначного статуса и полной трассировкой попыток и ошибок."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    status: StepStatus
    output: str = ""
    error: str | None = None
    assigned_role: str | None = None
    vote: Literal["approve", "reject", "abstain"] | None = None
    retryable: bool = False
    attempt: int = Field(default=1, ge=1)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def successful(self) -> bool:
        """Определяет успешное завершение шага, позволяя корректно интерпретировать результаты и принимать решения о дальнейшем выполнении."""
        return self.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}


class PlanRevision(BaseModel):
    """Обеспечивает прозрачную историю изменений плана с указанием причин и затронутых ролей для аудита и отката."""

    from_version: int
    to_version: int
    reason: str
    changed_roles: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SynchronizationResult(BaseModel):
    """Гарантирует однозначную фиксацию достижения кворума и консенсуса между участниками распределённой оркестрации."""

    required: int = 0
    received: int = 0
    successful: int = 0
    quorum_reached: bool = True
    consensus: Literal["approve", "reject", "undetermined"] = "undetermined"
    cancelled_steps: list[str] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
    """Гарантирует целостное описание итогового состояния задания, включая план, шаги, ревизии, синхронизацию и ошибки."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    answer: str = ""
    plan: ExecutionPlan
    step_results: list[StepResult] = Field(default_factory=list)
    revisions: list[PlanRevision] = Field(default_factory=list)
    synchronization: SynchronizationResult = Field(
        default_factory=SynchronizationResult
    )
    duration_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    completed_at: datetime = Field(default_factory=utc_now)


class JobRecord(BaseModel):
    """Фиксирует жизненный цикл задания с полной историей статусов, попыток, результатов и ошибок для мониторинга и восстановления."""

    model_config = ConfigDict(extra="forbid")

    job: OrchestrationJob
    status: JobStatus = JobStatus.QUEUED
    task_id: str | None = None
    attempts: int = Field(default=0, ge=0)
    result: OrchestrationResult | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobEvent(BaseModel):
    """Гарантирует последовательную и воспроизводимую фиксацию событий жизненного цикла задания для аудита и реактивных обработчиков."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(default=0, ge=0)
    job_id: str
    kind: Literal[
        "submitted",
        "started",
        "step",
        "retry",
        "replanned",
        "completed",
        "failed",
        "cancelled",
        "expired",
        "lease_lost",
    ]
    status: JobStatus
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class QueueStatus(BaseModel):
    """Хранит состояние очереди заданий и информацию о рабочих процессах для мониторинга и управления распределённой оркестрацией."""

    backend: Literal["inline", "celery"]
    ready: bool
    status_counts: dict[str, int] = Field(default_factory=dict)
    workers: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JobSubmission(BaseModel):
    """Передаёт зарегистрированное задание и признак повторного запроса с тем же idempotency key."""

    record: JobRecord
    deduplicated: bool = False
