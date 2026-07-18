from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OrchestrationPattern(StrEnum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    QUORUM = "quorum"
    DYNAMIC = "dynamic"


class JobPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

    @property
    def broker_value(self) -> int:
        return {self.LOW: 1, self.NORMAL: 5, self.HIGH: 9}[self]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self in {
            self.COMPLETED,
            self.FAILED,
            self.CANCELLED,
            self.EXPIRED,
        }


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class OrchestrationJob(BaseModel):
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
        return self.deadline_at is not None and utc_now() >= self.deadline_at


class PlanStep(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    pattern: OrchestrationPattern
    steps: list[PlanStep] = Field(min_length=1)
    reason: str = "Первичный план"
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_graph(self) -> ExecutionPlan:
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
        return self.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}


class PlanRevision(BaseModel):
    from_version: int
    to_version: int
    reason: str
    changed_roles: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class SynchronizationResult(BaseModel):
    required: int = 0
    received: int = 0
    successful: int = 0
    quorum_reached: bool = True
    consensus: Literal["approve", "reject", "undetermined"] = "undetermined"
    cancelled_steps: list[str] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
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
    ]
    status: JobStatus
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class QueueStatus(BaseModel):
    backend: Literal["inline", "celery"]
    ready: bool
    status_counts: dict[str, int] = Field(default_factory=dict)
    workers: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JobSubmission(BaseModel):
    record: JobRecord
    deduplicated: bool = False
