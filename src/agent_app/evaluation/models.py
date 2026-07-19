"""Типизированные модели данных для оценки качества агентной системы."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    """Возвращает текущее время в UTC с информацией о часовом поясе."""
    return datetime.now(timezone.utc)


class EvaluationCase(BaseModel):
    """Определяет структуру и обязательные проверки для кейса оценки, гарантируя наличие хотя бы одного критерия для объективной проверки качества работы агентной системы."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    title: str
    request: str
    expected_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    expected_roles: list[str] = Field(default_factory=list)
    require_citations: bool = False
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_verifiable_expectation(self) -> EvaluationCase:
        """Гарантирует, что кейс содержит хотя бы одно проверяемое ожидание, иначе инициирует ошибку валидации."""
        if not any(
            (
                self.expected_facts,
                self.forbidden_facts,
                self.expected_tools,
                self.expected_roles,
                self.require_citations,
            )
        ):
            raise ValueError("Evaluation case должен содержать проверяемое ожидание")
        return self


class EvaluationSuite(BaseModel):
    """Объединяет набор кейсов оценки в единую коллекцию с указанием режима работы, обеспечивая организацию и последовательное выполнение тестов качества агентной системы."""

    name: str
    description: str = ""
    mode: Literal["single", "multi"] = "multi"
    cases: list[EvaluationCase] = Field(min_length=1)


class EvaluationOutput(BaseModel):
    """Фиксирует контракт вывода агента для оценки: гарантирует наличие всех признаков, необходимых для автоматической проверки качества."""

    answer: str
    tool_calls: list[str] = Field(default_factory=list)
    selected_roles: list[str] = Field(default_factory=list)
    citations_count: int = 0
    latency_ms: float = 0.0
    estimated_cost: float = 0.0
    estimated_cost_currency: str = "RUB"
    costs_by_currency: dict[str, float] = Field(default_factory=dict)
    estimated_cost_rub: float | None = None
    exchange_rates_to_rub: dict[str, float] = Field(default_factory=dict)
    exchange_rate_dates: dict[str, str] = Field(default_factory=dict)
    exchange_rate_source: str | None = None
    exchange_rate_stale: bool = False
    currency_conversion_errors: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    guardrail_action: str = "allow"
    guardrail_findings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_rub_cost_for_rub_amount(self) -> EvaluationOutput:
        """Синхронизирует legacy-поле с RUB, если результат уже задан в рублях."""
        self.estimated_cost_currency = self.estimated_cost_currency.upper()
        if not self.costs_by_currency and self.estimated_cost:
            self.costs_by_currency = {self.estimated_cost_currency: self.estimated_cost}
        if self.estimated_cost_currency == "RUB" and self.estimated_cost_rub is None:
            self.estimated_cost_rub = self.estimated_cost
        return self


class EvaluationExecutor(Protocol):
    """Обеспечивает интерфейс для выполнения оценки кейса с гарантией возврата структурированного результата, позволяя реализовать различные стратегии оценки."""

    def execute(self, case: EvaluationCase, *, repetition: int) -> EvaluationOutput:
        """Выполняет один повтор evaluation-кейса и возвращает наблюдаемый результат."""
        ...


class EvaluationCaseResult(BaseModel):
    """Фиксирует полный контракт результата проверки одного тест-кейса, включая все метрики и соответствие ожиданиям."""

    case_id: str
    repetition: int
    output: EvaluationOutput
    task_success: bool
    fact_precision: float
    fact_recall: float
    fact_f1: float
    unsupported_claim_rate: float
    citations_ok: bool
    tools_ok: bool
    roles_ok: bool
    matched_expected_facts: list[str] = Field(default_factory=list)
    matched_forbidden_facts: list[str] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    """Гарантирует вызывающему коду агрегированные метрики качества и стоимости выполнения тестовой сессии для последующего анализа или автоматизации."""

    executions: int
    task_success_rate: float
    fact_precision: float
    fact_recall: float
    fact_f1: float
    consistency: float
    average_latency_ms: float
    p95_latency_ms: float
    average_cost: float
    total_cost: float
    cost_currency: Literal["RUB"] = "RUB"
    average_cost_rub: float | None = None
    total_cost_rub: float | None = None
    currency_conversion_complete: bool = True
    unconverted_cost_count: int = 0
    human_reviews_pending: int = 0


class QualityGateResult(BaseModel):
    """Гарантирует однозначную оценку прохождения контрольных порогов качества и предоставляет детализацию по каждому критерию."""

    passed: bool
    checks: dict[str, bool]


class EvaluationReport(BaseModel):
    """Фиксирует полный контракт результатов оценки, включая идентификатор запуска, параметры среды, метрики, подробности кейсов и итоговое решение о качестве."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utc_now)
    suite_name: str
    mode: Literal["single", "multi"]
    provider: str
    model: str
    config_hash: str
    results: list[EvaluationCaseResult] = Field(min_length=1)
    summary: EvaluationSummary
    quality_gate: QualityGateResult
    run_dir: str | None = None
