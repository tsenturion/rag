"""Типизированные модели данных для проверочных сценариев агента."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agent_app.models import AgentResponse, MemoryType, utc_now

ScenarioType = Literal[
    "main",
    "alternative",
    "error",
    "recovery",
    "tool_failure",
    "loop_guard",
]


class MemorySeed(BaseModel):
    """Гарантирует, что каждая запись памяти для сценария содержит валидные ключ, значение, тип, теги, важность и метаданные для инициализации состояния агента."""

    key: str
    value: str
    memory_type: MemoryType = "fact"
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryExpectation(BaseModel):
    """Определяет ожидаемое состояние памяти агента для проверки корректности работы с памятью в сценариях."""

    key_contains: str | None = None
    value_contains: str | None = None
    tag: str | None = None
    memory_type: MemoryType | None = None


class ScenarioCriteria(BaseModel):
    """Формализует набор требований к успешному прохождению сценария, обеспечивая прозрачный контракт проверки результата."""

    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    memory_contains: list[MemoryExpectation] = Field(default_factory=list)
    memory_not_contains: list[MemoryExpectation] = Field(default_factory=list)
    require_memory_created: bool = False
    require_memory_updated: bool = False
    allow_tool_errors: bool = False
    require_loop_guard: bool = False
    forbid_loop_guard: bool = False
    require_citations: bool = False
    forbid_citations: bool = False
    require_retrieval: bool = False
    require_retrieval_success: bool = False
    expected_retrieval_status: Literal["ok", "empty", "unavailable"] | None = None
    min_tool_calls: int = 0
    max_tool_calls: int | None = None


class ScenarioStep(BaseModel):
    """Определяет шаг сценария проверки агента, обеспечивая структуру для описания запроса пользователя, ожидаемого результата и критериев оценки корректности."""

    id: str
    test_case_id: str
    title: str
    user_request: str
    expected_result: str
    action_chain: list[str] = Field(default_factory=list)
    decision_points: list[str] = Field(default_factory=list)
    criteria: ScenarioCriteria = Field(default_factory=ScenarioCriteria)


class AgentScenario(BaseModel):
    """Описывает структуру тестового сценария, гарантируя воспроизводимость условий и критериев проверки для агента."""

    id: str
    test_case_id: str
    title: str
    type: ScenarioType
    goal: str
    user_request: str
    expected_result: str
    llm_role: str = ""
    tools_role: str = ""
    memory_role: str = ""
    action_chain: list[str] = Field(default_factory=list)
    decision_points: list[str] = Field(default_factory=list)
    transition_rules: list[str] = Field(default_factory=list)
    initial_memory: list[MemorySeed] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    steps: list[ScenarioStep]
    pass_criteria: ScenarioCriteria = Field(default_factory=ScenarioCriteria)
    reset_memory: bool = True


class ScenarioSuite(BaseModel):
    """Объединяет набор сценариев агента с параметрами сессии и путём отчёта, гарантируя целостность и удобство управления тестами."""

    default_user_id: str = "mvp_agent_scenarios"
    session_prefix: str = "scenario"
    report_path: Path = Path("data/agent/scenario_report.json")
    scenarios: list[AgentScenario] = Field(min_length=1)


class ScenarioCheck(BaseModel):
    """Фиксирует результат отдельной проверки внутри сценария, обеспечивая прозрачность причин успеха или неудачи шага."""

    name: str
    passed: bool
    details: str = ""


class ScenarioStepResult(BaseModel):
    """Гарантирует воспроизводимый результат выполнения шага сценария с деталями проверки и ответом агента."""

    scenario_id: str
    step_id: str
    test_case_id: str
    title: str
    passed: bool
    checks: list[ScenarioCheck] = Field(default_factory=list)
    response: AgentResponse


class ScenarioResult(BaseModel):
    """Формирует полный отчёт о прохождении сценария, фиксируя все шаги, проверки и итоговое состояние памяти."""

    id: str
    test_case_id: str
    title: str
    type: ScenarioType
    goal: str
    user_request: str
    expected_result: str
    llm_role: str = ""
    tools_role: str = ""
    memory_role: str = ""
    action_chain: list[str] = Field(default_factory=list)
    decision_points: list[str] = Field(default_factory=list)
    transition_rules: list[str] = Field(default_factory=list)
    passed: bool
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)
    step_results: list[ScenarioStepResult] = Field(default_factory=list)
    checks: list[ScenarioCheck] = Field(default_factory=list)
    memory_after: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioRunReport(BaseModel):
    """Гарантирует целостный отчёт о запуске набора сценариев с фиксацией времени, пользователя и итогового статуса."""

    created_at: datetime = Field(default_factory=utc_now)
    config_path: str
    user_id: str
    passed: bool = False
    results: list[ScenarioResult]

    @model_validator(mode="after")
    def empty_report_cannot_pass(self) -> "ScenarioRunReport":
        """Проверяет, что отчёт не может считаться успешным без хотя бы одного результата сценария."""
        if self.passed and not self.results:
            raise ValueError("Пустой набор результатов не может считаться успешным")
        return self
