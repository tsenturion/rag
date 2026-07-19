"""Регрессионные тесты для подсистемы agent_quality."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hypothesis import given, strategies as st

from agent_app.config import AgentAppConfig, AgentConfig, EvaluationConfig
from agent_app.evaluation.metrics import evaluate_output, fact_present
from agent_app.evaluation.models import (
    EvaluationCase,
    EvaluationOutput,
    EvaluationSuite,
)
from agent_app.evaluation.runner import EvaluationRunner, RuntimeEvaluationExecutor


class DeterministicExecutor:
    """Обеспечивает детерминированное выполнение кейсов оценки качества агента, гарантируя воспроизводимость и соответствие эталонному ответу без вариативности."""

    def execute(self, case: EvaluationCase, *, repetition: int) -> EvaluationOutput:
        """Проверяет, что результат выполнения кейса всегда детерминирован и соответствует эталонному ответу для оценки качества агента."""
        del repetition
        return EvaluationOutput(
            answer="Первая реакция занимает 15 минут. Нужно уведомить администратора.",
            selected_roles=["knowledge_agent"],
            citations_count=1,
            latency_ms=25.0,
            estimated_cost=0.01,
        )


def test_evaluation_runner_exports_reproducible_report_and_passes_gates() -> None:
    """Проверяет, что запуск оценки качества экспортирует воспроизводимый отчёт, сохраняет результаты и успешно проходит все установленные пороговые критерии качества."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir) / "evaluation"
        config = AgentAppConfig(
            agent=AgentConfig(provider="local", model="test-model"),
            evaluation=EvaluationConfig(
                output_dir=output_dir,
                repeats=2,
                min_task_success_rate=1.0,
                min_fact_f1=1.0,
                min_consistency=1.0,
                max_p95_latency_ms=100.0,
                max_average_cost=0.02,
                mlflow_enabled=False,
            ),
        )
        suite = EvaluationSuite(
            name="deterministic",
            mode="multi",
            cases=[
                EvaluationCase(
                    id="sla",
                    title="SLA",
                    request="Каков SLA?",
                    expected_facts=["15 минут", "уведомить администратора"],
                    expected_roles=["knowledge_agent"],
                    require_citations=True,
                )
            ],
        )
        report = EvaluationRunner(config, DeterministicExecutor()).run(suite)

        run_dir = Path(report.run_dir or "")
        assert report.quality_gate.passed
        assert report.summary.task_success_rate == 1.0
        assert report.summary.consistency == 1.0
        assert (run_dir / "report.json").exists()
        assert (run_dir / "results.jsonl").exists()
        assert (run_dir / "manifest.json").exists()


def test_russian_inflections_do_not_cause_false_recall_drop() -> None:
    """Проверяет, что морфологические изменения русских слов не приводят к ошибочному снижению метрики полноты фактов в ответах агента."""
    answer = "Укажите краткую тему. Затем заявка переходит в ожидание уточнения."
    assert fact_present("краткая тема", answer)
    assert fact_present("ожидает уточнения", answer)


def test_unlisted_claim_reduces_precision() -> None:
    """Считает произвольное дополнительное утверждение неподтверждённым."""
    result = evaluate_output(
        EvaluationCase(
            id="groundedness",
            title="Groundedness",
            request="Каков SLA?",
            expected_facts=["15 минут"],
        ),
        EvaluationOutput(
            answer=(
                "Первая реакция занимает 15 минут. "
                "Система автоматически перезапускает спутниковый канал."
            )
        ),
        repetition=1,
    )

    assert result.fact_precision == 0.5
    assert result.unsupported_claim_rate == 0.5
    assert result.unsupported_claims == [
        "Система автоматически перезапускает спутниковый канал"
    ]
    assert not result.task_success


def test_runtime_evaluation_uses_unique_user_and_session_namespace() -> None:
    """Не переиспользует память между независимыми evaluation run_id."""
    calls: list[tuple[str, str]] = []

    def ask(*, user_id: str, session_id: str, message: str):
        """Фиксирует memory namespace без запуска реальной LLM."""
        del message
        calls.append((user_id, session_id))
        return SimpleNamespace(answer="15 минут", tool_calls=[], citations=[]), None

    runtime = SimpleNamespace(
        ask=ask,
        config=SimpleNamespace(
            agent=SimpleNamespace(model="stub"),
            multi_agent=SimpleNamespace(
                cost=SimpleNamespace(
                    input_cost_per_million=0.0,
                    output_cost_per_million=0.0,
                    currency="RUB",
                )
            ),
        ),
        currency_converter=None,
        guardrails=SimpleNamespace(
            inspect_output=lambda text: SimpleNamespace(
                text=text,
                action=SimpleNamespace(value="allow"),
                findings=[],
            )
        ),
    )
    usage = SimpleNamespace(
        estimated_cost=0.0,
        estimated_cost_currency="RUB",
        costs_by_currency={},
        estimated_cost_rub=0.0,
        exchange_rates_to_rub={},
        exchange_rate_dates={},
        exchange_rate_source=None,
        exchange_rate_stale=False,
        currency_conversion_errors=[],
        input_tokens=1,
        output_tokens=1,
    )
    executor = RuntimeEvaluationExecutor(runtime, "single")
    case = EvaluationCase(
        id="sla",
        title="SLA",
        request="Каков SLA?",
        expected_facts=["15 минут"],
    )
    with patch("agent_app.evaluation.runner.estimate_mode_usage", return_value=usage):
        executor.begin_run("run-one")
        executor.execute(case, repetition=1)
        executor.begin_run("run-two")
        executor.execute(case, repetition=1)

    assert calls == [
        ("evaluation-run-one", "eval-run-one-sla-1"),
        ("evaluation-run-two", "eval-run-two-sla-1"),
    ]


@given(
    expected_count=st.integers(min_value=1, max_value=10),
    matched_count=st.integers(min_value=0, max_value=10),
)
def test_fact_metrics_are_bounded(expected_count: int, matched_count: int) -> None:
    """Проверяет, что вычисленные метрики точности, полноты, F1 и доли неподдерживаемых утверждений находятся в корректных пределах от 0 до 1 независимо от входных данных."""
    matched_count = min(matched_count, expected_count)
    expected = [f"факт-{index}" for index in range(expected_count)]
    answer = " ".join(expected[:matched_count])
    result = evaluate_output(
        EvaluationCase(
            id="property",
            title="Property",
            request="Проверка",
            expected_facts=expected,
        ),
        EvaluationOutput(answer=answer),
        repetition=1,
    )
    assert 0.0 <= result.fact_precision <= 1.0
    assert 0.0 <= result.fact_recall <= 1.0
    assert 0.0 <= result.fact_f1 <= 1.0
    assert 0.0 <= result.unsupported_claim_rate <= 1.0
