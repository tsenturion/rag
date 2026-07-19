"""Регрессионные тесты для подсистемы agent_quality."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, strategies as st

from agent_app.config import AgentAppConfig, AgentConfig, EvaluationConfig
from agent_app.evaluation.metrics import evaluate_output, fact_present
from agent_app.evaluation.models import (
    EvaluationCase,
    EvaluationOutput,
    EvaluationSuite,
)
from agent_app.evaluation.runner import EvaluationRunner


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
