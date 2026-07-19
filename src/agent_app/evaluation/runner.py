"""Запуск воспроизводимых сценариев для оценки качества агентной системы."""

from __future__ import annotations

import hashlib
import json
from time import perf_counter

import mlflow

from agent_app.config import AgentAppConfig
from agent_app.evaluation.exporting import EvaluationExporter
from agent_app.evaluation.metrics import evaluate_output, summarize
from agent_app.evaluation.models import (
    EvaluationCase,
    EvaluationExecutor,
    EvaluationOutput,
    EvaluationReport,
    EvaluationSuite,
    QualityGateResult,
)
from agent_app.guardrails import HumanReviewStore
from agent_app.multi_agent.usage import estimate_mode_usage
from agent_app.service.runtime import SupportApplicationRuntime


class RuntimeEvaluationExecutor:
    """Реализует выполнение оценки кейсов в заданном режиме, обеспечивая контроль времени, стоимости и безопасности ответов в рамках жизненного цикла рантайма."""

    def __init__(self, runtime: SupportApplicationRuntime, mode: str):
        """Готовит экземпляр к запуску оценки, устанавливая режим работы и связывая с жизненным циклом тестируемого рантайма."""
        self.runtime = runtime
        self.mode = mode

    def execute(self, case: EvaluationCase, *, repetition: int) -> EvaluationOutput:
        """Гарантирует получение структурированного результата оценки для одного тест-кейса с учётом выбранного режима и контроля ошибок."""
        session_id = f"eval-{case.id}-{repetition}"
        started = perf_counter()
        if self.mode == "single":
            response, _ = self.runtime.ask(
                user_id="evaluation", session_id=session_id, message=case.request
            )
            latency_ms = (perf_counter() - started) * 1000
            usage = estimate_mode_usage(
                request=case.request,
                answer=response.answer,
                model=self.runtime.config.agent.model,
                llm_calls=max(1, 1 + len(response.tool_calls)),
                tool_calls=len(response.tool_calls),
                duration_ms=latency_ms,
                input_cost_per_million=(
                    self.runtime.config.multi_agent.cost.input_cost_per_million
                ),
                output_cost_per_million=(
                    self.runtime.config.multi_agent.cost.output_cost_per_million
                ),
                cost_currency=self.runtime.config.multi_agent.cost.currency,
                currency_converter=self.runtime.currency_converter,
            )
            guardrail = self.runtime.guardrails.inspect_output(response.answer)
            return EvaluationOutput(
                answer=guardrail.text,
                tool_calls=response.tool_calls,
                citations_count=len(response.citations),
                latency_ms=latency_ms,
                estimated_cost=usage.estimated_cost,
                estimated_cost_currency=usage.estimated_cost_currency,
                costs_by_currency=usage.costs_by_currency,
                estimated_cost_rub=usage.estimated_cost_rub,
                exchange_rates_to_rub=usage.exchange_rates_to_rub,
                exchange_rate_dates=usage.exchange_rate_dates,
                exchange_rate_source=usage.exchange_rate_source,
                exchange_rate_stale=usage.exchange_rate_stale,
                currency_conversion_errors=usage.currency_conversion_errors,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                guardrail_action=guardrail.action.value,
                guardrail_findings=[item.code for item in guardrail.findings],
            )
        result, _ = self.runtime.ask_multi(
            user_id="evaluation", session_id=session_id, message=case.request
        )
        response = result.response
        guardrail = self.runtime.guardrails.inspect_output(response.answer)
        return EvaluationOutput(
            answer=guardrail.text,
            tool_calls=[
                tool for item in response.task_results for tool in item.tool_calls
            ],
            selected_roles=response.selected_agents,
            citations_count=len(response.citations),
            latency_ms=(perf_counter() - started) * 1000,
            estimated_cost=response.usage.estimated_cost,
            estimated_cost_currency=response.usage.estimated_cost_currency,
            costs_by_currency=response.usage.costs_by_currency,
            estimated_cost_rub=response.usage.estimated_cost_rub,
            exchange_rates_to_rub=response.usage.exchange_rates_to_rub,
            exchange_rate_dates=response.usage.exchange_rate_dates,
            exchange_rate_source=response.usage.exchange_rate_source,
            exchange_rate_stale=response.usage.exchange_rate_stale,
            currency_conversion_errors=(response.usage.currency_conversion_errors),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            guardrail_action=guardrail.action.value,
            guardrail_findings=[item.code for item in guardrail.findings],
        )


class EvaluationRunner:
    """Организует воспроизводимое выполнение и экспорт оценки качества агентной системы с контролем инвариантов и политикой обработки ошибок."""

    def __init__(
        self,
        config: AgentAppConfig,
        executor: EvaluationExecutor,
    ):
        """Готовит экземпляр к запуску оценки, связывая с конфигурацией, исполнителем и экспортёром результатов."""
        self.config = config
        self.executor = executor
        self.exporter = EvaluationExporter(config.evaluation.output_dir)

    def run(self, suite: EvaluationSuite) -> EvaluationReport:
        """Гарантирует воспроизводимое выполнение всей тестовой выборки с агрегацией результатов, проверкой порогов качества и экспортом отчёта."""
        results = []
        for case in suite.cases:
            for repetition in range(1, self.config.evaluation.repeats + 1):
                try:
                    output = self.executor.execute(case, repetition=repetition)
                except Exception as exc:
                    output = EvaluationOutput(answer="", error=str(exc))
                results.append(evaluate_output(case, output, repetition=repetition))
        summary = summarize(results, reviews_pending=self._pending_reviews())
        checks = {
            "task_success_rate": (
                summary.task_success_rate
                >= self.config.evaluation.min_task_success_rate
            ),
            "fact_f1": summary.fact_f1 >= self.config.evaluation.min_fact_f1,
            "consistency": (
                summary.consistency >= self.config.evaluation.min_consistency
            ),
            "p95_latency_ms": (
                summary.p95_latency_ms <= self.config.evaluation.max_p95_latency_ms
            ),
            "average_cost": (
                summary.currency_conversion_complete
                and summary.average_cost_rub is not None
                and summary.average_cost_rub <= self.config.evaluation.max_average_cost
            ),
            "currency_conversion": summary.currency_conversion_complete,
        }
        report = EvaluationReport(
            suite_name=suite.name,
            mode=suite.mode,
            provider=self.config.agent.provider,
            model=self.config.agent.model,
            config_hash=self._config_hash(),
            results=results,
            summary=summary,
            quality_gate=QualityGateResult(passed=all(checks.values()), checks=checks),
        )
        exported = self.exporter.export(report)
        self._log_mlflow(exported)
        return exported

    def _config_hash(self) -> str:
        """Гарантирует уникальный идентификатор конфигурации запуска для отслеживания воспроизводимости и версионирования отчётов."""
        payload = json.dumps(
            self.config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _pending_reviews(self) -> int:
        """Гарантирует получение актуального количества ожидающих ручной проверки оценок для контроля очереди ревью."""
        store = HumanReviewStore(self.config.guardrails.review_sqlite_path)
        try:
            return len(store.list(status="pending", limit=10_000))
        finally:
            store.close()

    def _log_mlflow(self, report: EvaluationReport) -> None:
        """Гарантирует сохранение параметров и метрик оценки в MLflow для воспроизводимости и аудита экспериментов."""
        config = self.config.evaluation
        if not config.mlflow_enabled:
            return
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        with mlflow.start_run(run_name=report.run_id):
            mlflow.log_params(
                {
                    "suite": report.suite_name,
                    "mode": report.mode,
                    "provider": report.provider,
                    "model": report.model,
                    "config_hash": report.config_hash,
                }
            )
            summary_metrics = {
                key: value
                for key, value in report.summary.model_dump(mode="python").items()
                if isinstance(value, (int, float, bool))
            }
            mlflow.log_metrics(
                {
                    **summary_metrics,
                    "quality_gate_passed": float(report.quality_gate.passed),
                }
            )
            if report.run_dir:
                mlflow.log_artifacts(report.run_dir)
