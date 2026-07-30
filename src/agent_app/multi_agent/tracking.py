"""Логирование запусков в MLflow для мультиагентной системы."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

from agent_app.config import MultiAgentConfig
from agent_app.multi_agent.models import (
    MultiAgentComparisonReport,
    MultiAgentRunResult,
)
from rag_prep.mlflow_uri import (
    ensure_mlflow_tracking_parent,
    resolve_mlflow_tracking_uri,
)

LOGGER = logging.getLogger(__name__)


class MultiAgentTracker:
    """Обеспечивает централизованный аудит и сохранение метрик запусков и сравнений мультиагентной системы с гарантией согласованности данных."""

    def __init__(self, config: MultiAgentConfig):
        """Гарантирует готовность трекера к логированию запусков и сравнений с учётом политики конфигурации."""
        self.config = config

    def log_run(self, result: MultiAgentRunResult) -> None:
        """Гарантирует запись метрик и параметров запуска в MLflow только при активированной поддержке трекинга и наличии каталога запуска."""
        if not self.config.mlflow_enabled or result.run_dir is None:
            return
        response = result.response
        metrics = {
            "quality": response.quality.score if response.quality else 0.0,
            "duration_ms": response.usage.duration_ms,
            "llm_calls": response.usage.llm_calls,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "tool_calls": response.usage.tool_calls,
            "estimated_cost": response.usage.estimated_cost,
            "estimated_cost_rub": response.usage.estimated_cost_rub or 0.0,
            "tasks": len(response.tasks),
            "history_messages_used": response.history_messages_used,
            "failed_tasks": sum(
                item.state != "completed" for item in response.task_results
            ),
        }
        params = {
            "execution_mode": response.execution_mode,
            "selected_agents": ",".join(response.selected_agents),
            "degraded": response.degraded,
            "summary_used": response.summary_used,
            "llm_routes": json.dumps(
                [route.model_dump(mode="json") for route in response.llm_routes],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "costs_by_currency": json.dumps(
                response.usage.costs_by_currency,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "exchange_rates_to_rub": json.dumps(
                response.usage.exchange_rates_to_rub,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "currency_conversion_complete": str(
                response.usage.estimated_cost_rub is not None
            ).lower(),
            "estimated_cost_currency": response.usage.estimated_cost_currency,
        }
        self._log(response.run_id, metrics, params, Path(result.run_dir))

    def log_comparison(self, report: MultiAgentComparisonReport) -> None:
        """Гарантирует сохранение результатов сравнения режимов работы агентов в MLflow при корректной конфигурации и наличии каталога."""
        if not self.config.mlflow_enabled or report.run_dir is None:
            return
        self._log(
            report.run_id,
            {
                "single_quality": report.average_single_quality,
                "multi_quality": report.average_multi_quality,
                "quality_delta": report.quality_delta,
                "single_cost": report.total_single_cost,
                "multi_cost": report.total_multi_cost,
                "cost_delta": report.total_cost_delta,
                "single_cost_rub": report.total_single_cost_rub or 0.0,
                "multi_cost_rub": report.total_multi_cost_rub or 0.0,
                "cost_delta_rub": report.total_cost_delta_rub or 0.0,
                "cases": len(report.cases),
            },
            {
                "provider": report.provider,
                "model": report.model,
                "cost_currency": report.cost_currency,
                "llm_routes": json.dumps(
                    [route.model_dump(mode="json") for route in report.llm_routes],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "single_costs_by_currency": json.dumps(
                    report.total_single_costs_by_currency,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "multi_costs_by_currency": json.dumps(
                    report.total_multi_costs_by_currency,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            Path(report.run_dir),
        )

    def _log(
        self,
        run_name: str,
        metrics: dict[str, float | int],
        params: dict[str, object],
        artifact_dir: Path,
    ) -> None:
        """Записывает параметры, метрики и артефакты в MLflow, не прерывая основной запуск при ошибке трекинга."""
        try:
            tracking_uri = self._tracking_uri()
            ensure_mlflow_tracking_parent(tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)
            with mlflow.start_run(
                run_name=run_name,
                nested=mlflow.active_run() is not None,
            ):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifacts(str(artifact_dir))
        except Exception:
            LOGGER.exception("Не удалось записать multi-agent запуск в MLflow")

    def _tracking_uri(self) -> str:
        """Гарантирует получение абсолютного URI для трекинга MLflow, разрешая относительные пути относительно корня проекта."""
        project_root = Path(__file__).resolve().parents[3]
        uri = resolve_mlflow_tracking_uri(
            self.config.mlflow_tracking_uri,
            base_dir=project_root,
        )
        if "://" in uri:
            return uri
        return Path(uri).resolve().as_uri()
