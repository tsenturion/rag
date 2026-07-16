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

LOGGER = logging.getLogger(__name__)


class MultiAgentTracker:
    def __init__(self, config: MultiAgentConfig):
        self.config = config

    def log_run(self, result: MultiAgentRunResult) -> None:
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
        }
        self._log(response.run_id, metrics, params, Path(result.run_dir))

    def log_comparison(self, report: MultiAgentComparisonReport) -> None:
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
                "cases": len(report.cases),
            },
            {
                "provider": report.provider,
                "model": report.model,
                "llm_routes": json.dumps(
                    [route.model_dump(mode="json") for route in report.llm_routes],
                    ensure_ascii=False,
                    separators=(",", ":"),
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
        try:
            mlflow.set_tracking_uri(self._tracking_uri())
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
        uri = self.config.mlflow_tracking_uri
        if "://" in uri:
            return uri
        return Path(uri).expanduser().resolve().as_uri()
