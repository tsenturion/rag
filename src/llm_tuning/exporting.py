from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_tuning.config import FineTuningPipelineConfig
from llm_tuning.models import (
    ComparisonResult,
    DatasetValidationResult,
    DeviceReport,
    EvaluationReport,
    FineTuningExportResult,
    TrainingResult,
)
from rag_prep.utils import json_dump


class FineTuningExportStage:
    def __init__(self, config: FineTuningPipelineConfig):
        self.config = config

    def write_evaluation(
        self,
        report: EvaluationReport,
        *,
        filename: str,
    ) -> Path:
        path = self.config.paths.reports_dir / filename
        json_dump(path, report.model_dump(mode="json"))
        return path

    def write_comparison(self, comparison: ComparisonResult) -> Path:
        path = self.config.paths.reports_dir / self.config.paths.comparison_report_filename
        json_dump(path, comparison.model_dump(mode="json"))
        return path

    def write_manifest(
        self,
        *,
        run_id: str,
        dataset_validation: DatasetValidationResult,
        device: DeviceReport,
        training: TrainingResult | None = None,
        adapter_path: Path | None = None,
        baseline_report_path: Path | None = None,
        tuned_report_path: Path | None = None,
        comparison_report_path: Path | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> FineTuningExportResult:
        self.config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.paths.reports_dir / self.config.paths.manifest_filename
        effective_adapter_path = training.adapter_path if training else adapter_path
        payload = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.model_dump(mode="json"),
            "dataset_validation": dataset_validation.model_dump(mode="json"),
            "device": device.model_dump(mode="json"),
            "training": training.model_dump(mode="json") if training else None,
            "diagnostics": diagnostics or {},
            "outputs": {
                "baseline_report": str(baseline_report_path)
                if baseline_report_path
                else None,
                "tuned_report": str(tuned_report_path) if tuned_report_path else None,
                "comparison_report": str(comparison_report_path)
                if comparison_report_path
                else None,
                "adapter": str(effective_adapter_path) if effective_adapter_path else None,
            },
        }
        json_dump(manifest_path, payload)
        return FineTuningExportResult(
            manifest_path=manifest_path,
            baseline_report_path=baseline_report_path,
            tuned_report_path=tuned_report_path,
            comparison_report_path=comparison_report_path,
            adapter_path=effective_adapter_path,
            run_id=run_id,
        )

    @staticmethod
    def load_evaluation_report(path: Path) -> EvaluationReport:
        with path.open("r", encoding="utf-8") as file:
            return EvaluationReport.model_validate(json.load(file))
