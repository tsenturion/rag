from __future__ import annotations

import logging
import random
from pathlib import Path

from llm_tuning.comparison import FineTuningComparisonStage
from llm_tuning.config import FineTuningPipelineConfig
from llm_tuning.dataset import FineTuningDatasetLoader
from llm_tuning.device import build_device_report
from llm_tuning.evaluation import FineTuningEvaluationStage
from llm_tuning.exporting import FineTuningExportStage
from llm_tuning.metrics import build_fine_tuning_counts
from llm_tuning.models import (
    ComparisonResult,
    DatasetValidationResult,
    DeviceReport,
    EvaluationReport,
    FineTuningPipelineResult,
)
from llm_tuning.training import FineTuningTrainingStage
from rag_prep.tracking import MLflowTracker
from rag_prep.utils import new_run_id

LOGGER = logging.getLogger(__name__)


class FineTuningPipeline:
    def __init__(self, config: FineTuningPipelineConfig):
        self.config = config
        self.dataset_loader = FineTuningDatasetLoader(config)
        self.trainer = FineTuningTrainingStage(config)
        self.evaluator = FineTuningEvaluationStage(config)
        self.comparer = FineTuningComparisonStage()
        self.exporter = FineTuningExportStage(config)
        self.tracker = MLflowTracker(config)

    def validate(self) -> tuple[DatasetValidationResult, DeviceReport]:
        dataset_validation = self.dataset_loader.validate()
        device = build_device_report(self.config.model)
        return dataset_validation, device

    def run_baseline(self, *, run_id: str | None = None) -> FineTuningPipelineResult:
        run_id = run_id or new_run_id()
        random.seed(self.config.run.seed)
        dataset_validation, device = self.validate()
        self._ensure_dataset_ok(dataset_validation)

        baseline = self.evaluator.run(run_id=run_id, report_label="baseline")
        baseline_path = self.exporter.write_evaluation(
            baseline,
            filename=self.config.paths.baseline_report_filename,
        )
        export = self.exporter.write_manifest(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            baseline_report_path=baseline_path,
        )
        self.tracker.log_run(
            build_fine_tuning_counts(dataset_validation, device, baseline=baseline),
            export,
        )
        return FineTuningPipelineResult(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            baseline=baseline,
            export=export,
        )

    def run_training(self, *, run_id: str | None = None) -> FineTuningPipelineResult:
        run_id = run_id or new_run_id()
        random.seed(self.config.run.seed)
        dataset_validation, device = self.validate()
        self._ensure_dataset_ok(dataset_validation)

        baseline: EvaluationReport | None = None
        baseline_path: Path | None = None
        if self.config.training.run_baseline_before_train:
            baseline = self.evaluator.run(run_id=run_id, report_label="baseline")
            baseline_path = self.exporter.write_evaluation(
                baseline,
                filename=self.config.paths.baseline_report_filename,
            )

        training = self.trainer.run(run_id=run_id)

        tuned: EvaluationReport | None = None
        tuned_path: Path | None = None
        comparison: ComparisonResult | None = None
        comparison_path: Path | None = None
        if self.config.training.run_evaluation_after_train:
            tuned = self.evaluator.run(
                run_id=run_id,
                adapter_path=training.adapter_path,
                report_label="tuned",
            )
            tuned_path = self.exporter.write_evaluation(
                tuned,
                filename=self.config.paths.tuned_report_filename,
            )
            if baseline is not None and baseline_path is not None:
                comparison = self.comparer.compare(
                    run_id=run_id,
                    baseline=baseline,
                    tuned=tuned,
                    baseline_report_path=baseline_path,
                    tuned_report_path=tuned_path,
                )
                comparison_path = self.exporter.write_comparison(comparison)

        export = self.exporter.write_manifest(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            training=training,
            baseline_report_path=baseline_path,
            tuned_report_path=tuned_path,
            comparison_report_path=comparison_path,
        )
        self.tracker.log_run(
            build_fine_tuning_counts(
                dataset_validation,
                device,
                training=training,
                baseline=baseline,
                tuned=tuned,
                comparison=comparison,
            ),
            export,
        )
        return FineTuningPipelineResult(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            training=training,
            baseline=baseline,
            tuned=tuned,
            comparison=comparison,
            export=export,
        )

    def run_evaluation(
        self,
        *,
        adapter_path: Path,
        run_id: str | None = None,
    ) -> FineTuningPipelineResult:
        run_id = run_id or new_run_id()
        dataset_validation, device = self.validate()
        self._ensure_dataset_ok(dataset_validation)

        tuned = self.evaluator.run(
            run_id=run_id,
            adapter_path=adapter_path,
            report_label="tuned",
        )
        tuned_path = self.exporter.write_evaluation(
            tuned,
            filename=self.config.paths.tuned_report_filename,
        )
        export = self.exporter.write_manifest(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            adapter_path=adapter_path,
            tuned_report_path=tuned_path,
        )
        self.tracker.log_run(
            build_fine_tuning_counts(dataset_validation, device, tuned=tuned),
            export,
        )
        return FineTuningPipelineResult(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            tuned=tuned,
            export=export,
        )

    def compare_reports(
        self,
        *,
        baseline_report_path: Path,
        tuned_report_path: Path,
        run_id: str | None = None,
    ) -> FineTuningPipelineResult:
        run_id = run_id or new_run_id()
        dataset_validation, device = self.validate()
        baseline = self.exporter.load_evaluation_report(baseline_report_path)
        tuned = self.exporter.load_evaluation_report(tuned_report_path)
        comparison = self.comparer.compare(
            run_id=run_id,
            baseline=baseline,
            tuned=tuned,
            baseline_report_path=baseline_report_path,
            tuned_report_path=tuned_report_path,
        )
        comparison_path = self.exporter.write_comparison(comparison)
        export = self.exporter.write_manifest(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            baseline_report_path=baseline_report_path,
            tuned_report_path=tuned_report_path,
            comparison_report_path=comparison_path,
        )
        self.tracker.log_run(
            build_fine_tuning_counts(
                dataset_validation,
                device,
                baseline=baseline,
                tuned=tuned,
                comparison=comparison,
            ),
            export,
        )
        return FineTuningPipelineResult(
            run_id=run_id,
            dataset_validation=dataset_validation,
            device=device,
            baseline=baseline,
            tuned=tuned,
            comparison=comparison,
            export=export,
        )

    @staticmethod
    def _ensure_dataset_ok(validation: DatasetValidationResult) -> None:
        if validation.has_errors:
            raise ValueError(f"Датасет fine-tuning не прошёл проверку: {validation}")
