from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llm_tuning.config import FineTuningPipelineConfig
from llm_tuning.dataset import FineTuningDatasetLoader
from llm_tuning.device import build_device_report
from llm_tuning.modeling import LocalCausalModelLoader, trainable_parameter_metrics
from llm_tuning.models import TrainingMetrics, TrainingResult, utc_now
from llm_tuning.tokenization import DataCollatorForCausalLM, SupervisedChatDataset

LOGGER = logging.getLogger(__name__)


class FineTuningTrainingStage:
    def __init__(self, config: FineTuningPipelineConfig):
        self.config = config
        self.dataset_loader = FineTuningDatasetLoader(config)
        self.model_loader = LocalCausalModelLoader(config)

    def run(self, *, run_id: str) -> TrainingResult:
        from transformers import Trainer, TrainingArguments, set_seed

        started_at = utc_now()
        set_seed(self.config.run.seed)

        device = build_device_report(self.config.model)
        tokenizer = self.model_loader.load_tokenizer()
        base_model = self.model_loader.load_base_model(device=device)
        model = self.model_loader.prepare_peft_model(base_model)
        parameter_metrics = trainable_parameter_metrics(model)

        train_examples = self.dataset_loader.load_train()
        eval_examples = self.dataset_loader.load_eval()
        train_dataset = SupervisedChatDataset(
            train_examples,
            tokenizer,
            max_seq_length=self.config.model.max_seq_length,
        )
        eval_dataset = SupervisedChatDataset(
            eval_examples,
            tokenizer,
            max_seq_length=self.config.model.max_seq_length,
        )

        trainer_output_dir = self.config.paths.output_dir / run_id / "trainer"
        adapter_path = self.config.paths.adapter_output_dir / run_id
        trainer_output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path.mkdir(parents=True, exist_ok=True)

        training_args = self._training_args(
            TrainingArguments,
            output_dir=trainer_output_dir,
            device_name=device.selected_device,
            dtype_name=device.selected_dtype,
            run_id=run_id,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForCausalLM(tokenizer),
            processing_class=tokenizer,
        )

        LOGGER.info(
            "Старт fine-tuning run_id=%s, model=%s",
            run_id,
            self.model_loader.active_model_id,
        )
        train_output = trainer.train(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint
        )
        eval_metrics = trainer.evaluate()
        trainer.save_model(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        metrics = self._collect_metrics(
            train_output.metrics,
            eval_metrics,
            parameter_metrics,
            global_step=int(trainer.state.global_step),
        )
        LOGGER.info(
            "Fine-tuning завершён: adapter=%s train_loss=%s eval_loss=%s",
            adapter_path,
            metrics.train_loss,
            metrics.eval_loss,
        )
        return TrainingResult(
            run_id=run_id,
            model_id=self.model_loader.active_model_id,
            method=self.config.peft.method,
            adapter_path=adapter_path,
            trainer_output_dir=trainer_output_dir,
            metrics=metrics,
            log_history=self._safe_log_history(trainer.state.log_history),
            started_at=started_at,
        )

    def _training_args(
        self,
        training_arguments_cls: type[Any],
        *,
        output_dir: Path,
        device_name: str,
        dtype_name: str,
        run_id: str,
    ) -> Any:
        training = self.config.training
        bf16 = dtype_name == "bf16" and device_name != "cpu"
        fp16 = dtype_name == "fp16" and device_name != "cpu"
        return training_arguments_cls(
            output_dir=str(output_dir),
            per_device_train_batch_size=training.per_device_train_batch_size,
            per_device_eval_batch_size=training.per_device_eval_batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            num_train_epochs=training.num_train_epochs,
            max_steps=training.max_steps,
            learning_rate=training.learning_rate,
            warmup_ratio=training.warmup_ratio,
            weight_decay=training.weight_decay,
            max_grad_norm=training.max_grad_norm,
            logging_steps=training.logging_steps,
            eval_steps=training.eval_steps,
            save_steps=training.save_steps,
            save_total_limit=training.save_total_limit,
            gradient_checkpointing=training.gradient_checkpointing,
            dataloader_num_workers=training.dataloader_num_workers,
            optim=training.optim,
            lr_scheduler_type=training.lr_scheduler_type,
            eval_strategy=training.eval_strategy,
            save_strategy=training.save_strategy,
            report_to=training.report_to,
            run_name=run_id,
            seed=self.config.run.seed,
            data_seed=self.config.run.seed,
            bf16=bf16,
            fp16=fp16,
            use_cpu=device_name == "cpu",
            remove_unused_columns=False,
        )

    @staticmethod
    def _collect_metrics(
        train_metrics: dict[str, Any],
        eval_metrics: dict[str, Any],
        parameter_metrics: TrainingMetrics,
        *,
        global_step: int,
    ) -> TrainingMetrics:
        return TrainingMetrics(
            train_loss=_optional_float(train_metrics.get("train_loss")),
            eval_loss=_optional_float(eval_metrics.get("eval_loss")),
            train_runtime=_optional_float(train_metrics.get("train_runtime")),
            train_samples_per_second=_optional_float(
                train_metrics.get("train_samples_per_second")
            ),
            train_steps_per_second=_optional_float(
                train_metrics.get("train_steps_per_second")
            ),
            global_step=global_step,
            trainable_parameters=parameter_metrics.trainable_parameters,
            total_parameters=parameter_metrics.total_parameters,
            trainable_ratio=parameter_metrics.trainable_ratio,
        )

    @staticmethod
    def _safe_log_history(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: _safe_scalar(value) for key, value in item.items()}
            for item in log_history
        ]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
