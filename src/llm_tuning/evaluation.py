"""Оценка результатов для PEFT fine-tuning локальной LLM."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from llm_tuning.config import FineTuningPipelineConfig
from llm_tuning.dataset import ChatFormatter, FineTuningDatasetLoader, examples_answer
from llm_tuning.device import build_device_report
from llm_tuning.modeling import LocalCausalModelLoader
from llm_tuning.models import EvaluationReport, FineTuningExample, GeneratedAnswer
from llm_tuning.tokenization import DataCollatorForCausalLM, SupervisedChatDataset

LOGGER = logging.getLogger(__name__)


class FineTuningEvaluationStage:
    """Обеспечивает выполнение этапа оценки fine-tuning модели, гарантируя корректную загрузку данных и воспроизводимость метрик качества для принятия решений о качестве адаптации."""

    def __init__(self, config: FineTuningPipelineConfig):
        """Гарантирует готовность экземпляра к запуску оценки, включая загрузку конфигурации, датасета и модели."""
        self.config = config
        self.dataset_loader = FineTuningDatasetLoader(config)
        self.model_loader = LocalCausalModelLoader(config)

    def run(
        self,
        *,
        run_id: str,
        adapter_path: Path | None = None,
        report_label: str = "baseline",
    ) -> EvaluationReport:
        """Гарантирует воспроизводимую оценку модели или адаптера на тестовом наборе с вычислением метрик качества и подробным отчётом."""
        import torch

        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter не найден: {adapter_path}")

        device = build_device_report(self.config.model)
        model = (
            self.model_loader.load_model_with_adapter(adapter_path)
            if adapter_path is not None
            else self.model_loader.load_base_model(device=device)
        )
        tokenizer = self.model_loader.load_tokenizer()
        model.eval()

        eval_examples = self._limited_examples(self.dataset_loader.load_eval())
        answers = []
        for example in eval_examples:
            with torch.no_grad():
                generated = self._generate_one(
                    model,
                    tokenizer,
                    example,
                    device_name=device.selected_device,
                )
            answers.append(self._score_answer(example, generated))

        eval_loss = self._compute_eval_loss(
            model,
            tokenizer,
            eval_examples,
            device_name=device.selected_device,
            dtype_name=device.selected_dtype,
            run_id=run_id,
            report_label=report_label,
        )
        passed_count = sum(1 for answer in answers if answer.passed)
        pass_rate = round(passed_count / len(answers), 6) if answers else 0.0
        report = EvaluationReport(
            run_id=run_id,
            model_id=self.model_loader.active_model_id,
            adapter_path=adapter_path,
            examples_count=len(answers),
            passed_count=passed_count,
            failed_count=len(answers) - passed_count,
            pass_rate=pass_rate,
            eval_loss=eval_loss,
            perplexity=round(math.exp(eval_loss), 6)
            if eval_loss is not None and eval_loss < 20
            else None,
            answers=answers,
        )
        LOGGER.info(
            "Оценка %s завершена: examples=%d pass_rate=%.3f eval_loss=%s",
            report_label,
            report.examples_count,
            report.pass_rate,
            report.eval_loss,
        )
        if self.config.evaluation.fail_on_failed_criteria and report.failed_count:
            raise ValueError(
                f"Оценка {report_label} не прошла критерии: {report.failed_count}"
            )
        return report

    def _limited_examples(
        self,
        examples: list[FineTuningExample],
    ) -> list[FineTuningExample]:
        """Гарантирует, что для оценки используются не более заданного числа примеров, обеспечивая воспроизводимость и контроль времени выполнения."""
        limit = self.config.evaluation.max_examples
        return examples[:limit] if limit is not None else examples

    def _generate_one(
        self,
        model: Any,
        tokenizer: Any,
        example: FineTuningExample,
        *,
        device_name: str,
    ) -> str:
        """Гарантирует генерацию одного ответа LLM по заданному примеру с учётом всех параметров инференса и корректной работы на выбранном устройстве."""
        generation = self.config.evaluation.generation
        prompt_text = ChatFormatter.apply_chat_template(
            tokenizer,
            ChatFormatter.prompt_messages(example),
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device_name) for key, value in encoded.items()}
        generate_kwargs = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.do_sample,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if generation.do_sample:
            generate_kwargs["temperature"] = generation.temperature
            generate_kwargs["top_p"] = generation.top_p
        output = model.generate(**encoded, **generate_kwargs)
        prompt_length = encoded["input_ids"].shape[-1]
        generated_ids = output[0][prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _score_answer(
        self,
        example: FineTuningExample,
        generated: str,
    ) -> GeneratedAnswer:
        """Гарантирует формирование результата проверки ответа LLM с учётом обязательных и запрещённых ключевых слов, фиксируя прохождение критериев."""
        expected = examples_answer(example)
        if self.config.evaluation.required_keywords_case_sensitive:
            haystack = generated
            required = example.required_keywords
            forbidden = example.forbidden_keywords
        else:
            haystack = generated.lower()
            required = [value.lower() for value in example.required_keywords]
            forbidden = [value.lower() for value in example.forbidden_keywords]

        passed_required = all(keyword in haystack for keyword in required)
        passed_forbidden = not any(keyword in haystack for keyword in forbidden)
        return GeneratedAnswer(
            example_id=example.id,
            prompt=ChatFormatter.prompt_text(example),
            expected_answer=expected,
            generated_answer=generated,
            required_keywords=example.required_keywords,
            forbidden_keywords=example.forbidden_keywords,
            passed_required_keywords=passed_required,
            passed_forbidden_keywords=passed_forbidden,
            passed=passed_required and passed_forbidden,
        )

    def _compute_eval_loss(
        self,
        model: Any,
        tokenizer: Any,
        examples: list[FineTuningExample],
        *,
        device_name: str,
        dtype_name: str,
        run_id: str,
        report_label: str,
    ) -> float | None:
        """Гарантирует вычисление метрики потерь на валидационной выборке с учётом всех параметров окружения и сохранением результатов для анализа."""
        from transformers import Trainer, TrainingArguments

        if not examples:
            return None
        output_dir = self.config.paths.output_dir / run_id / f"eval_{report_label}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            report_to=["none"],
            use_cpu=device_name == "cpu",
            bf16=dtype_name == "bf16" and device_name != "cpu",
            fp16=dtype_name == "fp16" and device_name != "cpu",
            remove_unused_columns=False,
        )
        dataset = SupervisedChatDataset(
            examples,
            tokenizer,
            max_seq_length=self.config.model.max_seq_length,
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=dataset,
            data_collator=DataCollatorForCausalLM(tokenizer),
            processing_class=tokenizer,
        )
        metrics = trainer.evaluate()
        value = metrics.get("eval_loss")
        return float(value) if value is not None else None
