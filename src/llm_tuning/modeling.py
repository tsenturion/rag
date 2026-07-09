from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

from llm_tuning.config import FineTuningPipelineConfig
from llm_tuning.device import build_device_report, torch_dtype
from llm_tuning.models import DeviceReport, TrainingMetrics

LOGGER = logging.getLogger(__name__)


class LocalCausalModelLoader:
    def __init__(self, config: FineTuningPipelineConfig):
        self.config = config
        self._configure_hf_hub_downloads()

    def load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer_id = self.config.model.tokenizer_id or self.config.model.model_id
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_base_model(self, device: DeviceReport | None = None) -> Any:
        from transformers import AutoModelForCausalLM

        device_report = device or build_device_report(self.config.model)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "low_cpu_mem_usage": self.config.model.low_cpu_mem_usage,
            "dtype": torch_dtype(device_report.selected_dtype),
        }
        if self.config.peft.method == "qlora":
            model_kwargs.update(self._qlora_quantization_kwargs(device_report))

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_id,
            **model_kwargs,
        )
        if self.config.peft.method != "qlora":
            model = model.to(device_report.selected_device)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return model

    def load_model_with_adapter(self, adapter_path: Path) -> Any:
        from peft import PeftModel

        device = build_device_report(self.config.model)
        model = self.load_base_model(device=device)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.to(device.selected_device)
        model.eval()
        return model

    def prepare_peft_model(self, model: Any) -> Any:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if self.config.peft.method == "qlora":
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.config.peft.r,
            lora_alpha=self.config.peft.lora_alpha,
            lora_dropout=self.config.peft.lora_dropout,
            bias=self.config.peft.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.peft.target_modules,
            use_rslora=self.config.peft.use_rslora,
        )
        model = get_peft_model(model, lora_config)
        LOGGER.info("Подключён PEFT adapter: %s", self.config.peft.method)
        return model

    def _qlora_quantization_kwargs(self, device: DeviceReport) -> dict[str, Any]:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError(
                "Выбран method=qlora, но пакет bitsandbytes не установлен. "
                "Для Intel Arc рекомендуется начать с method=lora."
            )
        if device.selected_device == "xpu":
            raise RuntimeError(
                "QLoRA через bitsandbytes не включён для torch.xpu в этом проекте. "
                "Используйте method=lora или CPU/CUDA-окружение с проверенной 4-bit поддержкой."
            )

        from transformers import BitsAndBytesConfig

        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=self.config.peft.qlora_load_in_4bit,
                bnb_4bit_quant_type=self.config.peft.qlora_quant_type,
                bnb_4bit_use_double_quant=self.config.peft.qlora_double_quant,
                bnb_4bit_compute_dtype=torch_dtype(device.selected_dtype),
            )
        }

    def _configure_hf_hub_downloads(self) -> None:
        if self.config.model.hub_disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if self.config.model.hub_disable_symlink_warning:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def trainable_parameter_metrics(model: Any) -> TrainingMetrics:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = round(trainable / total, 8) if total else 0.0
    return TrainingMetrics(
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_ratio=ratio,
    )
