"""Загрузка модели и PEFT-адаптера для PEFT fine-tuning локальной LLM."""

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
    """Отвечает за загрузку и подготовку локальной LLM с учётом PEFT, обеспечивая согласованность конфигурации и управление версиями моделей."""

    def __init__(self, config: FineTuningPipelineConfig):
        """Инициализирует загрузчик модели с конфигурацией, устанавливая активный идентификатор модели и настраивая параметры загрузки из HF Hub."""
        self.config = config
        self.active_model_id = config.model.model_id
        self._configure_hf_hub_downloads()

    def load_tokenizer(self) -> Any:
        """Загружает токенизатор с учётом конфигурации, обеспечивая корректное паддинг и совместимость с активной моделью."""
        from transformers import AutoTokenizer

        using_fallback_model = self.active_model_id != self.config.model.model_id
        tokenizer_id = (
            self.active_model_id
            if using_fallback_model
            else self.config.model.tokenizer_id or self.active_model_id
        )
        tokenizer, loaded_id = self._load_with_fallback(
            AutoTokenizer.from_pretrained,
            tokenizer_id,
            component_name="tokenizer",
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self.config.model.tokenizer_id is None or using_fallback_model:
            self.active_model_id = loaded_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_base_model(self, device: DeviceReport | None = None) -> Any:
        """Загружает базовую модель с учётом устройства и параметров конфигурации, гарантируя готовность к дальнейшему fine-tuning."""
        from transformers import AutoModelForCausalLM

        device_report = device or build_device_report(self.config.model)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "low_cpu_mem_usage": self.config.model.low_cpu_mem_usage,
            "dtype": torch_dtype(device_report.selected_dtype),
        }
        if self.config.peft.method == "qlora":
            model_kwargs.update(self._qlora_quantization_kwargs(device_report))

        model, loaded_id = self._load_with_fallback(
            AutoModelForCausalLM.from_pretrained,
            self.active_model_id,
            component_name="model",
            **model_kwargs,
        )
        self.active_model_id = loaded_id
        if self.config.peft.method != "qlora":
            model = model.to(device_report.selected_device)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return model

    def load_model_with_adapter(self, adapter_path: Path) -> Any:
        """Загружает базовую модель и применяет к ней адаптер PEFT, обеспечивая корректное размещение на устройстве и режим оценки."""
        from peft import PeftModel

        device = build_device_report(self.config.model)
        model = self.load_base_model(device=device)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.to(device.selected_device)
        model.eval()
        return model

    def prepare_peft_model(self, model: Any) -> Any:
        """Гарантирует, что модель подготовлена для обучения с выбранной PEFT-методикой и корректно интегрирует LoRA-адаптер с параметрами из конфигурации."""
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

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
        """Гарантирует, что параметры квантования QLoRA возвращаются только при наличии поддержки bitsandbytes и совместимого устройства, иначе выбрасывает осмысленную ошибку."""
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
        """Обеспечивает корректную настройку переменных окружения для загрузки моделей с HuggingFace Hub согласно политике конфигурации."""
        if self.config.model.hub_disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if self.config.model.hub_disable_symlink_warning:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    def _load_with_fallback(
        self,
        loader: Any,
        model_id: str,
        *,
        component_name: str,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Гарантирует успешную загрузку модели или fallback-идентификатора, либо выбрасывает ошибку, если оба варианта недоступны."""
        try:
            return loader(model_id, **kwargs), model_id
        except OSError:
            fallback_id = self.config.model.fallback_model_id
            if not fallback_id or fallback_id == model_id:
                raise
            LOGGER.warning(
                "Не удалось загрузить %s %s; используется fallback %s",
                component_name,
                model_id,
                fallback_id,
            )
            return loader(fallback_id, **kwargs), fallback_id


def trainable_parameter_metrics(model: Any) -> TrainingMetrics:
    """Вычисляет метрики обучаемых параметров модели, гарантируя точный учёт и соотношение для оценки эффективности fine-tuning."""
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
