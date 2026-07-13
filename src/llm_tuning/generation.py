from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_tuning.config import FineTuningPipelineConfig
from llm_tuning.dataset import ChatFormatter
from llm_tuning.device import build_device_report
from llm_tuning.modeling import LocalCausalModelLoader
from llm_tuning.models import LocalGenerationResult


class LocalGenerationStage:
    """Генерирует ответ локальной causal LM без обращения к внешнему API."""

    def __init__(self, config: FineTuningPipelineConfig):
        self.config = config
        self.model_loader = LocalCausalModelLoader(config)

    def run(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        adapter_path: Path | None = None,
        max_new_tokens: int | None = None,
    ) -> LocalGenerationResult:
        import torch

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt не должен быть пустым")
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter не найден: {adapter_path}")

        device = build_device_report(self.config.model)
        tokenizer = self.model_loader.load_tokenizer()
        model = (
            self.model_loader.load_model_with_adapter(adapter_path)
            if adapter_path is not None
            else self.model_loader.load_base_model(device=device)
        )
        model.eval()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})
        prompt_text = ChatFormatter.apply_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        encoded = {
            key: value.to(device.selected_device) for key, value in encoded.items()
        }

        generation = self.config.evaluation.generation
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or generation.max_new_tokens,
            "do_sample": generation.do_sample,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if generation.do_sample:
            generate_kwargs["temperature"] = generation.temperature
            generate_kwargs["top_p"] = generation.top_p

        with torch.no_grad():
            output = model.generate(**encoded, **generate_kwargs)

        prompt_length = encoded["input_ids"].shape[-1]
        answer = tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        ).strip()
        return LocalGenerationResult(
            model_id=self.model_loader.active_model_id,
            adapter_path=adapter_path,
            prompt=prompt,
            answer=answer,
            device=device,
            max_new_tokens=int(generate_kwargs["max_new_tokens"]),
        )
