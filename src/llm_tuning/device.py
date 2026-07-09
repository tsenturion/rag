from __future__ import annotations

from typing import Any

from llm_tuning.config import LocalModelConfig
from llm_tuning.models import DeviceReport


def build_device_report(config: LocalModelConfig) -> DeviceReport:
    import torch

    xpu_available = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    cuda_available = torch.cuda.is_available()
    notes: list[str] = []

    if config.device == "auto":
        if xpu_available:
            selected_device = "xpu"
        elif cuda_available:
            selected_device = "cuda"
        else:
            selected_device = "cpu"
            notes.append("GPU не найден, выбран CPU-режим.")
    else:
        selected_device = config.device

    if selected_device == "xpu" and not xpu_available:
        raise RuntimeError(
            "В конфиге выбран device=xpu, но torch.xpu.is_available() вернул False."
        )
    if selected_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "В конфиге выбран device=cuda, но torch.cuda.is_available() вернул False."
        )

    selected_dtype = select_dtype_name(config.dtype, selected_device)
    if selected_device == "xpu":
        notes.append("Для Intel Arc используется torch.xpu; CUDA-зависимые режимы не включаются.")
    if selected_dtype == "fp32":
        notes.append("Выбран fp32: режим устойчивый, но потребляет больше памяти.")

    return DeviceReport(
        requested_device=config.device,
        selected_device=selected_device,
        torch_version=torch.__version__,
        xpu_available=xpu_available,
        cuda_available=cuda_available,
        selected_dtype=selected_dtype,
        notes=notes,
    )


def select_dtype_name(requested_dtype: str, selected_device: str) -> str:
    if requested_dtype != "auto":
        return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}[requested_dtype]
    if selected_device in {"xpu", "cuda"}:
        return "bf16"
    return "fp32"


def torch_dtype(dtype_name: str) -> Any:
    import torch

    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32
