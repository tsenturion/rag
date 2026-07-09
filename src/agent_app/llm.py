from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from agent_app.config import AgentConfig

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
JSON_TOOL_CALL_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\"name\".*?\"arguments\".*?\})\s*```",
    re.DOTALL,
)


def build_llm(config: AgentConfig) -> Any:
    if config.provider == "openai":
        return _build_openai_llm(config)
    if config.provider == "local":
        return LocalTransformersChatModel(config)
    raise ValueError(f"Неизвестный provider LLM: {config.provider}")


def _build_openai_llm(config: AgentConfig) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY не задан в .env или переменных окружения.")
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


class LocalTransformersChatModel:
    """Chat-интерфейс поверх локальной causal LM с Qwen-style tool calling."""

    supports_tool_calling = True

    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = self._select_device()
        self.dtype = self._select_dtype(self.device)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.bound_tools: list[BaseTool] = []

    def invoke(self, messages: str | list[BaseMessage], *_args, **_kwargs) -> AIMessage:
        import torch

        prompt_text = self._format_prompt(messages)
        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.temperature > 0:
            generate_kwargs["temperature"] = self.config.temperature

        with torch.no_grad():
            output = self.model.generate(**encoded, **generate_kwargs)
        prompt_length = encoded["input_ids"].shape[-1]
        generated = self.tokenizer.decode(
            output[0][prompt_length:],
            skip_special_tokens=False,
        ).strip()
        return self._to_ai_message(generated)

    def bind_tools(self, tools: list[Any]) -> "LocalTransformersChatModel":
        self.bound_tools = [tool for tool in tools if isinstance(tool, BaseTool)]
        return self

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_model(self) -> Any:
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "dtype": self.dtype,
        }
        model = AutoModelForCausalLM.from_pretrained(self.config.model, **kwargs)
        if self.config.adapter_path is not None:
            from peft import PeftModel

            adapter_path = Path(self.config.adapter_path)
            if not adapter_path.exists():
                raise FileNotFoundError(f"LoRA adapter не найден: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.to(self.device)
        model.eval()
        return model

    def _format_prompt(self, messages: str | list[BaseMessage]) -> str:
        if isinstance(messages, str):
            chat_messages = [{"role": "user", "content": messages}]
        else:
            chat_messages = [self._message_to_dict(message) for message in messages]

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                chat_messages,
                tools=[self._tool_schema(tool) for tool in self.bound_tools],
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = []
        for message in chat_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                lines.append(f"Системная инструкция: {content}")
            elif role == "assistant":
                lines.append(f"Ассистент: {content}")
            else:
                lines.append(f"Пользователь: {content}")
        lines.append("Ассистент:")
        return "\n".join(lines)

    @staticmethod
    def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
        message_type = getattr(message, "type", "")
        if isinstance(message, ToolMessage):
            return {"role": "tool", "content": str(message.content)}
        if message_type == "system":
            role = "system"
        elif message_type == "ai":
            role = "assistant"
        else:
            role = "user"
        payload: dict[str, Any] = {"role": role, "content": str(message.content)}
        tool_calls = getattr(message, "tool_calls", None)
        if role == "assistant" and tool_calls:
            payload["tool_calls"] = [
                {
                    "name": call.get("name"),
                    "arguments": call.get("args", {}),
                }
                for call in tool_calls
            ]
        return payload

    def _to_ai_message(self, generated: str) -> AIMessage:
        cleaned = generated.replace("<|im_end|>", "").strip()
        tool_calls = []
        for match in TOOL_CALL_RE.finditer(cleaned):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            tool_call = self._parse_tool_call_payload(payload)
            if tool_call is not None:
                tool_calls.append(tool_call)

        if not tool_calls:
            for match in JSON_TOOL_CALL_RE.finditer(cleaned):
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                tool_call = self._parse_tool_call_payload(payload)
                if tool_call is not None:
                    tool_calls.append(tool_call)

        if tool_calls:
            content = TOOL_CALL_RE.sub("", cleaned)
            content = JSON_TOOL_CALL_RE.sub("", content).strip()
            return AIMessage(content=content, tool_calls=tool_calls)
        return AIMessage(content=cleaned)

    def _parse_tool_call_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        known_tools = {tool.name for tool in self.bound_tools}
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        if known_tools and name not in known_tools:
            return None
        return {
            "name": name,
            "args": arguments,
            "id": f"call_{uuid4().hex}",
            "type": "tool_call",
        }

    @staticmethod
    def _tool_schema(tool: BaseTool) -> dict[str, Any]:
        if isinstance(tool.args_schema, type) and hasattr(tool.args_schema, "model_json_schema"):
            parameters = tool.args_schema.model_json_schema()
        elif isinstance(tool.args_schema, dict):
            parameters = tool.args_schema
        else:
            parameters = {
                "type": "object",
                "properties": tool.args,
            }
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            },
        }

    def _select_device(self) -> str:
        import torch

        if self.config.local_device != "auto":
            return self.config.local_device
        if torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _select_dtype(self, device: str) -> Any:
        import torch

        dtype = self.config.local_dtype
        if dtype == "auto":
            dtype = "bf16" if device in {"xpu", "cuda"} else "fp32"
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "fp16":
            return torch.float16
        if dtype == "fp32":
            return torch.float32
        raise ValueError(f"Неизвестный dtype локальной LLM: {dtype}")
