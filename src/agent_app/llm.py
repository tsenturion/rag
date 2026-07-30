"""Создание и адаптация языковых моделей для агентного приложения."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from agent_app.config import AgentConfig
from rag_prep.gigachat_tls import resolve_gigachat_ca_bundle

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
JSON_TOOL_CALL_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\"name\".*?\"arguments\".*?\})\s*```",
    re.DOTALL,
)


def build_llm(config: AgentConfig) -> Any:
    """Гарантирует создание и возврат LLM-объекта, совместимого с выбранным провайдером и параметрами конфигурации агента."""
    if config.provider == "openai":
        return _build_openai_llm(config)
    if config.provider == "gigachat":
        return _build_gigachat_llm(config)
    if config.provider == "local":
        return LocalTransformersChatModel(config)
    raise ValueError(f"Неизвестный provider LLM: {config.provider}")


def _build_openai_llm(config: AgentConfig) -> ChatOpenAI:
    """Гарантирует создание клиента OpenAI с параметрами из конфигурации и проверкой наличия ключа API в окружении."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY не задан в .env или переменных окружения.")
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def _build_gigachat_llm(config: AgentConfig) -> Any:
    """Гарантирует создание клиента GigaChat с параметрами из конфигурации и безопасным получением секретов из окружения."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    credentials = _resolve_env_secret(
        config.gigachat_auth_key_env,
        provider_name="GigaChat",
    )
    try:
        from langchain_gigachat.chat_models import GigaChat
    except ImportError as exc:
        raise RuntimeError(
            "Пакет langchain-gigachat не установлен. Выполните: "
            "python -m pip install langchain-gigachat gigachat"
        ) from exc

    return GigaChat(
        credentials=credentials,
        scope=config.gigachat_scope,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
        verify_ssl_certs=config.gigachat_verify_ssl_certs,
        ca_bundle_file=resolve_gigachat_ca_bundle(),
        profanity_check=config.gigachat_profanity_check,
    )


def _resolve_env_secret(env_name: str, *, provider_name: str) -> str:
    """Гарантирует получение и валидацию секрета из переменных окружения с информативной ошибкой при отсутствии."""
    value = os.getenv(env_name)
    if value:
        return _clean_env_secret(value, env_name)
    raise RuntimeError(
        f"{provider_name}: ключ авторизации не найден. Укажите {env_name} в .env "
        "или переменных окружения."
    )


def _clean_env_secret(value: str, env_name: str) -> str:
    """Гарантирует очистку значения секрета от лишних кавычек и префикса переменной окружения для безопасного использования."""
    cleaned = value.strip().strip("\"'")
    if "=" in cleaned and cleaned.startswith(env_name):
        cleaned = cleaned.split("=", 1)[1].strip().strip("\"'")
    return cleaned


class LocalTransformersChatModel:
    """Chat-интерфейс поверх локальной causal LM с Qwen-style tool calling."""

    supports_tool_calling = True

    def __init__(self, config: AgentConfig):
        """Гарантирует готовность экземпляра к генерации ответов локальной LLM с учётом выбранных настроек, устройства и связанных инструментов."""
        self.config = config
        self.device = self._select_device()
        self.dtype = self._select_dtype(self.device)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.bound_tools: list[BaseTool] = []

    def invoke(self, messages: str | list[BaseMessage], *_args, **_kwargs) -> AIMessage:
        """Гарантирует получение корректного AIMessage на основе истории диалога и параметров генерации, независимо от формата входных сообщений."""
        import torch

        prompt_text = self._format_prompt(messages)
        previous_truncation_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "left"
        try:
            encoded = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self._input_token_limit(),
            )
        finally:
            self.tokenizer.truncation_side = previous_truncation_side
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

    def _input_token_limit(self) -> int:
        """Резервирует место под ответ в пределах контекстного окна модели."""
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(model_limit, int) or model_limit <= 0:
            return self.config.max_input_tokens
        available = model_limit - self.config.max_new_tokens
        if available < 1:
            raise ValueError(
                "max_new_tokens не оставляет места для входа в контекстном окне "
                f"локальной модели ({model_limit} токенов)"
            )
        return min(self.config.max_input_tokens, available)

    def bind_tools(self, tools: list[Any]) -> "LocalTransformersChatModel":
        """Обеспечивает, что только поддерживаемые инструменты будут доступны для вызова в процессе генерации ответов."""
        self.bound_tools = [tool for tool in tools if isinstance(tool, BaseTool)]
        return self

    def _load_tokenizer(self) -> Any:
        """Гарантирует совместимость токенизатора с моделью и корректную обработку специальных токенов для локального запуска."""
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
        """Гарантирует загрузку и подготовку модели для инференса с учётом адаптеров и ограничений по ресурсам, либо сообщает о невозможности запуска."""
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "dtype": self.dtype,
        }
        model: Any = AutoModelForCausalLM.from_pretrained(self.config.model, **kwargs)
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
        """Гарантирует формирование промпта в формате, совместимом с моделью и выбранными инструментами, независимо от структуры исходных сообщений."""
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
        """Гарантирует преобразование сообщения в словарь с ролью и содержимым, пригодный для промпта и передачи в LLM."""
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
        """Гарантирует восстановление структуры AIMessage с корректным извлечением вызовов инструментов из сгенерированного текста."""
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

    def _parse_tool_call_payload(
        self, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Гарантирует фильтрацию и нормализацию вызова инструмента, возвращая None для неизвестных или некорректных payload."""
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
        """Гарантирует получение схемы аргументов инструмента в формате, пригодном для передачи в промпт или шаблон LLM."""
        schema_factory = getattr(tool.args_schema, "model_json_schema", None)
        if isinstance(tool.args_schema, type) and callable(schema_factory):
            parameters = schema_factory()
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
        """Гарантирует выбор устройства для инференса LLM согласно политике конфигурации и доступности ускорителей, обеспечивая совместимость с PyTorch."""
        import torch

        if self.config.local_device != "auto":
            return self.config.local_device
        if torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _select_dtype(self, device: str) -> Any:
        """Гарантирует согласованный выбор типа данных для тензоров LLM в зависимости от устройства и политики, предотвращая несовместимость и ошибки."""
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
