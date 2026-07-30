"""Проверки выбора LLM-провайдера и локального tool-calling адаптера."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import pytest
import torch

from agent_app.config import AgentConfig
from agent_app.llm import (
    LocalTransformersChatModel,
    _build_gigachat_llm,
    _build_openai_llm,
    _clean_env_secret,
    _resolve_env_secret,
    build_llm,
)


@tool
def lookup_incident(incident_id: str) -> str:
    """Возвращает состояние учебного инцидента."""
    return incident_id


def test_build_llm_routes_explicit_provider(monkeypatch) -> None:
    """Проверяет маршрутизацию OpenAI/GigaChat/local и отказ от неявного provider."""
    configs = {
        provider: AgentConfig(provider=provider, model="model")
        for provider in ("openai", "gigachat", "local")
    }
    sentinels = {provider: object() for provider in configs}
    with (
        patch("agent_app.llm._build_openai_llm", return_value=sentinels["openai"]),
        patch("agent_app.llm._build_gigachat_llm", return_value=sentinels["gigachat"]),
        patch(
            "agent_app.llm.LocalTransformersChatModel",
            return_value=sentinels["local"],
        ),
    ):
        for provider, config in configs.items():
            assert build_llm(config) is sentinels[provider]

    # Pydantic не допускает неизвестный provider, поэтому проверяем защитную
    # ветку на объекте с тем же runtime-контрактом.
    with pytest.raises(ValueError, match="Неизвестный provider"):
        build_llm(SimpleNamespace(provider="unknown"))


def test_openai_and_gigachat_builders_validate_secrets(monkeypatch) -> None:
    """Проверяет env-секреты и передачу TLS/генерационных параметров SDK."""
    openai_config = AgentConfig(
        provider="openai", model="gpt-test", temperature=0.2, max_retries=2
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _build_openai_llm(openai_config)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with patch("agent_app.llm.ChatOpenAI") as chat_openai:
        _build_openai_llm(openai_config)
    assert chat_openai.call_args.kwargs["model"] == "gpt-test"

    giga_config = AgentConfig(
        provider="gigachat",
        model="GigaChat-2-Pro",
        gigachat_auth_key_env="TEST_GIGA_KEY",
        gigachat_verify_ssl_certs=True,
        max_new_tokens=123,
    )
    monkeypatch.setenv("TEST_GIGA_KEY", "TEST_GIGA_KEY='authorization-key'")
    giga_client = Mock()
    fake_module = SimpleNamespace(GigaChat=giga_client)
    with (
        patch.dict(sys.modules, {"langchain_gigachat.chat_models": fake_module}),
        patch("agent_app.llm.resolve_gigachat_ca_bundle", return_value="ca.pem"),
    ):
        _build_gigachat_llm(giga_config)
    kwargs = giga_client.call_args.kwargs
    assert kwargs["credentials"] == "authorization-key"
    assert kwargs["verify_ssl_certs"] is True
    assert kwargs["ca_bundle_file"] == "ca.pem"
    assert kwargs["max_tokens"] == 123


def test_secret_helpers_strip_assignment_and_reject_missing(monkeypatch) -> None:
    """Проверяет единый разбор env и диагностическую ошибку без секрета."""
    assert _clean_env_secret(" KEY='value' ", "KEY") == "value"
    monkeypatch.setenv("CUSTOM_KEY", "'value'")
    assert _resolve_env_secret("CUSTOM_KEY", provider_name="Provider") == "value"
    monkeypatch.delenv("CUSTOM_KEY")
    with pytest.raises(RuntimeError, match="CUSTOM_KEY"):
        _resolve_env_secret("CUSTOM_KEY", provider_name="Provider")


def _local_model() -> LocalTransformersChatModel:
    """Создаёт локальный chat adapter без загрузки model weights."""
    llm = LocalTransformersChatModel.__new__(LocalTransformersChatModel)
    llm.config = AgentConfig(provider="local", model="local-model")
    llm.bound_tools = [lookup_incident]
    llm.tokenizer = SimpleNamespace(chat_template=None)
    return llm


def test_local_prompt_formats_messages_and_tool_schema() -> None:
    """Проверяет fallback prompt, роли сообщений и JSON schema инструмента."""
    llm = _local_model()
    prompt = llm._format_prompt(
        [
            SystemMessage(content="Соблюдай регламент"),
            HumanMessage(content="Проверь INC-42"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_incident",
                        "args": {"incident_id": "INC-42"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="open", tool_call_id="call-1"),
        ]
    )
    assert "Системная инструкция" in prompt
    assert "Пользователь: Проверь INC-42" in prompt
    assert prompt.endswith("Ассистент:")

    schema = llm._tool_schema(lookup_incident)
    assert schema["function"]["name"] == "lookup_incident"
    assert "incident_id" in schema["function"]["parameters"]["properties"]
    assistant = llm._message_to_dict(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "lookup_incident",
                    "args": {"incident_id": "INC-42"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
    )
    assert assistant["tool_calls"][0]["arguments"] == {"incident_id": "INC-42"}


def test_local_tool_call_parser_accepts_supported_structures() -> None:
    """Проверяет Hermes/XML и fenced JSON, фильтрацию tool и сломанный JSON."""
    llm = _local_model()
    xml = llm._to_ai_message(
        'Перед вызовом <tool_call>{"name":"lookup_incident",'
        '"arguments":{"incident_id":"INC-42"}}</tool_call>'
    )
    assert xml.content == "Перед вызовом"
    assert xml.tool_calls[0]["args"] == {"incident_id": "INC-42"}

    fenced = llm._to_ai_message(
        '```json\n{"name":"lookup_incident",'
        '"arguments":"{\\"incident_id\\":\\"INC-43\\"}"}\n```'
    )
    assert fenced.tool_calls[0]["args"]["incident_id"] == "INC-43"
    assert llm._parse_tool_call_payload({"name": "unknown", "arguments": {}}) is None
    assert llm._parse_tool_call_payload({"name": 1, "arguments": {}}) is None
    plain = llm._to_ai_message("Обычный ответ <tool_call>{broken}</tool_call>")
    assert not plain.tool_calls


def test_local_loader_and_device_helpers(tmp_path: Path) -> None:
    """Проверяет tokenizer/model lifecycle, adapter и явный dtype/device."""
    llm = _local_model()
    llm.config = AgentConfig(
        provider="local",
        model="local-model",
        local_device="cpu",
        local_dtype="fp32",
        trust_remote_code=False,
    )
    tokenizer = SimpleNamespace(
        pad_token=None,
        eos_token="<eos>",
        padding_side="left",
    )
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        assert llm._load_tokenizer() is tokenizer
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "right"

    model = Mock()
    model.to.return_value = model
    llm.device = "cpu"
    llm.dtype = torch.float32
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    llm.config = llm.config.model_copy(update={"adapter_path": adapter})
    adapted = Mock()
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch("peft.PeftModel.from_pretrained", return_value=adapted) as peft_loader,
    ):
        adapted.to.return_value = adapted
        assert llm._load_model() is adapted
    peft_loader.assert_called_once_with(model, str(adapter))
    adapted.eval.assert_called_once()
    assert llm._select_device() == "cpu"
    assert llm._select_dtype("cpu") == torch.float32


def test_chat_template_receives_bound_tool_schemas() -> None:
    """Проверяет, что provider chat template получает tools, а не текстовую вставку."""
    llm = _local_model()
    tokenizer = Mock()
    tokenizer.chat_template = "template"
    tokenizer.apply_chat_template.return_value = "formatted"
    llm.tokenizer = tokenizer

    assert llm._format_prompt("Проверь инцидент") == "formatted"
    kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == "lookup_incident"
    assert kwargs["add_generation_prompt"] is True
