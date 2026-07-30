"""Проверки каскада актуальных моделей GigaChat и изоляции OpenAI."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agent_app.config import AgentConfig, load_agent_config
from agent_app.llm import GigaChatCascadeModel, build_llm


class ProviderError(RuntimeError):
    """Имитирует HTTP-ошибку provider-а без сетевого запроса."""

    def __init__(self, status_code: int):
        """Сохраняет HTTP-статус в том же виде, в котором его читают fallback-правила."""
        self.status_code = status_code
        super().__init__(f"provider status {status_code}")


class FakeChatModel:
    """Фиксирует вызовы и возвращает заданный ответ либо исключение."""

    def __init__(
        self,
        name: str,
        calls: dict[str, int],
        *,
        error: Exception | None = None,
    ):
        """Настраивает тестовую модель с подсчётом вызовов и заданной ошибкой."""
        self.name = name
        self.calls = calls
        self.error = error
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any], **_kwargs: Any) -> FakeChatModel:
        """Фиксирует tools, чтобы проверить их перенос между ступенями каскада."""
        self.bound_tools = list(tools)
        return self

    def invoke(self, _messages: Any, *_args: Any, **_kwargs: Any) -> AIMessage:
        """Имитирует provider-вызов и возвращает имя модели либо заданную ошибку."""
        self.calls[self.name] += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.name)


def _config(provider: str, model: str) -> AgentConfig:
    """Создаёт минимальную конфигурацию с официальным порядком GigaChat."""
    return AgentConfig(
        provider=provider,
        model=model,
        gigachat_model_priority=[
            "GigaChat-3-Ultra",
            "GigaChat-2-Max",
            "GigaChat-2-Pro",
            "GigaChat-2-Lite",
        ],
    )


def test_openai_never_switches_to_another_provider() -> None:
    """OpenAI 429 возвращается вызывающему коду без перехода на GigaChat."""
    calls: dict[str, int] = defaultdict(int)
    openai = FakeChatModel("openai", calls, error=ProviderError(429))

    with (
        patch("agent_app.llm._build_openai_llm", return_value=openai),
        patch("agent_app.llm._build_gigachat_llm") as build_gigachat,
    ):
        llm = build_llm(_config("openai", "gpt-test"))
        with pytest.raises(ProviderError, match="429"):
            llm.invoke("первый запрос")

    assert not isinstance(llm, GigaChatCascadeModel)
    assert calls["openai"] == 1
    build_gigachat.assert_not_called()


def test_gigachat_route_uses_full_priority_and_preserves_tools() -> None:
    """Каскад доходит до Lite, сохраняет tools и не использует alias GigaChat-2."""
    calls: dict[str, int] = defaultdict(int)
    built: list[str] = []
    models: dict[str, FakeChatModel] = {}
    errors = {
        "GigaChat-3-Ultra": ProviderError(403),
        "GigaChat-2-Max": ProviderError(404),
        "GigaChat-2-Pro": ProviderError(429),
    }

    def build_gigachat(config: AgentConfig) -> FakeChatModel:
        """Создаёт ступень с ошибкой доступности, заданной для её модели."""
        built.append(config.model)
        model = FakeChatModel(config.model, calls, error=errors.get(config.model))
        models[config.model] = model
        return model

    with patch(
        "agent_app.llm._build_gigachat_llm",
        side_effect=build_gigachat,
    ):
        llm = build_llm(_config("gigachat", "GigaChat-3-Ultra"))
        tool = object()
        bound = llm.bind_tools([tool])
        response = bound.invoke("используй инструмент")
        repeated = bound.invoke("повтори")

    assert response.content == "GigaChat-2-Lite"
    assert repeated.content == "GigaChat-2-Lite"
    assert built == [
        "GigaChat-3-Ultra",
        "GigaChat-2-Max",
        "GigaChat-2-Pro",
        "GigaChat-2-Lite",
    ]
    assert models["GigaChat-2-Lite"].bound_tools == [tool]
    assert calls["GigaChat-3-Ultra"] == 1
    assert calls["GigaChat-2-Max"] == 1
    assert calls["GigaChat-2-Pro"] == 1
    assert calls["GigaChat-2-Lite"] == 2
    assert "GigaChat-2" not in built


def test_gigachat_authentication_error_does_not_start_cascade() -> None:
    """Один неверный Authorization key не повторяется на остальных моделях."""
    calls: dict[str, int] = defaultdict(int)
    built: list[str] = []

    def build_gigachat(config: AgentConfig) -> FakeChatModel:
        """Создаёт ступень, которая воспроизводит ошибку авторизации."""
        built.append(config.model)
        return FakeChatModel(config.model, calls, error=ProviderError(401))

    with patch(
        "agent_app.llm._build_gigachat_llm",
        side_effect=build_gigachat,
    ):
        llm = build_llm(_config("gigachat", "GigaChat-3-Ultra"))
        with pytest.raises(ProviderError, match="401"):
            llm.invoke("запрос")

    assert built == ["GigaChat-3-Ultra"]
    assert calls["GigaChat-3-Ultra"] == 1


def test_gigachat_primary_must_match_first_priority_item() -> None:
    """Не допускает скрытого запуска более слабой модели перед Ultra."""
    with pytest.raises(ValidationError, match="первым элементом"):
        _config("gigachat", "GigaChat-2-Max")


def test_gigachat_yaml_uses_official_model_route_and_endpoint() -> None:
    """Фиксирует официальный порядок GigaChat и новый единый API endpoint."""
    project_root = Path(__file__).resolve().parents[1]
    config = load_agent_config(project_root / "config" / "agent_gigachat.yaml")

    assert config.agent.model == "GigaChat-3-Ultra"
    assert config.agent.gigachat_model_priority == [
        "GigaChat-3-Ultra",
        "GigaChat-2-Max",
        "GigaChat-2-Pro",
        "GigaChat-2-Lite",
    ]
    assert config.agent.gigachat_base_url == "https://api.giga.chat/v1"
    assert config.agent.gigachat_verify_ssl_certs is True
    assert "GigaChat-2" not in config.agent.gigachat_model_priority
