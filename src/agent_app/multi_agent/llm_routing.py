"""Маршрутизация ролей по LLM-провайдерам для мультиагентной системы."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

from agent_app.config import AgentAppConfig, AgentConfig
from agent_app.llm import build_llm
from agent_app.multi_agent.models import LLMRouteInfo

LOGGER = logging.getLogger(__name__)
MULTI_AGENT_LLM_ROLES = (
    # Список задаёт полный контракт маршрутизации: роль, отсутствующая здесь,
    # не сможет неявно получить произвольный LLM-профиль.
    "planner",
    "knowledge_agent",
    "diagnostics_agent",
    "incident_agent",
    "tool_agent",
    "critic_agent",
    "coordinator",
)


@dataclass(frozen=True)
class LLMRoute:
    """Разрешённый LLM-маршрут роли вместе с лимитами и тарифами."""

    role: str
    profile: str
    provider: str
    model: str
    llm: Any
    input_cost_per_million: float
    output_cost_per_million: float
    cost_currency: str
    max_output_tokens: int
    serialize_calls: bool


class MultiAgentLLMRegistry:
    """Создаёт LLM один раз на профиль и маршрутизирует вызовы по ролям."""

    def __init__(
        self,
        config: AgentAppConfig,
        *,
        default_llm: Any | None = None,
        role_llms: dict[str, Any] | None = None,
    ):
        """Гарантирует, что маршрутизация LLM по ролям и профилям будет корректно настроена и все необходимые клиенты будут созданы или переиспользованы для мультиагентной сессии."""
        self.config = config
        self._owned_clients: list[Any] = []
        self.default_llm = default_llm or self._build_owned(
            config.agent,
            label="default",
        )
        self._profile_clients: dict[str, Any] = {}
        self._routes: dict[str, LLMRoute] = {}
        injected = role_llms or {}
        for role in MULTI_AGENT_LLM_ROLES:
            profile_name = config.multi_agent.role_llm_profiles.get(role)
            if profile_name is None:
                agent_config: AgentConfig = config.agent
                llm = injected.get(role, self.default_llm)
                input_cost = config.multi_agent.cost.input_cost_per_million
                output_cost = config.multi_agent.cost.output_cost_per_million
                cost_currency = config.multi_agent.cost.currency
                label = "default"
            else:
                profile = config.multi_agent.llm_profiles[profile_name]
                agent_config = profile
                llm = injected.get(role)
                if llm is None:
                    # Один профиль разделяет клиент между ролями: локальная модель
                    # не загружается в память повторно, а HTTP-клиенты переиспользуются.
                    llm = self._profile_clients.get(profile_name)
                if llm is None:
                    llm = self._build_owned(profile, label=profile_name)
                    self._profile_clients[profile_name] = llm
                input_cost = profile.input_cost_per_million
                output_cost = profile.output_cost_per_million
                cost_currency = profile.currency
                label = profile_name
            self._routes[role] = LLMRoute(
                role=role,
                profile=label,
                provider=agent_config.provider,
                model=agent_config.model,
                llm=llm,
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
                cost_currency=cost_currency,
                max_output_tokens=agent_config.max_new_tokens,
                # Один экземпляр локальной Transformers-модели не должен получать
                # конкурентные generate-вызовы; runtime использует этот флаг.
                serialize_calls=agent_config.provider == "local",
            )

    @property
    def has_local_routes(self) -> bool:
        """Показывает, требует ли хотя бы одна роль локального ускорителя."""
        return any(route.provider == "local" for route in self._routes.values())

    @property
    def provider_summary(self) -> str:
        """Возвращает единственного провайдера либо маркер смешанной схемы."""
        providers = sorted({route.provider for route in self._routes.values()})
        return providers[0] if len(providers) == 1 else "mixed"

    @property
    def model_summary(self) -> str:
        """Возвращает единственную модель либо маркер смешанной схемы."""
        models = sorted({route.model for route in self._routes.values()})
        return models[0] if len(models) == 1 else "mixed"

    def route(self, role: str) -> LLMRoute:
        """Возвращает явно настроенный маршрут и отвергает неизвестную роль."""
        route = self._routes.get(role)
        if route is None:
            raise ValueError(f"Для неизвестной роли нет LLM-маршрута: {role}")
        return route

    def route_info(self) -> list[LLMRouteInfo]:
        """Формирует безопасное описание маршрутов без клиентских объектов."""
        return [
            LLMRouteInfo(
                role=role,
                profile=route.profile,
                provider=route.provider,
                model=route.model,
                cost_currency=route.cost_currency,
            )
            for role, route in self._routes.items()
        ]

    def close(self) -> None:
        """Закрывает только принадлежащие registry клиенты и освобождает accelerator cache."""
        # Несколько ролей могут ссылаться на один клиент; закрывать его нужно один раз.
        clients = list({id(client): client for client in self._owned_clients}.values())
        owned_ids = {id(client) for client in clients}
        clear_accelerator = any(
            route.provider == "local" and id(route.llm) in owned_ids
            for route in self._routes.values()
        )
        self._owned_clients.clear()
        self._profile_clients.clear()
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    LOGGER.exception("Не удалось закрыть LLM client")
        self._routes.clear()
        gc.collect()
        if not clear_accelerator:
            return
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            LOGGER.debug("Не удалось очистить accelerator cache", exc_info=True)

    def _build_owned(self, config: AgentConfig, *, label: str) -> Any:
        """Создаёт клиент профиля и регистрирует владение его lifecycle."""
        try:
            llm = build_llm(config)
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось инициализировать LLM-профиль {label}: {exc}"
            ) from exc
        self._owned_clients.append(llm)
        return llm
