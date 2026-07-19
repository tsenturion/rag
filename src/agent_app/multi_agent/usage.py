"""Учёт токенов и стоимости для мультиагентной системы."""

from __future__ import annotations

import threading
from time import perf_counter
from typing import Any, Callable

import tiktoken

from agent_app.currency import CBRCurrencyConverter, CurrencyConversionResult
from agent_app.multi_agent.models import UsageMetrics
from agent_app.multi_agent.llm_routing import LLMRoute


class LLMCallTracker:
    """Собирает provider usage или воспроизводимую token-оценку при его отсутствии."""

    def __init__(
        self,
        llm: Any,
        *,
        model: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
        cost_currency: str = "RUB",
        currency_converter: CBRCurrencyConverter | None = None,
        serialize_calls: bool = False,
        token_budget: int | None = None,
        max_output_tokens: int = 0,
        route_resolver: Callable[[str], LLMRoute] | None = None,
    ):
        """Гарантирует потокобезопасное отслеживание использования LLM с учётом ограничений бюджета токенов и параметров маршрутизации."""
        self.llm = llm
        self.model = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.cost_currency = cost_currency
        self.currency_converter = currency_converter or CBRCurrencyConverter()
        self.serialize_calls = serialize_calls
        self.token_budget = token_budget
        self.max_output_tokens = max_output_tokens
        self.route_resolver = route_resolver
        self._lock = threading.RLock()
        self._invoke_locks: dict[int, threading.RLock] = {}
        self._usage = UsageMetrics()
        self._inflight_reserved_tokens = 0

    def invoke(self, messages: list[Any], role: str) -> str:
        """Гарантирует выполнение вызова LLM с учётом политики маршрутизации и учёта использования для последующего анализа."""
        response = self.invoke_response(messages, role)
        return self._content(response)

    def invoke_response(
        self,
        messages: list[Any],
        role: str,
        *,
        tools: list[Any] | None = None,
    ) -> Any:
        """Гарантирует атомарное резервирование токенов и корректный учёт стоимости при вызове LLM с учётом маршрутизации и ограничений бюджета в мультиагентной системе."""
        route = self.route_resolver(role) if self.route_resolver is not None else None
        llm = route.llm if route is not None else self.llm
        model = route.model if route is not None else self.model
        max_output_tokens = (
            route.max_output_tokens if route is not None else self.max_output_tokens
        )
        input_cost = (
            route.input_cost_per_million
            if route is not None
            else self.input_cost_per_million
        )
        output_cost = (
            route.output_cost_per_million
            if route is not None
            else self.output_cost_per_million
        )
        cost_currency = route.cost_currency if route is not None else self.cost_currency
        serialize = route.serialize_calls if route is not None else self.serialize_calls
        input_text = "\n".join(str(getattr(item, "content", item)) for item in messages)
        reserved_tokens = (
            self.estimate_tokens(input_text, model=model) + max_output_tokens
        )
        with self._lock:
            if (
                self.token_budget is not None
                and self._usage.total_tokens
                + self._inflight_reserved_tokens
                + reserved_tokens
                > self.token_budget
            ):
                raise RuntimeError(
                    f"Token budget исчерпан перед вызовом роли {role}: "
                    f"used={self._usage.total_tokens} "
                    f"inflight={self._inflight_reserved_tokens} "
                    f"reserve={reserved_tokens} "
                    f"limit={self.token_budget}"
                )
            self._inflight_reserved_tokens += reserved_tokens
        started = perf_counter()
        try:
            target: Any = llm
            if tools:
                bind_tools = getattr(llm, "bind_tools", None)
                if not callable(bind_tools):
                    raise RuntimeError(
                        f"LLM роли {role} не поддерживает function calling"
                    )
                if not serialize:
                    target = bind_tools(tools)
            if serialize:
                with self._invoke_lock(llm):
                    previous_tools = getattr(llm, "bound_tools", None)
                    try:
                        if tools:
                            target = llm.bind_tools(tools)
                        response = target.invoke(messages)
                    finally:
                        if tools and previous_tools is not None:
                            llm.bound_tools = previous_tools
            else:
                response = target.invoke(messages)
            duration_ms = (perf_counter() - started) * 1000
            content = self._content(response)
            input_tokens, output_tokens, estimated = self._tokens(
                response,
                input_text=input_text,
                output_text=content,
                model=model,
            )
            cost = (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000
            conversion = self.currency_converter.convert(cost, cost_currency)
            delta = UsageMetrics(
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_tokens=estimated,
                duration_ms=round(duration_ms, 3),
                estimated_cost=round(cost, 8),
                estimated_cost_currency=cost_currency,
                costs_by_currency={cost_currency: round(cost, 8)},
                **self._conversion_fields(conversion),
            )
        except BaseException:
            with self._lock:
                self._inflight_reserved_tokens -= reserved_tokens
            raise
        with self._lock:
            self._inflight_reserved_tokens -= reserved_tokens
            self._usage = self._usage.add(delta)
        return response

    def snapshot(self) -> UsageMetrics:
        """Гарантирует получение согласованной копии накопленной метрики использования LLM для анализа или мониторинга."""
        with self._lock:
            return self._usage.model_copy(deep=True)

    def delta_since(self, before: UsageMetrics) -> UsageMetrics:
        """Вычисляет разницу метрик использования LLM между двумя моментами времени для отслеживания инкрементальных затрат."""
        return self.snapshot().subtract(before)

    def _tokens(
        self,
        response: Any,
        *,
        input_text: str,
        output_text: str,
        model: str,
    ) -> tuple[int, int, int]:
        """Гарантирует получение фактического или оценочного количества токенов ввода и вывода для корректного учёта расходов."""
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        metadata = getattr(response, "response_metadata", None) or {}
        token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
        input_tokens = input_tokens or int(
            token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
        )
        output_tokens = output_tokens or int(
            token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        )
        if input_tokens and output_tokens:
            return input_tokens, output_tokens, 0
        estimated_input = self.estimate_tokens(input_text, model=model)
        estimated_output = self.estimate_tokens(output_text, model=model)
        return (
            input_tokens or estimated_input,
            output_tokens or estimated_output,
            (0 if input_tokens else estimated_input)
            + (0 if output_tokens else estimated_output),
        )

    def estimate_tokens(self, text: str, *, model: str | None = None) -> int:
        """Гарантирует воспроизводимую оценку числа токенов для текста с учётом модели или дефолтного кодека, даже при ошибках."""
        try:
            encoding = tiktoken.encoding_for_model(model or self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        try:
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def _invoke_lock(self, llm: Any) -> threading.RLock:
        """Гарантирует эксклюзивный доступ к вызовам LLM для конкретного экземпляра, предотвращая гонки при параллельных обращениях."""
        with self._lock:
            return self._invoke_locks.setdefault(id(llm), threading.RLock())

    @staticmethod
    def _conversion_fields(conversion: CurrencyConversionResult) -> dict[str, Any]:
        """Переносит проверяемые параметры конвертации в публичные usage-метрики."""
        currency = conversion.source_currency
        return {
            "estimated_cost_rub": (
                round(conversion.amount_rub, 8)
                if conversion.amount_rub is not None
                else None
            ),
            "exchange_rates_to_rub": (
                {currency: conversion.rate_to_rub}
                if conversion.rate_to_rub is not None
                else {}
            ),
            "exchange_rate_dates": (
                {currency: conversion.rate_date}
                if conversion.rate_date is not None
                else {}
            ),
            "exchange_rate_source": conversion.source_url,
            "exchange_rate_stale": conversion.stale,
            "currency_conversion_errors": (
                [f"{currency}: {conversion.error}"] if conversion.error else []
            ),
        }

    @staticmethod
    def _content(response: Any) -> str:
        """Гарантирует получение текстового представления содержимого ответа LLM независимо от структуры объекта."""
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)


def estimate_mode_usage(
    *,
    request: str,
    answer: str,
    model: str,
    llm_calls: int,
    tool_calls: int,
    duration_ms: float,
    input_cost_per_million: float,
    output_cost_per_million: float,
    cost_currency: str = "RUB",
    currency_converter: CBRCurrencyConverter | None = None,
) -> UsageMetrics:
    """Гарантирует воспроизводимую оценку токенов и стоимости выполнения запроса с учётом параметров модели и числа вызовов."""
    tracker = LLMCallTracker(
        llm=None,
        model=model,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        cost_currency=cost_currency,
        currency_converter=currency_converter,
    )
    input_tokens = tracker.estimate_tokens(request) * max(1, llm_calls)
    output_tokens = tracker.estimate_tokens(answer)
    cost = (
        input_tokens * input_cost_per_million + output_tokens * output_cost_per_million
    ) / 1_000_000
    conversion = tracker.currency_converter.convert(cost, cost_currency)
    return UsageMetrics(
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_tokens=input_tokens + output_tokens,
        tool_calls=tool_calls,
        duration_ms=round(duration_ms, 3),
        estimated_cost=round(cost, 8),
        estimated_cost_currency=cost_currency,
        costs_by_currency={cost_currency: round(cost, 8)},
        **tracker._conversion_fields(conversion),
    )
