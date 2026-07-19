"""Проверки конвертации API-расходов в рубли по данным Банка России."""

from __future__ import annotations

import httpx2
from dataclasses import replace
from time import monotonic

from agent_app.config import CurrencyConversionConfig, load_agent_config
from agent_app.currency import CBRCurrencyConverter
from agent_app.multi_agent.models import UsageMetrics

CBR_XML = b"""<?xml version="1.0" encoding="windows-1251"?>
<ValCurs Date="18.07.2026" name="Foreign Currency Market">
  <Valute ID="R01235">
    <CharCode>USD</CharCode><Nominal>1</Nominal><Value>80,5000</Value>
  </Valute>
  <Valute ID="R01375">
    <CharCode>CNY</CharCode><Nominal>10</Nominal><Value>110,0000</Value>
  </Valute>
</ValCurs>
"""


def _converter(handler: object) -> CBRCurrencyConverter:
    """Создаёт конвертер с управляемым HTTP transport без внешней сети."""
    transport = httpx2.MockTransport(handler)

    def client_factory(**kwargs: object) -> httpx2.Client:
        """Подключает тестовый transport, сохраняя production-параметры клиента."""
        return httpx2.Client(transport=transport, **kwargs)

    return CBRCurrencyConverter(client_factory=client_factory)


def test_cbr_conversion_accounts_for_nominal_and_caches_rates() -> None:
    """Проверяет курс за единицу, дату ЦБ и отсутствие повторной загрузки."""
    requests = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Возвращает фиксированный официальный XML-снимок для двух валют."""
        nonlocal requests
        requests += 1
        return httpx2.Response(200, content=CBR_XML, request=request)

    converter = _converter(handler)
    usd = converter.convert(2.0, "usd")
    cny = converter.convert(3.0, "CNY")

    assert usd.amount_rub == 161.0
    assert usd.rate_to_rub == 80.5
    assert usd.rate_date == "2026-07-18"
    assert cny.amount_rub == 33.0
    assert cny.rate_to_rub == 11.0
    assert requests == 1


def test_zero_and_rub_costs_do_not_call_cbr() -> None:
    """Проверяет, что математически однозначные суммы не создают HTTP-трафик."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Обнаруживает неожиданный сетевой запрос для нулевой или RUB-суммы."""
        raise AssertionError(f"Неожиданный запрос: {request.url}")

    converter = _converter(handler)

    assert converter.convert(0.0, "USD").amount_rub == 0.0
    assert converter.convert(12.5, "RUB").amount_rub == 12.5


def test_failed_conversion_never_invents_rub_amount() -> None:
    """Проверяет fail-open результат с диагностикой вместо фиктивного курса."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        """Имитирует недоступность официального сервиса курсов."""
        return httpx2.Response(503, request=request)

    result = _converter(handler).convert(1.0, "USD")

    assert result.amount_rub is None
    assert result.error
    assert result.source_amount == 1.0
    assert result.source_currency == "USD"


def test_expired_cache_is_marked_stale_when_cbr_is_temporarily_unavailable() -> None:
    """Проверяет явную маркировку fallback на ранее полученный курс."""
    available = True

    def handler(request: httpx2.Request) -> httpx2.Response:
        """После первой загрузки имитирует временную ошибку Банка России."""
        if available:
            return httpx2.Response(200, content=CBR_XML, request=request)
        return httpx2.Response(503, request=request)

    converter = _converter(handler)
    first = converter.convert(1.0, "USD")
    assert converter._table is not None
    converter._table = replace(
        converter._table,
        fetched_monotonic=(monotonic() - converter.config.cache_ttl_seconds - 1.0),
    )
    available = False

    stale = converter.convert(1.0, "USD")

    assert stale.amount_rub == first.amount_rub
    assert stale.stale is True
    assert stale.rate_date == "2026-07-18"


def test_usage_aggregation_keeps_source_currencies_and_sums_rub() -> None:
    """Проверяет, что mixed usage не складывает USD и RUB как одну валюту."""
    usd = UsageMetrics(
        estimated_cost=1.0,
        estimated_cost_currency="USD",
        costs_by_currency={"USD": 1.0},
        estimated_cost_rub=80.5,
        exchange_rates_to_rub={"USD": 80.5},
        exchange_rate_dates={"USD": "2026-07-18"},
    )
    rub = UsageMetrics(
        estimated_cost=65.0,
        estimated_cost_currency="RUB",
        costs_by_currency={"RUB": 65.0},
        estimated_cost_rub=65.0,
        exchange_rates_to_rub={"RUB": 1.0},
    )

    total = usd.add(rub)

    assert total.costs_by_currency == {"USD": 1.0, "RUB": 65.0}
    assert total.estimated_cost == 145.5
    assert total.estimated_cost_currency == "RUB"
    assert total.estimated_cost_rub == 145.5


def test_currency_config_rejects_non_cbr_rate_source() -> None:
    """Проверяет защиту от незаметной подмены официального источника курса."""
    try:
        CurrencyConversionConfig(cbr_daily_rates_url="https://example.com/rates")
    except ValueError as exc:
        assert "cbr.ru" in str(exc)
    else:
        raise AssertionError("Неофициальный источник курса должен быть отклонён")


def test_provider_profiles_declare_billing_currency() -> None:
    """Проверяет явную валюту OpenAI, GigaChat и локального тарифа."""
    openai = load_agent_config("config/support_agent_openai.yaml")
    gigachat = load_agent_config("config/support_agent_gigachat_local_embeddings.yaml")
    local = load_agent_config("config/support_agent_local.yaml")

    assert openai.multi_agent.cost.currency == "USD"
    assert gigachat.multi_agent.cost.currency == "RUB"
    assert local.multi_agent.cost.currency == "RUB"
