"""Конвертация стоимости внешних API в рубли по курсам Банка России."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from time import monotonic
from typing import Any, Callable
from xml.etree import ElementTree

import httpx2

from agent_app.config import CurrencyConversionConfig

LOGGER = logging.getLogger(__name__)


class CurrencyConversionError(RuntimeError):
    """Сообщает, что официальный курс не удалось получить или применить."""


@dataclass(frozen=True)
class CurrencyRateTable:
    """Хранит снимок курсов к RUB и дату их действия по данным ЦБ РФ."""

    rates_to_rub: dict[str, Decimal]
    rate_date: str
    fetched_at: str
    fetched_monotonic: float


@dataclass(frozen=True)
class CurrencyConversionResult:
    """Описывает исходную сумму и результат её проверяемой конвертации в RUB."""

    source_amount: float
    source_currency: str
    amount_rub: float | None
    rate_to_rub: float | None
    rate_date: str | None
    source_url: str | None
    stale: bool = False
    error: str | None = None


class CBRCurrencyConverter:
    """Получает официальные курсы ЦБ РФ и переиспользует их в пределах TTL.

    Клиент ленивый: RUB и нулевая стоимость не требуют HTTP-запроса. При
    недоступности Банка России ранее успешно загруженная таблица может быть
    использована как stale, но этот факт сохраняется в метаданных результата.
    """

    def __init__(
        self,
        config: CurrencyConversionConfig | None = None,
        *,
        client_factory: Callable[..., Any] = httpx2.Client,
    ):
        """Создаёт потокобезопасный конвертер без немедленного сетевого вызова."""
        self.config = config or CurrencyConversionConfig()
        self._client_factory = client_factory
        self._lock = threading.RLock()
        self._table: CurrencyRateTable | None = None

    def convert(self, amount: float, currency: str) -> CurrencyConversionResult:
        """Сохраняет исходную сумму и рассчитывает эквивалент в рублях."""
        normalized_currency = currency.strip().upper()
        if normalized_currency == "RUB":
            return CurrencyConversionResult(
                source_amount=amount,
                source_currency="RUB",
                amount_rub=amount,
                rate_to_rub=1.0,
                rate_date=None,
                source_url=None,
            )
        if amount == 0:
            # Нулевой расход равен нулю в любой валюте и не оправдывает
            # отдельный сетевой запрос только ради курса.
            return CurrencyConversionResult(
                source_amount=amount,
                source_currency=normalized_currency,
                amount_rub=0.0,
                rate_to_rub=None,
                rate_date=None,
                source_url=None,
            )
        if not self.config.enabled:
            return self._failure(
                amount,
                normalized_currency,
                "Конвертация валют отключена конфигурацией",
            )

        try:
            table, stale = self._rate_table()
            rate = table.rates_to_rub.get(normalized_currency)
            if rate is None:
                raise CurrencyConversionError(
                    f"ЦБ РФ не вернул курс валюты {normalized_currency}"
                )
            rub = Decimal(str(amount)) * rate
            return CurrencyConversionResult(
                source_amount=amount,
                source_currency=normalized_currency,
                amount_rub=float(rub),
                rate_to_rub=float(rate),
                rate_date=table.rate_date,
                source_url=self.config.cbr_daily_rates_url,
                stale=stale,
            )
        except Exception as exc:
            if self.config.fail_on_error:
                raise CurrencyConversionError(str(exc)) from exc
            LOGGER.warning(
                "Не удалось пересчитать стоимость %s в RUB по курсу ЦБ РФ: %s",
                normalized_currency,
                exc,
            )
            return self._failure(amount, normalized_currency, str(exc))

    def refresh(self) -> CurrencyRateTable:
        """Принудительно обновляет кэш и возвращает полученную таблицу курсов."""
        with self._lock:
            self._table = self._fetch_rate_table()
            return self._table

    def _rate_table(self) -> tuple[CurrencyRateTable, bool]:
        """Возвращает свежую таблицу либо явно помеченный stale-снимок."""
        with self._lock:
            if self._table is not None and not self._expired(self._table):
                return self._table, False
            previous = self._table
            try:
                self._table = self._fetch_rate_table()
                return self._table, False
            except Exception:
                if previous is not None and self.config.allow_stale_on_error:
                    LOGGER.warning(
                        "ЦБ РФ недоступен; используется устаревший снимок курсов от %s",
                        previous.rate_date,
                    )
                    return previous, True
                raise

    def _fetch_rate_table(self) -> CurrencyRateTable:
        """Загружает XML_daily.asp и приводит номинальные котировки к одной единице."""
        with self._client_factory(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = client.get(self.config.cbr_daily_rates_url)
            response.raise_for_status()
        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            raise CurrencyConversionError("ЦБ РФ вернул некорректный XML") from exc

        rates: dict[str, Decimal] = {"RUB": Decimal("1")}
        for item in root.findall("Valute"):
            code = (item.findtext("CharCode") or "").strip().upper()
            nominal_text = (item.findtext("Nominal") or "").strip()
            value_text = (item.findtext("Value") or "").strip().replace(",", ".")
            if not code or not nominal_text or not value_text:
                continue
            nominal = Decimal(nominal_text)
            if nominal <= 0:
                continue
            rates[code] = Decimal(value_text) / nominal
        if len(rates) == 1:
            raise CurrencyConversionError("В ответе ЦБ РФ отсутствуют курсы валют")

        date_text = root.attrib.get("Date", "").strip()
        try:
            rate_date = datetime.strptime(date_text, "%d.%m.%Y").date().isoformat()
        except ValueError as exc:
            raise CurrencyConversionError(
                "ЦБ РФ вернул некорректную дату курса"
            ) from exc
        return CurrencyRateTable(
            rates_to_rub=rates,
            rate_date=rate_date,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            fetched_monotonic=monotonic(),
        )

    def _expired(self, table: CurrencyRateTable) -> bool:
        """Проверяет TTL по монотонным часам, не зависящим от перевода системного времени."""
        return monotonic() - table.fetched_monotonic >= self.config.cache_ttl_seconds

    def _failure(
        self,
        amount: float,
        currency: str,
        error: str,
    ) -> CurrencyConversionResult:
        """Формирует fail-open результат без выдуманного рублёвого эквивалента."""
        return CurrencyConversionResult(
            source_amount=amount,
            source_currency=currency,
            amount_rub=None,
            rate_to_rub=None,
            rate_date=None,
            source_url=self.config.cbr_daily_rates_url,
            error=error,
        )
