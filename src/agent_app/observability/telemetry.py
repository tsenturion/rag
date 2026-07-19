"""Трассировка и инструментирование для наблюдаемости агентного сервиса."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

from agent_app.config import ObservabilityConfig

LOGGER = logging.getLogger(__name__)
_LOCK = threading.RLock()
_CONFIGURED_SERVICES: set[str] = set()


def configure_telemetry(config: ObservabilityConfig) -> bool:
    """Однократно настраивает глобальный OpenTelemetry provider процесса."""
    if not config.enabled:
        return False
    with _LOCK:
        # OpenTelemetry хранит provider глобально. Повторная инициализация при
        # создании нескольких runtime приводит к дублированию экспортируемых span.
        if config.service_name in _CONFIGURED_SERVICES:
            return True
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        current = trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            LOGGER.warning(
                "TracerProvider уже настроен; используется существующий provider."
            )
            _CONFIGURED_SERVICES.add(config.service_name)
            return True
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": config.service_name,
                    "deployment.environment.name": config.environment,
                }
            ),
            # Дочерние span наследуют решение родителя, чтобы один trace не
            # превращался в несвязанный набор частично записанных операций.
            sampler=ParentBased(TraceIdRatioBased(config.trace_sample_ratio)),
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=f"{config.otlp_http_endpoint}/v1/traces")
            )
        )
        trace.set_tracer_provider(provider)
        _CONFIGURED_SERVICES.add(config.service_name)
        return True


def instrument_fastapi(app: object, config: ObservabilityConfig) -> None:
    """Подключает автотрейсинг FastAPI и исходящих HTTP-запросов."""
    if not configure_telemetry(config):
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="/health,/metrics")
    if config.instrument_http_clients:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()


def instrument_celery(config: ObservabilityConfig) -> None:
    """Подключает распространение trace-контекста через задачи Celery."""
    if not config.enabled or not config.instrument_celery:
        return
    configure_telemetry(config)
    from opentelemetry.instrumentation.celery import CeleryInstrumentor

    CeleryInstrumentor().instrument()


@contextmanager
def traced(name: str, **attributes: object) -> Iterator[object | None]:
    """Создаёт span, сохраняя работоспособность без optional OTel-зависимостей."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("agent_app")
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span
    except ImportError:
        yield None


def current_trace_id() -> str | None:
    """Возвращает идентификатор текущего валидного trace для логов и ответа API."""
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        return format(context.trace_id, "032x") if context.is_valid else None
    except ImportError:
        return None


def inject_trace_headers(headers: dict[str, str] | None = None) -> dict[str, str]:
    """Добавляет W3C trace context в заголовки исходящего сообщения."""
    carrier = dict(headers or {})
    try:
        from opentelemetry.propagate import inject

        inject(carrier)
    except ImportError:
        pass
    return carrier


@contextmanager
def extracted_trace(headers: dict[str, str] | None) -> Iterator[None]:
    """Временно активирует входящий trace context и гарантированно отсоединяет его."""
    try:
        from opentelemetry import context as otel_context
        from opentelemetry.propagate import extract

        token = otel_context.attach(extract(headers or {}))
    except ImportError:
        token = None
        otel_context = None
    try:
        yield
    finally:
        # ContextVar обязательно отсоединяется: иначе worker может приписать
        # следующую независимую задачу к trace предыдущего сообщения.
        if token is not None and otel_context is not None:
            otel_context.detach(token)
