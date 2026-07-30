"""Публичный интерфейс для наблюдаемости агентного сервиса."""

from agent_app.observability.logging import configure_service_logging
from agent_app.observability.telemetry import (
    configure_telemetry,
    current_trace_id,
    instrument_celery,
    instrument_fastapi,
    traced,
)

__all__ = [
    "configure_service_logging",
    "configure_telemetry",
    "current_trace_id",
    "instrument_celery",
    "instrument_fastapi",
    "traced",
]
