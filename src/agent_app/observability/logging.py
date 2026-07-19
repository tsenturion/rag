"""Структурированное журналирование для наблюдаемости агентного сервиса."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from agent_app.support.security import redact_secrets


class JsonLogFormatter(logging.Formatter):
    """Гарантирует структурированный JSON-лог с метаданными и защитой секретов для интеграции с системами мониторинга."""

    def format(self, record: logging.LogRecord) -> str:
        """Гарантирует формирование структурированного JSON-лога с метаданными, трассировкой и защитой секретов для наблюдаемости."""
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
        }
        for name in (
            "event",
            "request_id",
            "user_id",
            "session_id",
            "run_id",
            "job_id",
            "method",
            "path",
            "status",
            "duration_ms",
        ):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        try:
            from opentelemetry import trace

            context = trace.get_current_span().get_span_context()
            if context.is_valid:
                payload["trace_id"] = format(context.trace_id, "032x")
                payload["span_id"] = format(context.span_id, "016x")
        except ImportError:
            pass
        if record.exc_info:
            payload["exception"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_service_logging(level: str, *, json_format: bool) -> None:
    """Гарантирует настройку формата и уровня логирования для всего сервиса с поддержкой структурированных и человекочитаемых логов."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    formatter: logging.Formatter
    if json_format:
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
    for handler in root.handlers:
        handler.setFormatter(formatter)
