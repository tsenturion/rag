"""Регрессионные тесты для подсистемы agent_observability."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

from agent_app.config import load_agent_config
from agent_app.observability.logging import JsonLogFormatter
from agent_app.observability.telemetry import (
    current_trace_id,
    extracted_trace,
    inject_trace_headers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_json_logs_are_structured_and_redact_secrets() -> None:
    """Проверяет, что JSON-логи структурированы, содержат необходимые поля и корректно скрывают секретные значения из сообщений логов."""
    record = logging.LogRecord(
        name="agent.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=private-value",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-1"
    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["request_id"] == "request-1"
    assert "private-value" not in payload["message"]


def test_every_docker_provider_has_observability_profile() -> None:
    """Проверяет, что для каждого Docker-провайдера существует корректный профиль наблюдаемости с включённым сбором метрик, правильными эндпоинтами и настройками логирования."""
    configs = sorted(
        (PROJECT_ROOT / "config").glob("support_agent_docker_*observability.yaml")
    )
    assert len(configs) == 4
    providers = set()
    for path in configs:
        config = load_agent_config(path)
        assert config.observability.enabled
        assert config.observability.otlp_http_endpoint == "http://otel-collector:4318"
        assert config.logging.json_format
        providers.add((config.agent.provider, config.rag.embedding.provider))
    assert providers == {
        ("openai", "openai"),
        ("local", "local"),
        ("gigachat", "openai"),
        ("gigachat", "local"),
    }

    multi_configs = sorted(
        (PROJECT_ROOT / "config").glob("multi_agent_docker_*observability.yaml")
    )
    assert len(multi_configs) == 5
    assert all(load_agent_config(path).multi_agent.enabled for path in multi_configs)

    openai = load_agent_config(
        PROJECT_ROOT / "config" / "multi_agent_docker_openai_observability.yaml"
    )
    assert openai.multi_agent.cost.input_cost_per_million == 0.20
    assert openai.multi_agent.cost.output_cost_per_million == 1.25

    gigachat = load_agent_config(
        PROJECT_ROOT
        / "config"
        / "multi_agent_docker_gigachat_local_embeddings_observability.yaml"
    )
    assert gigachat.multi_agent.cost.input_cost_per_million == 65.0
    assert gigachat.multi_agent.cost.output_cost_per_million == 65.0


def test_observability_assets_exist() -> None:
    """Проверяет наличие и корректность основных конфигурационных файлов и дашбордов для системы наблюдаемости, обеспечивая их доступность и валидность."""
    required = [
        "observability/otel-collector.yaml",
        "observability/prometheus.yaml",
        "observability/alerts.yaml",
        "observability/alertmanager.yaml",
        "observability/grafana/provisioning/datasources/datasources.yaml",
        "observability/grafana/dashboards/support-agent.json",
    ]
    assert all((PROJECT_ROOT / item).exists() for item in required)
    for item in required[:-1]:
        assert isinstance(
            yaml.safe_load((PROJECT_ROOT / item).read_text(encoding="utf-8")), dict
        )
    dashboard = json.loads((PROJECT_ROOT / required[-1]).read_text(encoding="utf-8"))
    prometheus = yaml.safe_load(
        (PROJECT_ROOT / "observability/prometheus.yaml").read_text(encoding="utf-8")
    )
    metrics_job = prometheus["scrape_configs"][0]
    alerts = yaml.safe_load(
        (PROJECT_ROOT / "observability/alerts.yaml").read_text(encoding="utf-8")
    )
    review_alert = next(
        rule
        for group in alerts["groups"]
        for rule in group["rules"]
        if rule["alert"] == "SupportAgentHumanReviewBacklog"
    )
    assert dashboard["uid"] == "rag-support-agent"
    assert len(dashboard["panels"]) >= 3
    assert metrics_job["http_headers"]["X-API-Key"]["files"] == [
        "/run/secrets/support_service_api_key"
    ]
    assert review_alert["expr"] == "max(support_agent_human_reviews_pending) > 10"
    assert "increase(" not in review_alert["expr"]


def test_w3c_trace_context_is_injected_and_extracted() -> None:
    """Проверяет, что контекст трассировки W3C корректно внедряется в заголовки и извлекается из них, обеспечивая сохранение идентификатора текущей трассировки."""
    span_context = SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
        trace_state=TraceState(),
    )
    with trace.use_span(NonRecordingSpan(span_context)):
        headers = inject_trace_headers()
        expected_trace_id = current_trace_id()

    assert headers["traceparent"].startswith("00-1234567890abcdef1234567890abcdef-")
    with extracted_trace(headers):
        assert current_trace_id() == expected_trace_id
