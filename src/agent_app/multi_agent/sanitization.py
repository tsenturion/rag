"""Централизованная очистка публичных и сохраняемых multi-agent результатов."""

from __future__ import annotations

from typing import Any

from agent_app.guardrails import GuardrailPipeline
from agent_app.multi_agent.models import (
    AgentEnvelope,
    AgentModeResult,
    AgentTask,
    AgentTaskResult,
    ComparisonCaseResult,
    LifecycleEvent,
    LLMRouteInfo,
    MultiAgentComparisonReport,
    MultiAgentResponse,
    MultiAgentRunResult,
    QualityAssessment,
    UsageMetrics,
)
from agent_app.rag.models import RagCitation


def sanitize_run_result(
    result: MultiAgentRunResult,
    guardrails: GuardrailPipeline,
) -> MultiAgentRunResult:
    """Очищает ответ, журналы сообщений и dead letters до экспорта или передачи."""
    return result.model_copy(
        update={
            "response": sanitize_response(result.response, guardrails),
            "messages": [
                _sanitize_envelope(envelope, guardrails) for envelope in result.messages
            ],
            "dead_letters": [
                _sanitize_envelope(envelope, guardrails)
                for envelope in result.dead_letters
            ],
        },
        deep=True,
    )


def sanitize_response(
    response: MultiAgentResponse,
    guardrails: GuardrailPipeline,
) -> MultiAgentResponse:
    """Применяет output guardrails ко всем полям с данными LLM, RAG и tools."""
    return response.model_copy(
        update={
            "answer": _text(response.answer, guardrails),
            "tasks": [_sanitize_task(task, guardrails) for task in response.tasks],
            "task_results": [
                _sanitize_task_result(result, guardrails)
                for result in response.task_results
            ],
            "citations": [
                _sanitize_citation(citation, guardrails)
                for citation in response.citations
            ],
            "review": _text(response.review, guardrails),
            "llm_routes": [
                _sanitize_route(route, guardrails) for route in response.llm_routes
            ],
            "lifecycle": [
                _sanitize_lifecycle(event, guardrails) for event in response.lifecycle
            ],
            "usage": _sanitize_usage(response.usage, guardrails),
            "quality": (
                _sanitize_quality(response.quality, guardrails)
                if response.quality is not None
                else None
            ),
        },
        deep=True,
    )


def sanitize_comparison_report(
    report: MultiAgentComparisonReport,
    guardrails: GuardrailPipeline,
) -> MultiAgentComparisonReport:
    """Очищает оба ответа и диагностические строки сравнительного отчёта."""
    return report.model_copy(
        update={
            "provider": _text(report.provider, guardrails),
            "model": _text(report.model, guardrails),
            "cases": [_sanitize_case(case, guardrails) for case in report.cases],
            "llm_routes": [
                _sanitize_route(route, guardrails) for route in report.llm_routes
            ],
        },
        deep=True,
    )


def public_run_reference(run_dir: str | None) -> str | None:
    """Возвращает только имя каталога запуска без локальной структуры файловой системы."""
    if not run_dir:
        return None
    normalized = run_dir.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] or None


def _sanitize_task(task: AgentTask, guardrails: GuardrailPipeline) -> AgentTask:
    """Очищает сформированные LLM название и инструкцию подзадачи."""
    return task.model_copy(
        update={
            "title": _text(task.title, guardrails),
            "instruction": _text(task.instruction, guardrails),
        }
    )


def _sanitize_task_result(
    result: AgentTaskResult,
    guardrails: GuardrailPipeline,
) -> AgentTaskResult:
    """Очищает результат специалиста, ошибку, citations и usage diagnostics."""
    return result.model_copy(
        update={
            "content": _text(result.content, guardrails),
            "error": _optional_text(result.error, guardrails),
            "citations": [
                _sanitize_citation(citation, guardrails)
                for citation in result.citations
            ],
            "usage": _sanitize_usage(result.usage, guardrails),
        },
        deep=True,
    )


def _sanitize_citation(
    citation: RagCitation,
    guardrails: GuardrailPipeline,
) -> RagCitation:
    """Удаляет секреты и локальные пути из отображаемых полей RAG citation."""
    return citation.model_copy(
        update={
            "source": _optional_text(citation.source, guardrails),
            "section": _optional_text(citation.section, guardrails),
            "excerpt": _text(citation.excerpt, guardrails),
        }
    )


def _sanitize_envelope(
    envelope: AgentEnvelope,
    guardrails: GuardrailPipeline,
) -> AgentEnvelope:
    """Рекурсивно очищает недоверенный payload и ошибку сообщения шины."""
    return envelope.model_copy(
        update={
            "payload": _sanitize_value(envelope.payload, guardrails),
            "error": _optional_text(envelope.error, guardrails),
        },
        deep=True,
    )


def _sanitize_lifecycle(
    event: LifecycleEvent,
    guardrails: GuardrailPipeline,
) -> LifecycleEvent:
    """Очищает произвольные детали lifecycle-события перед записью trace."""
    return event.model_copy(
        update={"details": _sanitize_value(event.details, guardrails)},
        deep=True,
    )


def _sanitize_usage(
    usage: UsageMetrics,
    guardrails: GuardrailPipeline,
) -> UsageMetrics:
    """Очищает сообщения об ошибках конвертации валют без изменения чисел."""
    return usage.model_copy(
        update={
            "currency_conversion_errors": [
                _text(error, guardrails) for error in usage.currency_conversion_errors
            ],
            "exchange_rate_source": _optional_text(
                usage.exchange_rate_source,
                guardrails,
            ),
        }
    )


def _sanitize_quality(
    quality: QualityAssessment,
    guardrails: GuardrailPipeline,
) -> QualityAssessment:
    """Очищает человекочитаемые замечания оценщика качества."""
    return quality.model_copy(
        update={"notes": [_text(note, guardrails) for note in quality.notes]}
    )


def _sanitize_route(
    route: LLMRouteInfo,
    guardrails: GuardrailPipeline,
) -> LLMRouteInfo:
    """Скрывает локальный путь, если он использован как идентификатор модели."""
    return route.model_copy(update={"model": _text(route.model, guardrails)})


def _sanitize_mode_result(
    result: AgentModeResult,
    guardrails: GuardrailPipeline,
) -> AgentModeResult:
    """Очищает содержимое и диагностические части одного режима сравнения."""
    return result.model_copy(
        update={
            "answer": _text(result.answer, guardrails),
            "quality": _sanitize_quality(result.quality, guardrails),
            "usage": _sanitize_usage(result.usage, guardrails),
        },
        deep=True,
    )


def _sanitize_case(
    case: ComparisonCaseResult,
    guardrails: GuardrailPipeline,
) -> ComparisonCaseResult:
    """Очищает вход и оба результата одного сравнительного сценария."""
    return case.model_copy(
        update={
            "title": _text(case.title, guardrails),
            "request": _text(case.request, guardrails),
            "single": _sanitize_mode_result(case.single, guardrails),
            "multi": _sanitize_mode_result(case.multi, guardrails),
        },
        deep=True,
    )


def _sanitize_value(value: Any, guardrails: GuardrailPipeline) -> Any:
    """Обходит JSON-подобное дерево, не меняя ключи и числовую структуру."""
    if isinstance(value, str):
        return _text(value, guardrails)
    if isinstance(value, dict):
        return {key: _sanitize_value(item, guardrails) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item, guardrails) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item, guardrails) for item in value)
    return value


def _text(value: str, guardrails: GuardrailPipeline) -> str:
    """Возвращает очищенный текст из единого output guardrail pipeline."""
    return guardrails.inspect_output(value).text


def _optional_text(
    value: str | None,
    guardrails: GuardrailPipeline,
) -> str | None:
    """Сохраняет отсутствие необязательного поля и очищает только строку."""
    return _text(value, guardrails) if value is not None else None
