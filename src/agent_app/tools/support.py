from __future__ import annotations

import json
import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.rag.runtime import OnlineRagRuntime
from agent_app.support.incidents import (
    IncidentPriority,
    IncidentStatus,
    IncidentStore,
)
from agent_app.support.security import redact_secrets


class KnowledgeSearchInput(BaseModel):
    query: str = Field(min_length=2)
    top_k: int | None = Field(default=None, ge=1, le=50)
    source: str | None = None
    section: str | None = None


class LogAnalysisInput(BaseModel):
    log_text: str = Field(min_length=1)
    component: str | None = None


class IncidentCreateInput(BaseModel):
    title: str = Field(min_length=2)
    description: str = Field(min_length=2)
    priority: IncidentPriority = "medium"
    component: str | None = None


class IncidentGetInput(BaseModel):
    incident_id: str


class IncidentUpdateInput(BaseModel):
    incident_id: str
    status: IncidentStatus


class IncidentListInput(BaseModel):
    status: IncidentStatus | None = None
    current_session_only: bool = False
    limit: int = Field(default=20, ge=1, le=100)


class DiagnosticChecklistInput(BaseModel):
    component: str
    symptoms: str


class RunbookSearchInput(BaseModel):
    problem: str
    component: str | None = None


def support_tools(
    *,
    rag_runtime: OnlineRagRuntime | None,
    incident_store: IncidentStore,
    user_id: str,
    session_id: str,
    max_log_chars: int,
) -> list[StructuredTool]:
    def search_knowledge_base(
        query: str,
        top_k: int | None = None,
        source: str | None = None,
        section: str | None = None,
    ) -> str:
        if rag_runtime is None:
            return _json(
                {
                    "status": "unavailable",
                    "query": query,
                    "error": "Online RAG отключён в конфигурации",
                    "citations": [],
                }
            )
        result = rag_runtime.retrieve(
            query,
            top_k=top_k,
            source=source,
            section=section,
        )
        return result.model_dump_json()

    def find_runbook(problem: str, component: str | None = None) -> str:
        query = f"runbook инструкция диагностика {component or ''} {problem}".strip()
        return search_knowledge_base(query=query, top_k=5)

    def analyze_log_fragment(log_text: str, component: str | None = None) -> str:
        return _json(
            analyze_log_fragment_payload(
                log_text,
                component=component,
                max_log_chars=max_log_chars,
            )
        )

    def create_incident(
        title: str,
        description: str,
        priority: IncidentPriority = "medium",
        component: str | None = None,
    ) -> str:
        try:
            record = incident_store.create(
                user_id=user_id,
                session_id=session_id,
                title=title,
                description=description,
                priority=priority,
                component=component,
            )
            return _json(
                {"status": "created", "incident": record.model_dump(mode="json")}
            )
        except Exception as exc:
            return _json({"status": "error", "message": str(exc)})

    def get_incident(incident_id: str) -> str:
        record = incident_store.get(incident_id, user_id=user_id)
        if record is None:
            return _json({"status": "not_found", "incident_id": incident_id})
        return _json({"status": "found", "incident": record.model_dump(mode="json")})

    def update_incident_status(incident_id: str, status: IncidentStatus) -> str:
        record = incident_store.update_status(
            incident_id,
            user_id=user_id,
            status=status,
        )
        if record is None:
            return _json({"status": "not_found", "incident_id": incident_id})
        return _json({"status": "updated", "incident": record.model_dump(mode="json")})

    def list_incidents(
        status: IncidentStatus | None = None,
        current_session_only: bool = False,
        limit: int = 20,
    ) -> str:
        records = incident_store.list(
            user_id=user_id,
            session_id=session_id if current_session_only else None,
            status=status,
            limit=limit,
        )
        return _json(
            {
                "status": "ok",
                "count": len(records),
                "incidents": [record.model_dump(mode="json") for record in records],
            }
        )

    def build_diagnostic_checklist(component: str, symptoms: str) -> str:
        return _json(diagnostic_checklist_payload(component, symptoms))

    return [
        StructuredTool.from_function(
            name="search_knowledge_base",
            description=(
                "Ищет техническую информацию в Qdrant knowledge base. Используй для "
                "ответов по документации, процедурам и диагностике. В финальном ответе "
                "сохраняй ссылки вида [Источник N], которые вернул tool."
            ),
            func=search_knowledge_base,
            args_schema=KnowledgeSearchInput,
        ),
        StructuredTool.from_function(
            name="find_runbook",
            description="Ищет в базе знаний runbook для проблемы или компонента.",
            func=find_runbook,
            args_schema=RunbookSearchInput,
        ),
        StructuredTool.from_function(
            name="analyze_log_fragment",
            description="Безопасно анализирует небольшой фрагмент лога и скрывает секреты.",
            func=analyze_log_fragment,
            args_schema=LogAnalysisInput,
        ),
        StructuredTool.from_function(
            name="create_incident",
            description="Создаёт инцидент текущего инженера и сессии.",
            func=create_incident,
            args_schema=IncidentCreateInput,
        ),
        StructuredTool.from_function(
            name="get_incident",
            description="Читает инцидент текущего инженера по id.",
            func=get_incident,
            args_schema=IncidentGetInput,
        ),
        StructuredTool.from_function(
            name="update_incident_status",
            description="Обновляет статус инцидента текущего инженера.",
            func=update_incident_status,
            args_schema=IncidentUpdateInput,
        ),
        StructuredTool.from_function(
            name="list_incidents",
            description="Показывает инциденты текущего инженера.",
            func=list_incidents,
            args_schema=IncidentListInput,
        ),
        StructuredTool.from_function(
            name="build_diagnostic_checklist",
            description="Формирует воспроизводимый чек-лист диагностики по симптомам.",
            func=build_diagnostic_checklist,
            args_schema=DiagnosticChecklistInput,
        ),
    ]


def _log_findings(log_text: str) -> list[dict[str, str]]:
    rules = (
        (
            r"(?i)out of memory|oom|нехватк[аи] памяти",
            "memory_exhaustion",
            "Проверить RSS, heap, лимиты контейнера и утечки памяти.",
        ),
        (
            r"(?i)timeout|timed out|таймаут",
            "timeout",
            "Проверить latency зависимостей, retry и timeout budget.",
        ),
        (
            r"(?i)connection refused|connection reset|соединение отклонено",
            "connection_failure",
            "Проверить endpoint, порт, сеть и состояние downstream-сервиса.",
        ),
        (
            r"(?i)unauthorized|forbidden|\b401\b|\b403\b",
            "authorization_failure",
            "Проверить credentials и права, не выводя секреты.",
        ),
        (
            r"(?i)no space left|disk full|место на диске",
            "disk_exhaustion",
            "Проверить заполнение диска, inode и политику ротации логов.",
        ),
        (
            r"(?i)traceback|exception|\berror\b|ошибка",
            "application_error",
            "Найти первый exception и корреляционный идентификатор запроса.",
        ),
    )
    findings = []
    for pattern, code, recommendation in rules:
        if re.search(pattern, log_text):
            findings.append({"code": code, "recommendation": recommendation})
    if not findings:
        findings.append(
            {
                "code": "no_known_pattern",
                "recommendation": "Нужны дополнительные строки до и после события и контекст компонента.",
            }
        )
    return findings


def analyze_log_fragment_payload(
    log_text: str,
    *,
    component: str | None = None,
    max_log_chars: int = 12000,
) -> dict[str, object]:
    """Безопасное ядро анализа логов для LangChain tool и MCP server."""
    if len(log_text) > max_log_chars:
        return {
            "status": "error",
            "error": "log_too_large",
            "max_log_chars": max_log_chars,
            "actual_chars": len(log_text),
        }
    redacted = redact_secrets(log_text)
    return {
        "status": "ok",
        "component": component,
        "findings": _log_findings(redacted),
        "redacted_log_preview": redacted[:1000],
        "secrets_redacted": redacted != log_text,
    }


def diagnostic_checklist_payload(
    component: str,
    symptoms: str,
) -> dict[str, object]:
    """Детерминированный чек-лист без зависимости от LLM runtime."""
    lowered = symptoms.lower()
    steps = [
        "Зафиксировать время начала и область влияния.",
        f"Проверить состояние и последние изменения компонента {component}.",
        "Собрать логи с корреляционным идентификатором и точными timestamps.",
        "Проверить зависимости, сеть, DNS и доступность downstream-сервисов.",
    ]
    if any(marker in lowered for marker in ("медлен", "timeout", "таймаут")):
        steps.extend(
            [
                "Сравнить latency p50/p95/p99 с нормальным интервалом.",
                "Проверить saturation CPU, памяти, диска и пулов соединений.",
            ]
        )
    if any(marker in lowered for marker in ("401", "403", "доступ", "auth")):
        steps.append("Проверить срок действия credentials и права без вывода секретов.")
    steps.append(
        "После изменения повторить проверку и зафиксировать результат в инциденте."
    )
    return {
        "status": "ok",
        "component": component,
        "symptoms": redact_secrets(symptoms),
        "steps": steps,
    }


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)
