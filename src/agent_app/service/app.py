"""HTTP-приложение и его обработчики для HTTP-сервиса поддержки."""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from agent_app.config import AgentAppConfig, load_agent_config
from agent_app.guardrails import GuardrailPipeline
from agent_app.guardrails.models import HumanReviewRecord, SecurityAuditEvent
from agent_app.multi_agent.protocols.a2a import install_a2a_routes
from agent_app.multi_agent.protocols.mcp import build_mcp_server
from agent_app.orchestration.errors import JobNotFoundError, QueueCapacityError
from agent_app.orchestration.models import (
    JobEvent,
    JobRecord,
    JobSubmission,
    QueueStatus,
)
from agent_app.orchestration.service import OrchestrationService
from agent_app.service.runtime import SupportApplicationRuntime
from agent_app.service.auth import AuthManager, Permission, Principal
from agent_app.service.rate_limit import TokenBucketRateLimiter
from agent_app.service.schemas import (
    ApiError,
    ChatRequest,
    ChatResponse,
    DeleteSessionResponse,
    HealthResponse,
    HumanReviewDecisionRequest,
    HumanReviewResponse,
    MultiAgentChatResponse,
    MultiAgentCompareRequest,
    MultiAgentCompareResponse,
    OrchestrationJobRequest,
    SecurityAuditResponse,
    SessionResponse,
)
from agent_app.observability import (
    configure_service_logging,
    current_trace_id,
    instrument_fastapi,
)
from agent_app.support.security import redact_local_paths, redact_secrets

LOGGER = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {
        "name": "Диалог",
        "description": "Запросы к агенту, RAG, tools и памяти.",
    },
    {
        "name": "Мультиагентная система",
        "description": "Supervisor-граф, сравнение режимов и артефакты запусков.",
    },
    {
        "name": "Оркестрация",
        "description": (
            "Постановка заданий, паттерны выполнения, очередь, события и отмена."
        ),
    },
    {
        "name": "Сессии",
        "description": "Просмотр и очистка контекста инженерного расследования.",
    },
    {
        "name": "Состояние",
        "description": "Liveness, readiness и метрики наблюдаемости.",
    },
    {
        "name": "Безопасность",
        "description": "Human review, RBAC и append-only журнал решений безопасности.",
    },
]

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    scheme_name="SupportApiKey",
    description=(
        "Сервисный API key из переменной SUPPORT_SERVICE_API_KEY. "
        "В локальном конфиге проверка может быть отключена."
    ),
    auto_error=False,
)
BEARER_SCHEME = HTTPBearer(
    scheme_name="SupportBearer",
    description="Bearer JWT с полями sub, roles, iat, exp, iss и aud.",
    auto_error=False,
)

REQUEST_ID_HEADER: dict[str, Any] = {
    "description": "Корреляционный идентификатор запроса.",
    "schema": {"type": "string", "format": "uuid"},
}


def _error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Формирует стандартные HTTP-ответы с описаниями ошибок для заданных кодов, обеспечивая единообразие обработки ошибок в API."""
    descriptions = {
        400: "Запрос отклонён guardrail-проверкой.",
        401: "API key отсутствует или некорректен.",
        403: "Роль не имеет требуемого разрешения.",
        404: "Задание или ресурс не найден.",
        409: "Операция конфликтует с текущим состоянием задания.",
        413: "Размер запроса или сообщения превышает установленный предел.",
        429: "Очередь заполнена: сработал backpressure.",
        422: "Запрос не соответствует OpenAPI-схеме.",
        500: "Непредвиденная ошибка выполнения агента.",
        503: "Сервис, LLM или RAG временно не готов к обработке запроса.",
    }
    responses: dict[int | str, dict[str, Any]] = {}
    for code in status_codes:
        responses[code] = {
            "model": ApiError,
            "description": descriptions[code],
            "headers": {"X-Request-ID": REQUEST_ID_HEADER},
        }
    return responses


def create_app(
    config_path: str | Path | None = None,
    *,
    runtime: SupportApplicationRuntime | None = None,
) -> FastAPI:
    """Создаёт и настраивает экземпляр FastAPI с конфигурацией и жизненным циклом, обеспечивая готовность сервиса поддержки к обработке запросов."""
    if runtime is not None:
        config = runtime.config
    else:
        resolved_config_path = config_path or os.getenv("SUPPORT_AGENT_CONFIG")
        if not resolved_config_path:
            raise ValueError(
                "Задайте config_path или переменную окружения SUPPORT_AGENT_CONFIG."
            )
        config = load_agent_config(resolved_config_path)
    configure_service_logging(
        config.logging.level, json_format=config.logging.json_format
    )
    owns_runtime = runtime is None
    mcp_server = (
        build_mcp_server(max_log_chars=config.tools.max_log_chars)
        if config.multi_agent.enabled and config.multi_agent.protocols.mcp_enabled
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Обеспечивает корректную инициализацию и завершение жизненного цикла приложения с управлением зависимостями и ресурсами."""
        app.state.runtime = runtime or SupportApplicationRuntime(config)
        async with AsyncExitStack() as stack:
            if mcp_server is not None:
                await stack.enter_async_context(mcp_server.session_manager.run())
            try:
                yield
            finally:
                if owns_runtime:
                    app.state.runtime.close()

    app = FastAPI(
        title="ИИ-агент поддержки инженера",
        summary="LLM-агент с online RAG, tools и памятью",
        description=(
            "API итогового агента инженерной поддержки. Агент выбирает tools через "
            "LLM, извлекает подтверждённый контекст из Qdrant, возвращает citations "
            "и сохраняет user-scoped память. Защищённые операции принимают "
            "сервисный `X-API-Key` или ролевой `Bearer JWT`."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "filter": True,
            "tryItOutEnabled": True,
        },
        lifespan=lifespan,
    )
    instrument_fastapi(app, config.observability)
    if mcp_server is not None:
        app.mount(
            config.multi_agent.protocols.mcp_path,
            mcp_server.streamable_http_app(),
            name="mcp-engineering-tools",
        )
    if config.service.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.service.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "X-API-Key",
                "X-Request-ID",
            ],
        )

    registry = CollectorRegistry()
    request_counter = Counter(
        "support_agent_requests_total",
        "Количество API-запросов агента",
        ["method", "path", "status"],
        registry=registry,
    )
    request_latency = Histogram(
        "support_agent_request_duration_seconds",
        "Длительность API-запросов агента",
        ["method", "path"],
        registry=registry,
    )
    retrieval_counter = Counter(
        "support_agent_retrieval_total",
        "Результаты online retrieval",
        ["status"],
        registry=registry,
    )
    multi_agent_counter = Counter(
        "support_agent_multi_runs_total",
        "Результаты мультиагентных запусков",
        ["status"],
        registry=registry,
    )
    orchestration_counter = Counter(
        "support_agent_orchestration_jobs_total",
        "Результаты постановки orchestration-заданий",
        ["pattern", "status", "deduplicated"],
        registry=registry,
    )
    guardrail_counter = Counter(
        "support_agent_guardrail_decisions_total",
        "Решения guardrails по этапам",
        ["stage", "action"],
        registry=registry,
    )
    review_counter = Counter(
        "support_agent_human_reviews_total",
        "Human-in-the-loop задачи по статусам",
        ["status"],
        registry=registry,
    )
    auth_manager = AuthManager(config.security)
    rate_limiter = TokenBucketRateLimiter(
        requests_per_minute=config.security.rate_limit_requests_per_minute,
        burst=config.security.rate_limit_burst,
    )
    guardrail_pipeline = GuardrailPipeline(config.guardrails)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Гарантирует уникальный идентификатор запроса, контроль доступа по API-ключу и защиту от перегрузки по размеру тела запроса."""
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        protected_protocol_paths = (
            config.multi_agent.protocols.a2a_rpc_path,
            config.multi_agent.protocols.a2a_rest_path,
            config.multi_agent.protocols.mcp_path,
        )
        protocol_auth_enabled = (
            config.security.require_api_key or config.security.jwt_enabled
        )
        is_protected_protocol = any(
            request.url.path == path or request.url.path.startswith(path + "/")
            for path in protected_protocol_paths
            if path != "/"
        )
        if (
            request.method != "OPTIONS"
            and protocol_auth_enabled
            and is_protected_protocol
        ):
            authorization = request.headers.get("Authorization", "")
            bearer_token = (
                authorization[7:].strip()
                if authorization.casefold().startswith("bearer ")
                else None
            )
            try:
                principal = auth_manager.authenticate(
                    api_key=request.headers.get("X-API-Key"),
                    bearer_token=bearer_token,
                )
                auth_manager.authorize(principal, Permission.CHAT)
                if config.security.rate_limit_enabled:
                    retry_after = rate_limiter.consume(principal.subject)
                    if retry_after is not None:
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Превышен лимит запросов к агенту.",
                            headers={
                                "Retry-After": str(max(1, math.ceil(retry_after)))
                            },
                        )
                request.state.principal = principal
            except HTTPException as exc:
                return _error_response(
                    request,
                    exc.status_code,
                    (
                        "unauthorized"
                        if exc.status_code == status.HTTP_401_UNAUTHORIZED
                        else "forbidden"
                        if exc.status_code == status.HTTP_403_FORBIDDEN
                        else "rate_limited"
                        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
                        else "security_not_configured"
                    ),
                    str(exc.detail),
                    headers=exc.headers,
                )
        content_length = request.headers.get("content-length")
        try:
            content_length_value = int(content_length) if content_length else 0
        except ValueError:
            content_length_value = 0
        if content_length_value > config.service.request_max_chars * 4:
            return _error_response(
                request,
                status.HTTP_413_CONTENT_TOO_LARGE,
                "request_too_large",
                "Размер HTTP-запроса превышает допустимый предел.",
            )
        if request.method in {"POST", "PUT", "PATCH"}:
            # Content-Length может отсутствовать при chunked transfer encoding.
            # Читаем тело один раз: Starlette кэширует bytes для FastAPI parser.
            body = await request.body()
            if len(body) > config.service.request_max_chars * 4:
                return _error_response(
                    request,
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    "request_too_large",
                    "Размер HTTP-запроса превышает допустимый предел.",
                )
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        elapsed = perf_counter() - started
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", request.url.path)
        request_counter.labels(request.method, route, str(response.status_code)).inc()
        request_latency.labels(request.method, route).observe(elapsed)
        LOGGER.info(
            "HTTP request",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": route,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000, 3),
            },
        )
        if response.status_code in {
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        }:
            audit_store = getattr(request.app.state.runtime, "security_audit", None)
            if audit_store is not None:
                principal = getattr(request.state, "principal", None)
                audit_store.append(
                    SecurityAuditEvent(
                        event_type="authorization",
                        action="deny",
                        principal_id=principal.subject if principal else None,
                        role=(
                            principal.roles[0]
                            if principal is not None and principal.roles
                            else None
                        ),
                        request_id=request_id,
                        trace_id=current_trace_id(),
                        details={"path": route, "status": response.status_code},
                    )
                )
        return response

    def authenticate(
        request: Request,
        supplied: str | None = Security(API_KEY_HEADER),
        credentials: HTTPAuthorizationCredentials | None = Security(BEARER_SCHEME),
    ) -> Principal:
        """Гарантирует аутентификацию пользователя по API-ключу или токену и сохраняет результат в состоянии запроса."""
        principal = auth_manager.authenticate(
            api_key=supplied,
            bearer_token=(credentials.credentials if credentials else None),
        )
        request.state.principal = principal
        return principal

    def require_permission(permission: Permission):
        """Проверяет, что вызывающий обладает требуемым разрешением, и прерывает выполнение при отсутствии прав."""

        def dependency(
            principal: Principal = Depends(authenticate),
        ) -> Principal:
            """Гарантирует, что вызывающий код получит аутентифицированного и авторизованного пользователя с требуемыми правами."""
            auth_manager.authorize(principal, permission)
            if permission is Permission.CHAT and config.security.rate_limit_enabled:
                retry_after = rate_limiter.consume(principal.subject)
                if retry_after is not None:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Превышен лимит запросов к агенту.",
                        headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
                    )
            return principal

        dependency.__name__ = f"require_{permission.value.replace(':', '_')}"
        return dependency

    def inspect_request(request: Request, payload: ChatRequest) -> str:
        """Проверяет, что входящее сообщение пользователя не нарушает guardrail-политику и блокирует опасные запросы."""
        principal: Principal = request.state.principal
        auth_manager.enforce_user_scope(principal, payload.user_id)
        decision = guardrail_pipeline.inspect_input(payload.message)
        guardrail_counter.labels("input", decision.action.value).inc()
        _audit_security(
            request,
            event_type="guardrail",
            action=decision.action.value,
            user_id=payload.user_id,
            session_id=payload.session_id,
            details={
                "stage": "input",
                "findings": [item.code for item in decision.findings],
            },
        )
        if decision.blocked:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Запрос заблокирован: обнаружена prompt injection.",
            )
        return decision.text

    def inspect_answer(
        request: Request,
        *,
        prompt: str,
        answer: str,
        user_id: str,
        session_id: str,
    ) -> tuple[str, str, str | None]:
        """Проверяет, что ответ ассистента соответствует guardrail-политике, инициирует ручную проверку при необходимости и фиксирует аудит."""
        decision = guardrail_pipeline.inspect_output(answer)
        guardrail_counter.labels("output", decision.action.value).inc()
        review_id = None
        if decision.action.value == "review":
            review_store = getattr(request.app.state.runtime, "review_store", None)
            if review_store is None:
                return decision.text, decision.action.value, None
            review = review_store.create(
                HumanReviewRecord(
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request.state.request_id,
                    trace_id=current_trace_id(),
                    prompt=prompt,
                    answer=decision.text,
                    reason=", ".join(item.code for item in decision.findings),
                )
            )
            review_id = review.id
            review_counter.labels("pending").inc()
            public_answer = (
                "Ответ временно удержан и передан на ручную проверку. "
                f"Идентификатор review: {review_id}."
            )
        else:
            public_answer = decision.text
        _audit_security(
            request,
            event_type="guardrail",
            action=decision.action.value,
            user_id=user_id,
            session_id=session_id,
            details={
                "stage": "output",
                "findings": [item.code for item in decision.findings],
                "review_id": review_id,
            },
        )
        return public_answer, decision.action.value, review_id

    def _audit_security(
        request: Request,
        *,
        event_type: str,
        action: str,
        user_id: str | None = None,
        session_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Гарантирует запись события безопасности в аудит с привязкой к пользователю, роли и идентификаторам запроса."""
        principal: Principal | None = getattr(request.state, "principal", None)
        audit_store = getattr(request.app.state.runtime, "security_audit", None)
        if audit_store is None:
            return
        audit_store.append(
            SecurityAuditEvent(
                event_type=event_type,
                action=action,
                principal_id=principal.subject if principal else None,
                role=principal.roles[0] if principal and principal.roles else None,
                user_id=user_id,
                session_id=session_id,
                request_id=getattr(request.state, "request_id", None),
                trace_id=current_trace_id(),
                details=details or {},
            )
        )

    def require_orchestration(request: Request) -> OrchestrationService:
        """Обеспечивает доступ к сервису оркестрации с гарантией его инициализации, иначе возвращает ошибку 503, предотвращая вызовы при недоступности."""
        service = request.app.state.runtime.orchestration_service
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Оркестрация отключена или не инициализирована.",
            )
        return service

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Преобразует HTTP-исключения в стандартизированный ответ с кодом и сообщением, обеспечивая единообразную обработку ошибок клиента."""
        return _error_response(
            request,
            exc.status_code,
            "http_error",
            str(exc.detail),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Логирует необработанные ошибки и возвращает безопасный ответ с кодом 500, скрывая чувствительные данные из сообщения об ошибке."""
        LOGGER.exception("Необработанная ошибка API")
        return _error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "Внутренняя ошибка сервиса. Используйте request_id для обращения к журналу.",
        )

    @app.post(
        "/v1/chat",
        response_model=ChatResponse,
        tags=["Диалог"],
        summary="Отправить запрос агенту",
        description=(
            "Выполняет один ход агента. В зависимости от запроса LLM использует "
            "RAG, инженерные tools и память. Ответ содержит trace, citations и "
            "retrieval diagnostics."
        ),
        response_description="Итоговый ответ агента и диагностика выполнения.",
        responses=_error_responses(401, 413, 422, 500, 503),
        dependencies=[Depends(require_permission(Permission.CHAT))],
    )
    def chat(request: Request, payload: ChatRequest) -> ChatResponse:
        """Обрабатывает запросы чата, гарантируя валидацию, инспекцию и подсчёт метрик, возвращая структурированный ответ с результатом и метаданными."""
        _validate_message_size(payload.message, config)
        message = inspect_request(request, payload)
        result, duration_ms = request.app.state.runtime.ask(
            user_id=payload.user_id,
            session_id=payload.session_id,
            message=message,
        )
        answer, guardrail_action, review_id = inspect_answer(
            request,
            prompt=message,
            answer=result.answer,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )
        result = result.model_copy(
            update={
                "answer": answer,
                "citations": _sanitize_citations(result.citations),
            }
        )
        retrieval_status = result.retrieval.status if result.retrieval else "not_used"
        retrieval_counter.labels(retrieval_status).inc()
        return ChatResponse(
            **result.model_dump(mode="python"),
            request_id=request.state.request_id,
            duration_ms=duration_ms,
            guardrail_action=guardrail_action,
            review_id=review_id,
        )

    @app.post(
        "/v1/multi-agent/chat",
        response_model=MultiAgentChatResponse,
        tags=["Мультиагентная система"],
        summary="Выполнить запрос через supervisor-граф",
        description=(
            "Декомпозирует запрос, делегирует подзадачи профильным агентам, "
            "проверяет отчёты критиком и возвращает итог с lifecycle и usage."
        ),
        responses=_error_responses(401, 413, 422, 500, 503),
        dependencies=[Depends(require_permission(Permission.CHAT))],
    )
    def multi_agent_chat(
        request: Request,
        payload: ChatRequest,
    ) -> MultiAgentChatResponse:
        """Обрабатывает мультиагентные чат-запросы с валидацией и инспекцией, обеспечивая учёт статуса выполнения и возврат расширенного ответа с метриками."""
        _validate_message_size(payload.message, config)
        message = inspect_request(request, payload)
        result, duration_ms = request.app.state.runtime.ask_multi(
            user_id=payload.user_id,
            session_id=payload.session_id,
            message=message,
        )
        answer, guardrail_action, review_id = inspect_answer(
            request,
            prompt=message,
            answer=result.response.answer,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )
        result.response = result.response.model_copy(
            update={
                "answer": answer,
                "citations": _sanitize_citations(result.response.citations),
            }
        )
        multi_agent_counter.labels(
            "degraded" if result.response.degraded else "completed"
        ).inc()
        return MultiAgentChatResponse(
            **result.response.model_dump(mode="python"),
            request_id=request.state.request_id,
            duration_ms=duration_ms,
            run_dir=result.run_dir,
            guardrail_action=guardrail_action,
            review_id=review_id,
        )

    @app.post(
        "/v1/multi-agent/compare",
        response_model=MultiAgentCompareResponse,
        tags=["Мультиагентная система"],
        summary="Сравнить single-agent и multi-agent",
        description=(
            "Выполняет один запрос в обоих режимах при общей модели и возвращает "
            "качество, latency, tokens, tool calls и настраиваемую стоимость."
        ),
        responses=_error_responses(401, 413, 422, 500, 503),
        dependencies=[Depends(require_permission(Permission.CHAT))],
    )
    def compare_agents(
        request: Request,
        payload: MultiAgentCompareRequest,
    ) -> MultiAgentCompareResponse:
        """Выполняет сравнение агентов по заданным параметрам с гарантией валидации и инспекции, возвращая отчёт с результатами и временем выполнения."""
        _validate_message_size(payload.message, config)
        message = inspect_request(request, payload)
        report, duration_ms = request.app.state.runtime.compare_multi(
            user_id=payload.user_id,
            session_id=payload.session_id,
            message=message,
            expected_terms=payload.expected_terms,
            expected_tools=payload.expected_tools,
            require_citations=payload.require_citations,
        )
        return MultiAgentCompareResponse(
            **report.model_dump(mode="python"),
            request_id=request.state.request_id,
            duration_ms=duration_ms,
        )

    @app.get(
        "/v1/multi-agent/runs/{run_id}",
        response_model=dict[str, object],
        tags=["Мультиагентная система"],
        summary="Получить сохранённый результат запуска",
        responses=_error_responses(401, 500, 503),
        dependencies=[Depends(require_permission(Permission.RUN_READ))],
    )
    def get_multi_agent_run(request: Request, run_id: str) -> dict[str, object]:
        """Возвращает сохранённые данные мультиагентного запуска по идентификатору, гарантируя ошибку 404 при отсутствии записи."""
        try:
            payload = request.app.state.runtime.load_multi_run(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload is None:
            raise HTTPException(status_code=404, detail="Multi-agent запуск не найден")
        owner = payload.get("user_id")
        if not isinstance(owner, str):
            raise HTTPException(
                status_code=500, detail="Артефакт запуска не содержит owner"
            )
        auth_manager.enforce_user_scope(request.state.principal, owner)
        return payload

    @app.post(
        "/v1/orchestration/jobs",
        response_model=JobSubmission,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Оркестрация"],
        summary="Поставить задание в оркестратор",
        description=(
            "Создаёт воспроизводимое задание с выбранным паттерном. В inline "
            "режиме оно выполняется до возврата ответа; в Celery-режиме попадает "
            "в приоритетную RabbitMQ-очередь."
        ),
        responses=_error_responses(401, 413, 422, 429, 500, 503),
        dependencies=[Depends(require_permission(Permission.ORCHESTRATION_WRITE))],
    )
    def submit_orchestration_job(
        request: Request,
        payload: OrchestrationJobRequest,
    ) -> JobSubmission:
        """Принимает и валидирует задание оркестрации, гарантирует обработку ошибок переполнения очереди и учитывает метрики по статусу и паттерну задания."""
        _validate_message_size(payload.message, config)
        message = inspect_request(request, payload)
        service = require_orchestration(request)
        try:
            submission = service.submit(
                payload.model_copy(update={"message": message}).to_job()
            )
        except QueueCapacityError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc
        orchestration_counter.labels(
            submission.record.job.pattern.value,
            submission.record.status.value,
            str(submission.deduplicated).lower(),
        ).inc()
        return submission

    @app.get(
        "/v1/orchestration/jobs/{job_id}",
        response_model=JobRecord,
        tags=["Оркестрация"],
        summary="Получить состояние задания",
        responses=_error_responses(401, 404, 500, 503),
        dependencies=[Depends(require_permission(Permission.ORCHESTRATION_READ))],
    )
    def get_orchestration_job(request: Request, job_id: str) -> JobRecord:
        """Возвращает запись оркестрационного задания по идентификатору с гарантией ошибки 404 при отсутствии, обеспечивая надёжный доступ к данным."""
        try:
            record = require_orchestration(request).get(job_id)
            auth_manager.enforce_user_scope(request.state.principal, record.job.user_id)
            return record
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/v1/orchestration/jobs/{job_id}/events",
        response_model=list[JobEvent],
        tags=["Оркестрация"],
        summary="Получить журнал событий задания",
        responses=_error_responses(401, 404, 500, 503),
        dependencies=[Depends(require_permission(Permission.ORCHESTRATION_READ))],
    )
    def get_orchestration_events(request: Request, job_id: str) -> list[JobEvent]:
        """Возвращает список событий оркестрационного задания с гарантией ошибки 404 при отсутствии задания, обеспечивая целостность истории событий."""
        try:
            service = require_orchestration(request)
            record = service.get(job_id)
            auth_manager.enforce_user_scope(request.state.principal, record.job.user_id)
            return service.events(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete(
        "/v1/orchestration/jobs/{job_id}",
        response_model=JobRecord,
        tags=["Оркестрация"],
        summary="Отменить незавершённое задание",
        responses=_error_responses(401, 404, 500, 503),
        dependencies=[Depends(require_permission(Permission.ORCHESTRATION_WRITE))],
    )
    def cancel_orchestration_job(request: Request, job_id: str) -> JobRecord:
        """Гарантирует корректное завершение или отмену оркестрационной задачи с явной ошибкой 404 при отсутствии указанной задачи."""
        try:
            service = require_orchestration(request)
            record = service.get(job_id)
            auth_manager.enforce_user_scope(request.state.principal, record.job.user_id)
            return service.cancel(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/v1/orchestration/queues/status",
        response_model=QueueStatus,
        tags=["Оркестрация"],
        summary="Проверить очередь и workers",
        responses=_error_responses(401, 503),
        dependencies=[Depends(require_permission(Permission.ORCHESTRATION_READ))],
    )
    def orchestration_queue_status(request: Request) -> QueueStatus:
        """Обеспечивает вызывающему коду актуальное состояние очереди оркестрации для мониторинга и диагностики."""
        return require_orchestration(request).status()

    @app.post(
        "/v1/chat/stream",
        tags=["Диалог"],
        summary="Отправить запрос и получить SSE-события",
        description=(
            "Возвращает событие `started`, затем `result` с полным ChatResponse. "
            "Это поток этапов выполнения, а не token-by-token streaming LLM."
        ),
        responses={
            200: {
                "description": "Поток Server-Sent Events.",
                "content": {
                    "text/event-stream": {
                        "example": (
                            "event: started\n"
                            'data: {"request_id":"..."}\n\n'
                            "event: result\n"
                            'data: {"answer":"...","request_id":"..."}\n\n'
                        )
                    }
                },
                "headers": {"X-Request-ID": REQUEST_ID_HEADER},
            },
            **_error_responses(401, 413, 422, 500, 503),
        },
        dependencies=[Depends(require_permission(Permission.CHAT))],
    )
    def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
        """Гарантирует интерактивную потоковую передачу ответов агента с событиями и метаданными для поддержки real-time UI и автоматизации."""
        _validate_message_size(payload.message, config)
        message = inspect_request(request, payload)

        def events():
            """Гарантирует потоковую передачу статуса и результата диалога для интерактивных клиентов с учётом аудита и метрик."""
            yield _sse("started", {"request_id": request.state.request_id})
            result, duration_ms = request.app.state.runtime.ask(
                user_id=payload.user_id,
                session_id=payload.session_id,
                message=message,
            )
            answer, guardrail_action, review_id = inspect_answer(
                request,
                prompt=message,
                answer=result.answer,
                user_id=payload.user_id,
                session_id=payload.session_id,
            )
            result = result.model_copy(
                update={
                    "answer": answer,
                    "citations": _sanitize_citations(result.citations),
                }
            )
            response = ChatResponse(
                **result.model_dump(mode="python"),
                request_id=request.state.request_id,
                duration_ms=duration_ms,
                guardrail_action=guardrail_action,
                review_id=review_id,
            )
            retrieval_status = (
                result.retrieval.status if result.retrieval else "not_used"
            )
            retrieval_counter.labels(retrieval_status).inc()
            yield _sse("result", response.model_dump(mode="json"))

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get(
        "/v1/sessions/{session_id}",
        response_model=SessionResponse,
        tags=["Сессии"],
        summary="Получить состояние сессии",
        description=(
            "Возвращает доступные текущему user_id записи памяти и связанные "
            "инциденты. Чужие user-scoped данные не возвращаются."
        ),
        responses=_error_responses(401, 422, 500, 503),
        dependencies=[Depends(require_permission(Permission.SESSION_READ))],
    )
    def get_session(request: Request, session_id: str, user_id: str) -> SessionResponse:
        """Гарантирует возврат информации о сессии пользователя только при наличии прав доступа, предотвращая несанкционированное раскрытие данных."""
        auth_manager.enforce_user_scope(request.state.principal, user_id)
        return request.app.state.runtime.session(user_id=user_id, session_id=session_id)

    @app.delete(
        "/v1/sessions/{session_id}",
        response_model=DeleteSessionResponse,
        tags=["Сессии"],
        summary="Очистить сессию",
        description=(
            "Удаляет session-scoped память и AgentRunner из локального кэша. "
            "Глобальная долговременная память пользователя сохраняется."
        ),
        responses=_error_responses(401, 422, 500, 503),
        dependencies=[Depends(require_permission(Permission.SESSION_DELETE))],
    )
    def delete_session(
        request: Request,
        session_id: str,
        user_id: str,
    ) -> DeleteSessionResponse:
        """Гарантирует удаление пользовательской сессии с проверкой полномочий, предотвращая несанкционированное вмешательство."""
        auth_manager.enforce_user_scope(request.state.principal, user_id)
        return request.app.state.runtime.delete_session(
            user_id=user_id,
            session_id=session_id,
        )

    @app.get(
        "/v1/reviews",
        response_model=list[HumanReviewResponse],
        tags=["Безопасность"],
        summary="Получить очередь human review",
        responses=_error_responses(401, 403),
        dependencies=[Depends(require_permission(Permission.REVIEW_READ))],
    )
    def list_reviews(
        request: Request,
        review_status: str | None = None,
        limit: int = 100,
    ) -> list[HumanReviewResponse]:
        """Гарантирует возврат ограниченного списка заявок на ручную модерацию для последующей обработки или аудита."""
        records = request.app.state.runtime.review_store.list(
            status=review_status, limit=min(max(limit, 1), 500)
        )
        return [
            HumanReviewResponse.model_validate(record.model_dump(mode="python"))
            for record in records
        ]

    @app.post(
        "/v1/reviews/{review_id}/decision",
        response_model=HumanReviewResponse,
        tags=["Безопасность"],
        summary="Подтвердить или отклонить ответ",
        responses=_error_responses(401, 403, 404),
        dependencies=[Depends(require_permission(Permission.REVIEW_WRITE))],
    )
    def decide_review(
        request: Request,
        review_id: str,
        payload: HumanReviewDecisionRequest,
    ) -> HumanReviewResponse:
        """Гарантирует принятие решения по заявке на ручную модерацию с аудированием события и ошибкой 404 при невозможности обработки."""
        principal: Principal = request.state.principal
        record = request.app.state.runtime.review_store.decide(
            review_id,
            approved=payload.approved,
            reviewer_id=principal.subject,
            comment=payload.comment,
        )
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="Pending review не найден или решение уже принято.",
            )
        review_counter.labels(record.status).inc()
        _audit_security(
            request,
            event_type="human_review",
            action=record.status,
            user_id=record.user_id,
            session_id=record.session_id,
            details={"review_id": record.id},
        )
        return HumanReviewResponse.model_validate(record.model_dump(mode="python"))

    @app.get(
        "/v1/security/audit",
        response_model=SecurityAuditResponse,
        tags=["Безопасность"],
        summary="Получить журнал решений безопасности",
        responses=_error_responses(401, 403),
        dependencies=[Depends(require_permission(Permission.AUDIT_READ))],
    )
    def security_audit(request: Request, limit: int = 100) -> SecurityAuditResponse:
        """Гарантирует возврат последних событий аудита безопасности для анализа действий пользователей и администраторов."""
        return SecurityAuditResponse(
            events=request.app.state.runtime.security_audit.list(
                limit=min(max(limit, 1), 500)
            )
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Состояние"],
        summary="Проверить liveness",
        description="Проверяет, что HTTP-процесс запущен.",
    )
    def health() -> HealthResponse:
        """Гарантирует быстрый ответ о работоспособности сервиса для систем мониторинга."""
        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=HealthResponse,
        tags=["Состояние"],
        summary="Проверить readiness",
        description=(
            "Проверяет конфигурацию безопасности, LLM runtime, Qdrant collection, "
            "embedding provider и размерность vectors без платного LLM-запроса."
        ),
        responses=_error_responses(503),
    )
    def ready(request: Request) -> Response:
        """Гарантирует корректный статус готовности сервиса с подробностями для балансировщиков и автоматизации деплоя."""
        details = request.app.state.runtime.readiness()
        payload = HealthResponse(
            status="ready" if details["ready"] else "not_ready",
            details=details,
        )
        return JSONResponse(
            status_code=(
                status.HTTP_200_OK
                if details["ready"]
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content=payload.model_dump(mode="json"),
        )

    @app.get(
        "/metrics",
        tags=["Состояние"],
        summary="Получить Prometheus metrics",
        description=(
            "Возвращает счётчики запросов, latency и статусы retrieval субъекту "
            "с разрешением metrics:read."
        ),
        responses={
            **_error_responses(401, 403, 503),
            200: {
                "description": "Метрики в Prometheus text exposition format.",
                "content": {
                    "text/plain": {
                        "example": 'support_agent_requests_total{status="200"} 1.0'
                    }
                },
            },
        },
        dependencies=[Depends(require_permission(Permission.METRICS_READ))],
    )
    def metrics() -> Response:
        """Гарантирует предоставление актуальных метрик Prometheus для мониторинга состояния сервиса через HTTP."""
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    if config.multi_agent.enabled and config.multi_agent.protocols.a2a_enabled:
        configured_base_url = os.getenv("SUPPORT_PUBLIC_BASE_URL")
        public_host = (
            "127.0.0.1" if config.service.host == "0.0.0.0" else config.service.host
        )
        base_url = (
            configured_base_url.rstrip("/")
            if configured_base_url
            else config.service.public_base_url
            or f"http://{public_host}:{config.service.port}"
        )

        def a2a_ask(**kwargs):
            """Адаптирует A2A-вызов к результату `MultiAgentRuntime`."""
            result, _ = app.state.runtime.ask_multi(**kwargs)
            return result

        install_a2a_routes(
            app,
            config,
            base_url=base_url,
            ask=a2a_ask,
        )

    return app


def _validate_message_size(message: str, config: AgentAppConfig) -> None:
    """Проверяет размер входящего сообщения и прерывает обработку с ошибкой, если оно превышает допустимый лимит, защищая сервис от перегрузок."""
    if len(message) > config.service.request_max_chars:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Сообщение превышает service.request_max_chars: "
                f"actual={len(message)} limit={config.service.request_max_chars}"
            ),
        )


def _sanitize_citations(citations: list[Any]) -> list[Any]:
    """Удаляет локальные пути из источников цитат, чтобы исключить утечку чувствительной информации в HTTP-ответах."""
    return [
        citation.model_copy(
            update={
                "source": (
                    redact_local_paths(citation.source) if citation.source else None
                )
            }
        )
        for citation in citations
    ]


def _error_response(
    request: Request,
    status_code: int,
    error: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Гарантирует стандартизированный и безопасный формат ошибок API с маскировкой секретов и трассировкой по request_id."""
    payload = ApiError(
        error=error,
        message=redact_secrets(message),
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


def _sse(event: str, payload: dict[str, object]) -> str:
    """Формирует событие Server-Sent Events в формате, совместимом с браузерами и клиентами SSE."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
