from __future__ import annotations

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from agent_app.config import AgentAppConfig, load_agent_config
from agent_app.service.runtime import SupportApplicationRuntime
from agent_app.service.schemas import (
    ApiError,
    ChatRequest,
    ChatResponse,
    DeleteSessionResponse,
    HealthResponse,
    SessionResponse,
)
from agent_app.support.security import redact_secrets
from rag_prep.utils import setup_logging

LOGGER = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {
        "name": "Диалог",
        "description": "Запросы к агенту, RAG, tools и памяти.",
    },
    {
        "name": "Сессии",
        "description": "Просмотр и очистка контекста инженерного расследования.",
    },
    {
        "name": "Состояние",
        "description": "Liveness, readiness и метрики наблюдаемости.",
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

REQUEST_ID_HEADER = {
    "description": "Корреляционный идентификатор запроса.",
    "schema": {"type": "string", "format": "uuid"},
}


def _error_responses(*status_codes: int) -> dict[int, dict[str, object]]:
    descriptions = {
        401: "API key отсутствует или некорректен.",
        413: "Размер запроса или сообщения превышает установленный предел.",
        422: "Запрос не соответствует OpenAPI-схеме.",
        500: "Непредвиденная ошибка выполнения агента.",
        503: "Сервис, LLM или RAG временно не готов к обработке запроса.",
    }
    return {
        code: {
            "model": ApiError,
            "description": descriptions[code],
            "headers": {"X-Request-ID": REQUEST_ID_HEADER},
        }
        for code in status_codes
    }


def create_app(
    config_path: str | Path | None = None,
    *,
    runtime: SupportApplicationRuntime | None = None,
) -> FastAPI:
    if runtime is not None:
        config = runtime.config
    else:
        resolved_config_path = config_path or os.getenv("SUPPORT_AGENT_CONFIG")
        if not resolved_config_path:
            raise ValueError(
                "Задайте config_path или переменную окружения SUPPORT_AGENT_CONFIG."
            )
        config = load_agent_config(resolved_config_path)
    setup_logging(config.logging.level)
    owns_runtime = runtime is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or SupportApplicationRuntime(config)
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
            "и сохраняет user-scoped память. Защищённые операции используют "
            "заголовок `X-API-Key`."
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
    if config.service.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.service.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
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

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        content_length = request.headers.get("content-length")
        try:
            content_length_value = int(content_length) if content_length else 0
        except ValueError:
            content_length_value = 0
        if content_length_value > config.service.request_max_chars * 4:
            return _error_response(
                request,
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                "Размер HTTP-запроса превышает допустимый предел.",
            )
        started = perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        elapsed = perf_counter() - started
        route = request.url.path
        request_counter.labels(request.method, route, str(response.status_code)).inc()
        request_latency.labels(request.method, route).observe(elapsed)
        LOGGER.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": route,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 3),
                },
                ensure_ascii=False,
            )
        )
        return response

    def require_api_key(
        request: Request,
        supplied: str | None = Security(API_KEY_HEADER),
    ) -> None:
        if not config.security.require_api_key:
            return
        expected = os.getenv(config.security.api_key_env)
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Сервисный API key не настроен.",
            )
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Некорректный API key.",
            )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return _error_response(
            request,
            exc.status_code,
            "http_error",
            str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        LOGGER.exception("Необработанная ошибка API")
        return _error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            redact_secrets(str(exc))[:500],
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
        dependencies=[Depends(require_api_key)],
    )
    def chat(request: Request, payload: ChatRequest) -> ChatResponse:
        _validate_message_size(payload.message, config)
        result, duration_ms = request.app.state.runtime.ask(
            user_id=payload.user_id,
            session_id=payload.session_id,
            message=payload.message,
        )
        retrieval_status = result.retrieval.status if result.retrieval else "not_used"
        retrieval_counter.labels(retrieval_status).inc()
        return ChatResponse(
            **result.model_dump(mode="python"),
            request_id=request.state.request_id,
            duration_ms=duration_ms,
        )

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
        dependencies=[Depends(require_api_key)],
    )
    def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
        _validate_message_size(payload.message, config)

        def events():
            yield _sse("started", {"request_id": request.state.request_id})
            result, duration_ms = request.app.state.runtime.ask(
                user_id=payload.user_id,
                session_id=payload.session_id,
                message=payload.message,
            )
            response = ChatResponse(
                **result.model_dump(mode="python"),
                request_id=request.state.request_id,
                duration_ms=duration_ms,
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
        dependencies=[Depends(require_api_key)],
    )
    def get_session(request: Request, session_id: str, user_id: str) -> SessionResponse:
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
        dependencies=[Depends(require_api_key)],
    )
    def delete_session(
        request: Request,
        session_id: str,
        user_id: str,
    ) -> DeleteSessionResponse:
        return request.app.state.runtime.delete_session(
            user_id=user_id,
            session_id=session_id,
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Состояние"],
        summary="Проверить liveness",
        description="Проверяет, что HTTP-процесс запущен.",
    )
    def health() -> HealthResponse:
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
        description="Возвращает счётчики запросов, latency и статусы retrieval.",
        responses={
            200: {
                "description": "Метрики в Prometheus text exposition format.",
                "content": {
                    "text/plain": {
                        "example": 'support_agent_requests_total{status="200"} 1.0'
                    }
                },
            }
        },
    )
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def _validate_message_size(message: str, config: AgentAppConfig) -> None:
    if len(message) > config.service.request_max_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Сообщение превышает service.request_max_chars: "
                f"actual={len(message)} limit={config.service.request_max_chars}"
            ),
        )


def _error_response(
    request: Request,
    status_code: int,
    error: str,
    message: str,
) -> JSONResponse:
    payload = ApiError(
        error=error,
        message=redact_secrets(message),
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        status_code=status_code, content=payload.model_dump(mode="json")
    )


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
