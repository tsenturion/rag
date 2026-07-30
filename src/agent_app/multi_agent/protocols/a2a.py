"""Реализация компонентов для межагентных протоколов."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any
from uuid import uuid4

from a2a.auth.user import UnauthenticatedUser, User
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.routes.common import ServerCallContextBuilder
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Artifact,
    CancelTaskRequest,
    DeleteTaskPushNotificationConfigRequest,
    GetExtendedAgentCardRequest,
    GetTaskPushNotificationConfigRequest,
    GetTaskRequest,
    ListTaskPushNotificationConfigsRequest,
    ListTaskPushNotificationConfigsResponse,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SubscribeToTaskRequest,
    Task,
    TaskPushNotificationConfig,
    TaskState,
    TaskStatus,
    UnsupportedOperationError,
)
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict, Parse
from google.protobuf.struct_pb2 import Struct, Value
from google.protobuf.timestamp_pb2 import Timestamp
from starlette.requests import Request

from agent_app.config import AgentAppConfig
from agent_app.guardrails import GuardrailPipeline
from agent_app.multi_agent.models import MultiAgentRunResult
from agent_app.multi_agent.sanitization import (
    public_run_reference,
    sanitize_run_result,
)
from agent_app.multi_agent.roles import default_role_definitions
from agent_app.multi_agent.protocols.a2a_store import A2ATaskStore
from agent_app.service.auth import Principal

A2AAsk = Callable[..., MultiAgentRunResult]


def build_agent_card(config: AgentAppConfig, *, base_url: str) -> AgentCard:
    """Формирует публичное описание возможностей мультиагентной системы для автоматического обнаружения и интеграции в A2A-протоколах."""
    skills = [
        AgentSkill(
            id=capability.name,
            name=definition.title,
            description=capability.description,
            tags=["engineering-support", capability.name],
            examples=[definition.goal],
            input_modes=["text/plain", "application/json"],
            output_modes=["text/plain", "application/json"],
        )
        for definition in default_role_definitions()
        for capability in definition.capabilities
    ]
    return AgentCard(
        name="Инженерная мультиагентная система",
        description=(
            "Координатор делегирует поиск знаний, диагностику и работу с "
            "инцидентами специализированным агентам."
        ),
        supported_interfaces=[
            AgentInterface(
                url=base_url.rstrip("/") + config.multi_agent.protocols.a2a_rpc_path,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
            AgentInterface(
                url=base_url.rstrip("/") + config.multi_agent.protocols.a2a_rest_path,
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
            ),
        ],
        provider=AgentProvider(
            organization="Учебный RAG-проект",
            url=base_url,
        ),
        version="1.0.0",
        documentation_url=base_url.rstrip("/") + "/docs",
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
            extended_agent_card=False,
        ),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=skills,
    )


class MultiAgentA2AHandler(RequestHandler):
    """Обеспечивает потокобезопасную маршрутизацию и обработку запросов межагентного взаимодействия с сохранением задач и их статусов."""

    def __init__(
        self,
        card: AgentCard,
        ask: A2AAsk,
        store: A2ATaskStore,
        guardrails: GuardrailPipeline | None = None,
        *,
        request_max_chars: int = 20_000,
    ):
        """Настраивает хранилище задач, guardrails и предел итогового A2A-текста."""
        self.card = card
        self.ask = ask
        self.store = store
        self.guardrails = guardrails
        if request_max_chars < 1:
            raise ValueError("request_max_chars должен быть положительным")
        self.request_max_chars = request_max_chars
        self._futures: dict[str, asyncio.Task[None]] = {}

    async def on_message_send(
        self,
        params: SendMessageRequest,
        context: ServerCallContext,
    ) -> Task | Message:
        """Регистрирует задачу до выполнения и запускает её в фоне."""
        text = "\n".join(
            part.text for part in params.message.parts if part.text
        ).strip()
        if not text:
            raise ValueError("A2A message не содержит текстовую часть")
        if len(text) > self.request_max_chars:
            raise ValueError(
                "Собранный текст A2A message превышает разрешённый лимит: "
                f"{len(text)} > {self.request_max_chars} символов"
            )
        metadata = (
            MessageToDict(params.message.metadata) if params.message.metadata else {}
        )
        requested_user = str(
            metadata.get("userId") or metadata.get("user_id") or ""
        ).strip()
        user_id = self._resolve_user_id(context, requested_user)
        session_id = str(
            metadata.get("sessionId")
            or metadata.get("session_id")
            or params.message.context_id
            or str(uuid4())
        )
        task_id = str(uuid4())
        task = Task(
            id=task_id,
            context_id=params.message.context_id or session_id,
            status=TaskStatus(
                state=TaskState.TASK_STATE_SUBMITTED,
                timestamp=_timestamp(),
            ),
            history=[params.message],
        )
        self.store.save(task, owner_id=user_id)
        future = asyncio.create_task(
            self._execute(
                task_id=task_id,
                request=params.message,
                user_id=user_id,
                session_id=session_id,
                text=text,
            )
        )
        self._futures[task_id] = future
        future.add_done_callback(lambda _future: self._futures.pop(task_id, None))
        return task

    async def on_get_task(
        self,
        params: GetTaskRequest,
        context: ServerCallContext,
    ) -> Task | None:
        """Возвращает задачу только её owner либо service/admin principal."""
        stored = self.store.get(params.id)
        if stored is None:
            return None
        task, owner = stored
        if not self._can_access(context, owner):
            return None
        return _copy_task(task)

    async def on_list_tasks(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        """Возвращает owner-scoped страницу задач с непрозрачным page token."""
        owner_scope = self._owner_scope(context)
        tasks = [task for task, _owner in self.store.list(owner_id=owner_scope)]
        if params.context_id:
            tasks = [task for task in tasks if task.context_id == params.context_id]
        if params.status:
            tasks = [task for task in tasks if task.status.state == params.status]
        if params.HasField("status_timestamp_after"):
            threshold = params.status_timestamp_after.ToDatetime()
            tasks = [
                task for task in tasks if task.status.timestamp.ToDatetime() > threshold
            ]
        offset = _decode_page_token(params.page_token)
        page_size = max(1, min(params.page_size or 100, 500))
        selected = tasks[offset : offset + page_size]
        next_offset = offset + len(selected)
        return ListTasksResponse(
            tasks=[_copy_task(task) for task in selected],
            next_page_token=(
                _encode_page_token(next_offset) if next_offset < len(tasks) else ""
            ),
            page_size=len(selected),
            total_size=len(tasks),
        )

    async def on_cancel_task(
        self,
        params: CancelTaskRequest,
        context: ServerCallContext,
    ) -> Task | None:
        """Отменяет только ещё не начатую owner-scoped задачу.

        Синхронный provider/tool вызов выполняется в отдельном потоке и не имеет
        переносимого механизма принудительной остановки. Поэтому работающая
        задача не помечается отменённой: такой статус скрывал бы продолжающиеся
        расходы и побочные действия.
        """
        stored = self.store.get(params.id)
        if stored is None:
            return None
        task, owner = stored
        if not self._can_access(context, owner):
            return None
        if task.status.state in {
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_REJECTED,
        }:
            return _copy_task(task)
        if task.status.state != TaskState.TASK_STATE_SUBMITTED:
            raise UnsupportedOperationError(
                "A2A-задача уже выполняется и не поддерживает безопасную отмену"
            )
        task.status.state = TaskState.TASK_STATE_CANCELED
        task.status.timestamp.CopyFrom(_timestamp())
        self.store.save(task, owner_id=owner)
        future = self._futures.get(params.id)
        if future is not None:
            future.cancel()
        return _copy_task(task)

    async def _execute(
        self,
        *,
        task_id: str,
        request: Message,
        user_id: str,
        session_id: str,
        text: str,
    ) -> None:
        """Выполняет зарегистрированную задачу и соблюдает сохранённую отмену."""
        stored = self.store.get(task_id)
        if stored is None:
            return
        task, owner = stored
        if task.status.state == TaskState.TASK_STATE_CANCELED:
            return
        task.status.state = TaskState.TASK_STATE_WORKING
        task.status.timestamp.CopyFrom(_timestamp())
        self.store.save(task, owner_id=owner)
        try:
            result = await asyncio.to_thread(
                self.ask,
                user_id=user_id,
                session_id=session_id,
                message=text,
            )
            if self.guardrails is not None:
                # Runtime уже очищает результат до экспорта. Повторная проверка
                # защищает A2A-границу и при пользовательской реализации ask.
                result = sanitize_run_result(result, self.guardrails)
        except asyncio.CancelledError:
            # cancel() используется только до запуска provider-вызова. Если
            # coroutine всё же отменена снаружи во время to_thread(), мы не
            # публикуем ложное конечное состояние: поток завершит вызов сам.
            return
        except Exception as exc:
            current = self.store.get(task_id)
            if (
                current is None
                or current[0].status.state == TaskState.TASK_STATE_CANCELED
            ):
                return
            failed = current[0]
            failed.status.state = TaskState.TASK_STATE_FAILED
            failed.status.timestamp.CopyFrom(_timestamp())
            failed.status.message.CopyFrom(
                Message(
                    message_id=str(uuid4()),
                    context_id=failed.context_id,
                    task_id=failed.id,
                    role=Role.ROLE_AGENT,
                    parts=[Part(text=f"Ошибка выполнения A2A: {type(exc).__name__}")],
                )
            )
            self.store.save(failed, owner_id=owner)
            return
        current = self.store.get(task_id)
        if current is None or current[0].status.state == TaskState.TASK_STATE_CANCELED:
            return
        self.store.save(
            self._task_from_result(request, result, task_id=task_id),
            owner_id=owner,
        )

    async def on_message_send_stream(
        self,
        params: SendMessageRequest,
        context: ServerCallContext,
    ) -> AsyncGenerator[Any]:
        """Гарантирует явное завершение с ошибкой при попытке использовать потоковую отправку сообщений, сохраняя контракт асинхронного генератора."""
        del params, context
        raise UnsupportedOperationError("Streaming A2A в этом модуле отключён")
        # Недостижимый yield сохраняет контракт AsyncGenerator официального SDK.
        yield  # noqa

    async def on_create_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext,
    ) -> TaskPushNotificationConfig:
        """Гарантирует отказ с ошибкой при попытке создать push-уведомление, поскольку поддержка не реализована."""
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_get_task_push_notification_config(
        self,
        params: GetTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> TaskPushNotificationConfig:
        """Гарантирует вызывающему коду явное исключение при попытке получить конфигурацию push-уведомлений, поскольку эта функция не поддерживается в системе."""
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_subscribe_to_task(
        self,
        params: SubscribeToTaskRequest,
        context: ServerCallContext,
    ) -> AsyncGenerator[Any]:
        """Гарантирует вызывающему коду немедленное исключение при попытке подписки на задачи, поскольку подписки не реализованы и не поддерживаются."""
        del params, context
        raise UnsupportedOperationError("Task subscription не поддерживается")
        # Недостижимый yield сохраняет контракт AsyncGenerator официального SDK.
        yield  # noqa

    async def on_list_task_push_notification_configs(
        self,
        params: ListTaskPushNotificationConfigsRequest,
        context: ServerCallContext,
    ) -> ListTaskPushNotificationConfigsResponse:
        """Гарантирует вызывающему коду отказ в доступе к списку push-конфигураций, поскольку push-уведомления не поддерживаются."""
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> None:
        """Гарантирует вызывающему коду невозможность удаления push-конфигурации, поскольку push-уведомления не реализованы."""
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_get_extended_agent_card(
        self,
        params: GetExtendedAgentCardRequest,
        context: ServerCallContext,
    ) -> AgentCard:
        """Гарантирует возврат полной карточки агента для межагентных протоколов без обращения к внешним источникам."""
        del params, context
        return self.card

    @staticmethod
    def _task_from_result(
        request: Message,
        result: MultiAgentRunResult,
        *,
        task_id: str,
    ) -> Task:
        """Гарантирует создание воспроизводимой задачи с историей и артефактами на основе результата мультиагентного запуска."""
        response = result.response
        context_id = request.context_id or response.session_id
        response_message = Message(
            message_id=str(uuid4()),
            context_id=context_id,
            task_id=task_id,
            role=Role.ROLE_AGENT,
            parts=[Part(text=response.answer, media_type="text/plain")],
            metadata=_struct(
                {
                    "run_id": response.run_id,
                    "quality": response.quality.score if response.quality else 0.0,
                    "selected_agents": response.selected_agents,
                }
            ),
        )
        artifact = Artifact(
            artifact_id=str(uuid4()),
            name="multi-agent-result",
            description="Структурированный результат координации агентов.",
            parts=[
                Part(
                    data=_value(response.model_dump(mode="json")),
                    media_type="application/json",
                )
            ],
        )
        return Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.TASK_STATE_COMPLETED,
                message=response_message,
                timestamp=_timestamp(),
            ),
            artifacts=[artifact],
            history=[request, response_message],
            metadata=_struct({"run_dir": public_run_reference(result.run_dir) or ""}),
        )

    @staticmethod
    def _owner_scope(context: ServerCallContext) -> str | None:
        """Возвращает owner-фильтр; service/admin получают административный scope."""
        roles = set(context.state.get("roles", []))
        if roles.intersection({"service", "admin"}):
            return None
        if context.user.is_authenticated:
            return context.user.user_name
        return None

    @classmethod
    def _can_access(cls, context: ServerCallContext, owner: str) -> bool:
        """Проверяет object-level authorization для A2A task."""
        scope = cls._owner_scope(context)
        return scope is None or scope == owner

    @classmethod
    def _resolve_user_id(cls, context: ServerCallContext, requested_user: str) -> str:
        """Связывает metadata.userId с аутентифицированным principal."""
        scope = cls._owner_scope(context)
        if scope is not None:
            if requested_user and requested_user != scope:
                raise ValueError("metadata.userId не совпадает с JWT subject")
            return scope
        if requested_user:
            return requested_user
        if context.user.is_authenticated:
            return context.user.user_name
        return "a2a-user"


class PrincipalA2AUser(User):
    """Адаптирует principal HTTP-сервиса к контракту официального A2A SDK."""

    def __init__(self, principal: Principal):
        """Сохраняет уже аутентифицированного principal без повторной проверки токена."""
        self.principal = principal

    @property
    def is_authenticated(self) -> bool:
        """Сообщает A2A SDK, что пользователя ранее проверил middleware сервиса."""
        return True

    @property
    def user_name(self) -> str:
        """Использует subject principal как неизменяемого владельца A2A-задач."""
        return self.principal.subject


class PrincipalContextBuilder(ServerCallContextBuilder):
    """Передаёт проверенный JWT/API-key principal из FastAPI middleware в A2A."""

    def build(self, request: Request) -> ServerCallContext:
        """Строит контекст A2A из principal и ролей, проверенных HTTP middleware."""
        principal = getattr(request.state, "principal", None)
        if principal is None:
            return ServerCallContext(
                user=UnauthenticatedUser(),
                state={"headers": dict(request.headers), "roles": []},
            )
        return ServerCallContext(
            user=PrincipalA2AUser(principal),
            state={
                "headers": dict(request.headers),
                "roles": list(principal.roles),
                "auth_method": principal.auth_method,
            },
        )


def install_a2a_routes(
    app: FastAPI,
    config: AgentAppConfig,
    *,
    base_url: str,
    ask: A2AAsk,
) -> MultiAgentA2AHandler:
    """Устанавливает все публичные маршруты межагентного взаимодействия и возвращает готовый обработчик для FastAPI-приложения."""
    card = build_agent_card(config, base_url=base_url)
    handler = MultiAgentA2AHandler(
        card,
        ask,
        A2ATaskStore(
            config.multi_agent.protocols.a2a_task_store_path,
            ttl_seconds=config.multi_agent.protocols.a2a_task_ttl_seconds,
            max_tasks=config.multi_agent.protocols.a2a_max_tasks,
        ),
        GuardrailPipeline(config.guardrails),
        request_max_chars=config.service.request_max_chars,
    )
    context_builder = PrincipalContextBuilder()
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(
            handler,
            rpc_url=config.multi_agent.protocols.a2a_rpc_path,
            context_builder=context_builder,
        ),
        rest_routes=create_rest_routes(
            handler,
            path_prefix=config.multi_agent.protocols.a2a_rest_path,
            context_builder=context_builder,
        ),
    )
    return handler


def agent_card_dict(card: AgentCard) -> dict[str, object]:
    """Преобразует описание агента в сериализуемый словарь для публикации через API или документацию."""
    return MessageToDict(card, preserving_proto_field_name=False)


def _timestamp() -> Timestamp:
    """Гарантирует получение актуального времени в формате, совместимом с межагентными протоколами."""
    timestamp = Timestamp()
    timestamp.GetCurrentTime()
    return timestamp


def _struct(payload: dict[str, object]) -> Struct:
    """Преобразует словарь в сериализуемую структуру для передачи данных между агентами без потери вложенности."""
    value = Struct()
    Parse(json.dumps(payload, ensure_ascii=False), value)
    return value


def _value(payload: object) -> Value:
    """Обеспечивает сериализацию произвольного значения для передачи в межагентных сообщениях с сохранением типа."""
    value = Value()
    Parse(json.dumps(payload, ensure_ascii=False), value)
    return value


def _copy_task(task: Task) -> Task:
    """Возвращает независимую protobuf-копию task для безопасной выдачи."""
    copied = Task()
    copied.CopyFrom(task)
    return copied


def _encode_page_token(offset: int) -> str:
    """Кодирует offset в непрозрачный URL-safe token."""
    return base64.urlsafe_b64encode(str(offset).encode("ascii")).decode("ascii")


def _decode_page_token(token: str) -> int:
    """Декодирует page token и отклоняет повреждённое или отрицательное значение."""
    if not token:
        return 0
    try:
        offset = int(base64.urlsafe_b64decode(token.encode("ascii")).decode("ascii"))
    except (binascii.Error, ValueError, UnicodeError) as exc:
        raise ValueError("Некорректный A2A page token") from exc
    if offset < 0:
        raise ValueError("Некорректный A2A page token")
    return offset
