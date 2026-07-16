from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Callable
from typing import Any
from uuid import uuid4

from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.rest_routes import create_rest_routes
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
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

from agent_app.config import AgentAppConfig
from agent_app.multi_agent.models import MultiAgentRunResult
from agent_app.multi_agent.roles import default_role_definitions

A2AAsk = Callable[..., MultiAgentRunResult]


def build_agent_card(config: AgentAppConfig, *, base_url: str) -> AgentCard:
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
    def __init__(self, card: AgentCard, ask: A2AAsk):
        self.card = card
        self.ask = ask
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()

    async def on_message_send(
        self,
        params: SendMessageRequest,
        context: ServerCallContext,
    ) -> Task | Message:
        del context
        text = "\n".join(
            part.text for part in params.message.parts if part.text
        ).strip()
        if not text:
            raise ValueError("A2A message не содержит текстовую часть")
        metadata = (
            MessageToDict(params.message.metadata) if params.message.metadata else {}
        )
        user_id = str(metadata.get("userId") or metadata.get("user_id") or "a2a-user")
        session_id = str(
            metadata.get("sessionId")
            or metadata.get("session_id")
            or params.message.context_id
            or str(uuid4())
        )
        result = await asyncio.to_thread(
            self.ask,
            user_id=user_id,
            session_id=session_id,
            message=text,
        )
        task = self._task_from_result(params.message, result)
        with self._lock:
            self._tasks[task.id] = task
        return task

    async def on_get_task(
        self,
        params: GetTaskRequest,
        context: ServerCallContext,
    ) -> Task | None:
        del context
        with self._lock:
            task = self._tasks.get(params.id)
            if task is None:
                return None
            copy = Task()
            copy.CopyFrom(task)
            return copy

    async def on_list_tasks(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        del context
        with self._lock:
            tasks = list(self._tasks.values())
        if params.context_id:
            tasks = [task for task in tasks if task.context_id == params.context_id]
        page_size = params.page_size or 100
        selected = tasks[:page_size]
        return ListTasksResponse(
            tasks=selected,
            page_size=len(selected),
            total_size=len(tasks),
        )

    async def on_cancel_task(
        self,
        params: CancelTaskRequest,
        context: ServerCallContext,
    ) -> Task | None:
        del context
        with self._lock:
            task = self._tasks.get(params.id)
            if task is None:
                return None
            task.status.state = TaskState.TASK_STATE_CANCELED
            task.status.timestamp.CopyFrom(_timestamp())
            copy = Task()
            copy.CopyFrom(task)
            return copy

    async def on_message_send_stream(
        self,
        params: SendMessageRequest,
        context: ServerCallContext,
    ) -> AsyncGenerator[Any]:
        del params, context
        raise UnsupportedOperationError("Streaming A2A в этом модуле отключён")
        yield

    async def on_create_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext,
    ) -> TaskPushNotificationConfig:
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_get_task_push_notification_config(
        self,
        params: GetTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> TaskPushNotificationConfig:
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_subscribe_to_task(
        self,
        params: SubscribeToTaskRequest,
        context: ServerCallContext,
    ) -> AsyncGenerator[Any]:
        del params, context
        raise UnsupportedOperationError("Task subscription не поддерживается")
        yield

    async def on_list_task_push_notification_configs(
        self,
        params: ListTaskPushNotificationConfigsRequest,
        context: ServerCallContext,
    ) -> ListTaskPushNotificationConfigsResponse:
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigRequest,
        context: ServerCallContext,
    ) -> None:
        del params, context
        raise UnsupportedOperationError("Push notifications не поддерживаются")

    async def on_get_extended_agent_card(
        self,
        params: GetExtendedAgentCardRequest,
        context: ServerCallContext,
    ) -> AgentCard:
        del params, context
        return self.card

    @staticmethod
    def _task_from_result(
        request: Message,
        result: MultiAgentRunResult,
    ) -> Task:
        response = result.response
        context_id = request.context_id or response.session_id
        response_message = Message(
            message_id=str(uuid4()),
            context_id=context_id,
            task_id=response.run_id,
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
            id=response.run_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.TASK_STATE_COMPLETED,
                message=response_message,
                timestamp=_timestamp(),
            ),
            artifacts=[artifact],
            history=[request, response_message],
            metadata=_struct({"run_dir": result.run_dir or ""}),
        )


def install_a2a_routes(
    app: FastAPI,
    config: AgentAppConfig,
    *,
    base_url: str,
    ask: A2AAsk,
) -> MultiAgentA2AHandler:
    card = build_agent_card(config, base_url=base_url)
    handler = MultiAgentA2AHandler(card, ask)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(
            handler,
            rpc_url=config.multi_agent.protocols.a2a_rpc_path,
        ),
        rest_routes=create_rest_routes(
            handler,
            path_prefix=config.multi_agent.protocols.a2a_rest_path,
        ),
    )
    return handler


def agent_card_dict(card: AgentCard) -> dict[str, object]:
    return MessageToDict(card, preserving_proto_field_name=False)


def _timestamp() -> Timestamp:
    timestamp = Timestamp()
    timestamp.GetCurrentTime()
    return timestamp


def _struct(payload: dict[str, object]) -> Struct:
    value = Struct()
    value.update(payload)
    return value


def _value(payload: object):
    from google.protobuf.struct_pb2 import Value

    value = Value()
    value.struct_value.update(payload)
    return value
