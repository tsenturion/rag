"""HTTP-доступ к пользовательским ресурсам и постоянным операциям frontend."""

from __future__ import annotations

import json
import logging
from typing import Any, Generic, TypeVar

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_app.models import MemoryRecord, MemoryType
from agent_app.orchestration.models import JobRecord
from agent_app.multi_agent.exporting import MultiAgentExporter
from agent_app.service.auth import AuthManager, Permission
from agent_app.service.operation_store import (
    OperationPage,
    OperationRecord,
    OperationRequest,
)
from agent_app.service.operations import (
    ArtifactRecord,
    OperationService,
    PublicProfile,
    SourceRecord,
    public_text,
    public_data,
)
from agent_app.support.incidents import IncidentPriority, IncidentRecord, IncidentStatus
from agent_app.tools.project import (
    CreateProjectInput,
    CreateTaskInput,
    UpdateTaskStatusInput,
    project_tools,
)

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


class ResourcePage(BaseModel, Generic[T]):
    """Страница ресурсов с непрозрачным указателем следующей выборки."""

    items: list[T]
    next_cursor: str | None = None


class MemoryInput(BaseModel):
    """Сохраняет происхождение и область памяти без возможности сменить владельца."""

    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=12000)
    memory_type: MemoryType = "fact"
    session_id: str | None = Field(default=None, max_length=128)
    tags: list[str] = Field(default_factory=list, max_length=30)
    importance: int = Field(default=3, ge=1, le=5)
    ttl_seconds: int | None = Field(default=None, ge=1, le=31536000)


class MemoryUpdate(BaseModel):
    """Редактирование не позволяет менять scope или обходить policy памяти."""

    model_config = ConfigDict(extra="forbid")
    value: str = Field(min_length=1, max_length=12000)
    importance: int = Field(default=3, ge=1, le=5)


class IncidentInput(BaseModel):
    """Инцидент привязан к текущему пользователю и выбранному диалогу."""

    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=12000)
    priority: IncidentPriority = "medium"
    component: str | None = Field(default=None, max_length=200)


class IncidentUpdate(BaseModel):
    """Ограничивает переход известными статусами инженерного инцидента."""

    model_config = ConfigDict(extra="forbid")
    status: IncidentStatus


class SearchInput(BaseModel):
    """Тестовый retrieval использует тот же embedding-профиль, что активный RAG."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class BrowserError(BaseModel):
    """Телеметрия принимает классификацию ошибки, но не текст диалога или токены."""

    model_config = ConfigDict(extra="forbid")
    category: str = Field(pattern=r"^[a-z_]{1,40}$")
    route: str = Field(pattern=r"^[a-z_-]{1,60}$")
    request_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{36}$")


def install_resource_routes(
    app: FastAPI, config, service: OperationService, require_permission, authenticate
) -> None:
    """Добавляет маршруты поверх существующих store и tool-функций, сохраняя RBAC."""

    def enabled() -> None:
        """Web-ресурсы доступны только при включённом persistent web runtime."""
        if not config.web.enabled:
            raise HTTPException(503, "Web API отключён.")

    router = APIRouter(prefix="/v1", dependencies=[Depends(enabled)])

    def owner(request: Request, user_id: str | None = None) -> str:
        """Администратор может явно выбрать user_id; остальные работают со своим."""
        principal = request.state.principal
        user_id = user_id or principal.subject
        AuthManager(config.security).enforce_user_scope(principal, user_id)
        return user_id

    def operation(request: Request, operation_id: str) -> OperationRecord:
        """Оператор не получает чужие результаты только из-за права запуска операций."""
        principal = request.state.principal
        return service.store.get(
            operation_id,
            None
            if "admin" in principal.roles or "service" in principal.roles
            else principal.subject,
        )

    @router.get(
        "/memories",
        response_model=ResourcePage[MemoryRecord],
        tags=["Память"],
        dependencies=[Depends(require_permission(Permission.MEMORY_READ))],
    )
    def memories(
        request: Request,
        user_id: str | None = None,
        cursor: str = "",
        limit: int = Query(50, ge=1, le=200),
        session_id: str | None = None,
        memory_type: MemoryType | None = None,
        projects_only: bool = False,
    ) -> ResourcePage[MemoryRecord]:
        """Пагинация выполняется по идентификатору; истёкшая память исключается store-политикой."""
        store = request.app.state.runtime.memory_store
        sql, args = (
            "SELECT id FROM memories WHERE user_id = ? AND id > ?",
            [owner(request, user_id), cursor],
        )
        if session_id is not None:
            sql += " AND (session_id IS NULL OR session_id = ?)"
            args.append(session_id)
        if memory_type:
            sql += " AND memory_type = ?"
            args.append(memory_type)
        if projects_only:
            sql += " AND (key LIKE ? OR key LIKE ?)"
            args.extend(["project:%", "task:%"])
        with store.database.connection(store.path) as conn:
            rows = conn.execute(
                sql + " ORDER BY id LIMIT ?", [*args, limit + 1]
            ).fetchall()
        records = [
            record
            for row in rows[:limit]
            if (record := store.get(row["id"], user_id=owner(request, user_id)))
            is not None
        ]
        return ResourcePage(
            items=records,
            next_cursor=rows[limit - 1]["id"] if len(rows) > limit else None,
        )

    @router.post(
        "/memories",
        response_model=MemoryRecord,
        status_code=201,
        tags=["Память"],
        dependencies=[Depends(require_permission(Permission.MEMORY_WRITE))],
    )
    def save_memory(payload: MemoryInput, request: Request) -> MemoryRecord:
        """Отклоняет секреты перед сохранением факта или предпочтения пользователя."""
        try:
            return request.app.state.runtime.memory_store.save(
                user_id=owner(request), source="user", **payload.model_dump()
            )
        except ValueError as exc:
            raise HTTPException(422, public_text(str(exc))) from exc

    @router.get(
        "/memories/{memory_id}",
        response_model=MemoryRecord,
        tags=["Память"],
        dependencies=[Depends(require_permission(Permission.MEMORY_READ))],
    )
    def memory_detail(memory_id: str, request: Request) -> MemoryRecord:
        """Карточка памяти доступна владельцу; чужая и истёкшая запись дают одинаковый 404."""
        record = request.app.state.runtime.memory_store.get(
            memory_id, user_id=owner(request)
        )
        if record is None:
            raise HTTPException(404, "Память не найдена.")
        return record

    @router.patch(
        "/memories/{memory_id}",
        response_model=MemoryRecord,
        tags=["Память"],
        dependencies=[Depends(require_permission(Permission.MEMORY_WRITE))],
    )
    def update_memory(
        memory_id: str, payload: MemoryUpdate, request: Request
    ) -> MemoryRecord:
        """Изменяет только запись проверенного владельца."""
        store = request.app.state.runtime.memory_store
        if store.get(memory_id, user_id=owner(request)) is None:
            raise HTTPException(404, "Память не найдена.")
        try:
            return store.update(
                memory_id, user_id=owner(request), **payload.model_dump()
            )
        except ValueError as exc:
            raise HTTPException(422, public_text(str(exc))) from exc

    @router.delete(
        "/memories/{memory_id}",
        status_code=204,
        tags=["Память"],
        dependencies=[Depends(require_permission(Permission.MEMORY_WRITE))],
    )
    def delete_memory(memory_id: str, request: Request) -> None:
        """Чужой UUID не позволяет удалить запись другого пользователя."""
        if not request.app.state.runtime.memory_store.delete(
            memory_id, user_id=owner(request)
        ):
            raise HTTPException(404, "Память не найдена.")

    @router.get(
        "/incidents",
        response_model=ResourcePage[IncidentRecord],
        tags=["Инциденты"],
        dependencies=[Depends(require_permission(Permission.INCIDENT_READ))],
    )
    def incidents(
        request: Request,
        cursor: str = "",
        limit: int = Query(50, ge=1, le=200),
        incident_status: IncidentStatus | None = None,
    ) -> ResourcePage[IncidentRecord]:
        """Возвращает стабильную страницу инцидентов текущего инженера."""
        store = request.app.state.runtime.incident_store
        sql, args = (
            "SELECT * FROM incidents WHERE user_id = ? AND id > ?",
            [owner(request), cursor],
        )
        if incident_status:
            sql += " AND status = ?"
            args.append(incident_status)
        with store.database.connection(store.path) as conn:
            rows = conn.execute(
                sql + " ORDER BY id LIMIT ?", [*args, limit + 1]
            ).fetchall()
        records = [store.get(row["id"], user_id=owner(request)) for row in rows[:limit]]
        return ResourcePage(
            items=records,
            next_cursor=rows[limit - 1]["id"] if len(rows) > limit else None,
        )

    @router.get(
        "/incidents/{incident_id}",
        response_model=IncidentRecord,
        tags=["Инциденты"],
        dependencies=[Depends(require_permission(Permission.INCIDENT_READ))],
    )
    def incident(incident_id: str, request: Request) -> IncidentRecord:
        """Карточка инцидента проверяет владельца на сервере."""
        record = request.app.state.runtime.incident_store.get(
            incident_id, user_id=owner(request)
        )
        if record is None:
            raise HTTPException(404, "Инцидент не найден.")
        return record

    @router.post(
        "/incidents",
        response_model=IncidentRecord,
        status_code=201,
        tags=["Инциденты"],
        dependencies=[Depends(require_permission(Permission.INCIDENT_WRITE))],
    )
    def create_incident(payload: IncidentInput, request: Request) -> IncidentRecord:
        """Создаёт инцидент тем же доменным сервисом, что инженерный tool."""
        try:
            return request.app.state.runtime.incident_store.create(
                user_id=owner(request), **payload.model_dump()
            )
        except ValueError as exc:
            raise HTTPException(422, public_text(str(exc))) from exc

    @router.patch(
        "/incidents/{incident_id}",
        response_model=IncidentRecord,
        tags=["Инциденты"],
        dependencies=[Depends(require_permission(Permission.INCIDENT_WRITE))],
    )
    def update_incident(
        incident_id: str, payload: IncidentUpdate, request: Request
    ) -> IncidentRecord:
        """Не создаёт отсутствующую запись при ошибке идентификатора."""
        record = request.app.state.runtime.incident_store.update_status(
            incident_id, user_id=owner(request), status=payload.status
        )
        if record is None:
            raise HTTPException(404, "Инцидент не найден.")
        return record

    def project_tool(request: Request, name: str, payload) -> dict[str, Any]:
        """HTTP и LLM используют одинаковые ключи и правила долговременных проектов."""
        tools = project_tools(
            request.app.state.runtime.memory_store,
            user_id=owner(request),
            session_id="web",
        )
        tool = next(item for item in tools if item.name == name)
        return public_data(json.loads(tool.invoke(payload.model_dump())))

    @router.post(
        "/projects",
        tags=["Проекты"],
        dependencies=[Depends(require_permission(Permission.MEMORY_WRITE))],
    )
    def create_project(payload: CreateProjectInput, request: Request) -> dict[str, Any]:
        """Создаёт проект, доступный в следующих диалогах пользователя."""
        return project_tool(request, "create_project", payload)

    @router.post(
        "/projects/tasks",
        tags=["Проекты"],
        dependencies=[Depends(require_permission(Permission.MEMORY_WRITE))],
    )
    def create_task(payload: CreateTaskInput, request: Request) -> dict[str, Any]:
        """Сохраняет задачу через существующий project tool."""
        return project_tool(request, "create_task", payload)

    @router.patch(
        "/projects/tasks",
        tags=["Проекты"],
        dependencies=[Depends(require_permission(Permission.MEMORY_WRITE))],
    )
    def update_task(payload: UpdateTaskStatusInput, request: Request) -> dict[str, Any]:
        """Обновляет статус по доменному ключу проекта и задачи."""
        return project_tool(request, "update_task_status", payload)

    @router.get(
        "/projects",
        response_model=ResourcePage[MemoryRecord],
        tags=["Проекты"],
        dependencies=[Depends(require_permission(Permission.MEMORY_READ))],
    )
    def projects(
        request: Request, cursor: str = "", limit: int = Query(50, ge=1, le=200)
    ) -> ResourcePage[MemoryRecord]:
        """Проекты и задачи представлены типизированными записями долговременной памяти."""
        return memories(request, cursor=cursor, limit=limit, projects_only=True)

    @router.get(
        "/multi-agent/runs",
        response_model=ResourcePage[dict[str, Any]],
        tags=["Мультиагентная система"],
        dependencies=[Depends(require_permission(Permission.RUN_READ))],
    )
    def runs(
        request: Request, cursor: str = "", limit: int = Query(50, ge=1, le=200)
    ) -> ResourcePage[dict[str, Any]]:
        """Находит сохранённые запуски после перезапуска API с фильтрацией владельца."""
        exporter = MultiAgentExporter(config.multi_agent.output_dir)
        records = []
        for directory in sorted(config.multi_agent.output_dir.glob("*")):
            if directory.name <= cursor or not directory.is_dir():
                continue
            try:
                data = exporter.load_result(directory.name)
            except (ValueError, OSError):
                continue
            if data and data.get("user_id") == owner(request):
                records.append(public_data(data))
                if len(records) > limit:
                    break
        return ResourcePage(
            items=records[:limit],
            next_cursor=records[limit - 1]["run_id"] if len(records) > limit else None,
        )

    @router.get(
        "/operations/profiles",
        response_model=list[PublicProfile],
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_READ))],
    )
    def profiles() -> list[PublicProfile]:
        """Возвращает зарегистрированные действия и реальную готовность worker."""
        return service.public_profiles()

    @router.get(
        "/orchestration/jobs",
        response_model=ResourcePage[JobRecord],
        tags=["Оркестрация"],
        dependencies=[Depends(require_permission(Permission.ORCHESTRATION_READ))],
    )
    def jobs(
        request: Request, cursor: str = "", limit: int = Query(50, ge=1, le=200)
    ) -> ResourcePage[JobRecord]:
        """Отображает также задания, поставленные через CLI, с текущими broker-статусами."""
        orchestration = request.app.state.runtime.orchestration_service
        if orchestration is None:
            raise HTTPException(503, "Оркестрация отключена.")
        records = orchestration.store.list_for_user(
            owner(request), after=cursor, limit=limit + 1
        )
        return ResourcePage(
            items=records[:limit],
            next_cursor=records[limit - 1].job.id if len(records) > limit else None,
        )

    @router.get(
        "/operations",
        response_model=OperationPage,
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_READ))],
    )
    def operations(
        request: Request,
        cursor: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        operation_status: str | None = None,
    ) -> OperationPage:
        """Список операций сохраняется независимо от вкладки браузера."""
        return service.store.list(
            owner(request), limit=limit, cursor=cursor, status=operation_status
        )

    @router.post(
        "/operations",
        response_model=OperationRecord,
        status_code=202,
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_WRITE))],
    )
    def submit(payload: OperationRequest, request: Request) -> OperationRecord:
        """Создаёт ровно одну операцию для пары владелец/ключ идемпотентности."""
        try:
            return service.submit(owner(request), payload)
        except ValueError:
            raise HTTPException(
                422, "Параметры не соответствуют конфигурации операции."
            ) from None

    @router.get(
        "/operations/{operation_id}",
        response_model=OperationRecord,
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_READ))],
    )
    def get_operation(operation_id: str, request: Request) -> OperationRecord:
        """Состояние читается из БД, а не из памяти HTTP worker."""
        return operation(request, operation_id)

    @router.delete(
        "/operations/{operation_id}",
        response_model=OperationRecord,
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_WRITE))],
    )
    def cancel(operation_id: str, request: Request) -> OperationRecord:
        """Отмена отмечается в БД и обрабатывается владельцем процесса."""
        record = operation(request, operation_id)
        return service.store.cancel(operation_id, record.user_id)

    @router.get(
        "/operations/{operation_id}/artifacts",
        response_model=list[ArtifactRecord],
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_READ))],
    )
    def artifacts(operation_id: str, request: Request) -> list[ArtifactRecord]:
        """Список опубликованных результатов доступен только владельцу запуска."""
        operation(request, operation_id)
        return service.artifacts(operation_id)

    @router.get(
        "/operations/{operation_id}/artifacts/{artifact_id}",
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_READ))],
    )
    def download_artifact(operation_id: str, artifact_id: str, request: Request):
        """Артефакт отдаётся как скачивание, а HTML не исполняется на origin приложения."""
        operation(request, operation_id)
        path = service.artifact(operation_id, artifact_id)
        if path.name == "manifest.json":
            # Machine-readable файлы корпуса остаются побайтно исходными. Только
            # диагностический manifest выдаётся без серверных настроек и DSN.
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.pop("config", None)
            return JSONResponse(
                data,
                headers={
                    "Content-Disposition": 'attachment; filename="manifest.json"',
                    "X-Content-Type-Options": "nosniff",
                },
            )
        return FileResponse(
            path,
            filename=path.name,
            media_type="application/octet-stream",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @router.get(
        "/operations/{operation_id}/preview/{artifact_id}",
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_READ))],
    )
    def preview_artifact(
        operation_id: str,
        artifact_id: str,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        """JSONL читается построчно, поэтому просмотр embedding-корпуса не загружает его целиком."""
        operation(request, operation_id)
        path = service.artifact(operation_id, artifact_id)
        if path.suffix == ".jsonl":
            import itertools

            with path.open(encoding="utf-8") as stream:
                rows = [
                    public_data(json.loads(line))
                    for line in itertools.islice(stream, offset, offset + limit + 1)
                ]
            return {
                "items": rows[:limit],
                "next_offset": offset + limit if len(rows) > limit else None,
            }
        if path.suffix == ".json" and path.stat().st_size <= 2_000_000:
            return {"data": public_data(json.loads(path.read_text(encoding="utf-8")))}
        raise HTTPException(413, "Для этого файла используйте скачивание.")

    @router.get(
        "/operations/{operation_id}/logs",
        tags=["Операции"],
        dependencies=[Depends(require_permission(Permission.OPERATION_READ))],
    )
    def logs(
        operation_id: str, request: Request, offset: int = Query(0, ge=0)
    ) -> dict[str, Any]:
        """Отдаёт ограниченный блок сохранённого журнала с курсором в байтах."""
        operation(request, operation_id)
        path = service.job_dir(operation_id) / "execution.log"
        if not path.exists():
            return {"text": "", "next_offset": offset}
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(65536)
            next_offset = stream.tell()
        return {
            "text": public_text(data.decode("utf-8", errors="replace")),
            "next_offset": next_offset,
        }

    @router.get(
        "/knowledge/sources",
        response_model=list[SourceRecord],
        tags=["База знаний"],
        dependencies=[Depends(require_permission(Permission.KNOWLEDGE_READ))],
    )
    def sources(
        request: Request, after: str = "", limit: int = Query(50, ge=1, le=200)
    ) -> list[SourceRecord]:
        """Возвращает только исходники текущего пользователя."""
        return service.sources(owner(request), after=after, limit=limit)

    @router.post(
        "/knowledge/sources",
        response_model=SourceRecord,
        status_code=201,
        tags=["База знаний"],
        dependencies=[Depends(require_permission(Permission.KNOWLEDGE_WRITE))],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def upload(
        request: Request, filename: str = Query(min_length=1, max_length=160)
    ) -> SourceRecord:
        """Лимит проверяется при чтении потока, включая запрос без Content-Length."""
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > config.web.upload_max_bytes:
                raise HTTPException(413, "Файл превышает лимит загрузки.")
            chunks.append(chunk)
        return service.add_source(owner(request), filename, b"".join(chunks))

    @router.get(
        "/knowledge/sources/{source_id}",
        tags=["База знаний"],
        dependencies=[Depends(require_permission(Permission.KNOWLEDGE_READ))],
    )
    def download_source(source_id: str, request: Request):
        """Проверяет hash источника перед скачиванием."""
        record, path = service.source(source_id, owner(request))
        return FileResponse(
            path,
            filename=record.name,
            media_type="application/octet-stream",
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @router.delete(
        "/knowledge/sources/{source_id}",
        status_code=204,
        tags=["База знаний"],
        dependencies=[Depends(require_permission(Permission.KNOWLEDGE_WRITE))],
    )
    def delete_source(source_id: str, request: Request) -> None:
        """Не удаляет исходник, пока worker использует его для подготовки корпуса."""
        service.delete_source(source_id, owner(request))

    @router.post(
        "/knowledge/search",
        tags=["База знаний"],
        dependencies=[
            Depends(require_permission(Permission.KNOWLEDGE_READ)),
            Depends(require_permission(Permission.CHAT)),
        ],
    )
    def search(payload: SearchInput, request: Request) -> dict[str, Any]:
        """Поиск вызывает только configured embedding-провайдер и проходит лимит платных запросов."""
        result = request.app.state.runtime.rag_runtime.retrieve(
            payload.query, top_k=payload.top_k
        )
        return public_data(result.model_dump(mode="json"))

    @router.get(
        "/integrations", tags=["Приложение"], dependencies=[Depends(authenticate)]
    )
    def integrations() -> dict[str, Any]:
        """Возвращает безопасный каталог подключений без адресов с credentials."""
        return {
            "llm": {"provider": config.agent.provider, "model": config.agent.model},
            "tools": config.tools.enabled,
            "mcp": [
                {"name": server.name, "transport": server.transport}
                for server in config.tools.mcp_servers
            ],
            "roles": config.multi_agent.role_llm_profiles,
            "configuration_mode": "server_profiles",
        }

    @router.post(
        "/telemetry/browser",
        status_code=204,
        tags=["Состояние"],
        dependencies=[Depends(require_permission(Permission.CHAT))],
    )
    def browser_error(payload: BrowserError) -> None:
        """Сохраняет диагностический код и связь с запросом без произвольного текста клиента."""
        LOGGER.warning(
            "Ошибка web-клиента: %s, страница %s",
            payload.category,
            payload.route,
            extra={"event": "browser_error", "request_id": payload.request_id},
        )

    app.include_router(router)
