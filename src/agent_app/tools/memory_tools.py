"""Инструменты долговременной памяти для инструментов агента."""

from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.memory.store import SQLiteMemoryStore
from agent_app.models import MemoryType

VALID_MEMORY_TYPES: set[str] = {"fact", "preference", "task", "summary", "note"}


class SaveMemoryInput(BaseModel):
    """Определяет параметры для сохранения записи в памяти агента с контролем типа, важности и срока жизни данных."""

    memory_type: str = Field(
        default="note",
        description=(
            "Тип записи памяти: fact, preference, task, summary или note. "
            "Если не уверен, используй note. Не записывай сюда key."
        ),
    )
    key: str = Field(description="Стабильный короткий ключ, например project_name")
    value: str = Field(description="Значение, которое нужно сохранить в памяти")
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    ttl_seconds: int | None = None


class SearchMemoryInput(BaseModel):
    """Формирует запросы к памяти агента с возможностью фильтрации по типу и ограничению количества результатов."""

    query: str
    memory_type: MemoryType | None = Field(
        default=None,
        description="Фильтр типа памяти. Не указывай, если не уверен в типе записи.",
    )
    limit: int | None = Field(default=None, ge=1)


class GetMemoryInput(BaseModel):
    """Обеспечивает однозначное получение записи из памяти агента по уникальному идентификатору."""

    memory_id: str


class UpdateMemoryInput(BaseModel):
    """Определяет параметры для обновления существующей записи памяти с возможностью изменения ключа, типа, значения и метаданных."""

    memory_id: str | None = None
    key: str | None = None
    memory_type: MemoryType | None = None
    value: str
    tags: list[str] | None = None
    importance: int | None = Field(default=None, ge=1, le=5)
    ttl_seconds: int | None = None


class DeleteMemoryInput(BaseModel):
    """Обеспечивает удаление записей из памяти агента по идентификатору, ключу или типу с гарантией корректного удаления."""

    memory_id: str | None = None
    key: str | None = None
    memory_type: MemoryType | None = None


class ListMemoryInput(BaseModel):
    """Определяет параметры фильтрации и ограничения для запроса списка элементов памяти, обеспечивая корректный и контролируемый доступ к данным памяти."""

    memory_type: MemoryType | None = None
    limit: int = Field(default=20, ge=1)


class ClearSessionMemoryInput(BaseModel):
    """Обеспечивает подтверждение очистки сессионной памяти, гарантируя, что операция выполняется только при явном согласии пользователя."""

    confirm: bool = Field(default=False)


def memory_tools(
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    session_id: str,
    default_search_limit: int,
) -> list[StructuredTool]:
    """Группирует операции работы с памятью пользователя, обеспечивая единый интерфейс для сохранения, поиска, обновления и удаления данных с учётом сессии и пользователя."""

    def save_memory(
        memory_type: str = "note",
        key: str = "",
        value: str = "",
        tags: list[str] | None = None,
        importance: int = 3,
        ttl_seconds: int | None = None,
    ) -> str:
        """Гарантирует сохранение записи памяти с валидацией типа и обработкой ошибок, возвращая статус операции в стандартизированном формате."""
        normalized_memory_type, normalized_warning = _normalize_memory_type(memory_type)
        try:
            record = store.save(
                user_id=user_id,
                # Этот tool предназначен для долговременной памяти пользователя:
                # запись должна быть доступна в его следующих сессиях.
                session_id=None,
                memory_type=normalized_memory_type,
                key=key,
                value=value,
                tags=tags or [],
                importance=importance,
                ttl_seconds=ttl_seconds,
                source="user",
            )
            payload = {"status": "saved", "record": record.model_dump(mode="json")}
            if normalized_warning:
                payload["warning"] = normalized_warning
            return _json(payload)
        except Exception as exc:
            return _json({"status": "error", "message": str(exc)})

    def search_memory(
        query: str,
        memory_type: MemoryType | None = None,
        limit: int | None = None,
    ) -> str:
        """Обеспечивает поиск памяти с учётом типа и лимита, реализуя стратегию fallback для повышения релевантности результатов."""
        effective_limit = limit or default_search_limit
        result = store.search(
            user_id=user_id,
            query=query,
            memory_type=memory_type,
            session_id=session_id,
            limit=effective_limit,
        )
        payload = result.model_dump(mode="json")
        payload["requested_memory_type"] = memory_type
        payload["fallback_to_all_types"] = False
        payload["fallback_to_recent"] = False
        if result.count == 0 and memory_type is not None:
            fallback = store.search(
                user_id=user_id,
                query=query,
                memory_type=None,
                session_id=session_id,
                limit=effective_limit,
            )
            payload = fallback.model_dump(mode="json")
            payload["requested_memory_type"] = memory_type
            payload["fallback_to_all_types"] = True
            payload["fallback_to_recent"] = False
            result = fallback
        if result.count == 0:
            recent = store.search(
                user_id=user_id,
                query="",
                memory_type=None,
                session_id=session_id,
                limit=effective_limit,
            )
            payload = recent.model_dump(mode="json")
            payload["query"] = query
            payload["requested_memory_type"] = memory_type
            payload["fallback_to_all_types"] = memory_type is not None
            payload["fallback_to_recent"] = True
        return _json(payload)

    def get_memory(memory_id: str) -> str:
        """Возвращает конкретную запись памяти по идентификатору с гарантией информирования о её наличии или отсутствии."""
        record = store.get(memory_id, user_id=user_id)
        if record is None:
            return _json({"status": "not_found", "memory_id": memory_id})
        return _json({"status": "found", "record": record.model_dump(mode="json")})

    def update_memory(
        value: str,
        memory_id: str | None = None,
        key: str | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        importance: int | None = None,
        ttl_seconds: int | None = None,
    ) -> str:
        """Обеспечивает обновление существующей записи памяти по идентификатору или ключу с обработкой ошибок и возвратом статуса операции."""
        try:
            target_id = memory_id
            if target_id is None and key is not None:
                record = store.find_by_key(
                    user_id=user_id,
                    key=key,
                    memory_type=memory_type,
                    session_id=None,
                )
                target_id = record.id if record else None
            if target_id is None:
                return _json(
                    {"status": "not_found", "key": key, "memory_id": memory_id}
                )
            updated = store.update(
                target_id,
                user_id=user_id,
                value=value,
                tags=tags,
                importance=importance,
                ttl_seconds=ttl_seconds,
            )
            return _json(
                {"status": "updated", "record": updated.model_dump(mode="json")}
            )
        except Exception as exc:
            return _json({"status": "error", "message": str(exc)})

    def delete_memory(
        memory_id: str | None = None,
        key: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> str:
        """Гарантирует удаление записи памяти по идентификатору или ключу с информированием о результате и обязательным указанием параметров удаления."""
        if memory_id:
            deleted = store.delete(memory_id, user_id=user_id)
            return _json(
                {
                    "status": "deleted" if deleted else "not_found",
                    "memory_id": memory_id,
                }
            )
        if key:
            deleted_count = store.delete_by_key(
                user_id=user_id,
                key=key,
                memory_type=memory_type,
                session_id=None,
            )
            return _json(
                {"status": "deleted", "deleted_count": deleted_count, "key": key}
            )
        return _json({"status": "error", "message": "нужно указать memory_id или key"})

    def list_memories(memory_type: MemoryType | None = None, limit: int = 20) -> str:
        """Позволяет получить список пользовательских воспоминаний с фильтрацией по типу и ограничением по количеству, гарантируя сериализацию результата для автоматизации."""
        records = store.list_memories(
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            limit=limit,
        )
        return _json(
            {
                "count": len(records),
                "records": [record.model_dump(mode="json") for record in records],
            }
        )

    def clear_session_memory(confirm: bool = False) -> str:
        """Гарантирует безопасное удаление всех воспоминаний текущей сессии пользователя только после явного подтверждения."""
        if not confirm:
            return _json(
                {
                    "status": "confirmation_required",
                    "message": "Для очистки памяти текущей сессии вызовите tool с confirm=true.",
                }
            )
        deleted_count = store.clear_session(user_id=user_id, session_id=session_id)
        return _json({"status": "cleared", "deleted_count": deleted_count})

    return [
        StructuredTool.from_function(
            name="save_memory",
            description=(
                "Сохраняет долговременную память пользователя. Используй, когда пользователь "
                "просит запомнить факт, предпочтение, задачу, резюме или заметку."
            ),
            func=save_memory,
            args_schema=SaveMemoryInput,
        ),
        StructuredTool.from_function(
            name="search_memory",
            description="Ищет в долговременной памяти факты, предпочтения, задачи, резюме или заметки пользователя.",
            func=search_memory,
            args_schema=SearchMemoryInput,
        ),
        StructuredTool.from_function(
            name="get_memory",
            description="Читает одну запись памяти по id.",
            func=get_memory,
            args_schema=GetMemoryInput,
        ),
        StructuredTool.from_function(
            name="update_memory",
            description="Обновляет запись памяти по id или key.",
            func=update_memory,
            args_schema=UpdateMemoryInput,
        ),
        StructuredTool.from_function(
            name="delete_memory",
            description="Удаляет запись памяти по id или key, когда пользователь просит забыть её.",
            func=delete_memory,
            args_schema=DeleteMemoryInput,
        ),
        StructuredTool.from_function(
            name="list_memories",
            description="Показывает сохранённые записи памяти текущего пользователя.",
            func=list_memories,
            args_schema=ListMemoryInput,
        ),
        StructuredTool.from_function(
            name="clear_session_memory",
            description="Очищает записи памяти текущей сессии. Требует confirm=true.",
            func=clear_session_memory,
            args_schema=ClearSessionMemoryInput,
        ),
    ]


def _json(payload: object) -> str:
    """Формирует корректное JSON-представление данных с гарантией сохранения юникода для взаимодействия с внешними системами."""
    return json.dumps(payload, ensure_ascii=False)


def _normalize_memory_type(
    value: str | MemoryType | None,
) -> tuple[MemoryType, str | None]:
    """Обеспечивает стандартизацию типа памяти, предотвращая ошибки из-за некорректных значений и информируя о замене."""
    if value in VALID_MEMORY_TYPES:
        return value, None  # type: ignore[return-value]
    return "note", f"memory_type={value!r} заменён на note"
