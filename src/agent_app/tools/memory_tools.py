from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.memory.store import SQLiteMemoryStore
from agent_app.models import MemoryType


class SaveMemoryInput(BaseModel):
    memory_type: MemoryType = Field(default="fact")
    key: str = Field(description="Stable short key, for example project_name")
    value: str = Field(description="Memory value to store")
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    ttl_seconds: int | None = None


class SearchMemoryInput(BaseModel):
    query: str
    memory_type: MemoryType | None = None
    limit: int | None = Field(default=None, ge=1)


class GetMemoryInput(BaseModel):
    memory_id: str


class UpdateMemoryInput(BaseModel):
    memory_id: str | None = None
    key: str | None = None
    memory_type: MemoryType | None = None
    value: str
    tags: list[str] | None = None
    importance: int | None = Field(default=None, ge=1, le=5)
    ttl_seconds: int | None = None


class DeleteMemoryInput(BaseModel):
    memory_id: str | None = None
    key: str | None = None
    memory_type: MemoryType | None = None


class ListMemoryInput(BaseModel):
    memory_type: MemoryType | None = None
    limit: int = Field(default=20, ge=1)


class ClearSessionMemoryInput(BaseModel):
    confirm: bool = Field(default=False)


def memory_tools(
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    session_id: str,
    default_search_limit: int,
) -> list[StructuredTool]:
    def save_memory(
        memory_type: MemoryType = "fact",
        key: str = "",
        value: str = "",
        tags: list[str] | None = None,
        importance: int = 3,
        ttl_seconds: int | None = None,
    ) -> str:
        try:
            record = store.save(
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type,
                key=key,
                value=value,
                tags=tags or [],
                importance=importance,
                ttl_seconds=ttl_seconds,
                source="user",
            )
            return _json({"status": "saved", "record": record.model_dump(mode="json")})
        except Exception as exc:
            return _json({"status": "error", "message": str(exc)})

    def search_memory(
        query: str,
        memory_type: MemoryType | None = None,
        limit: int | None = None,
    ) -> str:
        result = store.search(
            user_id=user_id,
            query=query,
            memory_type=memory_type,
            limit=limit or default_search_limit,
        )
        return _json(result.model_dump(mode="json"))

    def get_memory(memory_id: str) -> str:
        record = store.get(memory_id)
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
        try:
            target_id = memory_id
            if target_id is None and key is not None:
                record = store.find_by_key(
                    user_id=user_id,
                    key=key,
                    memory_type=memory_type,
                    session_id=session_id,
                )
                target_id = record.id if record else None
            if target_id is None:
                return _json({"status": "not_found", "key": key, "memory_id": memory_id})
            updated = store.update(
                target_id,
                value=value,
                tags=tags,
                importance=importance,
                ttl_seconds=ttl_seconds,
            )
            return _json({"status": "updated", "record": updated.model_dump(mode="json")})
        except Exception as exc:
            return _json({"status": "error", "message": str(exc)})

    def delete_memory(
        memory_id: str | None = None,
        key: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> str:
        if memory_id:
            deleted = store.delete(memory_id)
            return _json({"status": "deleted" if deleted else "not_found", "memory_id": memory_id})
        if key:
            deleted_count = store.delete_by_key(
                user_id=user_id,
                key=key,
                memory_type=memory_type,
            )
            return _json({"status": "deleted", "deleted_count": deleted_count, "key": key})
        return _json({"status": "error", "message": "memory_id or key is required"})

    def list_memories(memory_type: MemoryType | None = None, limit: int = 20) -> str:
        records = store.list_memories(
            user_id=user_id,
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
        if not confirm:
            return _json(
                {
                    "status": "confirmation_required",
                    "message": "Call with confirm=true to clear current session memory records.",
                }
            )
        deleted_count = store.clear_session(user_id=user_id, session_id=session_id)
        return _json({"status": "cleared", "deleted_count": deleted_count})

    return [
        StructuredTool.from_function(
            name="save_memory",
            description=(
                "Save durable user memory. Use this when the user asks to remember "
                "a fact, preference, task, summary, or note."
            ),
            func=save_memory,
            args_schema=SaveMemoryInput,
        ),
        StructuredTool.from_function(
            name="search_memory",
            description="Search durable memory for user facts, preferences, tasks, summaries, or notes.",
            func=search_memory,
            args_schema=SearchMemoryInput,
        ),
        StructuredTool.from_function(
            name="get_memory",
            description="Read one memory record by id.",
            func=get_memory,
            args_schema=GetMemoryInput,
        ),
        StructuredTool.from_function(
            name="update_memory",
            description="Update a memory record by id or by key.",
            func=update_memory,
            args_schema=UpdateMemoryInput,
        ),
        StructuredTool.from_function(
            name="delete_memory",
            description="Delete a memory record by id or by key when the user asks to forget it.",
            func=delete_memory,
            args_schema=DeleteMemoryInput,
        ),
        StructuredTool.from_function(
            name="list_memories",
            description="List stored memories for the current user.",
            func=list_memories,
            args_schema=ListMemoryInput,
        ),
        StructuredTool.from_function(
            name="clear_session_memory",
            description="Clear memory records scoped to the current session. Requires confirm=true.",
            func=clear_session_memory,
            args_schema=ClearSessionMemoryInput,
        ),
    ]


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)
