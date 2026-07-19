"""Безопасные файловые инструменты для инструментов агента."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_app.config import FileToolsConfig
from agent_app.support.security import redact_secrets


class ListWorkspaceInput(BaseModel):
    """Ограничивает запросы к файловой системе каталогом workspace, гарантируя доступ только к разрешённым относительным путям."""

    path: str = Field(
        default=".",
        description="Относительный путь каталога внутри разрешённого workspace.",
    )


class ReadWorkspaceFileInput(BaseModel):
    """Обеспечивает безопасный доступ к файлам внутри workspace с гарантией корректного относительного пути."""

    path: str = Field(description="Относительный путь файла внутри workspace.")


class WriteWorkspaceFileInput(BaseModel):
    """Позволяет записывать текстовые данные в файлы workspace с контролем перезаписи для предотвращения потери данных."""

    path: str = Field(description="Относительный путь файла внутри workspace.")
    content: str = Field(description="Текстовое содержимое файла.")
    overwrite: bool = Field(
        default=False,
        description="Разрешить замену существующего файла.",
    )


class WorkspaceFileService:
    """Управляет безопасным доступом к изолированному workspace с контролем разрешений, путей и ограничений на операции с файлами."""

    def __init__(self, config: FileToolsConfig):
        """Гарантирует готовность к работе с workspace, создавая корневой каталог и сохраняя конфигурацию для последующих операций."""
        self.config = config
        self.root = config.workspace_path.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def list_files(self, path: str = ".") -> str:
        """Возвращает список разрешённых файлов и каталогов в workspace с фильтрацией по расширениям и скрытым файлам, обеспечивая безопасность и ограничение объёма."""
        directory = self._resolve(path, require_extension=False)
        if not directory.exists():
            return self._json({"status": "not_found", "path": path})
        if not directory.is_dir():
            return self._json({"status": "not_directory", "path": path})
        entries: list[dict[str, Any]] = []
        for item in sorted(
            directory.iterdir(), key=lambda value: value.name.casefold()
        ):
            if not self.config.allow_hidden_files and self._is_hidden(item):
                continue
            if item.is_symlink():
                continue
            if item.is_file() and not self._extension_allowed(item):
                continue
            entries.append(
                {
                    "path": item.relative_to(self.root).as_posix(),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
            if len(entries) >= self.config.max_list_entries:
                break
        return self._json(
            {
                "status": "ok",
                "path": directory.relative_to(self.root).as_posix() or ".",
                "entries": entries,
                "truncated": len(entries) >= self.config.max_list_entries,
            }
        )

    def read_file(self, path: str) -> str:
        """Загружает содержимое файла из workspace с проверкой существования, типа, размера и безопасности, возвращая результат в стандартизированном формате."""
        file_path = self._resolve(path, require_extension=True)
        if not file_path.exists():
            return self._json({"status": "not_found", "path": path})
        if not file_path.is_file() or file_path.is_symlink():
            return self._json({"status": "not_file", "path": path})
        size = file_path.stat().st_size
        if size > self.config.max_file_bytes:
            return self._json(
                {
                    "status": "too_large",
                    "path": path,
                    "size": size,
                    "limit": self.config.max_file_bytes,
                }
            )
        content = file_path.read_text(encoding="utf-8")
        return self._json(
            {
                "status": "ok",
                "path": file_path.relative_to(self.root).as_posix(),
                "size": size,
                "content": redact_secrets(content),
            }
        )

    def write_file(self, path: str, content: str, overwrite: bool = False) -> str:
        """Обеспечивает атомарную запись файла в workspace с проверкой прав, размера и предотвращением записи вне корня, гарантируя целостность данных."""
        if not self.config.allow_write:
            return self._json(
                {
                    "status": "forbidden",
                    "message": "Запись в workspace отключена конфигурацией.",
                }
            )
        encoded = content.encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            return self._json(
                {
                    "status": "too_large",
                    "size": len(encoded),
                    "limit": self.config.max_file_bytes,
                }
            )
        file_path = self._resolve(path, require_extension=True)
        if file_path.exists() and (file_path.is_symlink() or not file_path.is_file()):
            return self._json({"status": "not_file", "path": path})
        if file_path.exists() and not overwrite:
            return self._json({"status": "already_exists", "path": path})
        file_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = file_path.parent.resolve()
        if self.root != resolved_parent and self.root not in resolved_parent.parents:
            raise ValueError("Родительский каталог выходит за пределы workspace")
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            dir=file_path.parent,
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary_name).replace(file_path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return self._json(
            {
                "status": "ok",
                "path": file_path.relative_to(self.root).as_posix(),
                "size": len(encoded),
                "overwritten": overwrite,
            }
        )

    def _resolve(self, value: str, *, require_extension: bool) -> Path:
        """Гарантирует, что путь указывает на разрешённый файл внутри workspace без скрытых элементов, симлинков и запрещённых расширений."""
        relative = Path(value.strip() or ".")
        if relative.is_absolute():
            raise ValueError("Разрешены только относительные пути внутри workspace")
        if not self.config.allow_hidden_files and any(
            part.startswith(".") and part not in {".", ".."} for part in relative.parts
        ):
            raise ValueError("Скрытые файлы и каталоги запрещены")
        unresolved = self.root / relative
        current = self.root
        for part in relative.parts:
            if part in {".", ".."}:
                continue
            current /= part
            if current.is_symlink():
                raise ValueError("Символические ссылки внутри workspace запрещены")
        candidate = unresolved.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Путь выходит за пределы workspace")
        if require_extension and not self._extension_allowed(candidate):
            raise ValueError(
                "Расширение файла не разрешено: "
                + (candidate.suffix or "без расширения")
            )
        return candidate

    def _extension_allowed(self, path: Path) -> bool:
        """Гарантирует, что путь соответствует политике разрешённых расширений файлов в workspace."""
        return path.suffix.casefold() in set(self.config.allowed_extensions)

    @staticmethod
    def _is_hidden(path: Path) -> bool:
        """Проверяет, что путь содержит скрытые элементы, чтобы обеспечить соблюдение политики доступа к скрытым файлам."""
        return any(part.startswith(".") for part in path.parts)

    @staticmethod
    def _json(payload: dict[str, Any]) -> str:
        """Гарантирует сериализацию структуры данных в корректный JSON для обмена с инструментами агента."""
        return json.dumps(payload, ensure_ascii=False)


def filesystem_tools(config: FileToolsConfig) -> list[StructuredTool]:
    """Формирует набор инструментов для работы с файловой системой workspace, учитывая конфигурацию разрешений и ограничений."""
    if not config.enabled:
        return []
    service = WorkspaceFileService(config)
    tools = [
        StructuredTool.from_function(
            name="list_workspace_files",
            description=(
                "Показать разрешённые файлы и каталоги внутри изолированного "
                "workspace. Абсолютные пути недопустимы."
            ),
            func=service.list_files,
            args_schema=ListWorkspaceInput,
        ),
        StructuredTool.from_function(
            name="read_workspace_file",
            description=(
                "Прочитать UTF-8 файл внутри изолированного workspace с проверкой "
                "пути, расширения и размера."
            ),
            func=service.read_file,
            args_schema=ReadWorkspaceFileInput,
        ),
    ]
    if config.allow_write:
        tools.append(
            StructuredTool.from_function(
                name="write_workspace_file",
                description=(
                    "Атомарно записать UTF-8 файл внутри workspace. Использовать "
                    "только когда пользователь явно просит изменить файл."
                ),
                func=service.write_file,
                args_schema=WriteWorkspaceFileInput,
            )
        )
    return tools
