"""Подключение внешних MCP-серверов для инструментов агента."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, cast

import httpx2
from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool

from agent_app.config import ExternalMCPServerConfig
from agent_app.support.security import redact_secrets

LOGGER = logging.getLogger(__name__)
TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


class ExternalMCPToolManager:
    """Поддерживает постоянные MCP-сессии и адаптирует удалённые tools для агента."""

    def __init__(self, servers: list[ExternalMCPServerConfig]):
        """Готовит менеджер к запуску, фильтруя активные серверы и инициализируя внутренние структуры для управления сессиями и инструментами."""
        self.servers = [server for server in servers if server.enabled]
        self._server_by_name = {server.name: server for server in self.servers}
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[BaseTool] = []
        self._errors: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._started = False

    @property
    def tools(self) -> list[BaseTool]:
        """Гарантирует получение актуального списка инструментов, доступных через внешние MCP-серверы."""
        return list(self._tools)

    @property
    def errors(self) -> dict[str, str]:
        """Гарантирует доступ к актуальному состоянию ошибок подключения к MCP-серверам."""
        return dict(self._errors)

    def start(self) -> list[BaseTool]:
        """Гарантирует запуск фонового клиента MCP, готовность инструментов и обработку ошибок и таймаутов старта."""
        if self._started:
            return self.tools
        self._started = True
        if not self.servers:
            return []

        self._thread = threading.Thread(
            target=self._run,
            name="external-mcp-client",
            daemon=True,
        )
        self._thread.start()
        startup_timeout = sum(server.timeout_seconds for server in self.servers) + 5
        if not self._ready.wait(timeout=startup_timeout):
            self.close()
            raise TimeoutError("Истекло время подключения к внешним MCP-серверам")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise RuntimeError(
                f"Не удалось подключить обязательный MCP-сервер: {error}"
            ) from error
        return self.tools

    def status(self) -> dict[str, object]:
        """Гарантирует получение полной информации о конфигурации, подключениях, доступных инструментах и ошибках MCP."""
        connected = sorted(self._sessions)
        return {
            "configured": len(self.servers),
            "connected": connected,
            "tools": [tool.name for tool in self._tools],
            "errors": self.errors,
        }

    def close(self) -> None:
        """Гарантирует корректное завершение фоновых потоков, освобождение ресурсов и сброс состояния менеджера MCP."""
        thread = self._thread
        loop = self._loop
        shutdown_event = self._shutdown_event
        if thread is None:
            self._started = False
            return
        if loop is not None and shutdown_event is not None and loop.is_running():
            loop.call_soon_threadsafe(shutdown_event.set)
        thread.join(timeout=10)
        if thread.is_alive():
            LOGGER.error("Не удалось штатно остановить MCP client event loop")
        self._thread = None
        self._loop = None
        self._shutdown_event = None
        self._sessions.clear()
        self._started = False

    def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Обеспечивает синхронный вызов внешнего MCP-инструмента с гарантией завершения в пределах таймаута, обеспечивая корректное взаимодействие с асинхронным циклом."""
        loop = self._loop
        server = self._server_by_name[server_name]
        if loop is None or not loop.is_running():
            raise RuntimeError("MCP client manager не запущен")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(server_name, tool_name, arguments),
            loop,
        )
        return future.result(timeout=server.timeout_seconds + 1)

    def _run(self) -> None:
        """Запускает отдельный асинхронный цикл для управления жизненным циклом MCP-соединений, гарантируя корректное создание и завершение ресурсов."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._lifecycle())
        except BaseException as exc:
            self._startup_error = self._startup_error or exc
            self._ready.set()
            LOGGER.exception("Ошибка внешнего MCP client runtime")
        finally:
            loop.close()

    async def _lifecycle(self) -> None:
        """Организует подключение к MCP-серверам и проверку доступности инструментов, устанавливая состояние готовности или ошибку запуска."""
        self._shutdown_event = asyncio.Event()
        async with AsyncExitStack() as stack:
            try:
                for server in self.servers:
                    await self._connect_server(stack, server)
                self._validate_tool_names()
            except BaseException as exc:
                self._startup_error = exc
                self._ready.set()
                return
            self._ready.set()
            await self._shutdown_event.wait()

    async def _connect_server(
        self,
        stack: AsyncExitStack,
        server: ExternalMCPServerConfig,
    ) -> None:
        """Устанавливает и поддерживает сессию с MCP-сервером, проверяя обязательность и доступность инструментов, обеспечивая корректное управление ошибками и ресурсами."""
        server_stack = AsyncExitStack()
        await server_stack.__aenter__()
        try:
            read_stream, write_stream = await self._open_transport(
                server_stack,
                server,
            )
            session = await server_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=server.timeout_seconds),
                )
            )
            await session.initialize()
            remote_tools = await self._list_tools(session)
            if server.tool_allowlist != ["*"]:
                available = {tool.name for tool in remote_tools}
                missing = sorted(set(server.tool_allowlist) - available)
                if missing:
                    raise ValueError(
                        f"MCP-сервер {server.name} не опубликовал tools: "
                        + ", ".join(missing)
                    )
        except BaseException as exc:
            await server_stack.aclose()
            message = redact_secrets(str(exc))[:500]
            self._errors[server.name] = message
            if server.required:
                raise RuntimeError(f"{server.name}: {message}") from exc
            LOGGER.warning(
                "Не удалось подключить необязательный MCP-сервер %s: %s",
                server.name,
                message,
            )
            return

        stack.push_async_callback(server_stack.aclose)
        self._sessions[server.name] = session
        self._tools.extend(
            self._adapt_tool(server, tool)
            for tool in remote_tools
            if server.tool_allowlist == ["*"] or tool.name in server.tool_allowlist
        )
        LOGGER.info(
            "Подключён внешний MCP-сервер %s; доступно tools: %d",
            server.name,
            len(remote_tools),
        )

    async def _open_transport(
        self,
        stack: AsyncExitStack,
        server: ExternalMCPServerConfig,
    ) -> tuple[Any, Any]:
        """Создаёт транспортный канал к MCP-серверу с проверкой необходимых переменных окружения и конфигураций, гарантируя готовность к обмену данными."""
        if server.transport == "stdio":
            environment = dict(server.env)
            for variable in server.env_from_host:
                value = os.getenv(variable)
                if value is None:
                    raise ValueError(
                        f"Не задана переменная окружения {variable} для {server.name}"
                    )
                environment[variable] = value
            transport = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=server.command or "",
                        args=server.args,
                        env=environment,
                        cwd=str(server.cwd) if server.cwd is not None else None,
                    )
                )
            )
            return transport

        headers = dict(server.headers)
        for header, variable in server.header_env.items():
            value = os.getenv(variable)
            if value is None:
                raise ValueError(
                    f"Не задана переменная окружения {variable} для {server.name}"
                )
            headers[header] = value
        http_client = await stack.enter_async_context(
            httpx2.AsyncClient(
                headers=headers,
                timeout=server.timeout_seconds,
                verify=server.verify_ssl,
                follow_redirects=True,
            )
        )
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamable_http_client(
                server.url or "",
                http_client=cast(Any, http_client),
                terminate_on_close=server.terminate_on_close,
            )
        )
        return read_stream, write_stream

    @staticmethod
    async def _list_tools(session: ClientSession) -> list[Tool]:
        """Получает полный список инструментов с MCP-сессии, обеспечивая полноту и актуальность данных для дальнейшего использования."""
        tools: list[Tool] = []
        cursor: str | None = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if cursor is None:
                return tools

    def _adapt_tool(
        self,
        server: ExternalMCPServerConfig,
        tool: Tool,
    ) -> BaseTool:
        """Преобразует удалённый MCP-инструмент в локальный вызов с синхронным и асинхронным интерфейсом, сохраняя метаданные и гарантируя корректность вызова."""
        exposed_name = self._exposed_name(server, tool.name)

        def invoke_external(**arguments: Any) -> str:
            """Обеспечивает синхронный вызов внешнего инструмента агента через менеджер, гарантируя корректную маршрутизацию и обработку аргументов."""
            return self.call_tool(
                server_name=server.name,
                tool_name=tool.name,
                arguments=arguments,
            )

        async def ainvoke_external(**arguments: Any) -> str:
            """Обеспечивает асинхронный вызов внешнего инструмента агента, позволяя не блокировать основной поток при выполнении операции."""
            return await asyncio.to_thread(invoke_external, **arguments)

        description = tool.description or f"Внешний MCP tool {tool.name}"
        return StructuredTool.from_function(
            func=invoke_external,
            coroutine=ainvoke_external,
            name=exposed_name,
            description=f"[{server.name}] {description}"[:1000],
            args_schema=tool.inputSchema,
            infer_schema=False,
            metadata={
                "mcp_server": server.name,
                "mcp_tool": tool.name,
                "mcp_transport": server.transport,
            },
        )

    async def _call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Выполняет асинхронный вызов MCP-инструмента с обработкой ошибок и ограничением размера ответа, обеспечивая надёжность и безопасность данных."""
        server = self._server_by_name[server_name]
        session = self._sessions.get(server_name)
        if session is None:
            raise RuntimeError(f"MCP-сервер не подключён: {server_name}")
        result = await session.call_tool(
            tool_name,
            arguments=arguments,
            read_timeout_seconds=timedelta(seconds=server.timeout_seconds),
        )
        payload = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        text = redact_secrets(json.dumps(payload, ensure_ascii=False))
        if len(text) > server.max_output_chars:
            text = text[: server.max_output_chars] + "... [ответ MCP сокращён]"
        if result.isError:
            raise RuntimeError(text)
        return text

    @staticmethod
    def _exposed_name(server: ExternalMCPServerConfig, tool_name: str) -> str:
        """Формирует уникальное и валидное имя для MCP-инструмента с учётом префиксов, предотвращая коллизии и ошибки именования."""
        prefix = server.tool_prefix or f"mcp_{server.name}"
        normalized = TOOL_NAME_RE.sub("_", f"{prefix}_{tool_name}").strip("_")
        if not normalized:
            raise ValueError(
                f"Не удалось сформировать имя для MCP tool {server.name}/{tool_name}"
            )
        return normalized[:64]

    def _validate_tool_names(self) -> None:
        """Проверяет уникальность имён всех MCP-инструментов после применения префиксов, предотвращая конфликты в системе."""
        names = [tool.name for tool in self._tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "После добавления префиксов совпали имена MCP tools: "
                + ", ".join(duplicates)
            )
