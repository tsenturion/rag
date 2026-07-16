from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agent_app.tools.support import (
    analyze_log_fragment_payload,
    diagnostic_checklist_payload,
)


def build_mcp_server(*, max_log_chars: int = 12000) -> FastMCP:
    server = FastMCP(
        "Инженерные tools",
        instructions=(
            "Безопасные детерминированные инструменты диагностики. Сервер не "
            "предоставляет shell-доступ и не хранит пользовательские секреты."
        ),
        json_response=True,
        stateless_http=True,
        streamable_http_path="/",
    )

    @server.tool()
    def analyze_log_fragment(
        log_text: str,
        component: str | None = None,
    ) -> dict[str, object]:
        """Анализирует лог, маскирует секреты и возвращает проверяемые находки."""
        return analyze_log_fragment_payload(
            log_text,
            component=component,
            max_log_chars=max_log_chars,
        )

    @server.tool()
    def build_diagnostic_checklist(
        component: str,
        symptoms: str,
    ) -> dict[str, object]:
        """Формирует воспроизводимый чек-лист диагностики."""
        return diagnostic_checklist_payload(component, symptoms)

    @server.resource("agent://roles")
    def agent_roles() -> str:
        """Возвращает краткое описание ролей мультиагентной системы."""
        return (
            "knowledge_agent: RAG и citations; diagnostics_agent: логи и чек-лист; "
            "incident_agent: память и инциденты; coordinator: делегирование и ответ."
        )

    return server
