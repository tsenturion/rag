"""Регрессионные тесты для подсистемы support_agent."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.config import (  # noqa: E402
    AgentConfig,
    AgentAppConfig,
    AgentRagConfig,
    AgentToolsConfig,
    MemoryConfig,
)
from agent_app.graph import AgentRunner  # noqa: E402
from agent_app.models import AgentToolResult  # noqa: E402
from agent_app.rag.models import RagCitation, RagRetrievalResult  # noqa: E402
from agent_app.support.incidents import IncidentStore  # noqa: E402
from agent_app.tools.support import support_tools  # noqa: E402
from rag_prep.config import EmbeddingConfig, VectorStoreConfig  # noqa: E402


def _test_rag_config() -> AgentRagConfig:
    """Проверяет, что создаётся валидная конфигурация поиска знаний для тестирования агентов."""
    return AgentRagConfig(
        enabled=True,
        tokenizer_model="cl100k_base",
        embedding=EmbeddingConfig(
            provider="local",
            model="test-embedding",
            dimensions=3,
            api_key_env="HF_TOKEN",
        ),
        vector_store=VectorStoreConfig(
            collection_name="test-knowledge",
            vector_size=3,
            local_storage_path=Path("unused-qdrant-storage"),
        ),
    )


class CitationModel:
    """Модель проверяет корректность инициирования вызова инструментов и возврата ответов, обеспечивая правильное взаимодействие с инструментами поддержки."""

    supports_tool_calling = True

    def bind_tools(self, _tools):
        """Проверяет, что привязка инструментов не изменяет состояние модели и не вызывает ошибок."""
        return self

    def invoke(self, messages):
        """Проверяет, что модель корректно инициирует вызов инструмента или возвращает ответ в зависимости от входящих сообщений."""
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Проверьте личность пользователя.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "сброс пароля"},
                    "id": "search-call",
                    "type": "tool_call",
                }
            ],
        )


class SerializedToolCallModel:
    """Модель проверяет правильность формирования и восстановления сериализованных вызовов инструментов, гарантируя корректный формат ответов."""

    supports_tool_calling = True

    def bind_tools(self, _tools):
        """Проверяет, что привязка инструментов не изменяет состояние модели и не вызывает ошибок."""
        return self

    def invoke(self, messages):
        """Проверяет, что модель возвращает корректный формат вызова инструмента или финальный ответ в зависимости от сообщений."""
        if any(
            "Сформируй окончательный ответ" in str(message.content)
            for message in messages
        ):
            return AIMessage(content="Проверка личности обязательна.")
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(
                content=(
                    '{"recipient_name":"functions.search_knowledge_base",'
                    '"parameters":{"query":"сброс пароля"}}'
                )
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "сброс пароля"},
                    "id": "search-call",
                    "type": "tool_call",
                }
            ],
        )


class StubRagRuntime:
    """Заглушка имитирует поведение подсистемы поиска с возвратом предсказуемых результатов и корректным управлением жизненным циклом."""

    def retrieve(self, query: str, **_kwargs) -> RagRetrievalResult:
        """Проверяет, что поиск возвращает ожидаемый результат с корректными цитатами и метаданными."""
        return RagRetrievalResult(
            query=query,
            context="[Источник 1] Проверить личность пользователя.",
            citations=[
                RagCitation(
                    reference="[Источник 1]",
                    point_id="point-1",
                    chunk_id="chunk-1",
                    document_id="document-1",
                    source="support.txt",
                    section="Сброс пароля",
                    position=0,
                    score=0.99,
                    excerpt="Проверить личность пользователя.",
                )
            ],
            retrieved_count=1,
            used_count=1,
            context_tokens=10,
            provider="test",
            model="test",
            collection_name="knowledge",
        )

    def close(self) -> None:
        """Проверяет, что вызов закрытия не вызывает побочных эффектов и не нарушает инварианты тестируемого рантайма."""
        return None


class SupportAgentTest(unittest.TestCase):
    """Тесты проверяют интеграцию и корректность работы подсистемы поддержки с инструментами, памятью и рантаймом поиска."""

    def test_support_question_automatically_requires_rag(self) -> None:
        """Проверяет, что для вопросов поддержки автоматически требуется использование подсистемы поиска (RAG) для получения релевантной информации."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                rag=_test_rag_config(),
                tools=AgentToolsConfig(
                    enabled=["search_knowledge_base"],
                    incident_sqlite_path=root / "incidents.sqlite",
                ),
            )
            runner = AgentRunner(
                config,
                user_id="engineer",
                session_id="incident",
                llm=CitationModel(),
                rag_runtime=StubRagRuntime(),
            )

            required = runner._requested_tools(
                "Какие обязательные поля нужны в заявке и что делать, если данных недостаточно?"
            )

        self.assertIn("search_knowledge_base", required)

    def test_required_tool_chain_adds_grounded_summary_to_short_answer(self) -> None:
        """Проверяет, что краткий ответ дополняется фактическими результатами явно заданной цепочки tools."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            runner = AgentRunner(
                AgentAppConfig(
                    agent=AgentConfig(provider="local", model="test-model"),
                    memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                    tools=AgentToolsConfig(
                        enabled=["calculate_travel_budget", "advise_packing"],
                        incident_sqlite_path=root / "incidents.sqlite",
                    ),
                ),
                user_id="engineer",
                session_id="trip",
                llm=CitationModel(),
            )
            answer = runner._append_required_tool_summary(
                (
                    "Обязательно используй calculate_travel_budget и "
                    "advise_packing для поездки."
                ),
                "План подготовлен.",
                [
                    AgentToolResult(
                        name="calculate_travel_budget",
                        content=('{"city":"Казань","total":29000,"currency":"RUB"}'),
                    ),
                    AgentToolResult(
                        name="advise_packing",
                        content='{"items":["ноутбук","паспорт"]}',
                    ),
                ],
            )
            runner.close()

        self.assertIn("Ключевые результаты tools", answer)
        self.assertIn("бюджет 29000 RUB", answer)
        self.assertIn("ноутбук", answer)

    def test_serialized_tool_call_is_repaired_before_response(self) -> None:
        """Проверяет, что сериализованные вызовы инструментов корректно восстанавливаются перед формированием ответа, исключая некорректные поля и обеспечивая валидность ответа."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                rag=_test_rag_config(),
                tools=AgentToolsConfig(
                    enabled=["search_knowledge_base"],
                    incident_sqlite_path=root / "incidents.sqlite",
                ),
            )
            response = AgentRunner(
                config,
                user_id="engineer",
                session_id="incident",
                llm=SerializedToolCallModel(),
                rag_runtime=StubRagRuntime(),
            ).ask("Как выполнить сброс пароля?")

        self.assertIn("Проверка личности", response.answer)
        self.assertNotIn("recipient_name", response.answer)
        self.assertNotIn("functions.search_knowledge_base", response.answer)
        self.assertEqual(response.retrieval.status, "ok")

    def test_agent_response_contains_retrieval_diagnostics_and_citations(self) -> None:
        """Проверяет, что ответ агента содержит корректные диагностические данные по поиску и включает ссылки на источники информации."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = AgentAppConfig(
                agent=AgentConfig(provider="local", model="test-model"),
                memory=MemoryConfig(sqlite_path=root / "memory.sqlite"),
                rag=_test_rag_config(),
                tools=AgentToolsConfig(
                    enabled=["search_knowledge_base"],
                    incident_sqlite_path=root / "incidents.sqlite",
                ),
            )
            response = AgentRunner(
                config,
                user_id="engineer",
                session_id="incident",
                llm=CitationModel(),
                rag_runtime=StubRagRuntime(),
            ).ask("Как выполнить сброс пароля?")

        self.assertEqual(response.tool_calls, ["search_knowledge_base"])
        self.assertEqual(response.retrieval.status, "ok")
        self.assertEqual(response.citations[0].chunk_id, "chunk-1")
        self.assertIn("Источники:", response.answer)
        self.assertIn("[Источник 1]", response.answer)

    def test_incidents_are_user_scoped_and_log_secrets_are_redacted(self) -> None:
        """Проверяет, что инциденты доступны только пользователю-автору и что в логах автоматически скрываются конфиденциальные данные."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            store = IncidentStore(Path(temporary_dir) / "incidents.sqlite")
            alice = {
                tool.name: tool
                for tool in support_tools(
                    rag_runtime=None,
                    incident_store=store,
                    user_id="alice",
                    session_id="session-a",
                    max_log_chars=1000,
                )
            }
            bob = {
                tool.name: tool
                for tool in support_tools(
                    rag_runtime=None,
                    incident_store=store,
                    user_id="bob",
                    session_id="session-b",
                    max_log_chars=1000,
                )
            }
            created = json.loads(
                alice["create_incident"].invoke(
                    {
                        "title": "Ошибка API",
                        "description": "token=private-value timeout",
                    }
                )
            )["incident"]
            bob_result = json.loads(
                bob["get_incident"].invoke({"incident_id": created["id"]})
            )
            analysis = json.loads(
                alice["analyze_log_fragment"].invoke(
                    {"log_text": "password=secret ERROR request timed out"}
                )
            )

        self.assertEqual(bob_result["status"], "not_found")
        self.assertNotIn("private-value", created["description"])
        self.assertNotIn("secret", analysis["redacted_log_preview"])
        self.assertTrue(analysis["secrets_redacted"])
        self.assertEqual(analysis["findings"][0]["code"], "timeout")


if __name__ == "__main__":
    unittest.main()
