"""Регрессионные тесты для подсистемы memory_isolation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_app.memory.store import SQLiteMemoryStore  # noqa: E402
from agent_app.tools.memory_tools import memory_tools  # noqa: E402


class MemoryIsolationTest(unittest.TestCase):
    """Проверяет корректность изоляции памяти между пользователями и сессиями, обеспечивая безопасность и целостность данных."""

    def test_search_falls_back_to_recent_accessible_memory(self) -> None:
        """Проверяет, что поиск памяти возвращает данные из наиболее недавней доступной сессии при отсутствии результатов в текущей."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(Path(temp_dir) / "memory.sqlite")
            store.save(
                user_id="user",
                session_id="session-1",
                key="current_component",
                value="billing-api",
            )
            store.save(
                user_id="user",
                session_id="session-2",
                key="other_component",
                value="private-worker",
            )
            tools = {
                tool.name: tool
                for tool in memory_tools(
                    store,
                    user_id="user",
                    session_id="session-1",
                    default_search_limit=5,
                )
            }

            result = json.loads(
                tools["search_memory"].invoke({"query": "текущий компонент"})
            )

            self.assertTrue(result["fallback_to_recent"])
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["records"][0]["value"], "billing-api")

    def test_memory_id_operations_are_scoped_to_current_user(self) -> None:
        """Проверяет, что операции с памятью по ID ограничены текущим пользователем и не влияют на данные других пользователей."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(Path(temp_dir) / "memory.sqlite")
            alice_record = store.save(
                user_id="alice",
                session_id="alice-session",
                key="private_note",
                value="Личная запись Alice",
            )
            bob_tools = {
                tool.name: tool
                for tool in memory_tools(
                    store,
                    user_id="bob",
                    session_id="bob-session",
                    default_search_limit=5,
                )
            }

            get_result = json.loads(
                bob_tools["get_memory"].invoke({"memory_id": alice_record.id})
            )
            update_result = json.loads(
                bob_tools["update_memory"].invoke(
                    {"memory_id": alice_record.id, "value": "Подмена"}
                )
            )
            delete_result = json.loads(
                bob_tools["delete_memory"].invoke({"memory_id": alice_record.id})
            )

            self.assertEqual(get_result["status"], "not_found")
            self.assertEqual(update_result["status"], "error")
            self.assertEqual(delete_result["status"], "not_found")
            persisted = store.get(alice_record.id, user_id="alice")
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted.value, "Личная запись Alice")

    def test_tool_memory_is_available_in_following_sessions(self) -> None:
        """Проверяет сквозной поиск и изменение долговременной записи из новой сессии."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(Path(temp_dir) / "memory.sqlite")
            first_session = {
                tool.name: tool
                for tool in memory_tools(
                    store,
                    user_id="user",
                    session_id="session-1",
                    default_search_limit=5,
                )
            }
            saved = json.loads(
                first_session["save_memory"].invoke(
                    {
                        "memory_type": "preference",
                        "key": "city",
                        "value": "Екатеринбург",
                    }
                )
            )
            second_session = {
                tool.name: tool
                for tool in memory_tools(
                    store,
                    user_id="user",
                    session_id="session-2",
                    default_search_limit=5,
                )
            }

            found = json.loads(
                second_session["search_memory"].invoke({"query": "Екатеринбург"})
            )
            updated = json.loads(
                second_session["update_memory"].invoke(
                    {"key": "city", "value": "Пермь"}
                )
            )
            deleted = json.loads(
                second_session["delete_memory"].invoke({"key": "city"})
            )

        self.assertIsNone(saved["record"]["session_id"])
        self.assertEqual(found["count"], 1)
        self.assertEqual(updated["record"]["value"], "Пермь")
        self.assertEqual(deleted["deleted_count"], 1)

    def test_global_and_session_records_with_same_key_do_not_overwrite_each_other(
        self,
    ) -> None:
        """Проверяет, что записи с одинаковым ключом, но разными сессиями, хранятся и удаляются независимо, не перезаписывая друг друга."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(Path(temp_dir) / "memory.sqlite")
            global_record = store.save(
                user_id="user",
                session_id=None,
                key="city",
                value="Москва",
            )
            session_record = store.save(
                user_id="user",
                session_id="session-1",
                key="city",
                value="Казань",
            )

            self.assertNotEqual(global_record.id, session_record.id)
            self.assertEqual(
                store.find_by_key(user_id="user", key="city", session_id=None).value,
                "Москва",
            )
            self.assertEqual(
                store.find_by_key(
                    user_id="user",
                    key="city",
                    session_id="session-1",
                ).value,
                "Казань",
            )
            deleted = store.delete_by_key(
                user_id="user",
                key="city",
                session_id="session-1",
            )
            self.assertEqual(deleted, 1)
            self.assertIsNotNone(store.get(global_record.id, user_id="user"))

    def test_list_memories_excludes_other_sessions_but_keeps_global_records(
        self,
    ) -> None:
        """Проверяет, что контекст сессии состоит только из global-памяти и её собственных записей."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(Path(temp_dir) / "memory.sqlite")
            store.save(user_id="user", session_id=None, key="global", value="Общее")
            store.save(user_id="user", session_id="s1", key="one", value="Первая")
            store.save(user_id="user", session_id="s2", key="two", value="Вторая")

            records = store.list_memories(
                user_id="user",
                session_id="s1",
                limit=20,
            )

        self.assertEqual({record.key for record in records}, {"global", "one"})


if __name__ == "__main__":
    unittest.main()
