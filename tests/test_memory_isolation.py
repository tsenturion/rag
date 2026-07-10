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
    def test_memory_id_operations_are_scoped_to_current_user(self) -> None:
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

    def test_global_and_session_records_with_same_key_do_not_overwrite_each_other(
        self,
    ) -> None:
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


if __name__ == "__main__":
    unittest.main()
