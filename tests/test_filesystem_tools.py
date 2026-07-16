from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_app.config import FileToolsConfig
from agent_app.tools.filesystem import WorkspaceFileService, filesystem_tools


class FilesystemToolsTest(unittest.TestCase):
    def test_workspace_read_write_and_secret_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = WorkspaceFileService(
                FileToolsConfig(
                    enabled=True,
                    workspace_path=Path(temporary_dir),
                    allow_write=True,
                )
            )
            written = json.loads(
                service.write_file("notes/result.txt", "token=secret", False)
            )
            read = json.loads(service.read_file("notes/result.txt"))
            listing = json.loads(service.list_files("notes"))

        self.assertEqual(written["status"], "ok")
        self.assertEqual(read["content"], "token=<redacted>")
        self.assertEqual(listing["entries"][0]["path"], "notes/result.txt")

    def test_path_traversal_and_forbidden_extension_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            service = WorkspaceFileService(
                FileToolsConfig(
                    enabled=True,
                    workspace_path=Path(temporary_dir),
                )
            )
            with self.assertRaisesRegex(ValueError, "за пределы workspace"):
                service.read_file("../outside.txt")
            with self.assertRaisesRegex(ValueError, "Расширение файла"):
                service.read_file("payload.exe")

    def test_write_tool_is_not_exposed_in_read_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tools = filesystem_tools(
                FileToolsConfig(
                    enabled=True,
                    workspace_path=Path(temporary_dir),
                    allow_write=False,
                )
            )

        self.assertEqual(
            {tool.name for tool in tools},
            {"list_workspace_files", "read_workspace_file"},
        )


if __name__ == "__main__":
    unittest.main()
