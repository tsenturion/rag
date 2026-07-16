from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from code_runner.app import app


class CodeRunnerTest(unittest.TestCase):
    def test_executes_safe_code_and_rejects_dangerous_import(self) -> None:
        with patch.dict(os.environ, {"CODE_RUNNER_API_KEY": "test-key"}):
            with TestClient(app) as client:
                success = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "test-key"},
                    json={"code": "print(sum(i * i for i in range(5)))"},
                )
                rejected = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "test-key"},
                    json={"code": "import os\nprint(os.getcwd())"},
                )

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["status"], "completed")
        self.assertEqual(success.json()["stdout"], "30\n")
        self.assertEqual(rejected.json()["status"], "rejected")
        self.assertIn("Запрещён импорт", rejected.json()["error"])

    def test_requires_internal_api_key(self) -> None:
        with patch.dict(os.environ, {"CODE_RUNNER_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "wrong"},
                    json={"code": "print(1)"},
                )

        self.assertEqual(response.status_code, 401)

    def test_timeout_stops_infinite_loop(self) -> None:
        with patch.dict(os.environ, {"CODE_RUNNER_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "test-key"},
                    json={
                        "code": "while True:\n    pass",
                        "timeout_seconds": 0.1,
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "timeout")

    def test_large_output_is_bounded(self) -> None:
        with patch.dict(os.environ, {"CODE_RUNNER_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "test-key"},
                    json={
                        "code": "print('x' * 1000000)",
                        "max_output_chars": 500,
                    },
                )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(len(payload["stdout"]) + len(payload["stderr"]), 500)


if __name__ == "__main__":
    unittest.main()
