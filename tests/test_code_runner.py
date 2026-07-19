"""Регрессионные тесты для подсистемы code_runner."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from code_runner.app import app
from code_runner.sandbox import SafeModule, _inplacevar, _safe_import, execute, main


class CodeRunnerTest(unittest.TestCase):
    """Проверяет безопасность и корректность выполнения кода в изолированной среде, включая авторизацию и ограничение ресурсов."""

    def test_executes_safe_code_and_rejects_dangerous_import(self) -> None:
        """Проверяет, что безопасный код выполняется успешно, а попытки импортировать запрещённые модули отклоняются с соответствующей ошибкой."""
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
        """Проверяет, что доступ к выполнению кода требует корректного внутреннего API-ключа, обеспечивая безопасность вызовов."""
        with patch.dict(os.environ, {"CODE_RUNNER_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "wrong"},
                    json={"code": "print(1)"},
                )

        self.assertEqual(response.status_code, 401)

    def test_rejects_private_attribute_escape_from_allowed_module(self) -> None:
        """Проверяет запрет обхода import-политики через внутренние объекты разрешённого модуля."""
        code = 'import re\nprint(re._compiler.sys.modules["os"].getcwd())'
        with patch.dict(os.environ, {"CODE_RUNNER_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "test-key"},
                    json={"code": code},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "rejected")
        self.assertIn("приватным атрибутам", response.json()["error"])

    def test_safe_module_proxy_blocks_public_transitive_module_reference(self) -> None:
        """Проверяет обход через statistics.sys, который AST сам по себе не видит."""
        code = 'import statistics\nprint(statistics.sys.modules["os"].getcwd())'
        with patch.dict(os.environ, {"CODE_RUNNER_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/v1/execute",
                    headers={"X-Code-Runner-Key": "test-key"},
                    json={"code": code},
                )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["stdout"], "")

    def test_timeout_stops_infinite_loop(self) -> None:
        """Проверяет, что выполнение кода прерывается по таймауту, предотвращая бесконечные циклы и зависания."""
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
        """Проверяет, что вывод выполнения кода ограничивается заданным максимальным размером, предотвращая переполнение буфера."""
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


def test_restricted_runtime_exposes_only_explicit_module_api() -> None:
    """Проверяет арифметику, импорт proxy-модуля и запрет транзитивных атрибутов."""
    output = execute("import math\nvalue = 3\nvalue += 6\nprint(math.sqrt(value))\n")

    assert output == "3.0\n"
    assert _safe_import("statistics").mean([2, 4]) == 3
    with pytest.raises(AttributeError, match="не разрешён"):
        SafeModule("demo", {"value": 1}).missing
    with pytest.raises(ImportError, match="не разрешён"):
        _safe_import("os")
    with pytest.raises(ImportError, match="не разрешён"):
        _safe_import("math", level=1)
    with pytest.raises(TypeError, match="не разрешена"):
        _inplacevar("&=", 1, 1)


def test_restricted_runtime_main_reports_usage_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Проверяет коды процесса sandbox без запуска отдельного Python interpreter."""
    monkeypatch.setattr(sys, "argv", ["sandbox"])
    assert main() == 2
    assert "Ожидался путь" in capsys.readouterr().err

    source = tmp_path / "safe.py"
    source.write_text("print(2 + 3)", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["sandbox", str(source)])
    assert main() == 0
    assert capsys.readouterr().out == "5\n"

    source.write_text("import os", encoding="utf-8")
    assert main() == 1
    assert "ImportError" in capsys.readouterr().err


if __name__ == "__main__":
    unittest.main()
