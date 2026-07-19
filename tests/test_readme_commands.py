"""Проверки исполняемых PowerShell-примеров из README."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
POWERSHELL_BLOCK = re.compile(r"```powershell\n(.*?)\n```", re.DOTALL)
PLACEHOLDER = re.compile(r"<[^>]+>")
DOCUMENTED_PATH = re.compile(
    r"(?<![\w./-])((?:config|scripts|tests)/[\w./-]+\.(?:yaml|py|ps1))"
)


def _readme() -> str:
    """Читает единственный источник пользовательских команд без изменения разметки."""
    return README_PATH.read_text(encoding="utf-8")


def _concrete_powershell_blocks() -> list[str]:
    """Отбирает команды без учебных placeholders, которые обещаны как готовые к запуску."""
    return [
        block
        for block in POWERSHELL_BLOCK.findall(_readme())
        if not PLACEHOLDER.search(block)
    ]


def _logical_commands(block: str) -> list[str]:
    """Склеивает строки PowerShell по backtick для проверки аргументов одной команды."""
    commands: list[str] = []
    current: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current.append(line.removesuffix("`").rstrip())
        if not line.endswith("`"):
            commands.append(" ".join(current))
            current = []
    if current:
        commands.append(" ".join(current))
    return commands


def test_concrete_agent_commands_select_config() -> None:
    """Не допускает запуск `rag-agent` без обязательного provider-конфига."""
    commands = [
        command
        for block in _concrete_powershell_blocks()
        for command in _logical_commands(block)
        if command.startswith("rag-agent ")
    ]

    assert commands
    assert all("--config " in command for command in commands), commands


def test_concrete_commands_reference_existing_project_files() -> None:
    """Находит опечатки в путях конфигов, скриптов и тестов готовых примеров."""
    missing = {
        relative_path
        for block in _concrete_powershell_blocks()
        for relative_path in DOCUMENTED_PATH.findall(block)
        if not (PROJECT_ROOT / relative_path).is_file()
    }

    assert not missing, sorted(missing)


def test_powershell_blocks_do_not_contain_dotenv_assignments() -> None:
    """Отделяет синтаксис `.env` от исполняемого синтаксиса PowerShell."""
    invalid_lines = [
        line
        for block in _concrete_powershell_blocks()
        for line in block.splitlines()
        if re.fullmatch(r"[A-Z][A-Z0-9_]*=.*", line.strip())
    ]

    assert not invalid_lines, invalid_lines


def test_readme_uses_full_pytest_suite_for_regression() -> None:
    """Не выдаёт частичный unittest discovery за полную регрессионную проверку."""
    readme = _readme()
    assert "python -m unittest discover" not in readme
    assert "python -m pytest -q" in readme


def test_certificate_download_keeps_tls_verification() -> None:
    """Не разрешает отключать TLS при загрузке доверенного корневого сертификата."""
    readme = _readme()
    assert "curl.exe -k" not in readme
    assert "curl.exe --fail --location" in readme


def test_concrete_powershell_blocks_parse_when_pwsh_is_available() -> None:
    """Передаёт готовые блоки штатному parser PowerShell без выполнения команд."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell не установлен в текущем CI-окружении")

    blocks = _concrete_powershell_blocks()
    combined = "\n\n".join(
        f"# README PowerShell block {index}\n{block}"
        for index, block in enumerate(blocks, start=1)
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".ps1",
        delete=False,
    ) as script:
        script.write(combined)
        script_path = Path(script.name)
    parser_command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script_path.as_posix()}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object Message; exit 1 }"
    )
    try:
        result = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", parser_command],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
    finally:
        script_path.unlink(missing_ok=True)

    assert result.returncode == 0, result.stdout + result.stderr
