from __future__ import annotations

import ast
import hmac
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

ALLOWED_IMPORTS = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
}
BLOCKED_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "vars",
    "__import__",
}


class ExecutionRequest(BaseModel):
    code: str = Field(min_length=1, max_length=100_000)
    timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    max_output_chars: int = Field(default=12_000, ge=100, le=100_000)


class ExecutionResponse(BaseModel):
    status: str
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    duration_ms: float
    truncated: bool = False
    error: str | None = None


app = FastAPI(
    title="Изолированный Python Code Runner",
    version="1.0.0",
    description=(
        "Выполняет ограниченный Python-код в отдельном контейнере. "
        "Сервис не предназначен для публикации во внешнюю сеть."
    ),
)


@app.get("/health", tags=["Состояние"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/execute",
    response_model=ExecutionResponse,
    tags=["Выполнение"],
    summary="Выполнить ограниченный Python-код",
)
def execute(
    payload: ExecutionRequest,
    supplied_key: str | None = Header(default=None, alias="X-Code-Runner-Key"),
) -> ExecutionResponse:
    _require_api_key(supplied_key)
    violation = _validate_code(payload.code)
    if violation is not None:
        return ExecutionResponse(
            status="rejected",
            duration_ms=0.0,
            error=violation,
        )
    started = perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="code-runner-") as directory:
            script = Path(directory) / "main.py"
            stdout_path = Path(directory) / "stdout.txt"
            stderr_path = Path(directory) / "stderr.txt"
            script.write_text(payload.code, encoding="utf-8")
            try:
                with (
                    stdout_path.open("wb") as stdout_stream,
                    stderr_path.open("wb") as stderr_stream,
                ):
                    process = subprocess.run(
                        [sys.executable, "-I", "-S", "-B", str(script)],
                        cwd=directory,
                        env={
                            "PYTHONHASHSEED": "0",
                            "PYTHONIOENCODING": "utf-8",
                            "PATH": os.path.dirname(sys.executable),
                        },
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        stdin=subprocess.DEVNULL,
                        timeout=payload.timeout_seconds,
                        preexec_fn=(
                            _resource_limits(payload) if os.name == "posix" else None
                        ),
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                return _response(
                    status_value="timeout",
                    stdout=_read_output(stdout_path, payload.max_output_chars),
                    stderr=_read_output(stderr_path, payload.max_output_chars),
                    return_code=None,
                    started=started,
                    max_output_chars=payload.max_output_chars,
                    error="Превышен лимит времени выполнения.",
                )
            stdout = _read_output(stdout_path, payload.max_output_chars)
            stderr = _read_output(stderr_path, payload.max_output_chars)
    except Exception as exc:
        return ExecutionResponse(
            status="failed",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            error=str(exc)[:500],
        )
    return _response(
        status_value="completed" if process.returncode == 0 else "failed",
        stdout=stdout,
        stderr=stderr,
        return_code=process.returncode,
        started=started,
        max_output_chars=payload.max_output_chars,
    )


def _require_api_key(supplied: str | None) -> None:
    expected = os.getenv("CODE_RUNNER_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CODE_RUNNER_API_KEY не настроен.",
        )
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Некорректный code runner key.",
        )


def _validate_code(code: str) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"Синтаксическая ошибка: {exc.msg}, строка {exc.lineno}."
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            roots = {name.split(".", 1)[0] for name in names}
            denied = sorted(roots - ALLOWED_IMPORTS)
            if denied:
                return "Запрещён импорт: " + ", ".join(denied)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                return f"Запрещён вызов: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "Доступ к dunder-атрибутам запрещён."
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "Доступ к dunder-именам запрещён."
    return None


def _resource_limits(payload: ExecutionRequest):
    def apply_limits() -> None:
        resource: Any = importlib.import_module("resource")

        cpu_seconds = max(1, int(payload.timeout_seconds) + 1)
        output_bytes = max(4_096, payload.max_output_chars * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
        if hasattr(resource, "RLIMIT_AS"):
            memory_bytes = 192 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    return apply_limits


def _response(
    *,
    status_value: str,
    stdout: str,
    stderr: str,
    return_code: int | None,
    started: float,
    max_output_chars: int,
    error: str | None = None,
) -> ExecutionResponse:
    stdout_value, stdout_truncated = _truncate(stdout, max_output_chars)
    remaining = max(0, max_output_chars - len(stdout_value))
    stderr_value, stderr_truncated = _truncate(stderr, remaining)
    return ExecutionResponse(
        status=status_value,
        stdout=stdout_value,
        stderr=stderr_value,
        return_code=return_code,
        duration_ms=round((perf_counter() - started) * 1000, 3),
        truncated=stdout_truncated or stderr_truncated,
        error=error,
    )


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = "... [вывод сокращён]"
    if limit <= 0:
        return "", True
    if limit <= len(marker):
        return marker[:limit], True
    return value[: limit - len(marker)] + marker, True


def _read_output(path: Path, limit: int) -> str:
    with path.open("rb") as stream:
        value = stream.read(limit + 1).decode("utf-8", errors="replace")
    return value.replace("\r\n", "\n").replace("\r", "\n")
