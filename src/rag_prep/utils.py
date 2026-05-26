from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, length: int = 24) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def new_run_id() -> str:
    return str(uuid4())


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(path) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


class atomic_text_writer:
    def __init__(self, path: Path):
        self.path = path
        self.temp_path: Path | None = None
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            delete=False,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        self.temp_path = Path(handle.name)
        self.file = handle
        return handle

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.file is not None:
            self.file.close()
        if self.temp_path is None:
            return False
        if exc_type is None:
            os.replace(self.temp_path, self.path)
        else:
            self.temp_path.unlink(missing_ok=True)
        return False


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=next_key))
        else:
            flattened[next_key] = value
    return flattened
