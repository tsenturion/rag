from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
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


@contextmanager
def artifact_set_transaction(targets: list[Path]) -> Iterator[dict[Path, Path]]:
    if not targets:
        raise ValueError("Набор артефактов не может быть пустым")
    resolved_targets = [target.resolve() for target in targets]
    if len(resolved_targets) != len(set(resolved_targets)):
        raise ValueError("Пути артефактов в транзакции должны быть уникальными")
    parent = resolved_targets[0].parent
    if any(target.parent != parent for target in resolved_targets):
        raise ValueError(
            "Все артефакты одной транзакции должны находиться в одной директории"
        )

    parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".artifact-set-", dir=parent))
    staged = {
        target: staging_dir / f"{index:03d}-{target.name}"
        for index, target in enumerate(resolved_targets)
    }
    try:
        yield staged
        _commit_artifact_set(staged, staging_dir=staging_dir)
    finally:
        remaining_backups = list(staging_dir.glob(".backup-*"))
        if remaining_backups:
            logging.getLogger(__name__).critical(
                "Не удалось полностью откатить набор артефактов; backups сохранены в %s",
                staging_dir,
            )
        else:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _commit_artifact_set(staged: dict[Path, Path], *, staging_dir: Path) -> None:
    missing = [path for path in staged.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Не все staged-артефакты созданы: {missing}")

    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for index, target in enumerate(staged):
            if target.exists():
                backup = staging_dir / f".backup-{index:03d}-{target.name}"
                os.replace(target, backup)
                backups[target] = backup
        for target, staged_path in staged.items():
            os.replace(staged_path, target)
            installed.append(target)
    except BaseException as original_error:
        rollback_errors: list[OSError] = []
        for target in reversed(installed):
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(exc)
        for target, backup in backups.items():
            if backup.exists():
                try:
                    os.replace(backup, target)
                except OSError as exc:
                    rollback_errors.append(exc)
        if rollback_errors:
            raise RuntimeError(
                "Не удалось полностью восстановить предыдущий набор артефактов"
            ) from original_error
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


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

    def __exit__(self, exc_type, exc, _traceback) -> bool:
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
