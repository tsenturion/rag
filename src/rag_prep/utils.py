"""Общие воспроизводимые утилиты для RAG-конвейера."""

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
    """Настраивает глобальное логирование с заданным уровнем, обеспечивая единообразие вывода сообщений в системе."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Гарантирует идентификацию содержимого файла по стабильному SHA-256-хешу для контроля целостности и дедупликации артефактов."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_integrity(
    staged: dict[Path, Path],
    outputs: dict[str, Path],
) -> dict[str, dict[str, int | str]]:
    """Вычисляет хэши staged-файлов до атомарной публикации всего набора."""
    integrity: dict[str, dict[str, int | str]] = {}
    for name, target in outputs.items():
        staged_path = staged[target.resolve()]
        integrity[name] = {
            "sha256": file_sha256(staged_path),
            "size_bytes": staged_path.stat().st_size,
        }
    return integrity


def verify_upstream_artifact(
    input_path: Path,
    *,
    manifest_filename: str = "manifest.json",
) -> dict[str, str]:
    """Проверяет вход по манифесту и возвращает связь с upstream-run."""
    resolved_input = input_path.resolve()
    manifest_path = resolved_input.parent / manifest_filename
    if not manifest_path.is_file():
        raise ValueError(
            f"Для входного артефакта отсутствует манифест: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Не удалось прочитать upstream-манифест: {manifest_path}"
        ) from exc

    outputs = manifest.get("outputs") or {}
    integrity = manifest.get("integrity") or {}
    matching_names = [
        name
        for name, configured_path in outputs.items()
        if _portable_basename(str(configured_path)) == resolved_input.name
    ]
    if len(matching_names) != 1:
        raise ValueError(
            f"Upstream-манифест не связывает входной файл однозначно: {resolved_input}"
        )
    output_name = matching_names[0]
    integrity_record = integrity.get(output_name) or {}
    expected_hash = integrity_record.get("sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"В upstream-манифесте нет SHA-256 для outputs.{output_name}")
    actual_hash = file_sha256(resolved_input)
    if actual_hash != expected_hash:
        raise ValueError(
            "Нарушена целостность upstream-артефакта: "
            f"ожидался {expected_hash}, получен {actual_hash}"
        )
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Upstream-манифест не содержит run_id")
    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "input_sha256": actual_hash,
        "output_name": output_name,
    }


def _portable_basename(value: str) -> str:
    """Извлекает имя файла из Windows/POSIX path независимо от текущей ОС."""
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def text_sha256(text: str) -> str:
    """Гарантирует однозначную идентификацию текстовых данных по SHA-256-хешу для отслеживания изменений и кэширования."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, length: int = 24) -> str:
    """Гарантирует воспроизводимый короткий идентификатор на основе набора параметров для устойчивой адресации сущностей в пайплайне."""
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def new_run_id() -> str:
    """Гарантирует уникальный идентификатор запуска для отслеживания и логирования операций в рамках одной сессии."""
    return str(uuid4())


def json_dump(path: Path, payload: Any) -> None:
    """Гарантирует атомарную и человекочитаемую сериализацию данных в JSON-файл с защитой от частичной записи."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(path) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


@contextmanager
def artifact_set_transaction(targets: list[Path]) -> Iterator[dict[Path, Path]]:
    """Гарантирует атомарную публикацию набора файлов-артефактов с откатом при ошибках и логированием неустранимых сбоев."""
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
    """Гарантирует замену целевого набора файлов на подготовленные версии с восстановлением исходного состояния при сбоях."""
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
    """Гарантирует атомарную запись текстовых файлов с автоматическим созданием временного файла и заменой целевого только при успешном завершении."""

    def __init__(self, path: Path):
        """Готовит экземпляр к атомарной записи в указанный путь без захвата файловых ресурсов до входа в контекст."""
        self.path = path
        self.temp_path: Path | None = None
        self.file = None

    def __enter__(self):
        """Открывает управляемый контекст ресурса."""
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
        """Закрывает управляемый контекст ресурса."""
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
    """Гарантирует преобразование вложенного словаря в плоскую структуру с уникальными ключами для удобства сериализации и логирования."""
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=next_key))
        else:
            flattened[next_key] = value
    return flattened
