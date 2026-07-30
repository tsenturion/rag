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

import portalocker


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
    recover_artifact_transactions(resolved_input.parent)
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
    """Публикует набор файлов с журналом восстановления после аварии процесса."""
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
    lock_path = parent / ".artifact-publish.lock"
    with portalocker.Lock(str(lock_path), timeout=30):
        _recover_artifact_transactions_unlocked(parent)
        staging_dir = Path(tempfile.mkdtemp(prefix=".artifact-set-", dir=parent))
        staged = {
            target: staging_dir / f"{index:03d}-{target.name}"
            for index, target in enumerate(resolved_targets)
        }
        try:
            yield staged
            _commit_artifact_set(staged, staging_dir=staging_dir)
        finally:
            if staging_dir.exists():
                if not (staging_dir / "journal.json").exists():
                    shutil.rmtree(staging_dir)
                else:
                    logging.getLogger(__name__).critical(
                        "Незавершённая публикация сохранена для recovery: %s",
                        staging_dir,
                    )


def recover_artifact_transactions(parent: Path) -> None:
    """Восстанавливает целостный набор после прерванной публикации файлов."""
    resolved_parent = parent.resolve()
    if not resolved_parent.exists():
        return
    # Downstream-контейнеры монтируют готовые входные артефакты read-only. Если
    # журналов публикации нет, создавать lock-файл не требуется: целостность
    # входа ниже всё равно подтверждается SHA-256 из manifest. При наличии
    # staging-каталога восстановление остаётся обязательным и требует writable
    # mount, чтобы незавершённый набор нельзя было принять за согласованный.
    if not any(resolved_parent.glob(".artifact-set-*")):
        return
    lock_path = resolved_parent / ".artifact-publish.lock"
    with portalocker.Lock(str(lock_path), timeout=30):
        _recover_artifact_transactions_unlocked(resolved_parent)


def _recover_artifact_transactions_unlocked(parent: Path) -> None:
    """Обрабатывает журналы транзакций под уже захваченным directory lock."""
    for staging_dir in sorted(parent.glob(".artifact-set-*")):
        if not staging_dir.is_dir():
            continue
        journal_path = staging_dir / "journal.json"
        if not journal_path.exists():
            # До появления durable journal целевые файлы ещё не изменяются.
            shutil.rmtree(staging_dir)
            continue
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Повреждён журнал публикации артефактов: {journal_path}"
            ) from exc
        state = journal.get("state")
        if state == "prepared":
            _restore_artifact_set(journal, staging_dir=staging_dir)
        elif state != "committed":
            raise RuntimeError(f"Неизвестное состояние журнала артефактов: {state!r}")
        shutil.rmtree(staging_dir)
    _fsync_directory(parent)


def _commit_artifact_set(staged: dict[Path, Path], *, staging_dir: Path) -> None:
    """Фиксирует набор через write-ahead journal и durable filesystem barriers."""
    missing = [path for path in staged.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Не все staged-артефакты созданы: {missing}")

    entries: list[dict[str, str | bool]] = []
    for index, target in enumerate(staged):
        staged_path = staged[target]
        _fsync_file(staged_path)
        backup = staging_dir / f".backup-{index:03d}-{target.name}"
        existed = target.exists()
        if existed:
            shutil.copy2(target, backup)
            _fsync_file(backup)
        entries.append(
            {
                "target": str(target),
                "backup": str(backup),
                "existed": existed,
            }
        )

    journal_path = staging_dir / "journal.json"
    journal = {"version": 1, "state": "prepared", "entries": entries}
    _write_durable_json(journal_path, journal)
    try:
        for target, staged_path in staged.items():
            os.replace(staged_path, target)
        _fsync_directory(staging_dir.parent)
        journal["state"] = "committed"
        _write_durable_json(journal_path, journal)
    except BaseException:
        try:
            _restore_artifact_set(journal, staging_dir=staging_dir)
            shutil.rmtree(staging_dir)
        except OSError as rollback_error:
            raise RuntimeError(
                "Не удалось полностью восстановить предыдущий набор артефактов"
            ) from rollback_error
        raise
    shutil.rmtree(staging_dir)
    _fsync_directory(staging_dir.parent)


def _restore_artifact_set(
    journal: dict[str, Any],
    *,
    staging_dir: Path,
) -> None:
    """Восстанавливает все старые цели либо удаляет ранее отсутствовавшие."""
    for index, entry in enumerate(journal.get("entries", [])):
        target = Path(str(entry["target"]))
        backup = Path(str(entry["backup"]))
        if bool(entry["existed"]):
            if not backup.is_file():
                raise OSError(f"Отсутствует backup артефакта: {backup}")
            restore_temp = staging_dir / f".restore-{index:03d}-{target.name}"
            shutil.copy2(backup, restore_temp)
            _fsync_file(restore_temp)
            os.replace(restore_temp, target)
        else:
            target.unlink(missing_ok=True)
    _fsync_directory(staging_dir.parent)


def _write_durable_json(path: Path, payload: dict[str, Any]) -> None:
    """Атомарно записывает JSON и синхронизирует файл с каталогом."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    """Сбрасывает содержимое файла из page cache на диск."""
    # Windows отклоняет os.fsync для read-only CRT descriptor, поэтому файл
    # открывается на добавление без фактической записи.
    with path.open("ab") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Синхронизирует metadata каталога там, где ОС это поддерживает."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
