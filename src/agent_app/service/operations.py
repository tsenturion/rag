"""Разрешённые операции CLI, изолированные входы и выдача артефактов по ID."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import time
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import yaml
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from agent_app.service.operation_store import (
    OperationRecord,
    OperationRequest,
    OperationStore,
)
from agent_app.support.security import redact_local_paths, redact_secrets

OperationKind = Literal[
    "prepare",
    "chunk",
    "embed",
    "index",
    "evaluation",
    "scenarios",
    "tuning-inspect",
    "tuning-validate",
    "tuning-baseline",
    "tuning-train",
    "tuning-evaluate",
    "tuning-compare",
]


class OperationProfile(BaseModel):
    """Каталог задаёт доверенные конфиги; произвольные команды здесь отсутствуют."""

    model_config = ConfigDict(extra="forbid")
    title: str
    kind: OperationKind
    config: Path
    suite: Path | None = None
    parameters: list[str] = Field(default_factory=list)


class PublicProfile(BaseModel):
    """Браузер видит возможности и редактируемые поля, но не пути сервера."""

    id: str
    title: str
    kind: OperationKind
    parameters: list[str]
    parameter_schema: dict[str, dict] = Field(default_factory=dict)
    worker_ready: bool


class SourceRecord(BaseModel):
    """Источник хранится отдельно от результатов преобразования и имеет SHA-256."""

    id: str
    name: str
    sha256: str
    size_bytes: int
    created_at: float


class ArtifactRecord(BaseModel):
    """Непрозрачный идентификатор заменяет доступ по пользовательскому пути."""

    id: str
    name: str
    size_bytes: int


# Из браузера изменяются только ограниченные числовые/булевы настройки.
EDITABLE_PARAMETERS = {
    "chunking.chunk_size",
    "chunking.chunk_overlap",
    "chunking.min_chunk_tokens",
    "chunking.max_chunk_tokens",
    "embedding.batch_size",
    "embedding.normalize",
    "embedding.max_batch_tokens",
    "training.learning_rate",
    "training.num_train_epochs",
    "training.per_device_train_batch_size",
    "training.max_steps",
    "peft.r",
    "peft.lora_alpha",
    "peft.lora_dropout",
    "evaluation.repeats",
}


class OperationService:
    """Переиспользует реальные конфиги и CLI, сохраняя границы пользователя и запуска."""

    def __init__(self, store: OperationStore):
        """Корень каталога конфигураций определяется относительно файла регистрации."""
        self.store, self.config = store, store.config
        self.root = self.config.data_dir.resolve()

    def catalog(self) -> dict[str, OperationProfile]:
        """Валидирует серверный YAML и разрешает его относительные пути."""
        path = self.config.operations_catalog
        from rag_prep.config_composition import load_composed_yaml

        raw = load_composed_yaml(path)
        profiles = {}
        for key, value in raw.items():
            profile = OperationProfile.model_validate(value)
            if not set(profile.parameters) <= EDITABLE_PARAMETERS:
                raise ValueError(f"Недопустимые параметры профиля {key}")
            profile.config = (path.parent / profile.config).resolve()
            if profile.suite:
                profile.suite = (path.parent / profile.suite).resolve()
            profiles[key] = profile
        return profiles

    def public_profiles(self) -> list[PublicProfile]:
        """Готовность определяется актуальным heartbeat worker, а не наличием API."""
        ready = self.store.capabilities()
        return [
            PublicProfile(
                id=key,
                title=value.title,
                kind=value.kind,
                parameters=value.parameters,
                parameter_schema=self.parameter_schema(value),
                worker_ready=key in ready,
            )
            for key, value in self.catalog().items()
        ]

    def parameter_schema(self, profile: OperationProfile) -> dict[str, dict]:
        """Публикует тип, допустимый диапазон и текущее значение только разрешённых числовых настроек."""
        model = self._load_config(profile)
        result = {}
        for name in profile.parameters:
            section, field = name.split(".")
            settings = getattr(model, section)
            field_info = type(settings).model_fields[field]
            schema = TypeAdapter(field_info.rebuild_annotation()).json_schema()
            result[name] = {**schema, "default": getattr(settings, field)}
        return result

    def available_profiles(self) -> list[str]:
        """Worker сообщает только операции, зависимости которых установлены."""
        required = {
            "prepare": ["unstructured", "spacy", "llama_index"],
            "chunk": ["llama_index", "tiktoken"],
            "embed": ["numpy"],
            "index": ["qdrant_client"],
            "evaluation": ["langgraph"],
            "scenarios": ["langgraph"],
            "tuning-inspect": ["torch"],
            "tuning-validate": ["torch"],
            "tuning-compare": ["torch"],
        }
        result = []
        for name, profile in self.catalog().items():
            modules = required.get(profile.kind, ["torch", "transformers", "peft"])
            if profile.config.is_file() and all(
                importlib.util.find_spec(module) is not None for module in modules
            ):
                result.append(name)
        return result

    def cleanup_logs(self) -> None:
        """Удаляет только журналы старше 30 дней, сохраняя результаты экспериментов."""
        for path in (self.root / "operations").glob("*/execution.log"):
            if (
                not path.is_symlink()
                and path.resolve().is_relative_to(self.root)
                and path.stat().st_mtime < time.time() - 30 * 86400
            ):
                path.unlink(missing_ok=True)

    def submit(self, owner: str, payload: OperationRequest) -> OperationRecord:
        """Проверяет права на входы до записи задания, исключая скрытый межпользовательский импорт."""
        existing = self.store.existing(owner, payload)
        if existing is not None:
            return existing
        profile = self.catalog().get(payload.profile)
        if profile is None:
            raise HTTPException(422, "Неизвестный профиль операции.")
        if not set(payload.parameters) <= set(profile.parameters):
            raise HTTPException(422, "Параметр не разрешён выбранным профилем.")
        if payload.profile not in self.store.capabilities():
            raise HTTPException(503, "Нет готового worker для этого профиля.")
        if payload.source_ids and profile.kind != "prepare":
            raise HTTPException(
                422, "Исходные файлы принимаются только этапом подготовки."
            )
        if payload.baseline_id and profile.kind != "tuning-compare":
            raise HTTPException(
                422, "Baseline выбирается только для сравнения отчётов."
            )
        for source_id in payload.source_ids:
            self.source(source_id, owner)
        if payload.upstream_id:
            previous = self.store.get(payload.upstream_id, owner)
            if previous.status != "completed":
                raise HTTPException(409, "Входной запуск ещё не завершён успешно.")
            expected = {
                "chunk": "prepare",
                "embed": "chunk",
                "index": "embed",
                "tuning-evaluate": "tuning-train",
                "tuning-compare": "tuning-evaluate",
            }.get(profile.kind)
            if (
                expected is None
                or self.store.definition(previous.id)["profile"]["kind"] != expected
            ):
                raise HTTPException(422, "Неверный этап входного запуска.")
        if profile.kind == "tuning-evaluate" and not payload.upstream_id:
            raise HTTPException(422, "Выберите успешно завершённое обучение.")
        if profile.kind == "tuning-compare":
            if not payload.baseline_id or not payload.upstream_id:
                raise HTTPException(422, "Выберите baseline и оценку адаптера.")
            baseline = self.store.get(payload.baseline_id, owner)
            if (
                baseline.status != "completed"
                or self.store.definition(baseline.id)["profile"]["kind"]
                != "tuning-baseline"
            ):
                raise HTTPException(422, "Нужен успешно завершённый baseline.")
        # Формальная валидация параметров выполняется до постановки, а не после расходов.
        model = self._load_config(profile)
        resolved = self._apply_parameters(model, payload.parameters)
        definition = {
            "profile": profile.model_dump(mode="json"),
            "config": resolved.model_dump(mode="json"),
            "suite": profile.suite.read_text(encoding="utf-8")
            if profile.suite
            else None,
        }
        return self.store.submit(owner, payload, definition=definition)

    def job_dir(self, job_id: str) -> Path:
        """UUID и resolve исключают traversal и выход через символьную ссылку."""
        try:
            canonical = str(UUID(job_id))
        except ValueError:
            raise HTTPException(404, "Операция не найдена.") from None
        result = (self.root / "operations" / canonical).resolve()
        if not result.is_relative_to(self.root):
            raise HTTPException(404, "Артефакт недоступен.")
        return result

    def command(self, record: OperationRecord) -> tuple[str, list[str]]:
        """Создаёт абсолютный snapshot конфига и аргументы фиксированной CLI-функции."""
        definition = self.store.definition(record.id)
        profile = OperationProfile.model_validate(definition["profile"])
        data = definition["config"]
        directory = self.job_dir(record.id)
        output = directory / "artifacts"
        output.mkdir(parents=True, exist_ok=True)
        if definition["suite"]:
            profile.suite = directory / "suite.yaml"
            profile.suite.write_text(definition["suite"], encoding="utf-8")
        if profile.kind in {"prepare", "chunk", "embed", "index"}:
            data["paths"]["output_dir"] = str(output)
            if record.request.source_ids:
                inputs = directory / "inputs"
                inputs.mkdir(exist_ok=True)
                for source_id in record.request.source_ids:
                    source, path = self.source(source_id, record.user_id)
                    shutil.copyfile(
                        path, inputs / f"{source_id}{Path(source.name).suffix}"
                    )
                data["paths"]["input_dir"] = str(inputs)
            if record.request.upstream_id:
                filename = {
                    "chunk": "documents.jsonl",
                    "embed": "chunks.jsonl",
                    "index": "embeddings.jsonl",
                }[profile.kind]
                data["paths"]["input_jsonl"] = str(
                    self.job_dir(record.request.upstream_id) / "artifacts" / filename
                )
            if profile.kind == "index":
                # Web-загрузка дополняет общую базу знаний, не удаляя чужие точки.
                data["vector_store"]["recreate_collection"] = False
                data["vector_store"]["prune_stale_points"] = False
        elif profile.kind.startswith("tuning-"):
            data["paths"]["output_dir"] = str(output / "runs")
            data["paths"]["reports_dir"] = str(output / "reports")
            data["paths"]["adapter_output_dir"] = str(output / "adapter")
        else:
            data["evaluation"]["output_dir"] = str(output)
            data["multi_agent"]["output_dir"] = str(output / "multi_agent")
        snapshot = directory / "config.yaml"
        snapshot.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        if profile.kind in {"prepare", "chunk", "embed"}:
            return "rag_prep.cli", [
                profile.kind,
                "--config",
                str(snapshot),
                "--no-prefect",
            ]
        if profile.kind == "index":
            return "rag_prep.vector_store_cli", ["--config", str(snapshot)]
        if profile.kind == "evaluation":
            return "agent_app.evaluation.cli", [
                "--config",
                str(snapshot),
                "--suite",
                str(profile.suite),
            ]
        if profile.kind == "scenarios":
            return "agent_app.cli", [
                "--config",
                str(snapshot),
                "--run-scenarios",
                "--scenarios-config",
                str(profile.suite),
                "--scenario-report",
                str(output / "scenario_report.json"),
            ]
        action = {
            "tuning-inspect": "inspect-env",
            "tuning-validate": "validate-data",
            "tuning-baseline": "baseline",
            "tuning-train": "train",
            "tuning-evaluate": "evaluate",
            "tuning-compare": "compare",
        }[profile.kind]
        args = ["--config", str(snapshot), action]
        if action == "evaluate":
            adapters = list(
                (
                    self.job_dir(record.request.upstream_id) / "artifacts" / "adapter"
                ).glob("*/adapter_config.json")
            )
            if len(adapters) != 1:
                raise ValueError(
                    "Запуск обучения должен содержать один сохранённый адаптер."
                )
            args += ["--adapter-path", str(adapters[0].parent)]
        if action == "compare":
            for flag, previous, name in (
                (
                    "--baseline-report",
                    record.request.baseline_id,
                    "baseline_report.json",
                ),
                ("--tuned-report", record.request.upstream_id, "tuned_report.json"),
            ):
                matches = list((self.job_dir(previous) / "artifacts").rglob(name))
                if len(matches) != 1:
                    raise ValueError(
                        "Операция должна содержать один отчёт соответствующего типа."
                    )
                args += [flag, str(matches[0])]
        return "llm_tuning.cli", args

    def sources(
        self, owner: str, *, after: str = "", limit: int = 50
    ) -> list[SourceRecord]:
        """Список исходников ограничен владельцем и листается по ID."""
        with self.store.database.connection(self.store.path) as conn:
            rows = conn.execute(
                "SELECT * FROM web_sources WHERE user_id = ? AND id > ? ORDER BY id LIMIT ?",
                (owner, after, limit),
            ).fetchall()
        return [SourceRecord.model_validate(dict(row)) for row in rows]

    def add_source(self, owner: str, name: str, body: bytes) -> SourceRecord:
        """Сохраняет файл под UUID, не доверяя имени и MIME-заголовку загрузчика."""
        if (
            Path(name).name != name
            or any(c in name for c in "/\\\x00:")
            or len(name) > 160
            or Path(name).suffix.lower() not in {".pdf", ".txt", ".html", ".csv"}
        ):
            raise HTTPException(
                422, "Разрешены имена PDF, TXT, HTML и CSV без каталогов."
            )
        if not body or len(body) > self.config.upload_max_bytes:
            raise HTTPException(413, "Недопустимый размер файла.")
        if name.lower().endswith(".pdf") and not body.startswith(b"%PDF-"):
            raise HTTPException(422, "Файл не содержит заголовок PDF.")
        record = SourceRecord(
            id=str(uuid4()),
            name=name,
            sha256=hashlib.sha256(body).hexdigest(),
            size_bytes=len(body),
            created_at=time.time(),
        )
        root = self.root / "sources"
        root.mkdir(parents=True, exist_ok=True)
        path = root / record.id
        path.write_bytes(body)
        try:
            with self.store.database.connection(self.store.path) as conn:
                conn.execute(
                    "INSERT INTO web_sources VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        owner,
                        name,
                        record.sha256,
                        record.size_bytes,
                        record.created_at,
                    ),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return record

    def source(self, source_id: str, owner: str) -> tuple[SourceRecord, Path]:
        """Получение и скачивание всегда проверяют владельца и целостность содержимого."""
        with self.store.database.connection(self.store.path) as conn:
            row = conn.execute(
                "SELECT * FROM web_sources WHERE id = ? AND user_id = ?",
                (source_id, owner),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Источник не найден.")
        record = SourceRecord.model_validate(dict(row))
        path = (self.root / "sources" / record.id).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise HTTPException(404, "Файл источника недоступен.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256:
            raise HTTPException(409, "Нарушена целостность источника.")
        return record, path

    def delete_source(self, source_id: str, owner: str) -> None:
        """Удаляет исходник; уже подготовленные результаты операции сохраняются отдельно."""
        _, path = self.source(source_id, owner)
        with self.store.database.connection(self.store.path) as conn:
            conn.execute("UPDATE web_worker_lock SET id = id WHERE id = 1")
            rows = conn.execute(
                "SELECT request_json FROM web_operations WHERE user_id = ? AND status IN ('queued', 'running', 'cancel_requested')",
                (owner,),
            ).fetchall()
            if any(
                source_id in json.loads(row["request_json"])["source_ids"]
                for row in rows
            ):
                raise HTTPException(
                    409, "Источник используется незавершённой операцией."
                )
            conn.execute(
                "DELETE FROM web_sources WHERE id = ? AND user_id = ?",
                (source_id, owner),
            )
        path.unlink(missing_ok=True)

    def artifacts(self, job_id: str) -> list[ArtifactRecord]:
        """Выдаёт метаданные файлов результата без snapshot credentials и логов."""
        root = self.job_dir(job_id) / "artifacts"
        if not root.exists():
            return []
        result = []
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.resolve().is_relative_to(root.resolve())
            ):
                name = path.relative_to(root).as_posix()
                # Снимки конфигурации в manifest могут содержать DSN: они не
                # публикуются как необработанные загрузки, доступен JSON-preview.
                result.append(
                    ArtifactRecord(
                        id=hashlib.sha256(name.encode()).hexdigest()[:32],
                        name=name,
                        size_bytes=path.stat().st_size,
                    )
                )
        return result

    def artifact(self, job_id: str, artifact_id: str) -> Path:
        """Находит файл через серверный каталог, не декодируя пользовательский путь."""
        record = next(
            (item for item in self.artifacts(job_id) if item.id == artifact_id), None
        )
        if record is None:
            raise HTTPException(404, "Артефакт не найден.")
        return self.job_dir(job_id) / "artifacts" / record.name

    @staticmethod
    def _load_config(profile: OperationProfile):
        """Использует те же загрузчики и абсолютные пути, что штатные CLI."""
        from rag_prep.config import (
            load_config,
            load_chunking_config,
            load_embedding_config,
            load_vector_store_config,
        )
        from agent_app.config import load_agent_config
        from llm_tuning.config import load_fine_tuning_config

        loaders = {
            "prepare": load_config,
            "chunk": load_chunking_config,
            "embed": load_embedding_config,
            "index": load_vector_store_config,
            "evaluation": load_agent_config,
            "scenarios": load_agent_config,
        }
        return loaders.get(profile.kind, load_fine_tuning_config)(profile.config)

    @staticmethod
    def _apply_parameters(model, parameters):
        """Повторная Pydantic-валидация сохраняет ограничения исходного пайплайна."""
        data = model.model_dump(mode="python")
        for dotted, value in parameters.items():
            section, name = dotted.split(".", 1)
            data[section][name] = value
        return type(model).model_validate(data)


def public_text(text: str) -> str:
    """Ответы журналов и JSON-preview маскируют секреты и локальные пути."""
    text = re.sub(r"(\w+(?:\+\w+)?://)[^\s/@]+:[^\s/@]+@", r"\1<redacted>@", text)
    return redact_local_paths(redact_secrets(text))


def public_data(value):
    """Редактирует строковые значения, не повреждая JSON-структуру и числовые векторы."""
    if isinstance(value, dict):
        return {key: public_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_data(item) for item in value]
    return public_text(value) if isinstance(value, str) else value
