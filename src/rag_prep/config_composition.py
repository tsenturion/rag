from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml

RagProfileTarget = Literal["agent", "chunking", "embedding", "vector_store"]


def load_composed_yaml(path: str | Path) -> dict[str, Any]:
    """Загружает YAML с рекурсивным extends и проверкой циклов."""
    return _load_composed_yaml(Path(path).expanduser().resolve(), stack=())


def apply_rag_profile(
    raw: dict[str, Any],
    *,
    config_path: str | Path,
    target: RagProfileTarget,
) -> dict[str, Any]:
    """Проецирует единый RAG-профиль в схему конкретного пайплайна."""
    result = deepcopy(raw)
    profile_reference = result.pop("rag_profile", None)
    if profile_reference is None:
        return result
    if not isinstance(profile_reference, str) or not profile_reference.strip():
        raise ValueError("rag_profile должен быть непустым путём к YAML-файлу")

    profile_path = _resolve_reference(
        Path(config_path).expanduser().resolve(),
        profile_reference,
    )
    profile = load_composed_yaml(profile_path)
    _validate_rag_profile(profile, profile_path)

    if target == "agent":
        projected = {
            "tokenizer_model": profile["tokenizer_model"],
            "embedding": profile["embedding"],
            "vector_store": profile["vector_store"],
        }
        result["rag"] = deep_merge(projected, _mapping(result.get("rag"), "rag"))
    elif target == "chunking":
        projected = {
            "tokenizer_model": profile["tokenizer_model"],
            "embedding_model": profile["embedding"]["model"],
        }
        result["chunking"] = deep_merge(
            projected,
            _mapping(result.get("chunking"), "chunking"),
        )
    elif target == "embedding":
        result["embedding"] = deep_merge(
            profile["embedding"],
            _mapping(result.get("embedding"), "embedding"),
        )
    else:
        result["vector_store"] = deep_merge(
            profile["vector_store"],
            _mapping(result.get("vector_store"), "vector_store"),
        )
    return result


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно объединяет словари; списки и скаляры заменяются целиком."""
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_composed_yaml(
    path: Path,
    *,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(item.name for item in (*stack, path))
        raise ValueError(f"Обнаружен цикл extends: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"YAML-конфиг не найден: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Корень YAML-конфига должен быть mapping: {path}")

    references = _extends_references(loaded.pop("extends", None), path)
    merged: dict[str, Any] = {}
    for reference in references:
        parent_path = _resolve_reference(path, reference)
        parent = _load_composed_yaml(parent_path, stack=(*stack, path))
        merged = deep_merge(merged, parent)
    return deep_merge(merged, loaded)


def _extends_references(value: Any, path: Path) -> list[str]:
    if value is None:
        return []
    references = [value] if isinstance(value, str) else value
    if not isinstance(references, list) or not all(
        isinstance(item, str) and item.strip() for item in references
    ):
        raise ValueError(
            f"extends должен быть строкой или списком непустых строк: {path}"
        )
    return references


def _resolve_reference(owner_path: Path, reference: str) -> Path:
    reference_path = Path(reference).expanduser()
    if not reference_path.is_absolute():
        reference_path = owner_path.parent / reference_path
    return reference_path.resolve()


def _validate_rag_profile(profile: dict[str, Any], path: Path) -> None:
    required = ("tokenizer_model", "embedding", "vector_store")
    unexpected = sorted(set(profile) - set(required))
    if unexpected:
        raise ValueError(
            f"RAG-профиль {path} содержит неизвестные поля: {', '.join(unexpected)}"
        )
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"RAG-профиль {path} не содержит поля: {', '.join(missing)}")
    if not isinstance(profile["tokenizer_model"], str):
        raise ValueError(f"tokenizer_model в RAG-профиле должен быть строкой: {path}")
    embedding = _mapping(profile["embedding"], "embedding")
    vector_store = _mapping(profile["vector_store"], "vector_store")
    if not embedding.get("model"):
        raise ValueError(f"RAG-профиль не содержит embedding.model: {path}")
    if not vector_store.get("collection_name"):
        raise ValueError(
            f"RAG-профиль не содержит vector_store.collection_name: {path}"
        )
    dimensions = embedding.get("dimensions")
    vector_size = vector_store.get("vector_size")
    if dimensions is None or vector_size is None or dimensions != vector_size:
        raise ValueError(
            "Размерности RAG-профиля не совпадают: "
            f"embedding.dimensions={dimensions} "
            f"vector_store.vector_size={vector_size} profile={path}"
        )


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Поле {field} должно быть mapping")
    return value
