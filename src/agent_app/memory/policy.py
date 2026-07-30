"""Политики валидации и хранения для памяти агента."""

from __future__ import annotations

import re

from agent_app.support.security import contains_secret


def normalize_key(key: str) -> str:
    """Обеспечивает каноническое представление ключа памяти для предотвращения коллизий и ошибок поиска независимо от регистра и пробелов."""
    normalized = re.sub(r"\s+", "_", key.strip().lower())
    normalized = re.sub(r"[^a-zа-яё0-9_.:-]+", "_", normalized, flags=re.IGNORECASE)
    return normalized.strip("_") or "memory"


def validate_memory_key(key: str) -> str:
    """Гарантирует, что ключ памяти не содержит секретов и пригоден для безопасного хранения и поиска."""
    cleaned = key.strip()
    if contains_secret(cleaned):
        raise ValueError("ключ похож на секрет и не может использоваться для памяти")
    return cleaned


def validate_memory_value(value: str) -> str:
    """Гарантирует, что значение памяти не пустое и не содержит вероятных секретов, предотвращая утечку чувствительных данных."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("значение памяти не может быть пустым")
    if contains_secret(cleaned):
        raise ValueError("вероятный секрет нельзя сохранять в память")
    return cleaned


def clamp_importance(value: int) -> int:
    """Гарантирует, что важность памяти всегда находится в допустимом диапазоне от 1 до 5 включительно."""
    return max(1, min(5, int(value)))
