from __future__ import annotations

import re

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\b(api[_ -]?key|token|password|secret)\b\s*[:=]"),
)


def normalize_key(key: str) -> str:
    normalized = re.sub(r"\s+", "_", key.strip().lower())
    normalized = re.sub(r"[^a-zа-яё0-9_.:-]+", "_", normalized, flags=re.IGNORECASE)
    return normalized.strip("_") or "memory"


def validate_memory_value(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("memory value cannot be empty")
    for pattern in SECRET_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError("refusing to store likely secret value in memory")
    return cleaned


def clamp_importance(value: int) -> int:
    return max(1, min(5, int(value)))
