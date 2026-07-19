"""Потокобезопасное ограничение частоты дорогостоящих запросов к агенту."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import monotonic


@dataclass
class _Bucket:
    """Хранит число доступных токенов и момент последнего пополнения."""

    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Ограничивает запросы каждого субъекта с контролируемым кратким burst."""

    def __init__(self, *, requests_per_minute: int, burst: int):
        """Создаёт token bucket с независимым состоянием для каждого principal."""
        self.capacity = float(burst)
        self.refill_per_second = requests_per_minute / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def consume(self, subject: str) -> float | None:
        """Списывает запрос или возвращает секунды до появления следующего токена."""
        now = monotonic()
        with self._lock:
            bucket = self._buckets.get(subject)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, updated_at=now)
                self._buckets[subject] = bucket
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(
                self.capacity,
                bucket.tokens + elapsed * self.refill_per_second,
            )
            bucket.updated_at = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return None
            return (1.0 - bucket.tokens) / self.refill_per_second
