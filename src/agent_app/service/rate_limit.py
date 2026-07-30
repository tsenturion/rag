"""Потокобезопасное ограничение частоты дорогостоящих запросов к агенту."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from time import monotonic

import redis


@dataclass
class _Bucket:
    """Хранит число доступных токенов и момент последнего пополнения."""

    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Ограничивает запросы каждого субъекта с контролируемым кратким burst."""

    _REDIS_CONSUME = """
    local now_parts = redis.call('TIME')
    local now = tonumber(now_parts[1]) + tonumber(now_parts[2]) / 1000000
    local values = redis.call('HMGET', KEYS[1], 'tokens', 'updated_at')
    local tokens = tonumber(values[1]) or tonumber(ARGV[1])
    local updated_at = tonumber(values[2]) or now
    local elapsed = math.max(0, now - updated_at)
    tokens = math.min(tonumber(ARGV[1]), tokens + elapsed * tonumber(ARGV[2]))
    local retry_ms = 0
    if tokens >= 1 then
      tokens = tokens - 1
    else
      retry_ms = math.ceil(((1 - tokens) / tonumber(ARGV[2])) * 1000)
    end
    redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated_at', now)
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
    return retry_ms
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        burst: int,
        redis_url: str | None = None,
        key_prefix: str = "rag-support:rate-limit",
        bucket_ttl_seconds: int = 3600,
    ):
        """Создаёт process-local либо атомарный распределённый token bucket."""
        self.capacity = float(burst)
        self.refill_per_second = requests_per_minute / 60.0
        self.key_prefix = key_prefix.rstrip(":")
        self.bucket_ttl_seconds = bucket_ttl_seconds
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._consume_count = 0
        self._redis = (
            redis.Redis.from_url(redis_url, decode_responses=True)
            if redis_url
            else None
        )
        if self._redis is not None:
            # Redis-профиль не должен незаметно деградировать до process-local
            # квоты: при масштабировании это умножило бы разрешённый расход.
            self._redis.ping()

    def consume(self, subject: str) -> float | None:
        """Списывает запрос или возвращает секунды до появления следующего токена."""
        if self._redis is not None:
            digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()
            retry_ms = int(
                self._redis.eval(
                    self._REDIS_CONSUME,
                    1,
                    f"{self.key_prefix}:{digest}",
                    self.capacity,
                    self.refill_per_second,
                    self.bucket_ttl_seconds,
                )
            )
            return retry_ms / 1000.0 if retry_ms > 0 else None

        now = monotonic()
        with self._lock:
            self._consume_count += 1
            if self._consume_count % 128 == 0:
                cutoff = now - self.bucket_ttl_seconds
                self._buckets = {
                    key: value
                    for key, value in self._buckets.items()
                    if value.updated_at >= cutoff
                }
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

    def close(self) -> None:
        """Закрывает пул Redis; memory backend дополнительных ресурсов не имеет."""
        if self._redis is not None:
            self._redis.close()
