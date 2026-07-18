from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import Protocol

import redis

from agent_app.orchestration.models import (
    JobEvent,
    JobRecord,
    JobStatus,
)


class JobStore(Protocol):
    def save(self, record: JobRecord) -> None: ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def append_event(self, event: JobEvent) -> JobEvent: ...

    def events(self, job_id: str) -> list[JobEvent]: ...

    def claim_idempotency(self, key: str, job_id: str, ttl_seconds: int) -> str: ...

    def counts(self) -> dict[str, int]: ...

    def acquire_slot(
        self, name: str, token: str, *, limit: int, lease_seconds: int
    ) -> bool: ...

    def release_slot(self, name: str, token: str) -> None: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...


class InMemoryJobStore:
    def __init__(self):
        self._records: dict[str, JobRecord] = {}
        self._events: dict[str, list[JobEvent]] = {}
        self._idempotency: dict[str, str] = {}
        self._slots: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def save(self, record: JobRecord) -> None:
        with self._lock:
            self._records[record.job.id] = record.model_copy(deep=True)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            return record.model_copy(deep=True) if record is not None else None

    def append_event(self, event: JobEvent) -> JobEvent:
        with self._lock:
            events = self._events.setdefault(event.job_id, [])
            saved = event.model_copy(update={"sequence": len(events) + 1})
            events.append(saved)
            return saved.model_copy(deep=True)

    def events(self, job_id: str) -> list[JobEvent]:
        with self._lock:
            return [
                event.model_copy(deep=True) for event in self._events.get(job_id, [])
            ]

    def claim_idempotency(self, key: str, job_id: str, ttl_seconds: int) -> str:
        del ttl_seconds
        with self._lock:
            existing = self._idempotency.setdefault(key, job_id)
            return existing

    def counts(self) -> dict[str, int]:
        with self._lock:
            counts = {status.value: 0 for status in JobStatus}
            for record in self._records.values():
                counts[record.status.value] += 1
            return counts

    def acquire_slot(
        self, name: str, token: str, *, limit: int, lease_seconds: int
    ) -> bool:
        del lease_seconds
        with self._lock:
            slots = self._slots.setdefault(name, set())
            if token in slots:
                return True
            if len(slots) >= limit:
                return False
            slots.add(token)
            return True

    def release_slot(self, name: str, token: str) -> None:
        with self._lock:
            self._slots.setdefault(name, set()).discard(token)

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


class RedisJobStore:
    """Redis-хранилище состояний, событий и распределённых лимитов."""

    def __init__(
        self,
        url: str,
        *,
        prefix: str = "rag:orchestration",
        state_ttl_seconds: int = 86_400,
        event_limit: int = 500,
    ):
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix.rstrip(":")
        self.state_ttl_seconds = state_ttl_seconds
        self.event_limit = event_limit

    def save(self, record: JobRecord) -> None:
        job_id = record.job.id
        key = self._job_key(job_id)
        pipe = self.client.pipeline(transaction=True)
        pipe.set(
            key,
            record.model_dump_json(),
            ex=self.state_ttl_seconds,
        )
        for status in JobStatus:
            pipe.srem(self._status_key(status), job_id)
        pipe.sadd(self._status_key(record.status), job_id)
        pipe.expire(self._status_key(record.status), self.state_ttl_seconds)
        pipe.execute()

    def get(self, job_id: str) -> JobRecord | None:
        payload = self.client.get(self._job_key(job_id))
        return JobRecord.model_validate_json(payload) if payload else None

    def append_event(self, event: JobEvent) -> JobEvent:
        sequence = int(self.client.incr(self._sequence_key(event.job_id)))
        saved = event.model_copy(update={"sequence": sequence})
        key = self._event_key(event.job_id)
        pipe = self.client.pipeline(transaction=True)
        pipe.rpush(key, saved.model_dump_json())
        pipe.ltrim(key, -self.event_limit, -1)
        pipe.expire(key, self.state_ttl_seconds)
        pipe.expire(self._sequence_key(event.job_id), self.state_ttl_seconds)
        pipe.execute()
        return saved

    def events(self, job_id: str) -> list[JobEvent]:
        payloads: Iterable[str] = self.client.lrange(self._event_key(job_id), 0, -1)
        return [JobEvent.model_validate_json(payload) for payload in payloads]

    def claim_idempotency(self, key: str, job_id: str, ttl_seconds: int) -> str:
        redis_key = f"{self.prefix}:idempotency:{key}"
        claimed = self.client.set(redis_key, job_id, nx=True, ex=ttl_seconds)
        if claimed:
            return job_id
        existing = self.client.get(redis_key)
        return existing or job_id

    def counts(self) -> dict[str, int]:
        return {
            status.value: int(self.client.scard(self._status_key(status)))
            for status in JobStatus
        }

    def acquire_slot(
        self, name: str, token: str, *, limit: int, lease_seconds: int
    ) -> bool:
        key = f"{self.prefix}:slots:{name}"
        now = int(time.time() * 1000)
        expires_at = now + lease_seconds * 1000
        script = """
        redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
        if redis.call('ZSCORE', KEYS[1], ARGV[2]) then
          redis.call('ZADD', KEYS[1], ARGV[3], ARGV[2])
          return 1
        end
        if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[4]) then
          return 0
        end
        redis.call('ZADD', KEYS[1], ARGV[3], ARGV[2])
        redis.call('PEXPIRE', KEYS[1], ARGV[5])
        return 1
        """
        result = self.client.eval(
            script,
            1,
            key,
            now,
            token,
            expires_at,
            limit,
            lease_seconds * 2000,
        )
        return bool(result)

    def release_slot(self, name: str, token: str) -> None:
        self.client.zrem(f"{self.prefix}:slots:{name}", token)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def close(self) -> None:
        self.client.close()

    def _job_key(self, job_id: str) -> str:
        return f"{self.prefix}:job:{job_id}"

    def _event_key(self, job_id: str) -> str:
        return f"{self.prefix}:events:{job_id}"

    def _sequence_key(self, job_id: str) -> str:
        return f"{self.prefix}:event-sequence:{job_id}"

    def _status_key(self, status: JobStatus) -> str:
        return f"{self.prefix}:status:{status.value}"
