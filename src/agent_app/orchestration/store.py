"""Хранилище состояния для распределённой оркестрации."""

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
    """Определяет интерфейс для хранения и управления состояниями заданий, обеспечивая атомарность операций и контроль ресурсов."""

    def save(self, record: JobRecord) -> None:
        """Сохраняет актуальное состояние задания."""
        ...

    def get(self, job_id: str) -> JobRecord | None:
        """Возвращает задание по идентификатору."""
        ...

    def append_event(self, event: JobEvent) -> JobEvent:
        """Добавляет упорядоченное событие задания."""
        ...

    def events(self, job_id: str) -> list[JobEvent]:
        """Возвращает журнал событий задания."""
        ...

    def claim_idempotency(self, key: str, job_id: str, ttl_seconds: int) -> str:
        """Атомарно закрепляет idempotency key за первым заданием."""
        ...

    def rebind_idempotency(
        self,
        key: str,
        *,
        expected_job_id: str,
        new_job_id: str,
        ttl_seconds: int,
    ) -> bool:
        """Атомарно заменяет ссылку на отсутствующее задание."""
        ...

    def counts(self) -> dict[str, int]:
        """Возвращает количества заданий, сгруппированные по статусам."""
        ...

    def acquire_slot(
        self,
        name: str,
        token: str,
        *,
        limit: int,
        lease_seconds: int,
    ) -> bool:
        """Захватывает ограниченный lease-слот указанного ресурса."""
        ...

    def release_slot(self, name: str, token: str) -> None:
        """Освобождает ранее захваченный lease-слот."""
        ...

    def renew_slot(self, name: str, token: str, *, lease_seconds: int) -> bool:
        """Продлевает только принадлежащий заданию lease-слот."""
        ...

    def ping(self) -> bool:
        """Проверяет доступность backend хранилища."""
        ...

    def close(self) -> None:
        """Закрывает соединения и освобождает backend-ресурсы."""
        ...


class InMemoryJobStore:
    """Потокобезопасный backend для локального запуска и тестов."""

    def __init__(self):
        """Готовит потокобезопасное хранилище для заданий, событий и идемпотентности в памяти процесса."""
        self._records: dict[str, JobRecord] = {}
        self._events: dict[str, list[JobEvent]] = {}
        self._idempotency: dict[str, tuple[str, float]] = {}
        self._slots: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def save(self, record: JobRecord) -> None:
        """Гарантирует атомарное сохранение полной копии состояния задания в памяти."""
        with self._lock:
            self._records[record.job.id] = record.model_copy(deep=True)

    def get(self, job_id: str) -> JobRecord | None:
        """Возвращает неизменяемую копию состояния задания или None, если оно отсутствует."""
        with self._lock:
            record = self._records.get(job_id)
            return record.model_copy(deep=True) if record is not None else None

    def append_event(self, event: JobEvent) -> JobEvent:
        """Добавляет событие к истории задания с уникальным порядковым номером и возвращает его копию."""
        with self._lock:
            events = self._events.setdefault(event.job_id, [])
            saved = event.model_copy(update={"sequence": len(events) + 1})
            events.append(saved)
            return saved.model_copy(deep=True)

    def events(self, job_id: str) -> list[JobEvent]:
        """Гарантирует получение полной истории событий для задания в виде независимых копий."""
        with self._lock:
            return [
                event.model_copy(deep=True) for event in self._events.get(job_id, [])
            ]

    def claim_idempotency(self, key: str, job_id: str, ttl_seconds: int) -> str:
        """Закрепляет ключ до истечения TTL и после этого разрешает новое задание."""
        if ttl_seconds <= 0:
            raise ValueError("TTL idempotency key должен быть положительным")
        now = time.monotonic()
        with self._lock:
            existing = self._idempotency.get(key)
            if existing is not None:
                existing_job_id, expires_at = existing
                if expires_at > now:
                    return existing_job_id
            self._idempotency[key] = (job_id, now + ttl_seconds)
            return job_id

    def rebind_idempotency(
        self,
        key: str,
        *,
        expected_job_id: str,
        new_job_id: str,
        ttl_seconds: int,
    ) -> bool:
        """Перепривязывает ключ только если его значение не изменилось конкурентно."""
        if ttl_seconds <= 0:
            raise ValueError("TTL idempotency key должен быть положительным")
        now = time.monotonic()
        with self._lock:
            existing = self._idempotency.get(key)
            if existing is None:
                return False
            existing_job_id, expires_at = existing
            if expires_at <= now or existing_job_id != expected_job_id:
                return False
            self._idempotency[key] = (new_job_id, now + ttl_seconds)
            return True

    def counts(self) -> dict[str, int]:
        """Возвращает актуальное распределение заданий по статусам для мониторинга и диагностики."""
        with self._lock:
            counts = {status.value: 0 for status in JobStatus}
            for record in self._records.values():
                counts[record.status.value] += 1
            return counts

    def acquire_slot(
        self, name: str, token: str, *, limit: int, lease_seconds: int
    ) -> bool:
        """Гарантирует атомарное выделение слота для задачи с учётом лимита одновременных исполнителей в памяти процесса."""
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
        """Гарантирует немедленное освобождение слота для задачи, позволяя другим исполнителям занимать его без задержки."""
        with self._lock:
            self._slots.setdefault(name, set()).discard(token)

    def renew_slot(self, name: str, token: str, *, lease_seconds: int) -> bool:
        """Подтверждает владение in-memory слотом; TTL здесь не требуется."""
        del lease_seconds
        with self._lock:
            return token in self._slots.get(name, set())

    def ping(self) -> bool:
        """Гарантирует, что in-memory хранилище доступно для операций оркестрации."""
        return True

    def close(self) -> None:
        """Гарантирует отсутствие необходимости явного освобождения ресурсов для in-memory хранилища."""
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
        """Гарантирует готовность экземпляра к работе с Redis и корректное управление TTL и префиксами ключей."""
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix.rstrip(":")
        self.state_ttl_seconds = state_ttl_seconds
        self.event_limit = event_limit

    def save(self, record: JobRecord) -> None:
        """Атомарно обновляет запись и индекс её текущего статуса."""
        job_id = record.job.id
        key = self._job_key(job_id)
        # Запись и status sets должны измениться одной Redis-транзакцией, иначе
        # мониторинг может увидеть задание одновременно в двух состояниях.
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
        """Гарантирует получение полной информации о задаче по идентификатору или отсутствие результата, если задача не найдена."""
        payload = self.client.get(self._job_key(job_id))
        return JobRecord.model_validate_json(payload) if payload else None

    def append_event(self, event: JobEvent) -> JobEvent:
        """Добавляет событие с монотонным sequence и ограничивает журнал."""
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
        """Гарантирует получение всех событий жизненного цикла задачи в порядке их возникновения."""
        payloads: Iterable[str] = self.client.lrange(self._event_key(job_id), 0, -1)
        return [JobEvent.model_validate_json(payload) for payload in payloads]

    def claim_idempotency(self, key: str, job_id: str, ttl_seconds: int) -> str:
        """Закрепляет ключ за первым заданием атомарной операцией SET NX."""
        redis_key = f"{self.prefix}:idempotency:{key}"
        # SET NX не оставляет окна между проверкой ключа и его созданием даже при
        # одновременной отправке одинаковых задач несколькими API-процессами.
        claimed = self.client.set(redis_key, job_id, nx=True, ex=ttl_seconds)
        if claimed:
            return job_id
        existing = self.client.get(redis_key)
        return existing or job_id

    def rebind_idempotency(
        self,
        key: str,
        *,
        expected_job_id: str,
        new_job_id: str,
        ttl_seconds: int,
    ) -> bool:
        """Сравнивает и заменяет Redis binding одной Lua-операцией."""
        redis_key = f"{self.prefix}:idempotency:{key}"
        script = """
        if redis.call('GET', KEYS[1]) ~= ARGV[1] then
          return 0
        end
        redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
        return 1
        """
        return bool(
            self.client.eval(
                script,
                1,
                redis_key,
                expected_job_id,
                new_job_id,
                ttl_seconds,
            )
        )

    def counts(self) -> dict[str, int]:
        """Гарантирует получение актуального количества задач в каждом статусе для мониторинга нагрузки."""
        return {
            status.value: int(self.client.scard(self._status_key(status)))
            for status in JobStatus
        }

    def acquire_slot(
        self, name: str, token: str, *, limit: int, lease_seconds: int
    ) -> bool:
        """Атомарно захватывает слот с lease через Redis sorted set и Lua."""
        key = f"{self.prefix}:slots:{name}"
        now = int(time.time() * 1000)
        expires_at = now + lease_seconds * 1000
        # Lua объединяет удаление истёкших lease, проверку лимита и ZADD. Без него
        # параллельные workers могли бы одновременно пройти проверку ZCARD.
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
            # TTL ключа длиннее одного lease, чтобы активный sorted set не исчезал
            # на границе обновления; отдельные token всё равно чистятся по score.
            lease_seconds * 2000,
        )
        return bool(result)

    def release_slot(self, name: str, token: str) -> None:
        """Гарантирует немедленное освобождение слота в Redis, позволяя другим процессам занимать его без гонок."""
        self.client.zrem(f"{self.prefix}:slots:{name}", token)

    def renew_slot(self, name: str, token: str, *, lease_seconds: int) -> bool:
        """Продлевает Redis lease без повторного захвата и обхода лимита."""
        key = f"{self.prefix}:slots:{name}"
        now = int(time.time() * 1000)
        expires_at = now + lease_seconds * 1000
        script = """
        local current = redis.call('ZSCORE', KEYS[1], ARGV[1])
        if not current or tonumber(current) <= tonumber(ARGV[2]) then
          return 0
        end
        redis.call('ZADD', KEYS[1], ARGV[3], ARGV[1])
        redis.call('PEXPIRE', KEYS[1], ARGV[4])
        return 1
        """
        return bool(
            self.client.eval(
                script,
                1,
                key,
                token,
                now,
                expires_at,
                lease_seconds * 2000,
            )
        )

    def ping(self) -> bool:
        """Гарантирует доступность Redis для операций оркестрации и обнаружение сбоев соединения."""
        return bool(self.client.ping())

    def close(self) -> None:
        """Гарантирует корректное освобождение сетевых ресурсов и соединения с Redis при завершении работы хранилища заданий."""
        self.client.close()

    def _job_key(self, job_id: str) -> str:
        """Обеспечивает уникальное пространство имён для хранения состояния задания в Redis, предотвращая коллизии ключей между разными инстансами."""
        return f"{self.prefix}:job:{job_id}"

    def _event_key(self, job_id: str) -> str:
        """Гарантирует уникальность ключа для событий конкретного задания, обеспечивая изоляцию истории событий в Redis."""
        return f"{self.prefix}:events:{job_id}"

    def _sequence_key(self, job_id: str) -> str:
        """Обеспечивает уникальный ключ для хранения последовательности событий задания, исключая пересечения между заданиями."""
        return f"{self.prefix}:event-sequence:{job_id}"

    def _status_key(self, status: JobStatus) -> str:
        """Гарантирует уникальное отображение статуса задания в пространстве ключей Redis для эффективной фильтрации и поиска."""
        return f"{self.prefix}:status:{status.value}"
