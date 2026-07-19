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
    utc_now,
)

PENDING_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING}


class JobStore(Protocol):
    """Определяет интерфейс для хранения и управления состояниями заданий, обеспечивая атомарность операций и контроль ресурсов."""

    def save(self, record: JobRecord) -> None:
        """Сохраняет актуальное состояние задания."""
        ...

    def get(self, job_id: str) -> JobRecord | None:
        """Возвращает задание по идентификатору."""
        ...

    def create_if_capacity(self, record: JobRecord, *, max_pending: int) -> bool:
        """Атомарно создаёт queued-задание, если лимит незавершённых работ не исчерпан."""
        ...

    def claim_run(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: int,
    ) -> JobRecord | None:
        """Атомарно закрепляет одну попытку выполнения за worker до истечения lease."""
        ...

    def renew_run_claim(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: int,
    ) -> bool:
        """Продлевает lease только для worker, который владеет текущей попыткой."""
        ...

    def compare_and_save(
        self,
        record: JobRecord,
        *,
        expected_statuses: set[JobStatus],
        expected_attempt: int | None = None,
        claim_token: str | None = None,
        release_claim: bool = False,
    ) -> bool:
        """Сохраняет переход только при совпадении статуса, попытки и claim token."""
        ...

    def set_task_id(self, job_id: str, task_id: str) -> JobRecord | None:
        """Атомарно дописывает broker task ID, не заменяя конкурентный статус."""
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
        self._run_claims: dict[str, tuple[str, float]] = {}
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

    def create_if_capacity(self, record: JobRecord, *, max_pending: int) -> bool:
        """Проверяет capacity и создаёт запись внутри одной критической секции."""
        with self._lock:
            if record.job.id in self._records:
                return False
            pending = sum(
                item.status in PENDING_STATUSES for item in self._records.values()
            )
            if pending >= max_pending:
                return False
            self._records[record.job.id] = record.model_copy(deep=True)
            return True

    def claim_run(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: int,
    ) -> JobRecord | None:
        """Не позволяет двум потокам одновременно исполнять один job ID."""
        if lease_seconds <= 0:
            raise ValueError("Lease выполнения должен быть положительным")
        now = time.monotonic()
        with self._lock:
            record = self._records.get(job_id)
            if record is None or record.status.terminal:
                return None
            existing = self._run_claims.get(job_id)
            if existing is not None and existing[1] <= now:
                self._run_claims.pop(job_id, None)
                existing = None
            if existing is not None:
                return None
            if record.status not in {
                JobStatus.QUEUED,
                JobStatus.RETRYING,
                JobStatus.RUNNING,
            }:
                return None

            timestamp = utc_now()
            claimed = record.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "attempts": record.attempts + 1,
                    "started_at": record.started_at or timestamp,
                    "updated_at": timestamp,
                    "finished_at": None,
                    "result": None,
                    "error": None,
                },
                deep=True,
            )
            self._records[job_id] = claimed
            self._run_claims[job_id] = (token, now + lease_seconds)
            return claimed.model_copy(deep=True)

    def renew_run_claim(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: int,
    ) -> bool:
        """Продлевает in-memory claim без возможности присвоить чужой lease."""
        now = time.monotonic()
        with self._lock:
            existing = self._run_claims.get(job_id)
            if existing is None or existing[0] != token or existing[1] <= now:
                self._run_claims.pop(job_id, None)
                return False
            self._run_claims[job_id] = (token, now + lease_seconds)
            return True

    def compare_and_save(
        self,
        record: JobRecord,
        *,
        expected_statuses: set[JobStatus],
        expected_attempt: int | None = None,
        claim_token: str | None = None,
        release_claim: bool = False,
    ) -> bool:
        """Выполняет CAS над записью и, при необходимости, освобождает run claim."""
        now = time.monotonic()
        with self._lock:
            current = self._records.get(record.job.id)
            if current is None or current.status not in expected_statuses:
                return False
            if expected_attempt is not None and current.attempts != expected_attempt:
                return False
            if claim_token is not None:
                claim = self._run_claims.get(record.job.id)
                if claim is None or claim[0] != claim_token or claim[1] <= now:
                    return False
            self._records[record.job.id] = record.model_copy(deep=True)
            if release_claim:
                self._run_claims.pop(record.job.id, None)
            return True

    def set_task_id(self, job_id: str, task_id: str) -> JobRecord | None:
        """Меняет только task_id, сохраняя статус, установленный worker или отменой."""
        with self._lock:
            current = self._records.get(job_id)
            if current is None:
                return None
            updated = current.model_copy(
                update={"task_id": task_id, "updated_at": utc_now()},
                deep=True,
            )
            self._records[job_id] = updated
            return updated.model_copy(deep=True)

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
        if record.status in PENDING_STATUSES:
            pipe.zadd(
                self._pending_key(),
                {job_id: self._pending_expiry_ms()},
            )
            pipe.expire(self._pending_key(), self.state_ttl_seconds)
        else:
            pipe.zrem(self._pending_key(), job_id)
        pipe.execute()

    def get(self, job_id: str) -> JobRecord | None:
        """Гарантирует получение полной информации о задаче по идентификатору или отсутствие результата, если задача не найдена."""
        payload = self.client.get(self._job_key(job_id))
        return JobRecord.model_validate_json(payload) if payload else None

    def create_if_capacity(self, record: JobRecord, *, max_pending: int) -> bool:
        """Одной Lua-операцией резервирует capacity и создаёт queued-запись."""
        job_id = record.job.id
        status_keys = [self._status_key(status) for status in JobStatus]
        script = """
        redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', ARGV[5])
        if redis.call('EXISTS', KEYS[1]) == 1 then
          return 0
        end
        if redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[4]) then
          return 0
        end
        redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[3])
        for index = 3, #KEYS do
          redis.call('SREM', KEYS[index], ARGV[2])
        end
        redis.call('SADD', KEYS[3], ARGV[2])
        redis.call('EXPIRE', KEYS[3], ARGV[3])
        redis.call('ZADD', KEYS[2], ARGV[6], ARGV[2])
        redis.call('EXPIRE', KEYS[2], ARGV[3])
        return 1
        """
        return bool(
            self.client.eval(
                script,
                2 + len(status_keys),
                self._job_key(job_id),
                self._pending_key(),
                *status_keys,
                record.model_dump_json(),
                job_id,
                self.state_ttl_seconds,
                max_pending,
                self._now_ms(),
                self._pending_expiry_ms(),
            )
        )

    def claim_run(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: int,
    ) -> JobRecord | None:
        """Атомарно переводит queued/retrying либо abandoned running job в RUNNING."""
        status_keys = [self._status_key(status) for status in JobStatus]
        timestamp = utc_now().isoformat()
        script = """
        local raw = redis.call('GET', KEYS[1])
        if not raw then
          return nil
        end
        local data = cjson.decode(raw)
        local status = data['status']
        local active_claim = redis.call('GET', KEYS[2])
        if active_claim then
          return nil
        end
        if status ~= 'queued' and status ~= 'retrying' and status ~= 'running' then
          return nil
        end
        data['status'] = 'running'
        data['attempts'] = (data['attempts'] or 0) + 1
        if data['started_at'] == nil or data['started_at'] == cjson.null then
          data['started_at'] = ARGV[4]
        end
        data['updated_at'] = ARGV[4]
        data['finished_at'] = cjson.null
        data['result'] = cjson.null
        data['error'] = cjson.null
        local encoded = cjson.encode(data)
        redis.call('SET', KEYS[1], encoded, 'EX', ARGV[3])
        redis.call('SET', KEYS[2], ARGV[1], 'PX', ARGV[2])
        for index = 4, #KEYS do
          redis.call('SREM', KEYS[index], ARGV[5])
        end
        redis.call('SADD', KEYS[5], ARGV[5])
        redis.call('EXPIRE', KEYS[5], ARGV[3])
        redis.call('ZADD', KEYS[3], ARGV[6], ARGV[5])
        redis.call('EXPIRE', KEYS[3], ARGV[3])
        return encoded
        """
        payload = self.client.eval(
            script,
            3 + len(status_keys),
            self._job_key(job_id),
            self._run_claim_key(job_id),
            self._pending_key(),
            *status_keys,
            token,
            lease_seconds * 1000,
            self.state_ttl_seconds,
            timestamp,
            job_id,
            self._pending_expiry_ms(),
        )
        return JobRecord.model_validate_json(payload) if payload else None

    def renew_run_claim(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: int,
    ) -> bool:
        """Продлевает Redis claim только при точном совпадении token."""
        script = """
        if redis.call('GET', KEYS[1]) ~= ARGV[1] then
          return 0
        end
        redis.call('PEXPIRE', KEYS[1], ARGV[2])
        return 1
        """
        return bool(
            self.client.eval(
                script,
                1,
                self._run_claim_key(job_id),
                token,
                lease_seconds * 1000,
            )
        )

    def compare_and_save(
        self,
        record: JobRecord,
        *,
        expected_statuses: set[JobStatus],
        expected_attempt: int | None = None,
        claim_token: str | None = None,
        release_claim: bool = False,
    ) -> bool:
        """Выполняет Redis CAS и синхронно обновляет status/pending индексы."""
        job_id = record.job.id
        status_keys = [self._status_key(status) for status in JobStatus]
        expected = "|" + "|".join(item.value for item in expected_statuses) + "|"
        script = """
        local raw = redis.call('GET', KEYS[1])
        if not raw then
          return 0
        end
        local current = cjson.decode(raw)
        if not string.find(ARGV[4], '|' .. current['status'] .. '|', 1, true) then
          return 0
        end
        if ARGV[5] ~= '' and tonumber(current['attempts'] or 0) ~= tonumber(ARGV[5]) then
          return 0
        end
        if ARGV[6] ~= '' and redis.call('GET', KEYS[2]) ~= ARGV[6] then
          return 0
        end
        redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[3])
        for index = 4, #KEYS do
          redis.call('SREM', KEYS[index], ARGV[2])
        end
        local target_index = tonumber(ARGV[9])
        redis.call('SADD', KEYS[target_index], ARGV[2])
        redis.call('EXPIRE', KEYS[target_index], ARGV[3])
        if ARGV[8] == '1' then
          redis.call('ZADD', KEYS[3], ARGV[10], ARGV[2])
          redis.call('EXPIRE', KEYS[3], ARGV[3])
        else
          redis.call('ZREM', KEYS[3], ARGV[2])
        end
        if ARGV[7] == '1' then
          redis.call('DEL', KEYS[2])
        end
        return 1
        """
        target_index = 4 + list(JobStatus).index(record.status)
        return bool(
            self.client.eval(
                script,
                3 + len(status_keys),
                self._job_key(job_id),
                self._run_claim_key(job_id),
                self._pending_key(),
                *status_keys,
                record.model_dump_json(),
                job_id,
                self.state_ttl_seconds,
                expected,
                "" if expected_attempt is None else expected_attempt,
                claim_token or "",
                int(release_claim),
                int(record.status in PENDING_STATUSES),
                target_index,
                self._pending_expiry_ms(),
            )
        )

    def set_task_id(self, job_id: str, task_id: str) -> JobRecord | None:
        """Дополняет JSON broker ID без read-modify-write гонки со статусом."""
        status_keys = [self._status_key(status) for status in JobStatus]
        script = """
        local raw = redis.call('GET', KEYS[1])
        if not raw then
          return nil
        end
        local data = cjson.decode(raw)
        data['task_id'] = ARGV[1]
        data['updated_at'] = ARGV[2]
        local encoded = cjson.encode(data)
        redis.call('SET', KEYS[1], encoded, 'EX', ARGV[3])
        return encoded
        """
        payload = self.client.eval(
            script,
            1 + len(status_keys),
            self._job_key(job_id),
            *status_keys,
            task_id,
            utc_now().isoformat(),
            self.state_ttl_seconds,
        )
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

    def _run_claim_key(self, job_id: str) -> str:
        """Возвращает отдельный lease-ключ владельца текущей попытки."""
        return f"{self.prefix}:run-claim:{job_id}"

    def _pending_key(self) -> str:
        """Возвращает sorted set незавершённых заданий с TTL каждого member."""
        return f"{self.prefix}:pending"

    @staticmethod
    def _now_ms() -> int:
        """Возвращает текущее Unix-время в миллисекундах для Redis lease."""
        return int(time.time() * 1000)

    def _pending_expiry_ms(self) -> int:
        """Задаёт срок жизни pending-member одновременно с job state."""
        return self._now_ms() + self.state_ttl_seconds * 1000
