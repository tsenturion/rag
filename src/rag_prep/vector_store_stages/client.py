"""Создание и жизненный цикл внешнего клиента для индексации в Qdrant."""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from uuid import NAMESPACE_URL, uuid5

import portalocker
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from rag_prep.config import VectorStoreConfig

LOGGER = logging.getLogger(__name__)


def _make_qdrant_client(config: VectorStoreConfig) -> QdrantClient:
    """Создаёт Qdrant-клиент; вызывать только внутри защищённого контекста."""
    if config.mode == "local":
        config.local_storage_path.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            (
                "Используется embedded Qdrant local mode в %s. Это хранилище рассчитано "
                "на один процесс за раз; не запускайте параллельные записи в один "
                "local_storage_path."
            ),
            config.local_storage_path,
        )
        return QdrantClient(path=str(config.local_storage_path))

    api_key = os.getenv(config.api_key_env) if config.api_key_env else None
    return QdrantClient(
        host=config.host,
        port=config.port,
        https=config.https,
        api_key=api_key,
        timeout=max(1, math.ceil(config.timeout_seconds)),
    )


@contextmanager
def qdrant_client_context(config: VectorStoreConfig):
    """Выдаёт клиент и сериализует доступ к embedded-хранилищу между процессами."""
    lock = None
    if config.mode == "local":
        config.local_storage_path.mkdir(parents=True, exist_ok=True)
        lock_path = config.local_storage_path / ".rag_prep.lock"
        # Файл может остаться после аварийного завершения, но блокировкой является
        # дескриптор ОС, а не наличие файла. После смерти процесса lock освобождается.
        lock = portalocker.Lock(str(lock_path), timeout=0)
        try:
            lock.acquire()
        except portalocker.exceptions.LockException as exc:
            raise RuntimeError(
                (
                    "Embedded Qdrant local storage уже используется другим процессом: "
                    f"{config.local_storage_path}. Остановите другой запуск или используйте "
                    "vector_store.mode=http с Qdrant server для конкурентного доступа."
                )
            ) from exc

    client = None
    try:
        client = _make_qdrant_client(config)
        yield client
    finally:
        if client is not None:
            client.close()
        if lock is not None:
            lock.release()


def qdrant_distance(name: str) -> qdrant_models.Distance:
    """Преобразует имя метрики в enum Qdrant с поддержкой разных версий SDK."""
    try:
        return qdrant_models.Distance(name)
    except ValueError:
        return qdrant_models.Distance[name.upper()]


def point_id_for_chunk(collection_name: str, chunk_id: str) -> str:
    """Строит воспроизводимый UUID точки из коллекции и идентификатора чанка."""
    # Имя коллекции входит в namespace, поэтому одинаковые chunk_id в независимых
    # индексах не конкурируют за одну и ту же точку.
    return str(uuid5(NAMESPACE_URL, f"qdrant:{collection_name}:{chunk_id}"))


def qdrant_url(config: VectorStoreConfig) -> str | None:
    """Возвращает сетевой URL Qdrant или ``None`` для embedded-режима."""
    if config.mode != "http":
        return None
    scheme = "https" if config.https else "http"
    return f"{scheme}://{config.host}:{config.port}"
