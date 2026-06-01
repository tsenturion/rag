from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from uuid import NAMESPACE_URL, uuid5

import portalocker
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from rag_prep.config import VectorStoreConfig

LOGGER = logging.getLogger(__name__)


def _make_qdrant_client(config: VectorStoreConfig) -> QdrantClient:
    if config.mode == "local":
        config.local_storage_path.mkdir(parents=True, exist_ok=True)
        LOGGER.warning(
            (
                "Using embedded Qdrant local mode at %s. This storage is intended "
                "for one process at a time; do not run parallel writers against "
                "the same local_storage_path."
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
        timeout=config.timeout_seconds,
    )


@contextmanager
def qdrant_client_context(config: VectorStoreConfig):
    lock = None
    if config.mode == "local":
        config.local_storage_path.mkdir(parents=True, exist_ok=True)
        lock_path = config.local_storage_path / ".rag_prep.lock"
        lock = portalocker.Lock(str(lock_path), timeout=0)
        try:
            lock.acquire()
        except portalocker.exceptions.LockException as exc:
            raise RuntimeError(
                (
                    "Embedded Qdrant local storage is already in use by another "
                    f"process: {config.local_storage_path}. Stop the other run or "
                    "use vector_store.mode=http with a Qdrant server for concurrent access."
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
    try:
        return qdrant_models.Distance(name)
    except ValueError:
        return qdrant_models.Distance[name.upper()]


def point_id_for_chunk(collection_name: str, chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"qdrant:{collection_name}:{chunk_id}"))


def qdrant_url(config: VectorStoreConfig) -> str | None:
    if config.mode != "http":
        return None
    scheme = "https" if config.https else "http"
    return f"{scheme}://{config.host}:{config.port}"
