from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import tiktoken
from openai import OpenAI, OpenAIError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from rag_prep.config import EmbeddingConfig
from rag_prep.models import EmbeddedChunk, EmbeddedChunkMetadata, PreparedChunk, utc_now

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingBatchResult:
    vectors: list[list[float]]
    embedded_at: datetime


def resolve_openai_api_key(config: EmbeddingConfig) -> str:
    value = os.getenv(config.api_key_env)
    if value:
        return _clean_api_key(value, config.api_key_env)

    raise RuntimeError(
        (
            f"OpenAI API key not found. Set {config.api_key_env} in the environment"
            " or in the env_file configured for the embeddings pipeline."
        )
    )


class OpenAIEmbeddingStage:
    """Calculate OpenAI embeddings for prepared chunks."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        api_key = resolve_openai_api_key(config)
        with self._openai_client_env():
            self.client = OpenAI(
                api_key=api_key,
                timeout=config.timeout_seconds,
            )
        try:
            self.encoding = tiktoken.encoding_for_model(config.model)
        except KeyError:
            LOGGER.warning(
                "Unknown tokenizer model %s; falling back to cl100k_base",
                config.model,
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def run(self, chunks: list[PreparedChunk], *, run_id: str) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        batches = list(self._batches(chunks))
        for batch_number, batch in enumerate(batches, start=1):
            texts = [chunk.text for chunk in batch]
            result = self._embed_texts(texts)
            if len(result.vectors) != len(batch):
                raise ValueError(
                    f"OpenAI returned {len(result.vectors)} embeddings for {len(batch)} inputs"
                )
            for chunk, vector in zip(batch, result.vectors, strict=True):
                embedded.append(
                    self._build_embedded_chunk(
                        chunk,
                        vector,
                        run_id=run_id,
                        embedded_at=result.embedded_at,
                    )
                )
            LOGGER.info(
                "Embedded batch %d/%d with %d chunks",
                batch_number,
                len(batches),
                len(batch),
            )

        LOGGER.info("Calculated embeddings for %d chunks", len(embedded))
        return embedded

    def _embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        request: dict[str, object] = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.config.dimensions is not None:
            request["dimensions"] = self.config.dimensions

        retryer = Retrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_exception_type((OpenAIError, TimeoutError, ConnectionError)),
            reraise=True,
        )
        return retryer(self._request_embeddings, request)

    def _request_embeddings(self, request: dict[str, object]) -> EmbeddingBatchResult:
        response = self.client.embeddings.create(**request)
        embedded_at = utc_now()
        data = sorted(response.data, key=lambda item: item.index)
        return EmbeddingBatchResult(
            vectors=[list(item.embedding) for item in data],
            embedded_at=embedded_at,
        )

    def _batches(self, chunks: list[PreparedChunk]) -> Iterable[list[PreparedChunk]]:
        batch: list[PreparedChunk] = []
        batch_tokens = 0
        for chunk in chunks:
            token_count = self._token_count(chunk.text)
            if token_count > self.config.max_input_tokens:
                raise ValueError(
                    (
                        "Chunk exceeds embedding model token limit: "
                        f"id={chunk.metadata.id} tokens={token_count} "
                        f"limit={self.config.max_input_tokens}"
                    )
                )

            would_exceed_size = len(batch) >= self.config.batch_size
            would_exceed_tokens = (
                bool(batch)
                and batch_tokens + token_count > self.config.max_batch_tokens
            )
            if would_exceed_size or would_exceed_tokens:
                yield batch
                batch = []
                batch_tokens = 0

            batch.append(chunk)
            batch_tokens += token_count

        if batch:
            yield batch

    def _build_embedded_chunk(
        self,
        chunk: PreparedChunk,
        vector: list[float],
        *,
        run_id: str,
        embedded_at: datetime,
    ) -> EmbeddedChunk:
        embedding = self._normalize(vector) if self.config.normalize else vector
        metadata_payload = chunk.metadata.model_dump(mode="python")
        metadata_payload.update(
            {
                "embedding_provider": self.config.provider,
                "embedding_model": self.config.model,
                "embedding_dimensions": len(embedding),
                "embedding_vector_hash": self._embedding_hash(embedding),
                "embedding_norm": round(self._norm(embedding), 8),
                "embedding_run_id": run_id,
                "embedded_at": embedded_at,
            }
        )
        return EmbeddedChunk(
            text=chunk.text,
            embedding=embedding,
            metadata=EmbeddedChunkMetadata.model_validate(metadata_payload),
        )

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        array = np.array(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm == 0.0 or not math.isfinite(norm):
            LOGGER.warning("Cannot normalize embedding with invalid norm: %s", norm)
            return [float(value) for value in vector]
        return (array / norm).astype(float).tolist()

    @staticmethod
    def _norm(vector: list[float]) -> float:
        return float(np.linalg.norm(np.array(vector, dtype=np.float32)))

    @staticmethod
    def _embedding_hash(vector: list[float]) -> str:
        payload = json.dumps(vector, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def ensure_api_key(config: EmbeddingConfig) -> None:
        resolve_openai_api_key(config)

    @contextmanager
    def _openai_client_env(self):
        if not self.config.clear_no_proxy_for_openai:
            yield
            return

        previous = {name: os.environ.get(name) for name in ("NO_PROXY", "no_proxy")}
        try:
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _clean_api_key(value: str, env_name: str) -> str:
    cleaned = value.strip().strip("\"'")
    if "=" in cleaned and cleaned.startswith(env_name):
        cleaned = cleaned.split("=", 1)[1].strip().strip("\"'")
    return cleaned
