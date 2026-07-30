"""Расчёт векторных представлений для расчёта embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import tiktoken
from openai import OpenAI, OpenAIError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag_prep.config import EmbeddingConfig
from rag_prep.gigachat_tls import resolve_gigachat_ca_bundle
from rag_prep.models import EmbeddedChunk, EmbeddedChunkMetadata, PreparedChunk, utc_now

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingBatchResult:
    """Векторы одного provider-вызова и момент получения его результата."""

    vectors: list[list[float]]
    embedded_at: datetime


def resolve_openai_api_key(config: EmbeddingConfig) -> str:
    """Читает OpenAI API key только из уже загруженного окружения."""
    # Разбор .env сосредоточен в load_embedding_config. Здесь намеренно нет
    # ручного поиска строк, чтобы комментарий или чужое значение не приняли за ключ.
    value = os.getenv(config.api_key_env)
    if value:
        return _clean_api_key(value, config.api_key_env)

    raise RuntimeError(
        (
            f"OpenAI API-ключ не найден. Укажите {config.api_key_env} в переменных окружения"
            " или в env_file, настроенном для пайплайна embeddings."
        )
    )


def resolve_gigachat_auth_key(config: EmbeddingConfig) -> str:
    """Читает GigaChat Authorization key из настроенной переменной окружения."""
    value = os.getenv(config.api_key_env)
    if value:
        return _clean_api_key(value, config.api_key_env)

    raise RuntimeError(
        (
            f"GigaChat Authorization key не найден. Укажите {config.api_key_env} "
            "в переменных окружения или в env_file, настроенном для пайплайна embeddings."
        )
    )


class EmbeddingRecordMixin:
    """Формирует единый формат embedding-записи для всех провайдеров."""

    config: EmbeddingConfig

    def _build_embedded_chunk(
        self,
        chunk: PreparedChunk,
        vector: list[float],
        *,
        run_id: str,
        embedded_at: datetime,
    ) -> EmbeddedChunk:
        """Связывает вектор с исходным чанком и аудиторскими метаданными."""
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

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """Выполняет L2-нормализацию, не скрывая нулевые и нечисловые нормы."""
        array = np.array(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm == 0.0 or not math.isfinite(norm):
            LOGGER.warning(
                "Нельзя нормализовать embedding с некорректной нормой: %s", norm
            )
            return [float(value) for value in vector]
        return (array / norm).astype(float).tolist()

    @staticmethod
    def _norm(vector: list[float]) -> float:
        """Гарантирует корректное вычисление евклидовой нормы вектора embeddings для последующей нормализации и сравнения."""
        return float(np.linalg.norm(np.array(vector, dtype=np.float32)))

    @staticmethod
    def _embedding_hash(vector: list[float]) -> str:
        """Строит hash сериализованного вектора для проверки целостности."""
        payload = json.dumps(vector, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _query_vector(self, vector: list[float]) -> list[float]:
        """Применяет к query ту же нормализацию, что и к индексируемым векторам."""
        numeric = [float(value) for value in vector]
        return self._normalize(numeric) if self.config.normalize else numeric


class OpenAIEmbeddingStage(EmbeddingRecordMixin):
    """Считает OpenAI embeddings для подготовленных чанков."""

    def __init__(self, config: EmbeddingConfig):
        """Готовит экземпляр к безопасному и воспроизводимому взаимодействию с OpenAI API, включая выбор токенизатора и обработку неизвестных моделей."""
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
                "Неизвестная модель токенизатора %s; используется cl100k_base",
                config.model,
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def run(self, chunks: list[PreparedChunk], *, run_id: str) -> list[EmbeddedChunk]:
        """Гарантирует получение embeddings для всех входных чанков с контролем соответствия размеров и логированием прогресса."""
        embedded: list[EmbeddedChunk] = []
        batches = list(self._batches(chunks))
        for batch_number, batch in enumerate(batches, start=1):
            texts = [chunk.text for chunk in batch]
            result = self._embed_texts(texts)
            if len(result.vectors) != len(batch):
                raise ValueError(
                    f"OpenAI вернул {len(result.vectors)} embeddings для {len(batch)} входов"
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
                "Посчитан batch embeddings %d/%d; чанков: %d",
                batch_number,
                len(batches),
                len(batch),
            )

        LOGGER.info("Посчитаны embeddings для чанков: %d", len(embedded))
        return embedded

    def _embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Выполняет OpenAI-запрос с retry только для временных ошибок транспорта/API."""
        retryer = Retrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_exception_type((OpenAIError, TimeoutError, ConnectionError)),
            reraise=True,
        )
        return retryer(self._request_embeddings, texts)

    def embed_query(self, text: str) -> list[float]:
        """Гарантирует получение ровно одного embedding для запроса или сообщает об ошибке API."""
        result = self._embed_texts([text])
        if len(result.vectors) != 1:
            raise ValueError("OpenAI должен вернуть ровно один query embedding")
        return self._query_vector(result.vectors[0])

    def _request_embeddings(self, texts: list[str]) -> EmbeddingBatchResult:
        """Запрашивает batch и фиксирует время сразу после ответа API."""
        # Не все модели разрешают управлять размерностью. При dimensions=None
        # параметр полностью исключается, и размер выбирает сам provider.
        if self.config.dimensions is None:
            response = self.client.embeddings.create(
                model=self.config.model,
                input=texts,
                encoding_format="float",
            )
        else:
            response = self.client.embeddings.create(
                model=self.config.model,
                input=texts,
                encoding_format="float",
                dimensions=self.config.dimensions,
            )
        embedded_at = utc_now()
        data = sorted(response.data, key=lambda item: item.index)
        return EmbeddingBatchResult(
            vectors=[list(item.embedding) for item in data],
            embedded_at=embedded_at,
        )

    def _batches(self, chunks: list[PreparedChunk]) -> Iterable[list[PreparedChunk]]:
        """Формирует batch с ограничением и по числу входов, и по токенам."""
        batch: list[PreparedChunk] = []
        batch_tokens = 0
        for chunk in chunks:
            token_count = self._token_count(chunk.text)
            if token_count > self.config.max_input_tokens:
                raise ValueError(
                    (
                        "Чанк превышает token limit модели embeddings: "
                        f"id={chunk.metadata.id} tokens={token_count} "
                        f"limit={self.config.max_input_tokens}"
                    )
                )

            would_exceed_size = len(batch) >= self.config.batch_size
            # Общий token budget защищает API даже тогда, когда каждый отдельный
            # чанк укладывается в max_input_tokens.
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

    def _token_count(self, text: str) -> int:
        """Гарантирует точный подсчёт токенов текста согласно выбранному токенизатору модели."""
        return len(self.encoding.encode(text))

    @staticmethod
    def ensure_api_key(config: EmbeddingConfig) -> None:
        """Проверяет наличие и корректность API-ключа для предотвращения ошибок при обращении к OpenAI."""
        resolve_openai_api_key(config)

    @contextmanager
    def _openai_client_env(self):
        """Временно меняет proxy-env только во время создания HTTP-клиента."""
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
    """Удаляет случайные кавычки и префикс имени из значения env-переменной."""
    cleaned = value.strip().strip("\"'")
    if "=" in cleaned and cleaned.startswith(env_name):
        cleaned = cleaned.split("=", 1)[1].strip().strip("\"'")
    return cleaned


class LocalEmbeddingStage(EmbeddingRecordMixin):
    """Считает локальные embeddings через transformers encoder model."""

    def __init__(self, config: EmbeddingConfig):
        """Готовит экземпляр к локальному вычислению embeddings, включая загрузку модели, токенизатора и выбор устройства."""
        self.config = config
        self._configure_hf_hub_downloads()
        self.device = self._select_device()
        self.dtype = self._select_dtype(self.device)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def run(self, chunks: list[PreparedChunk], *, run_id: str) -> list[EmbeddedChunk]:
        """Гарантирует получение embeddings для всех входных чанков локальной моделью с контролем соответствия размеров и логированием прогресса."""
        embedded: list[EmbeddedChunk] = []
        batches = list(self._batches(chunks))
        for batch_number, batch in enumerate(batches, start=1):
            texts = [self._passage_text(chunk.text) for chunk in batch]
            result = self._embed_texts(texts)
            if len(result.vectors) != len(batch):
                raise ValueError(
                    f"Локальная модель вернула {len(result.vectors)} embeddings "
                    f"для {len(batch)} входов"
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
                "Посчитан локальный batch embeddings %d/%d; чанков: %d",
                batch_number,
                len(batches),
                len(batch),
            )

        LOGGER.info("Посчитаны локальные embeddings для чанков: %d", len(embedded))
        return embedded

    def _load_tokenizer(self) -> Any:
        """Гарантирует загрузку токенизатора, совместимого с выбранной моделью и политикой доверия к коду."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )

    def _load_model(self) -> Any:
        """Гарантирует загрузку и перевод embedding-модели на выбранное устройство с fallback на CPU при ошибках, обеспечивая готовность к инференсу."""
        from transformers import AutoModel

        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype

        model = AutoModel.from_pretrained(self.config.model, **kwargs)
        try:
            model = model.to(self.device)
        except Exception as exc:
            if self.device == "cpu":
                raise
            LOGGER.warning(
                "Не удалось перенести embedding-модель на %s: %s. Используется CPU.",
                self.device,
                exc,
            )
            self.device = "cpu"
            model = model.to("cpu")
        model.eval()
        return model

    def _embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Кодирует batch локально и применяет выбранный pooling."""
        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            vectors = self._pool(outputs, inputs["attention_mask"])
            if self.config.normalize:
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return EmbeddingBatchResult(
            vectors=vectors.detach().float().cpu().tolist(),
            embedded_at=utc_now(),
        )

    def embed_query(self, text: str) -> list[float]:
        """Гарантирует получение единственного embedding для запроса с учётом префикса, обеспечивая совместимость с локальной моделью."""
        query_text = (
            f"{self.config.query_prefix}{text}" if self.config.query_prefix else text
        )
        result = self._embed_texts([query_text])
        if len(result.vectors) != 1:
            raise ValueError(
                "Локальная модель должна вернуть ровно один query embedding"
            )
        return result.vectors[0]

    def _pool(self, outputs: Any, attention_mask: Any) -> Any:
        """Получает CLS либо mean pooling только по незаполненным padding токенам."""
        if self.config.pooling == "cls":
            return outputs.last_hidden_state[:, 0]

        token_embeddings = outputs.last_hidden_state
        # attention_mask исключает padding: простое среднее по длине batch-тензора
        # сместило бы короткие тексты в сторону нулевых представлений.
        mask = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(token_embeddings.dtype)
        )
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _batches(self, chunks: list[PreparedChunk]) -> Iterable[list[PreparedChunk]]:
        """Разбивает входные чанки на батчи, не превышающие ограничения по размеру и количеству токенов, чтобы избежать ошибок при обработке локальной моделью."""
        batch: list[PreparedChunk] = []
        batch_tokens = 0
        for chunk in chunks:
            token_count = self._token_count(self._passage_text(chunk.text))
            if token_count > self.config.max_input_tokens:
                raise ValueError(
                    (
                        "Чанк превышает token limit локальной embeddings-модели: "
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

    def _token_count(self, text: str) -> int:
        """Гарантирует точный подсчёт токенов в тексте согласно используемому токенизатору для соблюдения лимитов модели."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _passage_text(self, text: str) -> str:
        """Добавляет passage-префикс, ожидаемый асимметричными E5-моделями."""
        return (
            f"{self.config.passage_prefix}{text}"
            if self.config.passage_prefix
            else text
        )

    def _configure_hf_hub_downloads(self) -> None:
        """Обеспечивает корректную настройку переменных окружения для загрузки моделей HuggingFace в соответствии с политикой безопасности и совместимости."""
        if self.config.hub_disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if self.config.hub_disable_symlink_warning:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    def _select_device(self) -> str:
        """Выбирает XPU, затем CUDA и в последнюю очередь CPU."""
        import torch

        if self.config.local_device != "auto":
            return self.config.local_device
        if torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _select_dtype(self, device: str) -> Any:
        """Сопоставляет переносимый dtype конфига с типом PyTorch."""
        import torch

        dtype = self.config.local_dtype
        if dtype == "auto":
            dtype = "bf16" if device in {"xpu", "cuda"} else "fp32"
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "fp16":
            return torch.float16
        if dtype == "fp32":
            return torch.float32
        raise ValueError(f"Неизвестный dtype для локальных embeddings: {dtype}")


class GigaChatEmbeddingStage(EmbeddingRecordMixin):
    """Считает GigaChat embeddings для подготовленных чанков."""

    def __init__(self, config: EmbeddingConfig):
        """Готовит экземпляр к работе с GigaChat API, валидируя зависимости и аутентификацию для безопасного получения embeddings."""
        self.config = config
        credentials = resolve_gigachat_auth_key(config)
        try:
            from langchain_gigachat.embeddings import GigaChatEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Пакет langchain-gigachat не установлен. Выполните: "
                "python -m pip install langchain-gigachat gigachat"
            ) from exc

        self.client = GigaChatEmbeddings(
            credentials=credentials,
            scope=config.gigachat_scope,
            model=config.model,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            verify_ssl_certs=config.gigachat_verify_ssl_certs,
            ca_bundle_file=resolve_gigachat_ca_bundle(),
            prefix_query=config.gigachat_prefix_query,
            use_prefix_query=config.gigachat_use_prefix_query,
        )

    def run(self, chunks: list[PreparedChunk], *, run_id: str) -> list[EmbeddedChunk]:
        """Гарантирует получение embeddings для всех чанков с логированием прогресса и строгой проверкой соответствия входов и выходов."""
        embedded: list[EmbeddedChunk] = []
        batches = list(self._batches(chunks))
        for batch_number, batch in enumerate(batches, start=1):
            texts = [chunk.text for chunk in batch]
            result = self._embed_texts(texts)
            if len(result.vectors) != len(batch):
                raise ValueError(
                    f"GigaChat вернул {len(result.vectors)} embeddings "
                    f"для {len(batch)} входов"
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
                "Посчитан GigaChat batch embeddings %d/%d; чанков: %d",
                batch_number,
                len(batches),
                len(batch),
            )

        LOGGER.info("Посчитаны GigaChat embeddings для чанков: %d", len(embedded))
        return embedded

    def _embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Выполняет GigaChat-запрос с ограниченным retry временных ошибок."""
        try:
            from gigachat.exceptions import GigaChatException
        except ImportError:
            GigaChatException = Exception  # type: ignore[assignment]

        retryer = Retrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_exception_type(
                (GigaChatException, TimeoutError, ConnectionError)
            ),
            reraise=True,
        )
        return retryer(self._request_embeddings, texts)

    def _request_embeddings(self, texts: list[str]) -> EmbeddingBatchResult:
        """Гарантирует получение embeddings для заданных текстов с фиксацией времени генерации для последующей трассировки."""
        vectors = self.client.embed_documents(texts)
        return EmbeddingBatchResult(
            vectors=[[float(value) for value in vector] for vector in vectors],
            embedded_at=utc_now(),
        )

    def embed_query(self, text: str) -> list[float]:
        """Гарантирует получение embedding для запроса через GigaChat API с приведением к числовому формату."""
        vector = self.client.embed_query(text)
        return self._query_vector(vector)

    def _batches(self, chunks: list[PreparedChunk]) -> Iterable[list[PreparedChunk]]:
        """Разбивает входные чанки на батчи, строго соблюдая лимиты GigaChat по размеру и количеству токенов для предотвращения ошибок API."""
        batch: list[PreparedChunk] = []
        batch_tokens = 0
        for chunk in chunks:
            token_count = self._token_count(chunk.text)
            if token_count > self.config.max_input_tokens:
                raise ValueError(
                    (
                        "Чанк превышает token limit GigaChat embeddings-модели: "
                        f"id={chunk.metadata.id} tokens~={token_count} "
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

    def _token_count(self, text: str) -> int:
        """Оценивает токены по символам, когда provider не публикует tokenizer."""
        return max(1, math.ceil(len(text) / self.config.gigachat_chars_per_token))

    @staticmethod
    def ensure_api_key(config: EmbeddingConfig) -> None:
        """Проверяет наличие и корректность ключа доступа к GigaChat, предотвращая ошибки аутентификации при запуске."""
        resolve_gigachat_auth_key(config)


def build_embedding_stage(
    config: EmbeddingConfig,
) -> OpenAIEmbeddingStage | LocalEmbeddingStage | GigaChatEmbeddingStage:
    """Создаёт stage строго для явно выбранного embedding-провайдера."""
    if config.provider == "openai":
        return OpenAIEmbeddingStage(config)
    if config.provider == "local":
        return LocalEmbeddingStage(config)
    if config.provider == "gigachat":
        return GigaChatEmbeddingStage(config)
    raise ValueError(f"Неизвестный provider embeddings: {config.provider}")


def ensure_embedding_runtime(config: EmbeddingConfig) -> None:
    """Гарантирует, что для выбранного провайдера эмбеддингов доступны все необходимые ключи доступа перед запуском вычислений."""
    if config.provider == "openai":
        OpenAIEmbeddingStage.ensure_api_key(config)
    if config.provider == "gigachat":
        GigaChatEmbeddingStage.ensure_api_key(config)
