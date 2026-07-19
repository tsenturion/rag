"""Конфигурационные модели и загрузка настроек для RAG-конвейера."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_prep.config_composition import apply_rag_profile, load_composed_yaml
from rag_prep.mlflow_uri import resolve_mlflow_tracking_uri


class StrictConfigModel(BaseModel):
    """Запрещает неизвестные поля во всех вложенных секциях RAG-конфигурации."""

    model_config = ConfigDict(extra="forbid")


# Общие параметры первого этапа подготовки данных.
class RunConfig(StrictConfigModel):
    """Гарантирует воспроизводимость и идентифицируемость запуска пайплайна за счёт фиксированных имени и seed."""

    name: str = "rag_data_preparation"
    seed: int = 42


class PathConfig(StrictConfigModel):
    """Обеспечивает согласованное размещение входных и выходных данных пайплайна независимо от окружения."""

    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/prepared")
    json_filename: str = "documents.json"
    jsonl_filename: str = "documents.jsonl"
    manifest_filename: str = "manifest.json"


class LoaderConfig(StrictConfigModel):
    """Гарантирует фильтрацию и нормализацию списка файлов для загрузки, исключая скрытые и неподходящие форматы."""

    recursive: bool = True
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".html", ".htm", ".csv"]
    )
    exclude_hidden: bool = True
    num_files_limit: int | None = None

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        """Гарантирует, что расширения файлов заданы в унифицированном нижнем регистре с точкой для корректной фильтрации входных данных."""
        normalized = []
        for value in values:
            ext = value.strip().lower()
            normalized.append(ext if ext.startswith(".") else f".{ext}")
        return normalized


class ParserConfig(StrictConfigModel):
    """Обеспечивает единый контракт разбора документов с контролем ошибок и поддержкой мультиязычности."""

    strategy: Literal["auto", "fast", "hi_res", "ocr_only"] = "fast"
    encoding: str = "utf-8"
    languages: list[str] = Field(default_factory=lambda: ["rus", "eng"])
    pdf_infer_table_structure: bool = False
    fail_on_error: bool = False
    skip_infer_table_types: list[str] = Field(
        default_factory=lambda: ["pdf", "jpg", "png", "heic"]
    )


class CleaningConfig(StrictConfigModel):
    """Гарантирует удаление мусорных и нерелевантных фрагментов текста по заданным правилам."""

    min_chars: int = 12
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    drop_patterns: list[str] = Field(default_factory=list)
    boilerplate_patterns: list[str] = Field(default_factory=list)


class NormalizationConfig(StrictConfigModel):
    """Обеспечивает стандартизацию текста для дальнейшей обработки и анализа, включая нормализацию юникода и статистику предложений."""

    unicode_form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKC"
    lowercase: bool = False
    spacy_language: str = "ru"
    collect_sentence_stats: bool = True


class DeduplicationConfig(StrictConfigModel):
    """Гарантирует удаление дублирующихся текстов по настраиваемому порогу схожести и минимальному размеру."""

    enabled: bool = True
    threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    num_perm: int = Field(default=128, ge=16)
    shingle_size: int = Field(default=5, ge=1)
    min_tokens: int = Field(default=8, ge=1)


class StructuringConfig(StrictConfigModel):
    """Обеспечивает структурирование данных по секциям для унификации downstream-процессов."""

    group_by_section: bool = True
    default_section: str = "full_document"


class LoggingConfig(StrictConfigModel):
    """Гарантирует трассировку и аудит выполнения пайплайна с возможностью интеграции с MLflow."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment: str = "rag-data-preparation"


class PipelineConfig(StrictConfigModel):
    """Объединяет настройки подготовки данных и запрещает неизвестные параметры конфигурации."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    structuring: StructuringConfig = Field(default_factory=StructuringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# Параметры chunking отделены от подготовки, но используют тот же run/logging contract.
class ChunkingPathConfig(StrictConfigModel):
    """Гарантирует, что все пути для входных и выходных файлов этапа нарезки RAG-конвейера заданы явно и доступны вызывающему коду."""

    input_jsonl: Path
    output_dir: Path
    json_filename: str = "chunks.json"
    jsonl_filename: str = "chunks.jsonl"
    manifest_filename: str = "manifest.json"


class ChunkingConfig(StrictConfigModel):
    """Обеспечивает согласованность параметров нарезки текста и контроль инвариантов размера и перекрытия чанков для корректной работы пайплайна."""

    strategy: Literal["sentence", "token"] = "sentence"
    # Размер и overlap измеряются одним tokenizer_model; иначе metadata и фактические
    # ограничения следующей embedding-модели расходились бы.
    chunk_size: int = Field(default=220, ge=32)
    chunk_overlap: int = Field(default=40, ge=0)
    tokenizer_model: str
    embedding_model: str
    preserve_section_boundaries: bool = True
    preserve_block_boundaries: bool = True
    min_chunk_tokens: int = Field(default=20, ge=1)
    max_chunk_tokens: int = Field(default=280, ge=1)
    min_quality_score: float = Field(default=0.2, ge=0.0, le=1.0)
    fail_on_validation_error: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, value: int, info) -> int:
        """Гарантирует, что перекрытие чанков всегда строго меньше размера чанка, предотвращая некорректную нарезку."""
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        return value


class ChunkingPipelineConfig(StrictConfigModel):
    """Гарантирует целостную и валидированную конфигурацию запуска этапа нарезки, включая пути, параметры нарезки и настройки логирования."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=lambda: RunConfig(name="rag_chunking"))
    paths: ChunkingPathConfig
    chunking: ChunkingConfig
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(mlflow_experiment="rag-chunking")
    )


class EmbeddingPathConfig(StrictConfigModel):
    """Гарантирует, что все пути для входных и выходных файлов этапа эмбеддинга RAG-конвейера заданы явно и доступны вызывающему коду."""

    input_jsonl: Path
    output_dir: Path
    json_filename: str = "embeddings.json"
    jsonl_filename: str = "embeddings.jsonl"
    manifest_filename: str = "manifest.json"


class EmbeddingConfig(StrictConfigModel):
    """Параметры одного явно выбранного provider embeddings."""

    provider: Literal["openai", "local", "gigachat"]
    model: str
    # None означает provider-defined размерность; итог всё равно проверяется после API.
    dimensions: int | None = Field(ge=1)
    api_key_env: str
    env_file: Path | None = None
    batch_size: int = Field(default=64, ge=1)
    max_batch_tokens: int = Field(default=20000, ge=1)
    max_input_tokens: int = Field(default=8191, ge=1)
    max_retries: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    normalize: bool = False
    clear_no_proxy_for_openai: bool = True
    local_device: Literal["auto", "xpu", "cuda", "cpu"] = "auto"
    local_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = False
    pooling: Literal["mean", "cls"] = "mean"
    # Раздельные префиксы обязательны для retrieval-моделей семейства E5.
    passage_prefix: str = "passage: "
    query_prefix: str = "query: "
    hub_disable_xet: bool = True
    hub_disable_symlink_warning: bool = True
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_verify_ssl_certs: bool = True
    gigachat_use_prefix_query: bool = False
    gigachat_prefix_query: str = (
        "Дано предложение, необходимо найти его парафраз \nпредложение: "
    )
    gigachat_chars_per_token: int = Field(default=3, ge=1)
    fail_on_validation_error: bool = True


class EmbeddingPipelineConfig(StrictConfigModel):
    """Обеспечивает валидированную и самодостаточную конфигурацию запуска этапа эмбеддинга, включая пути, параметры эмбеддинга и логирование."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=lambda: RunConfig(name="rag_embeddings"))
    paths: EmbeddingPathConfig
    embedding: EmbeddingConfig
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(mlflow_experiment="rag-embeddings")
    )


# Размерности зафиксированы на уровне конфигурации, чтобы несовместимость с Qdrant
# обнаруживалась до расчёта и индексации векторов.
GIGACHAT_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "embeddings": 1024,
    "embeddings-2": 1024,
    "embeddingsgigar": 2560,
    "embeddings-3b-2025-09": 2048,
    "gigaembeddings-3b-2025-09": 2048,
}


class VectorStorePathConfig(StrictConfigModel):
    """Гарантирует, что все пути для файлов индексации, валидации и поиска векторного хранилища заданы явно и доступны вызывающему коду."""

    input_jsonl: Path
    output_dir: Path
    manifest_filename: str = "manifest.json"
    validation_filename: str = "validation.json"
    search_results_filename: str = "search_results.json"


class VectorStoreConfig(StrictConfigModel):
    """Параметры Qdrant, согласованные с размерностью embedding-профиля."""

    provider: Literal["qdrant"] = "qdrant"
    mode: Literal["local", "http"] = "local"
    collection_name: str
    vector_size: int = Field(ge=1)
    distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"] = "Cosine"
    recreate_collection: bool = False
    # Коллекция является materialized snapshot входного embeddings.jsonl. После
    # upsert удаляются только точки, ID которых отсутствуют в текущем наборе.
    prune_stale_points: bool = True
    batch_size: int = Field(default=128, ge=1)
    local_storage_path: Path
    host: str = "localhost"
    port: int = Field(default=6333, ge=1, le=65535)
    https: bool = False
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    search_limit: int = Field(default=5, ge=1)
    test_queries_count: int = Field(default=3, ge=0)
    score_threshold: float | None = None
    validation_sample_size: int = Field(default=1000, ge=1)
    fail_on_validation_error: bool = True

    @field_validator("score_threshold")
    @classmethod
    def score_threshold_must_be_positive(cls, value: float | None) -> float | None:
        """Гарантирует, что пороговое значение для поиска в векторном индексе неотрицательно, предотвращая логические ошибки."""
        if value is not None and value < 0.0:
            raise ValueError("score_threshold должен быть >= 0.0")
        return value


class VectorStorePipelineConfig(StrictConfigModel):
    """Обеспечивает валидированную и целостную конфигурацию запуска этапа индексации векторного хранилища, включая пути, параметры и логирование."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=lambda: RunConfig(name="rag_vector_store"))
    paths: VectorStorePathConfig
    vector_store: VectorStoreConfig
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(mlflow_experiment="rag-vector-store")
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Создаёт воспроизводимую и полностью валидированную конфигурацию пайплайна независимо от текущего рабочего каталога и окружения."""
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    raw = load_composed_yaml(config_path)
    config = PipelineConfig.model_validate(raw)
    return _resolve_logging_tracking_uri(
        _resolve_paths(config, base_dir=base_dir),
        base_dir=base_dir,
    )


def load_chunking_config(path: str | Path) -> ChunkingPipelineConfig:
    """Гарантирует корректную загрузку и валидацию конфигурации этапа нарезки с учётом профиля и разрешением всех путей относительно базовой директории."""
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    raw = apply_rag_profile(
        load_composed_yaml(config_path),
        config_path=config_path,
        target="chunking",
    )
    config = ChunkingPipelineConfig.model_validate(raw)
    return _resolve_logging_tracking_uri(
        _resolve_chunking_paths(config, base_dir=base_dir),
        base_dir=base_dir,
    )


def load_embedding_config(path: str | Path) -> EmbeddingPipelineConfig:
    """Загружает embedding-конфиг, env и provider-зависимые инварианты."""
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    raw = apply_rag_profile(
        load_composed_yaml(config_path),
        config_path=config_path,
        target="embedding",
    )
    config = EmbeddingPipelineConfig.model_validate(raw)
    env_file = config.embedding.env_file or base_dir / ".env"
    if not env_file.is_absolute():
        env_file = base_dir / env_file
    env_file = env_file.resolve()
    load_dotenv(env_file)
    embedding_update: dict[str, Any] = {"env_file": env_file}
    if config.embedding.provider == "local":
        embedding_update["model"] = _resolve_local_model_reference(
            config.embedding.model,
            base_dir=base_dir,
        )
    if config.embedding.provider == "gigachat":
        # Пользователь может задать размерность явно, но известное значение модели
        # остаётся проверяемым контрактом, а не молчаливой подсказкой.
        known_dimensions = _gigachat_embedding_dimensions(config.embedding.model)
        if known_dimensions is not None:
            if config.embedding.dimensions is None:
                embedding_update["dimensions"] = known_dimensions
            elif config.embedding.dimensions != known_dimensions:
                raise ValueError(
                    (
                        "Размерность GigaChat embeddings не соответствует модели: "
                        f"model={config.embedding.model} "
                        f"dimensions={config.embedding.dimensions} "
                        f"expected={known_dimensions}"
                    )
                )
    config = config.model_copy(
        update={"embedding": config.embedding.model_copy(update=embedding_update)}
    )
    return _resolve_logging_tracking_uri(
        _resolve_embedding_paths(config, base_dir=base_dir),
        base_dir=base_dir,
    )


def load_vector_store_config(path: str | Path) -> VectorStorePipelineConfig:
    """Гарантирует корректную загрузку и валидацию конфигурации этапа индексации векторного хранилища с учётом профиля и разрешением всех путей."""
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    raw = apply_rag_profile(
        load_composed_yaml(config_path),
        config_path=config_path,
        target="vector_store",
    )
    config = VectorStorePipelineConfig.model_validate(raw)
    return _resolve_logging_tracking_uri(
        _resolve_vector_store_paths(config, base_dir=base_dir),
        base_dir=base_dir,
    )


def _resolve_config_path(path: str | Path) -> Path:
    """Гарантирует, что путь к конфигурационному файлу будет найден относительно проекта или пользователя и приведён к абсолютному виду."""
    config_path = Path(path).expanduser()
    if config_path.is_absolute() or config_path.exists():
        return config_path.resolve()

    project_root = Path(__file__).resolve().parents[2]
    project_config_path = project_root / config_path
    if project_config_path.exists():
        return project_config_path.resolve()

    return config_path.resolve()


def _config_base_dir(config_path: Path) -> Path:
    """Определяет корень проекта для стандартной папки ``config``."""
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_paths(config: PipelineConfig, base_dir: Path) -> PipelineConfig:
    """Гарантирует, что все пути входных и выходных данных пайплайна будут абсолютными и независимыми от текущего каталога."""
    paths = config.paths
    input_dir = (
        paths.input_dir if paths.input_dir.is_absolute() else base_dir / paths.input_dir
    )
    output_dir = (
        paths.output_dir
        if paths.output_dir.is_absolute()
        else base_dir / paths.output_dir
    )
    return config.model_copy(
        update={
            "paths": paths.model_copy(
                update={
                    "input_dir": input_dir.resolve(),
                    "output_dir": output_dir.resolve(),
                }
            )
        }
    )


def _resolve_chunking_paths(
    config: ChunkingPipelineConfig, base_dir: Path
) -> ChunkingPipelineConfig:
    """Гарантирует, что пути к входным данным и выходной директории для этапа нарезки всегда абсолютны и корректны для запуска пайплайна."""
    paths = config.paths
    input_jsonl = (
        paths.input_jsonl
        if paths.input_jsonl.is_absolute()
        else base_dir / paths.input_jsonl
    )
    output_dir = (
        paths.output_dir
        if paths.output_dir.is_absolute()
        else base_dir / paths.output_dir
    )
    return config.model_copy(
        update={
            "paths": paths.model_copy(
                update={
                    "input_jsonl": input_jsonl.resolve(),
                    "output_dir": output_dir.resolve(),
                }
            )
        }
    )


def _resolve_embedding_paths(
    config: EmbeddingPipelineConfig, base_dir: Path
) -> EmbeddingPipelineConfig:
    """Гарантирует, что пути к входным данным и выходной директории для этапа эмбеддинга всегда абсолютны и корректны для запуска пайплайна."""
    paths = config.paths
    input_jsonl = (
        paths.input_jsonl
        if paths.input_jsonl.is_absolute()
        else base_dir / paths.input_jsonl
    )
    output_dir = (
        paths.output_dir
        if paths.output_dir.is_absolute()
        else base_dir / paths.output_dir
    )
    return config.model_copy(
        update={
            "paths": paths.model_copy(
                update={
                    "input_jsonl": input_jsonl.resolve(),
                    "output_dir": output_dir.resolve(),
                }
            )
        }
    )


def _resolve_vector_store_paths(
    config: VectorStorePipelineConfig, base_dir: Path
) -> VectorStorePipelineConfig:
    """Гарантирует, что все пути к данным и локальному хранилищу векторного индекса приведены к абсолютному виду и согласованы с базовой директорией."""
    paths = config.paths
    vector_store = config.vector_store
    input_jsonl = (
        paths.input_jsonl
        if paths.input_jsonl.is_absolute()
        else base_dir / paths.input_jsonl
    )
    output_dir = (
        paths.output_dir
        if paths.output_dir.is_absolute()
        else base_dir / paths.output_dir
    )
    local_storage_path = (
        vector_store.local_storage_path
        if vector_store.local_storage_path.is_absolute()
        else base_dir / vector_store.local_storage_path
    )
    return config.model_copy(
        update={
            "paths": paths.model_copy(
                update={
                    "input_jsonl": input_jsonl.resolve(),
                    "output_dir": output_dir.resolve(),
                }
            ),
            "vector_store": vector_store.model_copy(
                update={"local_storage_path": local_storage_path.resolve()}
            ),
        }
    )


def _resolve_local_model_reference(model: str, *, base_dir: Path) -> str:
    """Разрешает локальный путь модели, не изменяя Hugging Face model ID."""
    path = Path(model).expanduser()
    if path.is_absolute():
        return str(path.resolve()) if path.exists() else model

    candidate = base_dir / path
    if candidate.exists():
        return str(candidate.resolve())

    looks_like_path = (
        model.startswith(".")
        or model.startswith("data/")
        or model.startswith("data\\")
        or "\\" in model
    )
    if looks_like_path:
        return str(candidate.resolve())
    return model


def _resolve_logging_tracking_uri(config: Any, *, base_dir: Path) -> Any:
    """Гарантирует, что URI для отслеживания логов MLflow приведён к абсолютному виду и корректно интегрирован в конфигурацию."""
    logging_config = config.logging.model_copy(
        update={
            "mlflow_tracking_uri": resolve_mlflow_tracking_uri(
                config.logging.mlflow_tracking_uri,
                base_dir=base_dir,
            )
        }
    )
    return config.model_copy(update={"logging": logging_config})


def _gigachat_embedding_dimensions(model: str) -> int | None:
    """Возвращает известную размерность GigaChat без учёта регистра имени."""
    return GIGACHAT_EMBEDDING_DIMENSIONS.get(model.strip().lower())
