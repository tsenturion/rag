from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    name: str = "rag_data_preparation"
    seed: int = 42


class PathConfig(BaseModel):
    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/prepared")
    json_filename: str = "documents.json"
    jsonl_filename: str = "documents.jsonl"
    manifest_filename: str = "manifest.json"


class LoaderConfig(BaseModel):
    recursive: bool = True
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".html", ".htm", ".csv"]
    )
    exclude_hidden: bool = True
    num_files_limit: int | None = None

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            ext = value.strip().lower()
            normalized.append(ext if ext.startswith(".") else f".{ext}")
        return normalized


class ParserConfig(BaseModel):
    strategy: Literal["auto", "fast", "hi_res", "ocr_only"] = "fast"
    encoding: str = "utf-8"
    languages: list[str] = Field(default_factory=lambda: ["rus", "eng"])
    pdf_infer_table_structure: bool = False
    fail_on_error: bool = False
    skip_infer_table_types: list[str] = Field(
        default_factory=lambda: ["pdf", "jpg", "png", "heic"]
    )


class CleaningConfig(BaseModel):
    min_chars: int = 12
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    drop_patterns: list[str] = Field(default_factory=list)
    boilerplate_patterns: list[str] = Field(default_factory=list)


class NormalizationConfig(BaseModel):
    unicode_form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKC"
    lowercase: bool = False
    spacy_language: str = "ru"
    collect_sentence_stats: bool = True


class DeduplicationConfig(BaseModel):
    enabled: bool = True
    threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    num_perm: int = Field(default=128, ge=16)
    shingle_size: int = Field(default=5, ge=1)
    min_tokens: int = Field(default=8, ge=1)


class StructuringConfig(BaseModel):
    group_by_section: bool = True
    default_section: str = "full_document"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "rag-data-preparation"


class PipelineConfig(BaseModel):
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


class ChunkingPathConfig(BaseModel):
    input_jsonl: Path = Path("data/prepared/documents.jsonl")
    output_dir: Path = Path("data/chunks")
    json_filename: str = "chunks.json"
    jsonl_filename: str = "chunks.jsonl"
    manifest_filename: str = "manifest.json"


class ChunkingConfig(BaseModel):
    strategy: Literal["sentence", "token"] = "sentence"
    chunk_size: int = Field(default=220, ge=32)
    chunk_overlap: int = Field(default=40, ge=0)
    tokenizer_model: str = "text-embedding-3-small"
    embedding_model: str = "text-embedding-3-small"
    preserve_section_boundaries: bool = True
    preserve_block_boundaries: bool = True
    min_chunk_tokens: int = Field(default=20, ge=1)
    max_chunk_tokens: int = Field(default=280, ge=1)
    min_quality_score: float = Field(default=0.2, ge=0.0, le=1.0)
    fail_on_validation_error: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


class ChunkingPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=lambda: RunConfig(name="rag_chunking"))
    paths: ChunkingPathConfig = Field(default_factory=ChunkingPathConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(mlflow_experiment="rag-chunking")
    )


class EmbeddingPathConfig(BaseModel):
    input_jsonl: Path = Path("data/chunks/chunks.jsonl")
    output_dir: Path = Path("data/embeddings")
    json_filename: str = "embeddings.json"
    jsonl_filename: str = "embeddings.jsonl"
    manifest_filename: str = "manifest.json"


class EmbeddingConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int | None = Field(default=1536, ge=1)
    api_key_env: str = "OPENAI_API_KEY"
    env_file: Path | None = None
    batch_size: int = Field(default=64, ge=1)
    max_batch_tokens: int = Field(default=20000, ge=1)
    max_input_tokens: int = Field(default=8191, ge=1)
    max_retries: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    normalize: bool = False
    clear_no_proxy_for_openai: bool = True
    fail_on_validation_error: bool = True


class EmbeddingPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=lambda: RunConfig(name="rag_embeddings"))
    paths: EmbeddingPathConfig = Field(default_factory=EmbeddingPathConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(mlflow_experiment="rag-embeddings")
    )


class VectorStorePathConfig(BaseModel):
    input_jsonl: Path = Path("data/embeddings/embeddings.jsonl")
    output_dir: Path = Path("data/vector_store")
    manifest_filename: str = "manifest.json"
    validation_filename: str = "validation.json"
    search_results_filename: str = "search_results.json"


class VectorStoreConfig(BaseModel):
    provider: Literal["qdrant"] = "qdrant"
    mode: Literal["local", "http"] = "local"
    collection_name: str = "rag_chunks"
    vector_size: int = Field(default=1536, ge=1)
    distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"] = "Cosine"
    recreate_collection: bool = False
    batch_size: int = Field(default=128, ge=1)
    local_storage_path: Path = Path("data/qdrant_storage")
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
        if value is not None and value < 0.0:
            raise ValueError("score_threshold must be >= 0.0")
        return value


class VectorStorePipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig = Field(default_factory=lambda: RunConfig(name="rag_vector_store"))
    paths: VectorStorePathConfig = Field(default_factory=VectorStorePathConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(mlflow_experiment="rag-vector-store")
    )


def load_config(path: str | Path) -> PipelineConfig:
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    config = PipelineConfig.model_validate(raw)
    return _resolve_paths(config, base_dir=base_dir)


def load_chunking_config(path: str | Path) -> ChunkingPipelineConfig:
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    config = ChunkingPipelineConfig.model_validate(raw)
    return _resolve_chunking_paths(config, base_dir=base_dir)


def load_embedding_config(path: str | Path) -> EmbeddingPipelineConfig:
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    config = EmbeddingPipelineConfig.model_validate(raw)
    env_file = config.embedding.env_file or base_dir / ".env"
    if not env_file.is_absolute():
        env_file = base_dir / env_file
    env_file = env_file.resolve()
    load_dotenv(env_file)
    config = config.model_copy(
        update={
            "embedding": config.embedding.model_copy(
                update={"env_file": env_file}
            )
        }
    )
    return _resolve_embedding_paths(config, base_dir=base_dir)


def load_vector_store_config(path: str | Path) -> VectorStorePipelineConfig:
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    config = VectorStorePipelineConfig.model_validate(raw)
    return _resolve_vector_store_paths(config, base_dir=base_dir)


def _resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path).expanduser()
    if config_path.is_absolute() or config_path.exists():
        return config_path.resolve()

    project_root = Path(__file__).resolve().parents[2]
    project_config_path = project_root / config_path
    if project_config_path.exists():
        return project_config_path.resolve()

    return config_path.resolve()


def _config_base_dir(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_paths(config: PipelineConfig, base_dir: Path) -> PipelineConfig:
    paths = config.paths
    input_dir = paths.input_dir if paths.input_dir.is_absolute() else base_dir / paths.input_dir
    output_dir = (
        paths.output_dir if paths.output_dir.is_absolute() else base_dir / paths.output_dir
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
    paths = config.paths
    input_jsonl = (
        paths.input_jsonl if paths.input_jsonl.is_absolute() else base_dir / paths.input_jsonl
    )
    output_dir = (
        paths.output_dir if paths.output_dir.is_absolute() else base_dir / paths.output_dir
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
    paths = config.paths
    input_jsonl = (
        paths.input_jsonl if paths.input_jsonl.is_absolute() else base_dir / paths.input_jsonl
    )
    output_dir = (
        paths.output_dir if paths.output_dir.is_absolute() else base_dir / paths.output_dir
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
    paths = config.paths
    vector_store = config.vector_store
    input_jsonl = (
        paths.input_jsonl if paths.input_jsonl.is_absolute() else base_dir / paths.input_jsonl
    )
    output_dir = (
        paths.output_dir if paths.output_dir.is_absolute() else base_dir / paths.output_dir
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
