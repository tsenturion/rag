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


def load_config(path: str | Path) -> PipelineConfig:
    config_path = _resolve_config_path(path)
    base_dir = _config_base_dir(config_path)
    load_dotenv(base_dir / ".env")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    config = PipelineConfig.model_validate(raw)
    return _resolve_paths(config, base_dir=base_dir)


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
