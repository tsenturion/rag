from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from rag_prep.config import LoaderConfig
from rag_prep.models import SourceFile
from rag_prep.utils import file_sha256

LOGGER = logging.getLogger(__name__)


class LlamaIndexLoadingStage:
    """Находит поддерживаемые файлы через directory reader из LlamaIndex."""

    def __init__(self, config: LoaderConfig):
        self.config = config

    def run(self, input_dir: Path) -> list[SourceFile]:
        if not input_dir.exists():
            raise FileNotFoundError(f"Входная директория не существует: {input_dir}")

        try:
            reader = SimpleDirectoryReader(
                input_dir=str(input_dir),
                recursive=self.config.recursive,
                required_exts=self.config.allowed_extensions,
                exclude_hidden=self.config.exclude_hidden,
                num_files_limit=self.config.num_files_limit,
                filename_as_id=True,
            )
            files = [Path(path) for path in reader.input_files]
        except ValueError as exc:
            if "No files found" not in str(exc):
                raise
            files = []

        sources = [self._to_source_file(path) for path in sorted(files)]
        LOGGER.info("Загружено исходных файлов: %d из %s", len(sources), input_dir)
        return sources

    def _to_source_file(self, path: Path) -> SourceFile:
        stat = path.stat()
        return SourceFile(
            path=path.resolve(),
            source=str(path.resolve()),
            file_name=path.name,
            file_type=path.suffix.lower().lstrip("."),
            source_hash=file_sha256(path),
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

