"""Регрессии криптографической связи артефактов RAG-пайплайнов."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rag_prep.utils import file_sha256, verify_upstream_artifact


def test_upstream_manifest_detects_replaced_jsonl() -> None:
    """Подмена JSONL после экспорта должна останавливать следующий этап."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        artifact = root / "chunks.jsonl"
        artifact.write_text('{"id":"original"}\n', encoding="utf-8")
        manifest = {
            "run_id": "f2a55fa7-8b99-41b2-9cab-78a490754148",
            # Manifest может быть создан на Windows, а проверяться внутри Linux Docker.
            "outputs": {"jsonl": r"C:\bundle\chunks.jsonl"},
            "integrity": {
                "jsonl": {
                    "sha256": file_sha256(artifact),
                    "size_bytes": artifact.stat().st_size,
                }
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        lineage = verify_upstream_artifact(artifact)
        artifact.write_text('{"id":"replaced"}\n', encoding="utf-8")

        assert lineage["run_id"] == manifest["run_id"]
        with pytest.raises(ValueError, match="Нарушена целостность"):
            verify_upstream_artifact(artifact)
