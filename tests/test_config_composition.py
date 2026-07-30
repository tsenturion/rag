"""Регрессионные тесты для подсистемы config_composition."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_prep.config_composition import (  # noqa: E402
    apply_rag_profile,
    load_composed_yaml,
)


class ConfigCompositionTest(unittest.TestCase):
    """Проверяет корректность композиции конфигураций с учётом наследования, приоритетов и циклов для обеспечения надёжной загрузки настроек."""

    def test_extends_deep_merges_mappings_and_replaces_lists(self) -> None:
        """Проверяет, что при наследовании конфигураций словари сливаются рекурсивно, а списки полностью заменяются значениями потомка."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "base.yaml").write_text(
                "section:\n  inherited: true\n  value: base\nitems:\n  - base\n",
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                "extends: base.yaml\nsection:\n  value: child\nitems:\n  - child\n",
                encoding="utf-8",
            )

            result = load_composed_yaml(root / "child.yaml")

        self.assertEqual(
            result,
            {
                "section": {"inherited": True, "value": "child"},
                "items": ["child"],
            },
        )

    def test_later_parent_and_local_values_have_priority(self) -> None:
        """Проверяет, что при множественном наследовании приоритет имеют значения из более поздних родителей и локальные значения."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "first.yaml").write_text("value: first\n", encoding="utf-8")
            (root / "second.yaml").write_text("value: second\n", encoding="utf-8")
            (root / "result.yaml").write_text(
                "extends:\n  - first.yaml\n  - second.yaml\nlocal: true\n",
                encoding="utf-8",
            )

            result = load_composed_yaml(root / "result.yaml")

        self.assertEqual(result, {"value": "second", "local": True})

    def test_extends_cycle_is_rejected(self) -> None:
        """Проверяет, что при обнаружении циклической зависимости в наследовании конфигураций выбрасывается ошибка."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
            (root / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "цикл extends"):
                load_composed_yaml(root / "a.yaml")

    def test_rag_profile_rejects_dimension_mismatch(self) -> None:
        """Проверяет, что при применении RAG-профиля с несовпадающими размерностями эмбеддингов возникает ошибка валидации."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "profile.yaml").write_text(
                "tokenizer_model: test\n"
                "embedding:\n"
                "  model: test\n"
                "  dimensions: 3\n"
                "vector_store:\n"
                "  collection_name: test\n"
                "  vector_size: 4\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "rag_profile: profile.yaml\nembedding: {}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Размерности.*не совпадают"):
                apply_rag_profile(
                    load_composed_yaml(config_path),
                    config_path=config_path,
                    target="embedding",
                )


if __name__ == "__main__":
    unittest.main()
