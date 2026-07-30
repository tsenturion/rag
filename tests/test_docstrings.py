"""Проверки полноты и языка docstring во всех Python-файлах проекта."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("src", "scripts", "tests")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
TECHNICAL_COMMENT_PREFIXES = ("# type:", "# noqa", "# pragma:", "#!/", "#:")
PLACEHOLDER_DOCSTRING_RE = re.compile(
    r"реализует операци|"
    r"проверяет сценарий\s+``|"
    r"набор проверок\s+``|"
    r"проверенные входные данные\s+``|"
    r"допустимые значения\s+``|"
    r"типизированная сущность\s+``|"
    r"контракт\s+``|"
    r"исполняемый этап\s+``|"
    r"хранилище\s+``|"
    r"прикладная ошибка\s+``|"
    r"и его зависимости|"
    r"инкапсулирующ",
    re.IGNORECASE,
)


def _python_files() -> list[Path]:
    """Возвращает first-party Python-файлы без generated/cache директорий."""
    files: list[Path] = []
    for root_name in PYTHON_ROOTS:
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            normalized = path.as_posix()
            if "__pycache__" not in path.parts and ".egg-info" not in normalized:
                files.append(path)
    return sorted(files)


def _documented_nodes(tree: ast.Module) -> list[tuple[str, int, ast.AST]]:
    """Возвращает module/class/function scopes, для которых обязателен docstring."""
    nodes: list[tuple[str, int, ast.AST]] = [("module", 1, tree)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append(
                (getattr(node, "name", type(node).__name__), node.lineno, node)
            )
    return nodes


def test_every_python_scope_has_russian_docstring() -> None:
    """Требует русский docstring у модулей, классов и функций, включая tests."""
    failures: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for name, line, node in _documented_nodes(tree):
            docstring = ast.get_docstring(node)
            if not docstring:
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line} {name}: нет docstring"
                )
            elif not CYRILLIC_RE.search(docstring):
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line} {name}: docstring не на русском"
                )

    assert not failures, "\n".join(failures)


def test_docstrings_describe_behavior_instead_of_symbol_names() -> None:
    """Запрещает шаблоны, которые подменяют назначение объекта повторением имени."""
    failures: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for name, line, node in _documented_nodes(tree):
            docstring = ast.get_docstring(node) or ""
            if PLACEHOLDER_DOCSTRING_RE.search(docstring):
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line} {name}: "
                    "обнаружен фиктивный шаблон docstring"
                )
            if name != "module" and re.search(
                rf"(?<![\w]){re.escape(name)}(?![\w])",
                docstring,
            ):
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line} {name}: "
                    "docstring повторяет имя документируемого объекта"
                )

    assert not failures, "\n".join(failures)


def test_python_comments_are_in_russian() -> None:
    """Требует русский текст в обычных комментариях, кроме машинных директив."""
    failures: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8-sig")
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            comment = token.string
            if comment.startswith(TECHNICAL_COMMENT_PREFIXES):
                continue
            if not CYRILLIC_RE.search(comment):
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{token.start[0]}: "
                    "комментарий не на русском"
                )

    assert not failures, "\n".join(failures)
