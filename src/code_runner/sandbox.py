"""Минимальный RestrictedPython runtime без файловой системы и process API."""

from __future__ import annotations

import collections
import datetime
import decimal
import fractions
import functools
import itertools
import json
import math
import operator
import re
import statistics
import sys
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Any

from RestrictedPython import (
    compile_restricted,
    limited_builtins,
    safe_builtins,
    utility_builtins,
)
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector


class SafeModule:
    """Показывает пользовательскому коду только явно разрешённые API модуля."""

    def __init__(self, name: str, exports: dict[str, Any]):
        """Фиксирует неизменяемый набор публичных объектов разрешённого модуля."""
        self.name = name
        self.exports = MappingProxyType(exports)

    def __getattr__(self, name: str) -> Any:
        """Возвращает только явно экспортированный объект без доступа к module globals."""
        try:
            return self.exports[name]
        except KeyError as exc:
            raise AttributeError(f"{self.name}.{name} не разрешён") from exc


def _exports(module: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Извлекает фиксированный allowlist без транзитивных module attributes."""
    return {name: getattr(module, name) for name in names}


SAFE_MODULES = {
    "math": SafeModule(
        "math",
        {name: getattr(math, name) for name in dir(math) if not name.startswith("_")},
    ),
    "statistics": SafeModule(
        "statistics",
        _exports(
            statistics,
            (
                "StatisticsError",
                "NormalDist",
                "correlation",
                "covariance",
                "fmean",
                "geometric_mean",
                "harmonic_mean",
                "linear_regression",
                "mean",
                "median",
                "median_grouped",
                "median_high",
                "median_low",
                "mode",
                "multimode",
                "pstdev",
                "pvariance",
                "quantiles",
                "stdev",
                "variance",
            ),
        ),
    ),
    "json": SafeModule("json", _exports(json, ("JSONDecodeError", "dumps", "loads"))),
    "re": SafeModule(
        "re",
        _exports(
            re,
            (
                "A",
                "ASCII",
                "I",
                "IGNORECASE",
                "M",
                "MULTILINE",
                "S",
                "DOTALL",
                "X",
                "VERBOSE",
                "compile",
                "escape",
                "findall",
                "finditer",
                "fullmatch",
                "match",
                "search",
                "split",
                "sub",
                "subn",
            ),
        ),
    ),
    "datetime": SafeModule(
        "datetime",
        _exports(datetime, ("date", "datetime", "time", "timedelta", "timezone")),
    ),
    "decimal": SafeModule(
        "decimal",
        _exports(
            decimal,
            ("Decimal", "DecimalException", "Context", "getcontext", "localcontext"),
        ),
    ),
    "fractions": SafeModule("fractions", _exports(fractions, ("Fraction",))),
    "collections": SafeModule(
        "collections",
        _exports(collections, ("Counter", "OrderedDict", "defaultdict", "deque")),
    ),
    "itertools": SafeModule(
        "itertools",
        _exports(
            itertools,
            (
                "accumulate",
                "chain",
                "combinations",
                "combinations_with_replacement",
                "compress",
                "count",
                "cycle",
                "dropwhile",
                "filterfalse",
                "groupby",
                "islice",
                "pairwise",
                "permutations",
                "product",
                "repeat",
                "starmap",
                "takewhile",
                "tee",
                "zip_longest",
            ),
        ),
    ),
    "functools": SafeModule("functools", _exports(functools, ("partial", "reduce"))),
}


def _safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> SafeModule:
    """Возвращает proxy разрешённого модуля и запрещает relative imports."""
    del globals_, locals_, fromlist
    if level != 0 or name not in SAFE_MODULES:
        raise ImportError(f"Импорт {name!r} не разрешён")
    return SAFE_MODULES[name]


def _inplacevar(operator_name: str, left: Any, right: Any) -> Any:
    """Выполняет только известные арифметические inplace-операции."""
    operations = {
        "+=": operator.add,
        "-=": operator.sub,
        "*=": operator.mul,
        "/=": operator.truediv,
        "//=": operator.floordiv,
        "%=": operator.mod,
        "**=": operator.pow,
    }
    try:
        return operations[operator_name](left, right)
    except KeyError as exc:
        raise TypeError(f"Операция {operator_name} не разрешена") from exc


def execute(source: str) -> str:
    """Компилирует и выполняет код в restricted globals, возвращая print output."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        bytecode = compile_restricted(source, "<agent-code>", "exec")
    builtins = {
        **safe_builtins,
        **limited_builtins,
        **utility_builtins,
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "hex": hex,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "__name__": "__main__",
        "_print_": PrintCollector,
        "_getattr_": safer_getattr,
        "_getitem_": operator.getitem,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_write_": full_write_guard,
        "_inplacevar_": _inplacevar,
    }
    exec(bytecode, namespace)
    collector = namespace.get("_print")
    return collector() if collector is not None else ""


def main() -> int:
    """Читает source-файл, выполняет sandbox и пишет только контролируемый вывод."""
    if len(sys.argv) != 2:
        print("Ожидался путь к source-файлу", file=sys.stderr)
        return 2
    try:
        output = execute(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
