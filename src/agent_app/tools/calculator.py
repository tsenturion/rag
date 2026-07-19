"""Безопасные арифметические инструменты для инструментов агента."""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Обеспечивает проверку математического выражения, гарантируя корректность формата для последующих вычислений."""

    expression: str = Field(description="Математическое выражение, например: 128 * 47")


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression: str) -> str:
    """Гарантирует безопасное вычисление арифметического выражения с обработкой ошибок для CLI и инструментов агента."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except Exception as exc:
        return f"ошибка калькулятора: {exc}"
    return str(result)


def _eval_node(node: ast.AST) -> Any:
    """Гарантирует корректную и безопасную рекурсивную обработку AST арифметических выражений без сторонних эффектов."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"неподдерживаемый узел выражения: {type(node).__name__}")


def calculator_tool() -> StructuredTool:
    """Гарантирует регистрацию калькулятора как структурированного инструмента с контрактом безопасного вычисления выражений."""
    return StructuredTool.from_function(
        name="calculator",
        description=(
            "Используй этот tool для точной арифметики. На входе должно быть "
            "математическое выражение только с числами и операторами."
        ),
        func=calculate,
        args_schema=CalculatorInput,
    )
