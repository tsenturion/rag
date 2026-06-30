from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
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
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except Exception as exc:
        return f"ошибка калькулятора: {exc}"
    return str(result)


def _eval_node(node: ast.AST) -> Any:
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
    return StructuredTool.from_function(
        name="calculator",
        description=(
            "Используй этот tool для точной арифметики. На входе должно быть "
            "математическое выражение только с числами и операторами."
        ),
        func=calculate,
        args_schema=CalculatorInput,
    )
