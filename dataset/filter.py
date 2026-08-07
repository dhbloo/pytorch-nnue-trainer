from __future__ import annotations

import ast
import math
import operator
from typing import Any

import numpy as np


MAX_EXPRESSION_BYTES = 4096
MAX_AST_NODES = 128
MAX_AST_DEPTH = 16
MAX_LITERAL_ITEMS = 64
MAX_INTERMEDIATE_BYTES = 64 * 1024 * 1024

_FUNCTIONS = {
    "abs": np.abs,
    "isfinite": np.isfinite,
    "isnan": np.isnan,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "logical_not": np.logical_not,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Invert: operator.invert}
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_ALLOWED_DTYPES = frozenset("biuf")


class SafeFilterEvaluator:
    def __init__(self, expression: str, arrays: dict[str, np.ndarray]):
        self.expression = expression
        self.arrays = arrays
        try:
            self.length = len(next(iter(arrays.values())))
        except StopIteration as exc:
            raise ValueError("filter data dictionary must not be empty") from exc
        self.max_elements = max(4096, 8 * self.length)
        if len(expression.encode("utf-8")) > MAX_EXPRESSION_BYTES:
            raise ValueError("filter expression exceeds 4096 UTF-8 bytes")
        try:
            self.tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid filter expression {expression!r}: {exc.msg}") from exc
        nodes = list(ast.walk(self.tree))
        if len(nodes) > MAX_AST_NODES:
            raise ValueError(f"filter expression exceeds {MAX_AST_NODES} AST nodes")
        self._check_depth(self.tree, 1)
        for name, value in arrays.items():
            if name.startswith("__"):
                raise ValueError(f"filter input name {name!r} is forbidden")
            self._check_array(value, f"input {name}", allocated=False)

    def _error(self, node: ast.AST, message: str) -> ValueError:
        return ValueError(
            f"invalid filter expression {self.expression!r}: {message} "
            f"(node {type(node).__name__})"
        )

    def _check_depth(self, node: ast.AST, depth: int) -> None:
        if depth > MAX_AST_DEPTH:
            raise self._error(node, f"AST nesting exceeds {MAX_AST_DEPTH}")
        for child in ast.iter_child_nodes(node):
            self._check_depth(child, depth + 1)

    def _check_array(self, value: Any, context: str, *, allocated: bool = True) -> Any:
        if not isinstance(value, np.ndarray):
            return value
        if value.dtype.kind not in _ALLOWED_DTYPES:
            raise ValueError(f"{context} has unsupported dtype {value.dtype}")
        if allocated and (
            value.size > self.max_elements or value.nbytes > MAX_INTERMEDIATE_BYTES
        ):
            raise ValueError(
                f"{context} exceeds intermediate allocation limit "
                f"({value.size} elements, {value.nbytes} bytes)"
            )
        return value

    def evaluate(self) -> np.ndarray:
        result = self._eval(self.tree.body)
        if not isinstance(result, np.ndarray):
            raise ValueError(
                f"filter expression {self.expression!r} returned a scalar; "
                f"expected boolean vector of length {self.length}"
            )
        if result.dtype != np.bool_:
            raise ValueError(
                f"filter expression {self.expression!r} returned {result.dtype}; expected boolean"
            )
        if result.shape != (self.length,):
            raise ValueError(
                f"filter expression {self.expression!r} returned shape {result.shape}; "
                f"expected ({self.length},)"
            )
        return result

    def _literal(self, node: ast.Constant) -> Any:
        value = node.value
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            if not -(1 << 63) <= value < (1 << 63):
                raise self._error(node, "integer literal is outside signed 64-bit range")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise self._error(node, "float literal must be finite")
            return value
        raise self._error(node, f"constant type {type(value).__name__} is forbidden")

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return self._literal(node)
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise self._error(node, "dunder names are forbidden")
            if node.id in self.arrays:
                return self.arrays[node.id]
            if node.id == "np":
                return np
            if node.id in _FUNCTIONS:
                return _FUNCTIONS[node.id]
            raise self._error(node, f"unknown name {node.id!r}")
        if isinstance(node, (ast.Tuple, ast.List)):
            if len(node.elts) > MAX_LITERAL_ITEMS:
                raise self._error(node, f"literal exceeds {MAX_LITERAL_ITEMS} elements")
            return tuple(self._index_literal(item) for item in node.elts)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            operand = self._eval(node.operand)
            self._check_result_allocation(
                getattr(operand, "shape", ()),
                getattr(getattr(operand, "dtype", None), "itemsize", 8),
                node,
            )
            result = _UNARY[type(node.op)](operand)
            return self._check_array(result, "unary intermediate")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            if isinstance(node.op, ast.Pow):
                if not (
                    isinstance(node.right, ast.Constant)
                    and isinstance(node.right.value, int)
                    and not isinstance(node.right.value, bool)
                    and 0 <= node.right.value <= 8
                ):
                    raise self._error(node, "exponent must be a literal integer in [0, 8]")
            left = self._eval(node.left)
            right = self._eval(node.right)
            self._check_broadcast(left, right, node)
            try:
                result = _BINARY[type(node.op)](left, right)
            except Exception as exc:
                raise self._error(node, str(exc)) from exc
            return self._check_array(result, "binary intermediate")
        if isinstance(node, ast.Compare):
            if len(node.ops) != len(node.comparators):
                raise self._error(node, "malformed comparison")
            left = self._eval(node.left)
            result = None
            for op_node, comparator in zip(node.ops, node.comparators):
                if type(op_node) not in _COMPARE:
                    raise self._error(node, f"comparison {type(op_node).__name__} is forbidden")
                right = self._eval(comparator)
                self._check_broadcast(left, right, node)
                part = self._check_array(_COMPARE[type(op_node)](left, right), "comparison intermediate")
                result = part if result is None else np.logical_and(result, part)
                left = right
            return result
        if isinstance(node, ast.Subscript):
            value = self._eval(node.value)
            if not isinstance(value, np.ndarray):
                raise self._error(node, "only numeric arrays may be indexed")
            index = self._index_literal(node.slice)
            if self._contains_advanced_index(index):
                advanced_count = self._advanced_index_count(index)
                estimated_elements = max(1, advanced_count) * max(
                    1, value.size // max(1, value.shape[0])
                )
                self._check_result_allocation(
                    (estimated_elements,), value.dtype.itemsize, node
                )
            try:
                result = value[index]
            except Exception as exc:
                raise self._error(node, str(exc)) from exc
            return self._check_array(result, "subscript intermediate")
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "np"
                and node.attr in _FUNCTIONS
            ):
                return _FUNCTIONS[node.attr]
            raise self._error(node, f"attribute access {node.attr!r} is forbidden")
        if isinstance(node, ast.Call):
            function = self._eval(node.func)
            if function not in _FUNCTIONS.values():
                raise self._error(node, "only whitelisted NumPy functions may be called")
            if node.keywords:
                raise self._error(node, "keyword arguments are forbidden")
            args = [self._eval(arg) for arg in node.args]
            arrays = [arg for arg in args if isinstance(arg, np.ndarray)]
            if arrays:
                try:
                    shape = np.broadcast_shapes(*(arg.shape for arg in arrays))
                except ValueError as exc:
                    raise self._error(node, "incompatible function argument shapes") from exc
                self._check_result_allocation(
                    shape,
                    max(arg.dtype.itemsize for arg in arrays),
                    node,
                )
            try:
                result = function(*args)
            except Exception as exc:
                raise self._error(node, str(exc)) from exc
            return self._check_array(result, "function intermediate")
        raise self._error(node, "syntax is forbidden")

    def _check_broadcast(self, left: Any, right: Any, node: ast.AST) -> None:
        shapes = [value.shape for value in (left, right) if isinstance(value, np.ndarray)]
        if not shapes:
            return
        try:
            shape = np.broadcast_shapes(*shapes)
        except ValueError as exc:
            raise self._error(node, f"incompatible broadcast shapes {shapes}") from exc
        itemsize = max(
            (
                value.dtype.itemsize
                for value in (left, right)
                if isinstance(value, np.ndarray)
            ),
            default=8,
        )
        self._check_result_allocation(shape, itemsize, node)

    def _check_result_allocation(self, shape, itemsize, node):
        elements = math.prod(shape)
        byte_count = elements * int(itemsize)
        if elements > self.max_elements or byte_count > MAX_INTERMEDIATE_BYTES:
            raise self._error(
                node,
                "result exceeds intermediate allocation limit "
                f"({elements} elements, {byte_count} bytes)",
            )

    @staticmethod
    def _contains_advanced_index(index):
        if isinstance(index, list):
            return True
        return isinstance(index, tuple) and any(
            isinstance(item, list) for item in index
        )

    @staticmethod
    def _advanced_index_count(index):
        if isinstance(index, list):
            return len(index)
        if isinstance(index, tuple):
            return max(
                (len(item) for item in index if isinstance(item, list)),
                default=0,
            )
        return 0

    def _index_literal(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if node.value is Ellipsis:
                return Ellipsis
            value = self._literal(node)
            if value is None:
                raise self._error(node, "new-axis indexing is forbidden")
            if not isinstance(value, int) or isinstance(value, bool):
                raise self._error(node, "indexes must be literal integers")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._index_literal(node.operand)
            return -value
        if isinstance(node, ast.Slice):
            return slice(
                None if node.lower is None else self._index_literal(node.lower),
                None if node.upper is None else self._index_literal(node.upper),
                None if node.step is None else self._index_literal(node.step),
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            if len(node.elts) > MAX_LITERAL_ITEMS:
                raise self._error(node, f"index literal exceeds {MAX_LITERAL_ITEMS} elements")
            values = tuple(self._index_literal(item) for item in node.elts)
            if sum(value is Ellipsis for value in values) > 1:
                raise self._error(node, "at most one ellipsis is allowed")
            if isinstance(node, ast.List):
                if not all(isinstance(value, int) for value in values):
                    raise self._error(node, "advanced index lists require literal integers")
                return list(values)
            return values
        raise self._error(node, "subscripts require literal integers, lists, slices, or ellipsis")


def evaluate_filter_condition(expression: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
    return SafeFilterEvaluator(expression, arrays).evaluate()
