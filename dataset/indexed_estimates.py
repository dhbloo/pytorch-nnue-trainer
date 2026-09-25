"""Header-derived bounds for the legacy indexed NPZ materializers."""

from __future__ import annotations

import ast
import math
import struct
import sys

import numpy as np

from .execution import FileMemoryBound
from .filter import MAX_AST_NODES, MAX_INTERMEDIATE_BYTES


def _member(schemas, name, rank):
    try:
        shape, dtype = schemas[name]
    except KeyError as exc:
        raise ValueError(f"indexed NPZ is missing {name!r}") from exc
    if len(shape) != rank or any(dim < 0 for dim in shape):
        raise ValueError(f"indexed NPZ member {name!r} has an invalid shape")
    dtype = np.dtype(dtype)
    if dtype.kind not in "biuf" or dtype.hasobject:
        raise ValueError(f"indexed NPZ member {name!r} must be numeric")
    return shape, dtype


def _bytes(schemas):
    return sum(math.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in schemas.values())


def _division_itemsize(dtype):
    left = np.ones((), dtype=dtype)
    right = np.ones((), dtype=np.float32)
    return int(np.true_divide(left, right).dtype.itemsize)


def _python_index_list_bound(length):
    # CPython list growth stays below twice the logical length; include ints.
    return (
        sys.getsizeof([])
        + (2 * length + 8) * struct.calcsize("P")
        + length * sys.getsizeof(length)
    )


def raw_indexed_file_bound(
    schemas,
    *,
    has_pass_move=False,
    rule_index=None,
    filter_stm=None,
    filter_condition=None,
):
    packed, _ = _member(schemas, "binaryInputNCHWPacked", 3)
    global_input, global_dtype = _member(schemas, "globalInputNC", 2)
    targets, target_dtype = _member(schemas, "globalTargetsNC", 2)
    policy, policy_dtype = _member(schemas, "policyTargetsNCMove", 3)
    n = packed[0]
    if n <= 0 or any(shape[0] != n for shape in (global_input, targets, policy)):
        raise ValueError("indexed raw NPZ row counts are invalid")
    if packed[1] < 3 or global_input[1] < 1 or targets[1] < 3 or policy[1] < 1:
        raise ValueError("indexed raw NPZ is missing required channels")
    j = packed[2]
    board_area = math.isqrt(8 * j) ** 2
    policy_area = max(0, policy[2] - 1)
    pass_area = int(bool(has_pass_move))
    p = policy_area + pass_area
    t = target_dtype.itemsize
    g = global_dtype.itemsize
    q = _division_itemsize(policy_dtype)
    r = 8 if rule_index is not None else 0
    raw_bytes = sum(
        math.prod(schemas[name][0]) * np.dtype(schemas[name][1]).itemsize
        for name in (
            "binaryInputNCHWPacked",
            "globalInputNC",
            "globalTargetsNC",
            "policyTargetsNCMove",
        )
    )
    resident = n * (16 + 2 * board_area + 4 + 3 * t + p * q + r)
    scratch = (
        3 * n * j
        + 4 * n * board_area
        + n * (g + 1 + 8)
        + 4 * n * p
        + 8 * n
        + 16 * n
    )
    filter_bytes = 0
    if filter_stm is not None:
        filter_bytes = raw_bytes + 9 * n
    if filter_condition is not None:
        # Every evaluated AST node may retain one numeric intermediate while
        # its parent evaluates.  The evaluator checks each result before its
        # allocation against both its element and byte caps.
        node_count = len(list(ast.walk(ast.parse(filter_condition, mode="eval"))))
        if node_count > MAX_AST_NODES:
            raise ValueError("filter expression exceeds the supported AST size")
        maximum_itemsize = max(
            16, *(np.dtype(dtype).itemsize for _, dtype in schemas.values())
        )
        intermediate = min(
            MAX_INTERMEDIATE_BYTES,
            max(4096, 8 * n) * maximum_itemsize,
        )
        filter_bytes += raw_bytes + 9 * n + node_count * (intermediate + 1024)
    transient = (
        raw_bytes
        + 2 * resident
        + scratch
        + _python_index_list_bound(n)
        + 8 * n
        + filter_bytes
    )
    tuple_bound = sys.getsizeof((0, 0)) + 2 * sys.getsizeof(n) + 32
    catalog = 80 * n + n * tuple_bound + 2048
    return FileMemoryBound(resident, transient, catalog)


def sparse_indexed_file_bound(schemas):
    packed, _ = _member(schemas, "binaryInputNCHWPacked", 3)
    global_input, global_dtype = _member(schemas, "globalInputNC", 2)
    targets, target_dtype = _member(schemas, "globalTargetsNC", 2)
    policy, policy_dtype = _member(schemas, "policyTargetsNCHW", 3)
    u8, _ = _member(schemas, "sparseInputNCHWU8", 3)
    u16, u16_dtype = _member(schemas, "sparseInputNCHWU16", 3)
    dim, dim_dtype = _member(schemas, "sparseInputDim", 1)
    n = packed[0]
    if n <= 0 or any(shape[0] != n for shape in (global_input, targets, policy, u8, u16)):
        raise ValueError("indexed sparse NPZ row counts are invalid")
    if packed[1] < 3 or global_input[1] < 1 or targets[1] < 3 or policy[1] < 1:
        raise ValueError("indexed sparse NPZ is missing required channels")
    j = packed[2]
    board_area = math.isqrt(8 * j) ** 2
    policy_area = policy[2]
    feature_area = max(u8[2], u16[2])
    feature_count = u8[1] + u16[1]
    t = target_dtype.itemsize
    g = global_dtype.itemsize
    q = _division_itemsize(policy_dtype)
    f = np.result_type(np.uint16, u16_dtype).itemsize
    raw_bytes = _bytes(schemas)
    resident = (
        2 * n * board_area
        + 4 * n
        + 3 * n * t
        + n * policy_area * q
        + n * feature_count * feature_area * f
        + dim[0] * dim_dtype.itemsize
    )
    scratch = (
        2 * n * j
        + 2 * n * board_area
        + n * g
        + 4 * n * policy_area
        + 8 * n
        + 2 * n * u8[1] * feature_area
    )
    return FileMemoryBound(resident, 2 * raw_bytes + resident + scratch, 4096)


def raw_indexed_output_row_bytes(schemas, *, output_shape, has_pass_move=False, rule_index=None):
    _, target_dtype = _member(schemas, "globalTargetsNC", 2)
    _, policy_dtype = _member(schemas, "policyTargetsNCMove", 3)
    area = math.prod(output_shape)
    return (
        16 + 2 * area + 4 + 3 * target_dtype.itemsize
        + (area + int(bool(has_pass_move))) * _division_itemsize(policy_dtype)
        + (8 if rule_index is not None else 0)
    )


def sparse_indexed_output_row_bytes(schemas, *, output_shape):
    _, target_dtype = _member(schemas, "globalTargetsNC", 2)
    _, policy_dtype = _member(schemas, "policyTargetsNCHW", 3)
    u8, _ = _member(schemas, "sparseInputNCHWU8", 3)
    u16, _ = _member(schemas, "sparseInputNCHWU16", 3)
    area = math.prod(output_shape)
    feature_count = u8[1] + u16[1]
    return (
        2 + 2 * area + 4 + 3 * target_dtype.itemsize
        + area * _division_itemsize(policy_dtype)
        + 4 * feature_count * area + 4 * feature_count
    )
