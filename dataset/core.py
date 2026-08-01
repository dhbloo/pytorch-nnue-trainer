from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch


RNG_ALGORITHM = ("nnue-dataset-rng", 1, "blake2b-256-tlv-le")
_RNG_HEADER = b"NNUE-DATA-RNG\0" + struct.pack("<H", 1)
_RNG_PERSON = b"NNUEdsRNGv1" + b"\0" * 5
_THREE_RNG_INTS = struct.Struct("<BqBqBq")
_FOUR_RNG_INTS = struct.Struct("<BqBqBqBq")


@dataclass(frozen=True)
class DatasetRuntimeContext:
    global_batch_size: int
    local_batch_size: int
    world_size: int
    rank: int
    seed: int
    mode: Literal["train", "validate", "test", "export", "visualize"]

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be at least one, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is outside [0, {self.world_size})")
        if self.global_batch_size <= 0 or self.local_batch_size <= 0:
            raise ValueError(
                "global_batch_size and local_batch_size must be positive, got "
                f"{self.global_batch_size} and {self.local_batch_size}"
            )
        if self.local_batch_size * self.world_size != self.global_batch_size:
            raise ValueError(
                f"local batch {self.local_batch_size} * world size {self.world_size} "
                f"must equal global batch {self.global_batch_size}"
            )
        if self.mode not in {"train", "validate", "test", "export", "visualize"}:
            raise ValueError(f"unsupported dataset runtime mode {self.mode!r}")
        if not -(1 << 63) <= int(self.seed) < (1 << 63):
            raise ValueError(f"seed must be a signed 64-bit integer, got {self.seed}")

    @property
    def global_contract(self) -> "GlobalRuntimeContract":
        return GlobalRuntimeContract(
            global_batch_size=self.global_batch_size,
            local_batch_size=self.local_batch_size,
            world_size=self.world_size,
            seed=self.seed,
            mode=self.mode,
        )

    @property
    def rank_local_identity(self) -> "RankLocalIdentity":
        start = self.rank * self.local_batch_size
        return RankLocalIdentity(self.rank, start, start + self.local_batch_size)


@dataclass(frozen=True)
class GlobalRuntimeContract:
    global_batch_size: int
    local_batch_size: int
    world_size: int
    seed: int
    mode: str


@dataclass(frozen=True)
class RankLocalIdentity:
    rank: int
    slice_start: int
    slice_stop: int


def single_process_dataset_context(
    batch_size: int,
    *,
    seed: int = 0,
    mode: Literal["train", "validate", "test", "export", "visualize"] = "train",
) -> DatasetRuntimeContext:
    return DatasetRuntimeContext(batch_size, batch_size, 1, 0, seed, mode)


@dataclass(frozen=True)
class DatasetCapabilities:
    yields_batches: bool = False
    resumable: bool = False
    deterministic: bool = False
    supports_batch_pipeline: bool = False


@dataclass(frozen=True)
class FieldSpec:
    name: str
    required: bool
    scope: Literal["per_sample", "batch_shared"]
    spatial_axes: tuple[int, int] | None
    side_axis: int | None
    collate: Literal["stack", "broadcast", "list"]
    dtype_kinds: tuple[str, ...]
    transform: Literal[
        "plain", "board_policy", "flat_policy_with_pass", "moves"
    ]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must not be empty")
        if type(self.required) is not bool:
            raise TypeError(f"field {self.name} required must be a bool")
        if self.scope not in {"per_sample", "batch_shared"}:
            raise ValueError(f"invalid field scope {self.scope!r} for {self.name}")
        if self.collate not in {"stack", "broadcast", "list"}:
            raise ValueError(f"invalid collation {self.collate!r} for {self.name}")
        if self.transform not in {
            "plain",
            "board_policy",
            "flat_policy_with_pass",
            "moves",
        }:
            raise ValueError(f"invalid transform {self.transform!r} for {self.name}")
        if self.scope == "batch_shared" and self.collate == "stack":
            raise ValueError(f"batch-shared field {self.name} cannot use stack collation")
        if self.spatial_axes is not None:
            if (
                not isinstance(self.spatial_axes, tuple)
                or len(self.spatial_axes) != 2
                or any(type(axis) is not int for axis in self.spatial_axes)
            ):
                raise TypeError(
                    f"field {self.name} spatial_axes must be two integer axes"
                )
            if self.spatial_axes[0] == self.spatial_axes[1]:
                raise ValueError(f"field {self.name} spatial axes must be distinct")
        if self.side_axis is not None and type(self.side_axis) is not int:
            raise TypeError(f"field {self.name} side_axis must be an integer")
        if (
            not isinstance(self.dtype_kinds, tuple)
            or not self.dtype_kinds
            or any(
                not isinstance(kind, str)
                or len(kind) != 1
                or kind not in "?biufcOSUVmM"
                for kind in self.dtype_kinds
            )
            or len(set(self.dtype_kinds)) != len(self.dtype_kinds)
        ):
            raise ValueError(f"field {self.name} must declare at least one dtype kind")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError(f"field {self.name} schema_version must be positive")
        if (
            self.side_axis is not None
            and self.spatial_axes is not None
            and self.side_axis in self.spatial_axes
        ):
            raise ValueError(
                f"field {self.name} side axis cannot alias a spatial axis"
            )


CORE_FIELD_SPECS = (
    FieldSpec("board_size", True, "per_sample", None, None, "stack", ("i", "u"), "plain"),
    FieldSpec("board_input", True, "per_sample", (-2, -1), 0, "stack", ("b", "i", "u"), "plain"),
    FieldSpec("stm_input", True, "per_sample", None, None, "stack", ("i", "f"), "plain"),
    FieldSpec("value_target", True, "per_sample", None, 0, "stack", ("f",), "plain"),
    FieldSpec(
        "policy_target", True, "per_sample", None, None, "stack",
        ("b", "i", "u", "f"), "board_policy"
    ),
    FieldSpec("rule_index", False, "per_sample", None, None, "stack", ("i", "u"), "plain"),
    FieldSpec("ply", False, "per_sample", None, None, "stack", ("i", "u"), "plain"),
    FieldSpec("raw_eval", False, "per_sample", None, None, "stack", ("f",), "plain"),
    FieldSpec(
        "sparse_feature_input", False, "per_sample", (-2, -1), 0,
        "stack", ("b", "i", "u", "f"), "plain"
    ),
    FieldSpec(
        "sparse_feature_dim", False, "batch_shared", None, None,
        "broadcast", ("i", "u"), "plain"
    ),
    FieldSpec(
        "position", False, "per_sample", None, None,
        "list", ("O",), "moves"
    ),
    FieldSpec(
        "position_string", False, "per_sample", None, None,
        "list", ("U", "S", "O"), "moves"
    ),
    FieldSpec(
        "last_move", False, "per_sample", None, None,
        "stack", ("i", "u"), "plain"
    ),
    FieldSpec(
        "forbidden_point", False, "per_sample", (-2, -1), None,
        "stack", ("b", "i", "u"), "plain"
    ),
    FieldSpec(
        "line_encoding", False, "per_sample", (-2, -1), None,
        "stack", ("i", "u"), "plain"
    ),
    FieldSpec(
        "line_encoding_total_num", False, "batch_shared", None, None,
        "broadcast", ("i", "u"), "plain"
    ),
)

FIELD_SPECS = {spec.name: spec for spec in CORE_FIELD_SPECS}


def _field_dtype_kind(value, spec: FieldSpec) -> str:
    if spec.transform == "moves" and isinstance(value, (list, tuple)):
        if not value or not isinstance(value[0], (str, bytes, int, float, bool)):
            return "O"
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return "b"
        if value.dtype.is_floating_point:
            return "f"
        if value.dtype.is_complex:
            return "c"
        return "i"
    return np.asarray(value).dtype.kind


def validate_field_dict(data: dict, *, batched: bool) -> None:
    """Validate one registry-owned sample or batch before transformation."""
    if not isinstance(data, dict):
        raise TypeError("dataset fields must be provided as a dictionary")
    unknown = sorted(set(data).difference(FIELD_SPECS))
    if unknown:
        raise ValueError(f"undeclared dataset field(s): {unknown}")
    missing = sorted(
        spec.name
        for spec in CORE_FIELD_SPECS
        if spec.required and spec.name not in data
    )
    if missing:
        raise ValueError(f"missing required dataset field(s): {missing}")

    board_size = np.asarray(
        data["board_size"].detach().cpu()
        if isinstance(data["board_size"], torch.Tensor)
        else data["board_size"]
    )
    expected_board_shape = (None, 2) if batched else (2,)
    if (
        (batched and (board_size.ndim != 2 or board_size.shape[1] != 2))
        or (not batched and board_size.shape != expected_board_shape)
    ):
        raise ValueError(
            f"board_size has invalid {'batch' if batched else 'sample'} shape "
            f"{tuple(board_size.shape)}"
        )
    batch_size = int(board_size.shape[0]) if batched else None

    for key, value in data.items():
        spec = FIELD_SPECS[key]
        kind = _field_dtype_kind(value, spec)
        if kind not in spec.dtype_kinds:
            raise TypeError(
                f"field {key!r} has dtype kind {kind!r}, expected "
                f"{spec.dtype_kinds}"
            )
        array = (
            value
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        ndim = array.ndim
        if batched and spec.scope == "per_sample":
            if ndim == 0 or len(array) != batch_size:
                raise ValueError(
                    f"per-sample field {key!r} must contain {batch_size} rows"
                )
        if batched and spec.scope == "batch_shared":
            if ndim == 0 or len(array) != batch_size:
                raise ValueError(
                    f"batch-shared field {key!r} must retain leading "
                    f"dimension {batch_size}"
                )
            first = array[0]
            if isinstance(array, torch.Tensor):
                equal_rows = (
                    array == first.expand_as(array)
                ).reshape(batch_size, -1).all(dim=1).cpu().numpy()
            else:
                equal_rows = np.all(
                    np.asarray(array) == np.broadcast_to(first, array.shape),
                    axis=tuple(range(1, ndim)),
                )
            mismatched = np.flatnonzero(~equal_rows)
            if mismatched.size:
                raise ValueError(
                    f"batch-shared field {key!r} differs at row "
                    f"{int(mismatched[0])}"
                )
        if spec.spatial_axes is not None:
            axes = tuple(
                (
                    axis
                    + (1 if batched and spec.scope == "per_sample" else 0)
                    if axis >= 0
                    else ndim + axis
                )
                for axis in spec.spatial_axes
            )
            if any(axis < 0 or axis >= ndim for axis in axes):
                raise ValueError(
                    f"field {key!r} is missing declared spatial axes "
                    f"{spec.spatial_axes}"
                )
            if batched and spec.scope == "per_sample" and 0 in axes:
                raise ValueError(
                    f"field {key!r} spatial axes cannot target the batch axis"
                )
            spatial_shape = tuple(int(array.shape[axis]) for axis in axes)
            if batched:
                physical = np.asarray(spatial_shape, dtype=np.int64)
                compared_axes = min(len(physical), board_size.shape[1])
                spatial_mismatch = np.any(
                    physical[None, :compared_axes]
                    < board_size[:, :compared_axes],
                )
            else:
                spatial_mismatch = any(
                    physical < logical
                    for physical, logical in zip(spatial_shape, board_size)
                )
            if spatial_mismatch:
                raise ValueError(
                    f"field {key!r} spatial shape {spatial_shape} is smaller "
                    "than board_size"
                )
        if spec.side_axis is not None:
            side_axis = (
                spec.side_axis
                + (1 if batched and spec.scope == "per_sample" else 0)
                if spec.side_axis >= 0
                else ndim + spec.side_axis
            )
            if side_axis < 0 or side_axis >= ndim:
                raise ValueError(
                    f"field {key!r} is missing declared side axis "
                    f"{spec.side_axis}"
                )
            if batched and spec.scope == "per_sample" and side_axis == 0:
                raise ValueError(
                    f"field {key!r} side axis cannot target the batch axis"
                )
            if spec.spatial_axes is not None and side_axis in axes:
                raise ValueError(
                    f"field {key!r} side axis aliases a spatial axis"
                )

    policy = data["policy_target"]
    policy_shape = tuple(policy.shape)
    if batched:
        height, width = (
            int(v) for v in data["board_input"].shape[-2:]
        )
        valid_policy_shapes = {
            (batch_size, height, width),
            (batch_size, height * width + 1),
        }
    else:
        height, width = (int(v) for v in board_size)
        physical_height, physical_width = (
            int(v) for v in data["board_input"].shape[-2:]
        )
        valid_policy_shapes = {
            (height, width),
            (height * width + 1,),
            (physical_height, physical_width),
            (physical_height * physical_width + 1,),
        }
    if policy_shape not in valid_policy_shapes:
        raise ValueError(
            f"policy_target shape {policy_shape} does not match board_size"
        )


def _tlv(value: Any) -> bytes:
    if value is None:
        return b"\x00"
    if isinstance(value, bool):
        return b"\x01" + bytes([value])
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError(f"RNG integer is outside signed 64-bit range: {value}")
        return b"\x02" + struct.pack("<q", value)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"\x03" + struct.pack("<I", len(encoded)) + encoded
    if isinstance(value, bytes):
        return b"\x04" + struct.pack("<I", len(value)) + value
    if isinstance(value, (tuple, list)):
        return b"\x05" + struct.pack("<I", len(value)) + b"".join(_tlv(v) for v in value)
    raise TypeError(f"unsupported RNG key component {type(value).__name__}")


def rng_digest(root_seed: int, domain: str, key: tuple, counter: int = 0) -> bytes:
    if not 0 <= counter < (1 << 63):
        raise ValueError(f"RNG counter is outside [0, 2**63): {counter}")
    payload = _RNG_HEADER + _tlv((domain, int(root_seed), key, int(counter)))
    return hashlib.blake2b(payload, digest_size=32, person=_RNG_PERSON).digest()


def rng_u64(root_seed: int, domain: str, key: tuple, counter: int = 0) -> int:
    return int.from_bytes(rng_digest(root_seed, domain, key, counter)[:8], "little")


def _counter_rng_base(root_seed: int, domain: str, key: tuple):
    # This is byte-for-byte the prefix of
    # _tlv((domain, root_seed, key, counter)), stopping after the counter's
    # integer type tag. Cloning the preseeded BLAKE2 state avoids recursively
    # rebuilding the invariant key for every Fisher-Yates draw.
    prefix = (
        _RNG_HEADER
        + b"\x05"
        + struct.pack("<I", 4)
        + _tlv(domain)
        + _tlv(int(root_seed))
        + _tlv(key)
        + b"\x02"
    )
    return hashlib.blake2b(prefix, digest_size=32, person=_RNG_PERSON)


def _uniform_below_from_base(n: int, base, counter: int) -> tuple[int, int]:
    if n <= 0 or n > (1 << 64):
        raise ValueError(f"uniform_below bound must be in [1, 2**64], got {n}")
    limit = (1 << 64) - ((1 << 64) % n)
    while True:
        if not 0 <= counter < (1 << 63):
            raise ValueError(f"RNG counter is outside [0, 2**63): {counter}")
        digest = base.copy()
        digest.update(struct.pack("<q", counter))
        value = int.from_bytes(digest.digest()[:8], "little")
        counter += 1
        if value < limit:
            return value % n, counter


def uniform_below(n: int, root_seed: int, domain: str, key: tuple, counter: int = 0) -> tuple[int, int]:
    if n <= 0 or n > (1 << 64):
        raise ValueError(f"uniform_below bound must be in [1, 2**64], got {n}")
    limit = (1 << 64) - ((1 << 64) % n)
    while True:
        value = rng_u64(root_seed, domain, key, counter)
        counter += 1
        if value < limit:
            return value % n, counter


def _uniform_below_for_processed_sample_keys(
    n: int,
    root_seed: int,
    domain: str,
    epoch: int,
    sample_keys,
    occurrence: int = 0,
) -> list[int]:
    """Fast exact RNG for row and cycle-row processed sample keys.

    Each result is byte-for-byte identical to an independent
    ``uniform_below`` call. Only the invariant TLV prefix and hash state are
    shared per file/kind; every sample retains its own rejection counter.
    """
    if n <= 0 or n > (1 << 64):
        raise ValueError(f"uniform_below bound must be in [1, 2**64], got {n}")
    limit = (1 << 64) - ((1 << 64) % n)
    prefix = (
        _RNG_HEADER
        + b"\x05"
        + struct.pack("<I", 4)
        + _tlv(domain)
        + _tlv(int(root_seed))
        + b"\x05"
        + struct.pack("<I", 3)
        + _tlv(int(epoch))
    )
    results = []
    previous_base_key = None
    base = None
    for sample_key in sample_keys:
        namespace, content_digest, kind, address = sample_key
        if len(address) not in {1, 2}:
            raise ValueError(
                "processed sample RNG requires a row or cycle-row address"
            )
        base_key = (namespace, content_digest, kind, len(address))
        if base_key != previous_base_key:
            namespace_bytes = namespace.encode("utf-8")
            kind_bytes = kind.encode("utf-8")
            base = hashlib.blake2b(
                prefix
                + b"\x05"
                + struct.pack("<I", 4)
                + b"\x03"
                + struct.pack("<I", len(namespace_bytes))
                + namespace_bytes
                + b"\x04"
                + struct.pack("<I", len(content_digest))
                + content_digest
                + b"\x03"
                + struct.pack("<I", len(kind_bytes))
                + kind_bytes
                + b"\x05"
                + struct.pack("<I", len(address)),
                digest_size=32,
                person=_RNG_PERSON,
            )
            previous_base_key = base_key
        counter = 0
        while True:
            digest = base.copy()
            if len(address) == 1:
                digest.update(
                    _THREE_RNG_INTS.pack(
                        2,
                        int(address[0]),
                        2,
                        int(occurrence),
                        2,
                        counter,
                    )
                )
            else:
                digest.update(
                    _FOUR_RNG_INTS.pack(
                        2,
                        int(address[0]),
                        2,
                        int(address[1]),
                        2,
                        int(occurrence),
                        2,
                        counter,
                    )
                )
            value = int.from_bytes(digest.digest()[:8], "little")
            counter += 1
            if value < limit:
                results.append(value % n)
                break
    return results


def _uniform_below_for_processed_sample_keys_v2(
    n: int,
    root_seed: int,
    domain: str,
    epoch: int,
    sample_keys,
    occurrence: int = 0,
) -> list[int]:
    """Vectorized key-stable SplitMix RNG for power-of-two symmetries.

    This is a deliberately versioned augmentation stream. It preserves a
    uniform independent choice per immutable row key while avoiding one
    cryptographic hash object and Python loop body per training sample.
    """
    if n <= 0 or n > (1 << 64):
        raise ValueError(f"uniform_below bound must be in [1, 2**64], got {n}")
    if n & (n - 1):
        return _uniform_below_for_processed_sample_keys(
            n, root_seed, domain, epoch, sample_keys, occurrence
        )
    keys = tuple(sample_keys)
    if not keys:
        return []
    groups = {}
    for position, sample_key in enumerate(keys):
        namespace, content_digest, kind, address = sample_key
        if len(address) not in {1, 2}:
            raise ValueError(
                "processed sample RNG requires a row or cycle-row address"
            )
        base_key = (namespace, content_digest, kind, len(address))
        positions, cycles, rows = groups.setdefault(base_key, ([], [], []))
        positions.append(position)
        if len(address) == 1:
            cycles.append(0)
            rows.append(int(address[0]))
        else:
            cycles.append(int(address[0]))
            rows.append(int(address[1]))

    output = np.empty(len(keys), dtype=np.uint64)
    mask = np.uint64(n - 1)
    row_gamma = np.uint64(0x9E3779B97F4A7C15)
    cycle_gamma = np.uint64(0xD1B54A32D192ED03)
    for base_key, (positions, cycles, rows) in groups.items():
        base = np.uint64(
            rng_u64(
                root_seed,
                domain + "_processed_splitmix_v1",
                (epoch, *base_key, occurrence),
            )
        )
        with np.errstate(over="ignore"):
            values = (
                base
                + np.asarray(rows, dtype=np.uint64) * row_gamma
                + np.asarray(cycles, dtype=np.uint64) * cycle_gamma
            )
            values = (values ^ (values >> np.uint64(30))) * np.uint64(
                0xBF58476D1CE4E5B9
            )
            values = (values ^ (values >> np.uint64(27))) * np.uint64(
                0x94D049BB133111EB
            )
            values ^= values >> np.uint64(31)
        output[np.asarray(positions, dtype=np.intp)] = values & mask
    return output.tolist()


def _uniform_below_for_packed_processed_rows_v2(
    n: int,
    root_seed: int,
    domain: str,
    epoch: int,
    content_digests: tuple[bytes, ...],
    file_indices: np.ndarray,
    rows: np.ndarray,
    occurrence: int = 0,
) -> np.ndarray:
    """Structured form of processed-splitmix-v1 without Python row keys."""
    if n <= 0 or n > (1 << 64) or n & (n - 1):
        raise ValueError("packed processed RNG requires a power-of-two bound")
    file_indices = np.asarray(file_indices, dtype=np.intp)
    rows = np.asarray(rows, dtype=np.uint64)
    if file_indices.ndim != 1 or rows.shape != file_indices.shape:
        raise ValueError("packed processed RNG arrays must be matching vectors")
    output = np.empty(len(rows), dtype=np.uint64)
    row_gamma = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for file_index in np.unique(file_indices):
            if not 0 <= file_index < len(content_digests):
                raise ValueError("packed processed RNG file index is out of range")
            selected = file_indices == file_index
            base = np.uint64(
                rng_u64(
                    root_seed,
                    domain + "_processed_splitmix_v1",
                    (
                        epoch,
                        "root",
                        content_digests[file_index],
                        "npz-row",
                        1,
                        occurrence,
                    ),
                )
            )
            values = base + rows[selected] * row_gamma
            values = (values ^ (values >> np.uint64(30))) * np.uint64(
                0xBF58476D1CE4E5B9
            )
            values = (values ^ (values >> np.uint64(27))) * np.uint64(
                0x94D049BB133111EB
            )
            values ^= values >> np.uint64(31)
            output[selected] = values & np.uint64(n - 1)
    return output


def deterministic_permutation(length: int, root_seed: int, domain: str, key: tuple) -> list[int]:
    values = list(range(length))
    counter = 0
    base = _counter_rng_base(root_seed, domain, key)
    for i in range(length - 1, 0, -1):
        j, counter = _uniform_below_from_base(i + 1, base, counter)
        values[i], values[j] = values[j], values[i]
    return values


def sample_is_admitted(sample_rate: float, root_seed: int, epoch: int, sample_key: tuple) -> bool:
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate}")
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    return rng_u64(root_seed, "sample", (epoch, sample_key)) / (1 << 64) < sample_rate


_TENSOR_DTYPES = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}
_TENSOR_DTYPES_BY_NAME = {name: dtype for dtype, name in _TENSOR_DTYPES.items()}


def _canonical_pipeline_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError(f"pipeline integer is outside signed 64-bit range: {value}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("pipeline floats must be finite")
        return {"$f64": struct.pack(">d", value).hex()}
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_canonical_pipeline_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("pipeline state dictionaries require string keys")
        if any(key in {"$bytes", "$f64", "$tensor"} for key in value):
            raise ValueError("pipeline state collides with a reserved tag")
        return {key: _canonical_pipeline_value(value[key]) for key in sorted(value)}
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("pipeline tensors must not require gradients")
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_sparse:
            raise ValueError("pipeline tensors must be dense strided CPU tensors")
        if value.dtype not in _TENSOR_DTYPES:
            raise ValueError(f"unsupported pipeline tensor dtype {value.dtype}")
        tensor = value.detach().contiguous()
        if tensor.dtype == torch.bfloat16:
            data = tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
        else:
            array = tensor.numpy()
            dtype = array.dtype.newbyteorder("<")
            data = array.astype(dtype, copy=False).tobytes(order="C")
        return {
            "$tensor": {
                "dtype": _TENSOR_DTYPES[tensor.dtype],
                "shape": list(tensor.shape),
                "data_le": data.hex(),
            }
        }
    raise TypeError(f"unsupported pipeline state value {type(value).__name__}")


def canonical_pipeline_state_bytes(value: Any) -> bytes:
    canonical = _canonical_pipeline_value(value)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _decode_pipeline_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_pipeline_value(item) for item in value)
    if not isinstance(value, dict):
        return value
    if set(value) == {"$bytes"}:
        text = value["$bytes"]
        if not isinstance(text, str) or len(text) % 2 or text.lower() != text:
            raise ValueError("invalid canonical bytes tag")
        return bytes.fromhex(text)
    if set(value) == {"$f64"}:
        text = value["$f64"]
        if not isinstance(text, str) or len(text) != 16 or text.lower() != text:
            raise ValueError("invalid canonical float tag")
        result = struct.unpack(">d", bytes.fromhex(text))[0]
        if not math.isfinite(result):
            raise ValueError("canonical float must be finite")
        return result
    if set(value) == {"$tensor"}:
        state = value["$tensor"]
        if not isinstance(state, dict) or set(state) != {"dtype", "shape", "data_le"}:
            raise ValueError("invalid canonical tensor tag")
        dtype = _TENSOR_DTYPES_BY_NAME.get(state["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported canonical tensor dtype {state['dtype']!r}")
        shape = tuple(int(dim) for dim in state["shape"])
        if any(dim < 0 for dim in shape):
            raise ValueError("canonical tensor dimensions must be non-negative")
        data = bytes.fromhex(state["data_le"])
        if dtype == torch.bfloat16:
            array = np.frombuffer(data, dtype="<u2").copy()
            tensor = torch.from_numpy(array).view(torch.bfloat16)
        else:
            np_dtype = torch.empty((), dtype=dtype).numpy().dtype.newbyteorder("<")
            tensor = torch.from_numpy(np.frombuffer(data, dtype=np_dtype).copy())
        expected = math.prod(shape)
        if tensor.numel() != expected:
            raise ValueError(f"canonical tensor has {tensor.numel()} elements, expected {expected}")
        return tensor.reshape(shape)
    if any(key in {"$bytes", "$f64", "$tensor"} for key in value):
        raise ValueError("invalid or colliding canonical pipeline tag")
    return {key: _decode_pipeline_value(item) for key, item in value.items()}


def decode_canonical_pipeline_state(blob: bytes) -> Any:
    value = json.loads(blob.decode("utf-8"))
    if canonical_pipeline_state_bytes(_decode_pipeline_value(value)) != blob:
        raise ValueError("pipeline state is not in canonical form")
    return _decode_pipeline_value(value)


@dataclass(frozen=True)
class SumCount:
    scope: Literal["global_batch", "evaluation"]
    sum: torch.Tensor
    count: int

    def __post_init__(self) -> None:
        if self.scope not in {"global_batch", "evaluation"}:
            raise ValueError(f"invalid SumCount scope {self.scope!r}")
        if not isinstance(self.sum, torch.Tensor):
            raise TypeError("SumCount sum must be a tensor")
        if (
            not isinstance(self.count, int)
            or isinstance(self.count, bool)
            or self.count < 0
        ):
            raise ValueError("SumCount count must be non-negative")

    def combine(self, other: "SumCount") -> "SumCount":
        if (
            self.scope != other.scope
            or self.sum.shape != other.sum.shape
            or self.sum.dtype != other.sum.dtype
            or self.sum.device != other.sum.device
        ):
            raise ValueError("incompatible SumCount values")
        return SumCount(self.scope, self.sum + other.sum, self.count + other.count)

    def finalize(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("cannot finalize empty SumCount")
        return self.sum / self.count


@dataclass(frozen=True)
class Maximum:
    scope: Literal["evaluation"]
    value: torch.Tensor
    valid_count: int

    def __post_init__(self) -> None:
        if self.scope != "evaluation":
            raise ValueError("Maximum scope must be evaluation")
        if not isinstance(self.value, torch.Tensor):
            raise TypeError("Maximum value must be a tensor")
        if (
            not isinstance(self.valid_count, int)
            or isinstance(self.valid_count, bool)
            or self.valid_count < 0
        ):
            raise ValueError("Maximum valid_count must be non-negative")

    def combine(self, other: "Maximum") -> "Maximum":
        if (
            self.value.shape != other.value.shape
            or self.value.dtype != other.value.dtype
            or self.value.device != other.value.device
        ):
            raise ValueError("incompatible Maximum values")
        if self.valid_count == 0:
            return other
        if other.valid_count == 0:
            return self
        return Maximum("evaluation", torch.maximum(self.value, other.value), self.valid_count + other.valid_count)

    def finalize(self) -> torch.Tensor:
        if self.valid_count == 0:
            raise ValueError("cannot finalize empty Maximum")
        return self.value


@dataclass(frozen=True)
class SufficientStats:
    scope: Literal["global_batch", "evaluation"]
    finalizer_id: str
    tensors: dict[str, torch.Tensor]
    counts: dict[str, int]

    def __post_init__(self) -> None:
        if self.scope not in {"global_batch", "evaluation"}:
            raise ValueError(f"invalid SufficientStats scope {self.scope!r}")
        if self.finalizer_id not in FINALIZERS:
            raise ValueError(f"unknown sufficient-statistic finalizer {self.finalizer_id!r}")
        if any(not isinstance(key, str) for key in self.tensors):
            raise ValueError("sufficient-statistic tensor keys must be strings")
        if any(not isinstance(value, torch.Tensor) for value in self.tensors.values()):
            raise TypeError("sufficient-statistic values must be tensors")
        if any(not isinstance(key, str) for key in self.counts):
            raise ValueError("sufficient-statistic count keys must be strings")
        if any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in self.counts.values()
        ):
            raise ValueError("sufficient-statistic counts must be non-negative")
        _validate_sufficient_stats_schema(
            self.finalizer_id, self.tensors, self.counts
        )

    def combine(self, other: "SufficientStats") -> "SufficientStats":
        if (
            self.scope != other.scope
            or self.finalizer_id != other.finalizer_id
            or self.tensors.keys() != other.tensors.keys()
            or self.counts.keys() != other.counts.keys()
            or any(
                self.tensors[key].shape != other.tensors[key].shape
                or self.tensors[key].dtype != other.tensors[key].dtype
                or self.tensors[key].device != other.tensors[key].device
                for key in self.tensors
            )
        ):
            raise ValueError("incompatible SufficientStats values")
        return SufficientStats(
            self.scope,
            self.finalizer_id,
            {key: self.tensors[key] + other.tensors[key] for key in self.tensors},
            {key: self.counts[key] + other.counts[key] for key in self.counts},
        )

    def finalize(self) -> torch.Tensor:
        return FINALIZERS[self.finalizer_id](self.tensors, self.counts)


def _finalize_mean_square(tensors: dict[str, torch.Tensor], counts: dict[str, int]) -> torch.Tensor:
    if set(tensors) != {"valid_logit_sum"} or set(counts) != {"valid_cells"}:
        raise ValueError("policy_mean_square_v1 has an invalid tensor/count schema")
    count = counts["valid_cells"]
    if count == 0:
        raise ValueError("policy_mean_square requires valid cells")
    return (tensors["valid_logit_sum"] / count) ** 2


def _finalize_mean(tensors: dict[str, torch.Tensor], counts: dict[str, int]) -> torch.Tensor:
    if set(tensors) != {"sum"} or set(counts) != {"count"}:
        raise ValueError("mean_v1 has an invalid tensor/count schema")
    count = counts["count"]
    if count <= 0:
        raise ValueError("mean_v1 requires a positive count")
    return tensors["sum"] / count


def _constant_mean(
    tensors: dict[str, torch.Tensor],
    counts: dict[str, int],
    *,
    sum_key: str,
    square_sum_key: str,
    replica_key: str,
) -> torch.Tensor:
    replicas = counts.get(replica_key, 0)
    if replicas <= 0:
        raise ValueError(f"replicated constant {sum_key!r} has no replicas")
    mean = tensors[sum_key] / replicas
    variance = tensors[square_sum_key] / replicas - mean.square()
    tolerance = 16 * torch.finfo(mean.dtype).eps * torch.maximum(
        torch.ones_like(mean), mean.square()
    )
    if torch.any(variance.abs() > tolerance):
        raise ValueError(f"replicated constant {sum_key!r} differs across ranks")
    return mean


def _finalize_replicated_constant(
    tensors: dict[str, torch.Tensor], counts: dict[str, int]
) -> torch.Tensor:
    if set(tensors) != {"sum", "square_sum"} or set(counts) != {"replicas"}:
        raise ValueError("replicated_constant_v1 has an invalid schema")
    return _constant_mean(
        tensors,
        counts,
        sum_key="sum",
        square_sum_key="square_sum",
        replica_key="replicas",
    )


def _schema_ids(mapping, prefix):
    return {
        key[len(prefix) :]
        for key in mapping
        if key.startswith(prefix)
    }


def _vq_schema(
    tensors: dict[str, torch.Tensor], counts: dict[str, int]
) -> tuple[set[str], set[str]]:
    slot_ids = _schema_ids(tensors, "embed_sum:")
    if not slot_ids:
        raise ValueError("vq_composition_v1 requires at least one slot")
    expected_tensors = set()
    expected_counts = set()
    for slot_id in slot_ids:
        base_tensors = {
            f"embed_sum:{slot_id}",
            f"commit_sum:{slot_id}",
            f"slot_weight_sum:{slot_id}",
            f"slot_weight_square_sum:{slot_id}",
            f"commit_weight_sum:{slot_id}",
            f"commit_weight_square_sum:{slot_id}",
        }
        base_counts = {
            f"elements:{slot_id}",
            f"vectors:{slot_id}",
            f"constant_replicas:{slot_id}",
        }
        expected_tensors.update(base_tensors)
        expected_counts.update(base_counts)
        entropy_tensors = {
            f"probability_sum:{slot_id}",
            f"entropy_weight_sum:{slot_id}",
            f"entropy_weight_square_sum:{slot_id}",
        }
        present_entropy = entropy_tensors & set(tensors)
        if present_entropy:
            if present_entropy != entropy_tensors:
                raise ValueError(
                    f"VQ slot {slot_id!r} has an incomplete entropy schema"
                )
            expected_tensors.update(entropy_tensors)
    if set(tensors) != expected_tensors or set(counts) != expected_counts:
        raise ValueError("vq_composition_v1 has an invalid schema")
    return expected_tensors, expected_counts


def _composition_schema(
    tensors: dict[str, torch.Tensor],
    counts: dict[str, int],
    group: str,
) -> tuple[set[str], set[str], bool]:
    expected_tensors = set()
    expected_counts = set()
    has_terms = False
    for item_id in _schema_ids(tensors, f"mean_sum:{group}."):
        prefix = f"{group}.{item_id}"
        expected_tensors.update(
            {
                f"mean_sum:{prefix}",
                f"mean_weight_sum:{prefix}",
                f"mean_weight_square_sum:{prefix}",
            }
        )
        expected_counts.update(
            {f"mean_count:{prefix}", f"mean_replicas:{prefix}"}
        )
        has_terms = True
    for item_id in _schema_ids(tensors, f"reg_sum:{group}."):
        prefix = f"{group}.{item_id}"
        expected_tensors.update(
            {
                f"reg_sum:{prefix}",
                f"reg_weight_sum:{prefix}",
                f"reg_weight_square_sum:{prefix}",
            }
        )
        expected_counts.update(
            {f"reg_cells:{prefix}", f"reg_replicas:{prefix}"}
        )
        has_terms = True
    vq_prefix = f"vq:{group}."
    vq_ids = {
        key[len(vq_prefix) :].split(":", 1)[0]
        for key in tensors
        if key.startswith(vq_prefix)
    }
    for item_id in vq_ids:
        nested_tensors, nested_counts = _extract_nested_stats(
            tensors, counts, "vq", f"{group}.{item_id}"
        )
        _vq_schema(nested_tensors, nested_counts)
        expected_tensors.update(
            f"vq:{group}.{item_id}:{key}" for key in nested_tensors
        )
        expected_counts.update(
            f"vq:{group}.{item_id}:{key}" for key in nested_counts
        )
        has_terms = True
    return expected_tensors, expected_counts, has_terms


def _validate_sufficient_stats_schema(
    finalizer_id: str,
    tensors: dict[str, torch.Tensor],
    counts: dict[str, int],
) -> None:
    simple_schemas = {
        "policy_mean_square_v1": (
            {"valid_logit_sum"},
            {"valid_cells"},
        ),
        "mean_v1": ({"sum"}, {"count"}),
        "replicated_constant_v1": (
            {"sum", "square_sum"},
            {"replicas"},
        ),
    }
    if finalizer_id in simple_schemas:
        expected_tensors, expected_counts = simple_schemas[finalizer_id]
        if set(tensors) != expected_tensors or set(counts) != expected_counts:
            raise ValueError(
                f"{finalizer_id} has an invalid tensor/count schema"
            )
        return
    if finalizer_id == "vq_composition_v1":
        _vq_schema(tensors, counts)
        return
    if finalizer_id == "loss_composition_v1":
        expected_tensors, expected_counts, has_terms = _composition_schema(
            tensors, counts, "all"
        )
        if (
            not has_terms
            or set(tensors) != expected_tensors
            or set(counts) != expected_counts
        ):
            raise ValueError(
                "loss_composition_v1 has an invalid tensor/count schema"
            )
        return
    if finalizer_id == "kd_total_v1":
        expected_tensors = {
            "kd_alpha_sum",
            "kd_alpha_square_sum",
            "kd_temperature_sum",
            "kd_temperature_square_sum",
        }
        expected_counts = {"kd_config_replicas"}
        required_groups = {}
        for group in ("kd", "real", "independent"):
            group_tensors, group_counts, has_terms = _composition_schema(
                tensors, counts, group
            )
            expected_tensors.update(group_tensors)
            expected_counts.update(group_counts)
            required_groups[group] = has_terms
        if (
            not required_groups["kd"]
            or not required_groups["real"]
            or set(tensors) != expected_tensors
            or set(counts) != expected_counts
        ):
            raise ValueError("kd_total_v1 has an invalid tensor/count schema")
        return
    raise ValueError(f"unknown sufficient-statistic finalizer {finalizer_id!r}")


def _finalize_vq_composition(
    tensors: dict[str, torch.Tensor], counts: dict[str, int]
) -> torch.Tensor:
    slot_ids = _schema_ids(tensors, "embed_sum:")
    if not slot_ids:
        raise ValueError("vq_composition_v1 requires at least one slot")
    result = None
    expected_tensors = set()
    expected_counts = set()
    for slot_id in sorted(slot_ids):
        tensor_keys = {
            "embed_sum": f"embed_sum:{slot_id}",
            "commit_sum": f"commit_sum:{slot_id}",
            "slot_weight_sum": f"slot_weight_sum:{slot_id}",
            "slot_weight_square_sum": f"slot_weight_square_sum:{slot_id}",
            "commit_weight_sum": f"commit_weight_sum:{slot_id}",
            "commit_weight_square_sum": f"commit_weight_square_sum:{slot_id}",
        }
        count_keys = {
            "elements": f"elements:{slot_id}",
            "vectors": f"vectors:{slot_id}",
            "constant_replicas": f"constant_replicas:{slot_id}",
        }
        expected_tensors.update(tensor_keys.values())
        expected_counts.update(count_keys.values())
        elements = counts.get(count_keys["elements"], 0)
        vectors = counts.get(count_keys["vectors"], 0)
        if elements <= 0 or vectors <= 0:
            raise ValueError(f"VQ slot {slot_id!r} has no selected entries")
        slot_weight = _constant_mean(
            tensors,
            counts,
            sum_key=tensor_keys["slot_weight_sum"],
            square_sum_key=tensor_keys["slot_weight_square_sum"],
            replica_key=count_keys["constant_replicas"],
        )
        commitment_weight = _constant_mean(
            tensors,
            counts,
            sum_key=tensor_keys["commit_weight_sum"],
            square_sum_key=tensor_keys["commit_weight_square_sum"],
            replica_key=count_keys["constant_replicas"],
        )
        slot_value = (
            tensors[tensor_keys["embed_sum"]]
            + commitment_weight * tensors[tensor_keys["commit_sum"]]
        ) / elements

        probability_key = f"probability_sum:{slot_id}"
        entropy_sum_key = f"entropy_weight_sum:{slot_id}"
        entropy_square_key = f"entropy_weight_square_sum:{slot_id}"
        entropy_keys = {probability_key, entropy_sum_key, entropy_square_key}
        has_entropy = entropy_keys & set(tensors)
        if has_entropy:
            if has_entropy != entropy_keys:
                raise ValueError(f"VQ slot {slot_id!r} has an incomplete entropy schema")
            expected_tensors.update(entropy_keys)
            entropy_weight = _constant_mean(
                tensors,
                counts,
                sum_key=entropy_sum_key,
                square_sum_key=entropy_square_key,
                replica_key=count_keys["constant_replicas"],
            )
            probability = tensors[probability_key] / vectors
            entropy_regularizer = (
                probability * torch.log(probability.clamp_min(1e-7))
            ).sum()
            slot_value = slot_value + entropy_weight * entropy_regularizer
        weighted = slot_weight * slot_value
        result = weighted if result is None else result + weighted

    if set(tensors) != expected_tensors or set(counts) != expected_counts:
        raise ValueError("vq_composition_v1 has an invalid schema")
    return result


def _extract_nested_stats(
    tensors: dict[str, torch.Tensor],
    counts: dict[str, int],
    prefix: str,
    item_id: str,
):
    tensor_prefix = f"{prefix}:{item_id}:"
    count_prefix = f"{prefix}:{item_id}:"
    nested_tensors = {
        key[len(tensor_prefix) :]: value
        for key, value in tensors.items()
        if key.startswith(tensor_prefix)
    }
    nested_counts = {
        key[len(count_prefix) :]: value
        for key, value in counts.items()
        if key.startswith(count_prefix)
    }
    return nested_tensors, nested_counts


def _finalize_composition_group(
    tensors: dict[str, torch.Tensor],
    counts: dict[str, int],
    group: str,
    expected_tensors: set[str],
    expected_counts: set[str],
) -> torch.Tensor | None:
    result = None
    mean_prefix = f"mean_sum:{group}."
    mean_ids = _schema_ids(tensors, mean_prefix)
    for item_id in sorted(mean_ids):
        prefix = f"{group}.{item_id}"
        sum_key = f"mean_sum:{prefix}"
        weight_sum_key = f"mean_weight_sum:{prefix}"
        weight_square_key = f"mean_weight_square_sum:{prefix}"
        count_key = f"mean_count:{prefix}"
        replica_key = f"mean_replicas:{prefix}"
        expected_tensors.update({sum_key, weight_sum_key, weight_square_key})
        expected_counts.update({count_key, replica_key})
        if counts.get(count_key, 0) <= 0:
            raise ValueError(f"composition mean {prefix!r} has no values")
        weight = _constant_mean(
            tensors,
            counts,
            sum_key=weight_sum_key,
            square_sum_key=weight_square_key,
            replica_key=replica_key,
        )
        value = weight * tensors[sum_key] / counts[count_key]
        result = value if result is None else result + value

    reg_prefix = f"reg_sum:{group}."
    reg_ids = _schema_ids(tensors, reg_prefix)
    for item_id in sorted(reg_ids):
        prefix = f"{group}.{item_id}"
        sum_key = f"reg_sum:{prefix}"
        weight_sum_key = f"reg_weight_sum:{prefix}"
        weight_square_key = f"reg_weight_square_sum:{prefix}"
        cell_key = f"reg_cells:{prefix}"
        replica_key = f"reg_replicas:{prefix}"
        expected_tensors.update({sum_key, weight_sum_key, weight_square_key})
        expected_counts.update({cell_key, replica_key})
        if counts.get(cell_key, 0) <= 0:
            raise ValueError(f"composition regularizer {prefix!r} has no cells")
        weight = _constant_mean(
            tensors,
            counts,
            sum_key=weight_sum_key,
            square_sum_key=weight_square_key,
            replica_key=replica_key,
        )
        value = weight * (tensors[sum_key] / counts[cell_key]).square()
        result = value if result is None else result + value

    vq_prefix = f"vq:{group}."
    vq_ids = {
        key[len(vq_prefix) :].split(":", 1)[0]
        for key in tensors
        if key.startswith(vq_prefix)
    }
    for item_id in sorted(vq_ids):
        nested_tensors, nested_counts = _extract_nested_stats(
            tensors, counts, "vq", f"{group}.{item_id}"
        )
        expected_tensors.update(
            f"vq:{group}.{item_id}:{key}" for key in nested_tensors
        )
        expected_counts.update(
            f"vq:{group}.{item_id}:{key}" for key in nested_counts
        )
        value = _finalize_vq_composition(nested_tensors, nested_counts)
        result = value if result is None else result + value
    return result


def _finalize_loss_composition(
    tensors: dict[str, torch.Tensor], counts: dict[str, int]
) -> torch.Tensor:
    expected_tensors: set[str] = set()
    expected_counts: set[str] = set()
    result = _finalize_composition_group(
        tensors, counts, "all", expected_tensors, expected_counts
    )
    if result is None:
        raise ValueError("loss_composition_v1 requires at least one term")
    if set(tensors) != expected_tensors or set(counts) != expected_counts:
        raise ValueError("loss_composition_v1 has an invalid tensor/count schema")
    return result


def _finalize_kd_total(
    tensors: dict[str, torch.Tensor], counts: dict[str, int]
) -> torch.Tensor:
    config_tensors = {
        "kd_alpha_sum",
        "kd_alpha_square_sum",
        "kd_temperature_sum",
        "kd_temperature_square_sum",
    }
    config_counts = {"kd_config_replicas"}
    if not config_tensors <= set(tensors) or not config_counts <= set(counts):
        raise ValueError("kd_total_v1 has an invalid configuration schema")
    alpha = _constant_mean(
        tensors,
        counts,
        sum_key="kd_alpha_sum",
        square_sum_key="kd_alpha_square_sum",
        replica_key="kd_config_replicas",
    )
    temperature = _constant_mean(
        tensors,
        counts,
        sum_key="kd_temperature_sum",
        square_sum_key="kd_temperature_square_sum",
        replica_key="kd_config_replicas",
    )
    expected_tensors = set(config_tensors)
    expected_counts = set(config_counts)
    groups = {
        group: _finalize_composition_group(
            tensors, counts, group, expected_tensors, expected_counts
        )
        for group in ("kd", "real", "independent")
    }
    if groups["kd"] is None or groups["real"] is None:
        raise ValueError("kd_total_v1 requires KD and real target terms")
    zero = groups["kd"].new_zeros(())
    result = (
        alpha * temperature.square() * groups["kd"]
        + (1 - alpha) * groups["real"]
        + (groups["independent"] if groups["independent"] is not None else zero)
    )
    if set(tensors) != expected_tensors or set(counts) != expected_counts:
        raise ValueError("kd_total_v1 has an invalid tensor/count schema")
    return result


FINALIZERS = {
    "policy_mean_square_v1": _finalize_mean_square,
    "mean_v1": _finalize_mean,
    "kd_total_v1": _finalize_kd_total,
    "replicated_constant_v1": _finalize_replicated_constant,
    "vq_composition_v1": _finalize_vq_composition,
    "loss_composition_v1": _finalize_loss_composition,
}


@dataclass(frozen=True)
class PreparedPipelineResult:
    data: dict
    next_state: Any


@dataclass(frozen=True)
class PreparedPipelineBatch:
    data: dict
    composite_blob: bytes


class PipelineStateComposer:
    """Purely prepare an ordered pipeline chain and one immutable next blob."""

    def __init__(self, pipelines):
        self.pipelines = tuple(pipelines)
        identifiers = [pipeline.pipeline_id for pipeline in self.pipelines]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate pipeline identifiers: {identifiers}")
        states = []
        self._signature_fingerprints = []
        produced_fields = set()
        for pipeline in self.pipelines:
            if not getattr(pipeline, "pipeline_id", ""):
                raise ValueError(f"opaque pipeline {type(pipeline).__name__} lacks pipeline_id")
            if not isinstance(getattr(pipeline, "schema_version", None), int):
                raise ValueError(f"pipeline {pipeline.pipeline_id} lacks schema_version")
            signature = canonical_pipeline_state_bytes(pipeline.signature_state())
            self._signature_fingerprints.append(hashlib.sha256(signature).digest())
            input_names = {field.name for field in pipeline.input_fields}
            output_names = [field.name for field in pipeline.output_fields]
            if len(set(output_names)) != len(output_names):
                raise ValueError(
                    f"pipeline {pipeline.pipeline_id} declares duplicate output fields"
                )
            collisions = produced_fields.intersection(output_names)
            if collisions:
                raise ValueError(
                    f"pipeline {pipeline.pipeline_id} output collision: "
                    f"{sorted(collisions)}"
                )
            illegal_overwrite = set(output_names).intersection(input_names)
            if illegal_overwrite:
                raise ValueError(
                    f"pipeline {pipeline.pipeline_id} overwrites input fields: "
                    f"{sorted(illegal_overwrite)}"
                )
            produced_fields.update(output_names)
            state = pipeline.initial_state() if hasattr(pipeline, "initial_state") else None
            states.append(
                (
                    pipeline.pipeline_id,
                    pipeline.schema_version,
                    canonical_pipeline_state_bytes(state).hex(),
                )
            )
        self.initial_blob = canonical_pipeline_state_bytes(states)

    def _decode_blob(self, blob: bytes):
        values = decode_canonical_pipeline_state(blob)
        if len(values) != len(self.pipelines):
            raise ValueError("pipeline composite state length changed")
        states = []
        for pipeline, value in zip(self.pipelines, values):
            pipeline_id, schema_version, state_hex = value
            if pipeline_id != pipeline.pipeline_id or schema_version != pipeline.schema_version:
                raise ValueError(
                    f"pipeline composite descriptor changed for {pipeline.pipeline_id}"
                )
            states.append(decode_canonical_pipeline_state(bytes.fromhex(state_hex)))
        return states

    @staticmethod
    def _field_value_digest(value) -> bytes:
        digest = hashlib.sha256()

        def update(item):
            if isinstance(item, torch.Tensor):
                array = item.detach().cpu().contiguous().numpy()
                digest.update(b"tensor\0")
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(repr(array.shape).encode("ascii"))
                digest.update(array.tobytes())
            elif isinstance(item, np.ndarray):
                array = np.ascontiguousarray(item)
                digest.update(b"array\0")
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(repr(array.shape).encode("ascii"))
                if array.dtype.kind == "O":
                    for nested in array.flat:
                        update(nested)
                else:
                    digest.update(array.tobytes())
            elif isinstance(item, (list, tuple)):
                digest.update(type(item).__name__.encode("ascii") + b"\0")
                digest.update(len(item).to_bytes(8, "little"))
                for nested in item:
                    update(nested)
            elif isinstance(item, dict):
                digest.update(b"dict\0")
                for key in sorted(item):
                    update(key)
                    update(item[key])
            elif isinstance(item, (str, bytes, int, float, bool, type(None))):
                digest.update(type(item).__name__.encode("ascii") + b"\0")
                digest.update(repr(item).encode("utf-8"))
            else:
                digest.update(type(item).__qualname__.encode("utf-8") + b"\0")
                digest.update(repr(item).encode("utf-8"))

        update(value)
        return digest.digest()

    @staticmethod
    def _validate_fields(
        data, specs, batch_size, stage, *, reject_unknown=False
    ):
        specs = tuple(specs)
        if reject_unknown:
            unknown = sorted(
                set(data).difference(spec.name for spec in specs)
            )
            if unknown:
                raise ValueError(
                    f"{stage} contains undeclared field(s): {unknown}"
                )
        board_sizes = None
        if "board_size" in data:
            raw_board_sizes = (
                data["board_size"].detach().cpu().numpy()
                if isinstance(data["board_size"], torch.Tensor)
                else np.asarray(data["board_size"])
            )
            if raw_board_sizes.shape == (batch_size, 2):
                board_sizes = raw_board_sizes.astype(np.int64, copy=False)
        for spec in specs:
            if spec.name not in data:
                if spec.required:
                    raise ValueError(f"{stage} is missing required field {spec.name!r}")
                continue
            value = data[spec.name]
            if isinstance(value, torch.Tensor):
                shape = tuple(value.shape)
                dtype_kind = value.detach().cpu().numpy().dtype.kind
                if value.is_complex():
                    dtype_kind = "c"
            else:
                array = np.asarray(value)
                shape = array.shape
                dtype_kind = array.dtype.kind
            if dtype_kind not in spec.dtype_kinds:
                raise ValueError(
                    f"{stage} field {spec.name!r} has dtype kind {dtype_kind!r}; "
                    f"expected {spec.dtype_kinds}"
                )
            if spec.scope == "per_sample":
                if not shape or shape[0] != batch_size:
                    raise ValueError(
                        f"{stage} per-sample field {spec.name!r} must have leading "
                        f"dimension {batch_size}, got {shape}"
                    )
                if spec.collate == "list" and (
                    not isinstance(value, (list, tuple)) or len(value) != batch_size
                ):
                    raise ValueError(
                        f"{stage} list-collated field {spec.name!r} must contain "
                        f"{batch_size} rows"
                    )
            else:
                if spec.collate == "list" and not isinstance(
                    value, (list, tuple)
                ):
                    raise ValueError(
                        f"{stage} batch-shared list field {spec.name!r} "
                        "must be a list"
                    )
                if not shape or shape[0] != batch_size:
                    raise ValueError(
                        f"{stage} batch-shared field {spec.name!r} must "
                        f"retain leading dimension {batch_size}, got {shape}"
                    )
                first = value[0]
                for index in range(1, batch_size):
                    if isinstance(value, torch.Tensor):
                        equal = torch.equal(value[index], first)
                    else:
                        equal = np.array_equal(
                            np.asarray(value[index]), np.asarray(first)
                        )
                    if not equal:
                        raise ValueError(
                            f"{stage} batch-shared field {spec.name!r} differs "
                            f"at row {index}"
                        )
            spatial_axes = []
            for axis in spec.spatial_axes or ():
                normalized = (
                    axis + 1
                    if spec.scope == "per_sample" and axis >= 0
                    else (axis if axis >= 0 else len(shape) + axis)
                )
                if not 0 <= normalized < len(shape):
                    raise ValueError(
                        f"{stage} field {spec.name!r} spatial axis {axis} "
                        f"is invalid for shape {shape}"
                    )
                spatial_axes.append(normalized)
            if spatial_axes and board_sizes is not None:
                spatial_shape = tuple(shape[axis] for axis in spatial_axes)
                physical = np.asarray(spatial_shape, dtype=np.int64)
                compared_axes = min(len(physical), board_sizes.shape[1])
                mismatched = np.flatnonzero(
                    np.any(
                        physical[None, :compared_axes]
                        < board_sizes[:, :compared_axes],
                        axis=1,
                    )
                )
                if mismatched.size:
                    raise ValueError(
                        f"{stage} field {spec.name!r} spatial shape {spatial_shape} "
                        f"does not match board_size at row {int(mismatched[0])}"
                    )
            if spec.side_axis is not None:
                if spec.side_axis >= 0:
                    normalized = spec.side_axis + (
                        1 if spec.scope == "per_sample" else 0
                    )
                else:
                    normalized = len(shape) + spec.side_axis
                if not 0 <= normalized < len(shape):
                    raise ValueError(
                        f"{stage} field {spec.name!r} side axis {spec.side_axis} "
                        f"is invalid for shape {shape}"
                    )
                if shape[normalized] <= 0:
                    raise ValueError(
                        f"{stage} field {spec.name!r} has an empty side axis"
                    )
                if normalized in spatial_axes:
                    raise ValueError(
                        f"{stage} field {spec.name!r} side axis aliases "
                        "a spatial axis"
                    )
            if spec.transform == "board_policy":
                if board_sizes is None:
                    raise ValueError(
                        f"{stage} field {spec.name!r} requires board_size"
                    )
                spatial_policy = (
                    len(shape) == 3
                    and np.all(
                        board_sizes
                        == np.asarray(shape[-2:], dtype=np.int64)[None, :]
                    )
                )
                logical_policy_sizes = board_sizes[:, 0] * board_sizes[:, 1]
                physical_policy_size = (
                    int(data["board_input"].shape[-2])
                    * int(data["board_input"].shape[-1])
                )
                flat_policy = (
                    len(shape) == 2
                    and np.all(
                        (shape[1] == logical_policy_sizes)
                        | (shape[1] == logical_policy_sizes + 1)
                        | (shape[1] == physical_policy_size)
                        | (shape[1] == physical_policy_size + 1)
                    )
                )
                if not (spatial_policy or flat_policy):
                    raise ValueError(
                        f"{stage} field {spec.name!r} is neither a board plane "
                        "nor a flattened board policy"
                    )
            elif spec.transform == "flat_policy_with_pass":
                if board_sizes is None:
                    raise ValueError(
                        f"{stage} field {spec.name!r} requires board_size"
                    )
                if len(shape) != 2 or np.any(
                    shape[1] != board_sizes[:, 0] * board_sizes[:, 1] + 1
                ):
                    raise ValueError(
                        f"{stage} field {spec.name!r} must contain one flattened "
                        "board policy plus a pass entry"
                    )
            elif spec.transform == "moves":
                if spec.collate != "list":
                    raise ValueError(
                        f"{stage} move field {spec.name!r} requires list collation"
                    )

    def prepare_batch(
        self,
        batch: dict,
        *,
        sample_keys: tuple,
        rng_keys: tuple,
        current_blob: bytes | None = None,
    ) -> PreparedPipelineBatch:
        if len(sample_keys) != len(rng_keys):
            raise ValueError("sample_keys and rng_keys must have equal length")
        blob = self.initial_blob if current_blob is None else current_blob
        states = self._decode_blob(blob)
        data = dict(batch)
        next_entries = []
        for index, (pipeline, current_state) in enumerate(zip(self.pipelines, states)):
            before_signature = canonical_pipeline_state_bytes(pipeline.signature_state())
            if hashlib.sha256(before_signature).digest() != self._signature_fingerprints[index]:
                raise RuntimeError(f"pipeline {pipeline.pipeline_id} signature state mutated")
            input_state_bytes = canonical_pipeline_state_bytes(current_state)
            isolated_state = decode_canonical_pipeline_state(input_state_bytes)
            before_keys = set(data)
            before_field_digests = {
                key: self._field_value_digest(value)
                for key, value in data.items()
            }
            self._validate_fields(
                data,
                pipeline.input_fields,
                len(sample_keys),
                f"pipeline {pipeline.pipeline_id} input",
            )
            if hasattr(pipeline, "initial_state"):
                result = pipeline.prepare_batch(
                    data,
                    current_state=isolated_state,
                    sample_keys=sample_keys,
                    rng_keys=rng_keys,
                )
                if not isinstance(result, PreparedPipelineResult):
                    raise TypeError(
                        f"pipeline {pipeline.pipeline_id} returned non-prepared result"
                    )
                data = result.data
                next_state = result.next_state
            else:
                data = pipeline.process_batch(
                    data,
                    sample_keys=sample_keys,
                    rng_keys=rng_keys,
                )
                next_state = None
            if not isinstance(data, dict):
                raise TypeError(
                    f"pipeline {pipeline.pipeline_id} must return a field dictionary"
                )
            output_names = {field.name for field in pipeline.output_fields}
            overwritten = before_keys.intersection(output_names)
            undeclared = set(data).difference(before_keys).difference(output_names)
            removed = before_keys.difference(data)
            mutated = sorted(
                key
                for key in before_keys.difference(output_names)
                if key in data
                and self._field_value_digest(data[key])
                != before_field_digests[key]
            )
            if overwritten:
                raise ValueError(
                    f"pipeline {pipeline.pipeline_id} overwrote existing fields: "
                    f"{sorted(overwritten)}"
                )
            if undeclared:
                raise ValueError(
                    f"pipeline {pipeline.pipeline_id} produced undeclared fields: "
                    f"{sorted(undeclared)}"
                )
            if removed:
                raise ValueError(
                    f"pipeline {pipeline.pipeline_id} removed fields: {sorted(removed)}"
                )
            if mutated:
                raise ValueError(
                    f"pipeline {pipeline.pipeline_id} mutated undeclared fields: "
                    f"{mutated}"
                )
            self._validate_fields(
                data,
                pipeline.output_fields,
                len(sample_keys),
                f"pipeline {pipeline.pipeline_id} output",
            )
            if canonical_pipeline_state_bytes(isolated_state) != input_state_bytes:
                raise RuntimeError(f"pipeline {pipeline.pipeline_id} mutated its input state")
            after_signature = canonical_pipeline_state_bytes(pipeline.signature_state())
            if after_signature != before_signature:
                raise RuntimeError(f"pipeline {pipeline.pipeline_id} mutated declared state")
            next_entries.append(
                (
                    pipeline.pipeline_id,
                    pipeline.schema_version,
                    canonical_pipeline_state_bytes(next_state).hex(),
                )
            )
        return PreparedPipelineBatch(data, canonical_pipeline_state_bytes(next_entries))
