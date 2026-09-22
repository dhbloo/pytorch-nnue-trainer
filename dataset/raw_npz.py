"""Budgeted, vectorized loading of dense native KataGo NPZ archives."""

import math
import os
import zipfile

import numpy as np

from . import DATASETS
from .decoder import ProcessedNpzDecoder, _read_npy_array_header
from .katago import BatchedProcessedKatagoNumpyDataset
from .stream import _sha256_file
from .telemetry import ObservedProcessedNpzDecoder


class RawNpzDecoder(ProcessedNpzDecoder):
    """Normalize native arrays in memory and reuse the dense decoding runtime."""

    format_id = "batched-raw-katago-npz"
    schema_version = 1
    _raw_keys = (
        "binaryInputNCHWPacked", "globalInputNC", "globalTargetsNC",
        "policyTargetsNCMove",
    )

    def __init__(self, *args, value_td_level=0, **kwargs):
        if type(value_td_level) is not int or value_td_level < 0:
            raise ValueError("value_td_level must be a non-negative integer")
        for key in ("filter_stm", "filter_condition", "board_input_channels",
                    "stm_input_channel", "value_target_channels"):
            if kwargs.get(key) is not None:
                raise ValueError(f"batched raw NPZ does not support {key}; use iterative_katago_numpy")
        super().__init__(*args, **kwargs)
        self.value_td_level = value_td_level

    def signature_state(self):
        return {**super().signature_state(), "value_td_level": self.value_td_level}

    def configure_mapped_shards(self, catalog):
        raise ValueError("native raw NPZ uses the in-memory decoded cache, not processed mapped shards")

    def _native_schemas(self, path):
        schemas = {}
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            for key in self._raw_keys:
                # Native writers also use extensionless NPY member names.
                info = archive.getinfo(key if key in members else f"{key}.npy")
                with archive.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    shape, _, dtype = _read_npy_array_header(member, version)
                    offset = member.tell()
                if dtype.kind not in "biuf" or dtype.itemsize > 8 or any(n < 0 for n in shape):
                    raise ValueError(f"invalid native NPZ schema for {key}")
                if offset + math.prod(shape) * dtype.itemsize > info.file_size:
                    raise ValueError(f"truncated native NPZ array {key}")
                schemas[key] = (shape, dtype)
        packed, packed_dtype = schemas[self._raw_keys[0]]
        global_input, _ = schemas[self._raw_keys[1]]
        targets, _ = schemas[self._raw_keys[2]]
        policy, _ = schemas[self._raw_keys[3]]
        if len(policy) != 3 or policy[1] < 1 or policy[2] < 2:
            raise ValueError("native policy requires (N,C,H*W+1)")
        size = math.isqrt(policy[2] - 1)
        length = policy[0]
        if size * size + 1 != policy[2] or length <= 0:
            raise ValueError("native policy requires nonempty square boards")
        if (len(packed) != 3 or packed[1] < 3 or packed[2] != (size * size + 7) // 8
                or packed_dtype != np.dtype(np.uint8)):
            raise ValueError("native packed board has incompatible shape or dtype")
        if len(global_input) != 2 or not (global_input[1] == 1 or global_input[1] >= 6):
            raise ValueError("native global inputs do not contain STM")
        if len(targets) != 2 or targets[1] < self.value_td_level * 4 + 3:
            raise ValueError("native global targets do not contain the requested value TD level")
        if any(shape[0] != length for shape, _ in schemas.values()):
            raise ValueError("native NPZ fields have unequal row counts")
        return length, size, schemas

    def inspect(self, path, file_ordinal):
        canonical = os.path.abspath(path)
        length, size, raw_schemas = self._native_schemas(canonical)
        policy_shape = (size * size + 1,) if self.has_pass_move else (size, size)
        schemas = {
            "bf": ((length, 2, size, size), np.dtype(np.int8)),
            "gf": ((length, 1), np.dtype(np.float32)),
            "vt": ((length, 3), np.dtype(np.float32)),
            "pt": ((length, *policy_shape), np.dtype(np.float32)),
        }
        accepted = (size, size) in self.boardsizes
        decoded_bytes = self._schema_payload_bytes(schemas) if accepted else 0
        raw_bytes = self._schema_payload_bytes(raw_schemas)
        # Raw members, normalization output/copies, unpacked board/mask and
        # policy arithmetic temporaries can coexist during a cache miss.
        temporary_bytes = length * (8 * size * size + 8 * (size * size + 1) + 64)
        inflight_bytes = raw_bytes + 2 * decoded_bytes + temporary_bytes
        with self._array_cache_condition:
            self._decoded_file_size_catalog[canonical] = (decoded_bytes, inflight_bytes)
        stat = os.stat(canonical)
        return {
            "path": canonical, "file_ordinal": int(file_ordinal),
            "file_sha256": _sha256_file(canonical), "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "logical_row_count": length if accepted else 0,
            "board_size": self.fixed_board_size or (size, size),
            "decoded_file_bytes": decoded_bytes, "inflight_file_bytes": inflight_bytes,
            "output_row_bytes": self._output_row_payload_bytes(schemas, (size, size)),
        }

    def _load_uncached(self, canonical, *, before_heap_inflate=None, before_mmap_transform=None):
        if before_heap_inflate is not None:
            before_heap_inflate()
        length, size, _ = self._native_schemas(canonical)
        with np.load(canonical, allow_pickle=False) as source:
            packed = source["binaryInputNCHWPacked"]
            planes = np.unpackbits(packed[:, :3], axis=2, count=size * size, bitorder="big")
            if not np.all(planes[:, 0]):
                raise ValueError("batched raw NPZ requires full uniform boards; padded boards are unsupported")
            board = np.ascontiguousarray(planes[:, 1:3].reshape(length, 2, size, size), dtype=np.int8)
            global_input = source["globalInputNC"]
            stm = (global_input[:, :1] if global_input.shape[1] == 1
                   else np.where(global_input[:, 5:6] > 0, 1, -1))
            targets = source["globalTargetsNC"]
            base = 4 * self.value_td_level
            policy = np.array(source["policyTargetsNCMove"][:, 0, :size * size + int(self.has_pass_move)],
                              dtype=np.float32, order="C", copy=True)
            policy /= policy.sum(axis=1, keepdims=True) + 1e-9
            arrays = {
                "bf": board,
                "gf": np.ascontiguousarray(stm, dtype=np.float32),
                "vt": np.array(targets[:, base:base + 3], dtype=np.float32, order="C", copy=True),
                "pt": policy if self.has_pass_move else policy.reshape(length, size, size),
            }
        if (size, size) not in self.boardsizes:
            arrays = {key: np.empty((0, *value.shape[1:]), dtype=value.dtype)
                      for key, value in arrays.items()}
        for value in arrays.values():
            value.flags.writeable = False
        return arrays


class ObservedRawNpzDecoder(ObservedProcessedNpzDecoder, RawNpzDecoder):
    """Reuse cache telemetry for the native raw decoder."""


@DATASETS.register("batched_katago_numpy")
class BatchedKatagoNumpyDataset(BatchedProcessedKatagoNumpyDataset):
    """Dense native NPZ with bounded caching and packed batch prefetch."""

    def __init__(self, *args, value_td_level=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_kwargs["value_td_level"] = value_td_level

    def _make_record_decoder(self, telemetry_enabled, stats, **kwargs):
        if telemetry_enabled:
            return ObservedRawNpzDecoder(pipeline_stats=stats, **kwargs)
        return RawNpzDecoder(**kwargs)

    def _node_decoded_cache_is_supported(self):
        return False
