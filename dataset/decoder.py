from __future__ import annotations

import os
import copy
import threading
import struct
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import numpy as np

try:
    from numpy.lib._format_impl import _read_array_header as _read_npy_array_header
except ImportError:  # NumPy < 2.0
    from numpy.lib.format import _read_array_header as _read_npy_array_header

from utils.data_utils import post_process_batch, post_process_data

from .core import (
    DatasetRuntimeContext,
    _uniform_below_for_packed_processed_rows_v2,
    _uniform_below_for_processed_sample_keys_v2,
    uniform_below,
)
from .filter import evaluate_filter_condition
from .host_memory import (
    HostMemoryBudget,
    HostMemoryBudgetExceeded,
    HostMemoryCategory,
)
from .stream import (
    _sha256_file,
)


_CACHE_CONFIG_UNSET = object()


class ProcessedNpzCacheAdmissionError(RuntimeError):
    """Raised when configured logical memory cannot admit a decoded load."""


class ProcessedNpzDecoder:
    format_id = "processed-katago-npz"
    schema_version = 3
    file_exts = (".npz",)

    def __init__(
        self,
        *,
        boardsizes: frozenset[tuple[int, int]] | set[tuple[int, int]],
        runtime_context: DatasetRuntimeContext,
        fixed_side_input: bool = False,
        fixed_board_size: tuple[int, int] | None = None,
        has_pass_move: bool = False,
        apply_symmetry: bool | str = False,
        filter_stm: int | None = None,
        filter_condition: str | None = None,
        board_input_channels: list[int] | None = None,
        stm_input_channel: int | None = None,
        value_target_channels: list[int] | None = None,
    ):
        self.boardsizes = frozenset(boardsizes)
        self.runtime_context = runtime_context
        self.fixed_side_input = bool(fixed_side_input)
        self.fixed_board_size = fixed_board_size
        self.has_pass_move = bool(has_pass_move)
        self.apply_symmetry = apply_symmetry
        self.filter_stm = filter_stm
        self.filter_condition = filter_condition
        self.board_input_channels = board_input_channels
        self.stm_input_channel = stm_input_channel
        self.value_target_channels = value_target_channels
        # Keep one shared file per decode worker. Multi-file batches gather
        # one path at a time, so this avoids duplicate array loading without
        # retaining arrays in proportion to the dataset's file count.
        self._array_cache_capacity = 6
        self._array_cache_byte_capacity = 1152 * 1024 * 1024
        self._mmap_cache_capacity = 16
        self._array_cache_values = OrderedDict()
        self._array_cache_reservations = {}
        self._array_cache_loading = set()
        self._array_cache_condition = threading.Condition()
        self._host_memory_budget = None
        self._decoded_file_size_catalog = {}
        self._cache_closed = False
        self._validated_stored_npz_paths = set()
        self._mapped_shards = {}
        self._active_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._active_epoch = int(epoch)

    def configure_cache(
        self,
        *,
        entries: int,
        byte_capacity: int,
        host_memory_budget=_CACHE_CONFIG_UNSET,
        file_size_catalog=_CACHE_CONFIG_UNSET,
    ) -> None:
        """Update cache limits and optional logical-memory accounting.

        ``file_size_catalog`` accepts the dictionaries returned by ``inspect``.
        Omitting both accounting arguments preserves the current configuration;
        explicitly passing a null budget disables accounting. Accounting changes
        retire current cache ownership and are rejected while a load is active.
        """
        if type(entries) is not int or entries <= 0:
            raise ValueError("processed NPZ cache entries must be a positive integer")
        if type(byte_capacity) is not int or byte_capacity <= 0:
            raise ValueError("processed NPZ cache byte capacity must be a positive integer")
        budget_supplied = host_memory_budget is not _CACHE_CONFIG_UNSET
        catalog_supplied = file_size_catalog is not _CACHE_CONFIG_UNSET
        if budget_supplied and host_memory_budget is not None and not isinstance(
            host_memory_budget, HostMemoryBudget
        ):
            raise TypeError("host_memory_budget must be a HostMemoryBudget or null")
        normalized_catalog = (
            self._normalize_file_size_catalog(file_size_catalog)
            if catalog_supplied and file_size_catalog is not None
            else ({} if catalog_supplied else None)
        )
        with self._array_cache_condition:
            if self._cache_closed:
                raise RuntimeError("cannot configure a closed processed NPZ cache")
            next_budget = (
                host_memory_budget
                if budget_supplied
                else self._host_memory_budget
            )
            next_catalog = (
                normalized_catalog
                if catalog_supplied
                else self._decoded_file_size_catalog
            )
            accounting_changed = (
                next_budget is not self._host_memory_budget
                or next_catalog != self._decoded_file_size_catalog
            )
            if accounting_changed and self._array_cache_loading:
                raise RuntimeError(
                    "cannot reconfigure processed NPZ memory accounting while "
                    "a cache load is active"
                )
            if accounting_changed:
                self._retire_all_cache_entries()
                self._host_memory_budget = next_budget
                self._decoded_file_size_catalog = dict(next_catalog)
            self._array_cache_capacity = entries
            self._array_cache_byte_capacity = byte_capacity
            self._evict_array_cache()

    def configure_mapped_shards(self, catalog: Mapping[str, Mapping[str, str]]) -> None:
        """Use run-scoped read-only NPY files for the configured source paths."""

        if not isinstance(catalog, Mapping) or not catalog:
            raise ValueError("mapped shard catalog must be a non-empty mapping")
        if any(
            option is not None
            for option in (
                self.filter_stm,
                self.filter_condition,
                self.board_input_channels,
                self.stm_input_channel,
                self.value_target_channels,
            )
        ):
            raise ValueError("mapped shards do not support load-time array transforms")
        normalized = {}
        for source, arrays in catalog.items():
            if not isinstance(arrays, Mapping):
                raise TypeError("each mapped shard must be an array-path mapping")
            keys = set(arrays)
            missing = {"bf", "vt"}.difference(keys)
            unexpected = keys.difference({"bf", "gf", "vt", "pt"})
            if missing or unexpected:
                raise ValueError("mapped shard array keys are invalid")
            normalized[os.path.abspath(os.fspath(source))] = {
                key: os.path.abspath(os.fspath(path))
                for key, path in arrays.items()
            }
        with self._array_cache_condition:
            if self._cache_closed:
                raise RuntimeError("cannot configure a closed processed NPZ cache")
            if self._array_cache_loading:
                raise RuntimeError(
                    "cannot configure mapped shards while a cache load is active"
                )
            self._retire_all_cache_entries()
            self._mapped_shards = normalized

    def is_mapped_source(self, path: str) -> bool:
        return os.path.abspath(os.fspath(path)) in self._mapped_shards

    @staticmethod
    def _normalize_file_size_catalog(catalog) -> dict[str, tuple[int, int]]:
        if isinstance(catalog, (str, bytes)) or not isinstance(catalog, Iterable):
            raise TypeError("file_size_catalog must be an iterable of inspect catalogs")
        normalized = {}
        for item in catalog:
            if not isinstance(item, Mapping):
                raise TypeError("each file_size_catalog item must be a mapping")
            missing = {"path", "decoded_file_bytes"}.difference(item)
            if missing:
                raise ValueError(
                    "file_size_catalog item is missing " + ", ".join(sorted(missing))
                )
            decoded_bytes = item["decoded_file_bytes"]
            inflight_bytes = item.get("inflight_file_bytes", decoded_bytes)
            for name, value in (
                ("decoded_file_bytes", decoded_bytes),
                ("inflight_file_bytes", inflight_bytes),
            ):
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
            canonical = os.path.abspath(os.fspath(item["path"]))
            sizes = (decoded_bytes, max(decoded_bytes, inflight_bytes))
            previous = normalized.setdefault(canonical, sizes)
            if previous != sizes:
                raise ValueError(
                    "file_size_catalog has conflicting entries for one path"
                )
        return normalized

    def close(self) -> None:
        """Close the cache and retire ownership while preserving active leases."""
        with self._array_cache_condition:
            if self._cache_closed:
                return
            self._cache_closed = True
            self._retire_all_cache_entries()
            self._array_cache_condition.notify_all()

    def cache_state(self) -> dict[str, int]:
        """Return a low-frequency cache snapshot for optional telemetry."""
        with self._array_cache_condition:
            loaded = [
                cached
                for cached in self._array_cache_values.values()
                if not self._is_mmap_cache_entry(cached)
            ]
            return {
                "entries": len(loaded),
                "bytes": sum(
                    array.nbytes for cached in loaded for array in cached.values()
                ),
                "capacity_entries": self._array_cache_capacity,
                "capacity_bytes": self._array_cache_byte_capacity,
            }

    def signature_state(self) -> dict:
        return {
            "format_id": self.format_id,
            "schema_version": self.schema_version,
            "boardsizes": sorted(self.boardsizes),
            "fixed_side_input": self.fixed_side_input,
            "fixed_board_size": self.fixed_board_size,
            "has_pass_move": self.has_pass_move,
            "apply_symmetry": self.apply_symmetry,
            "filter_stm": self.filter_stm,
            "filter_condition": self.filter_condition,
            "board_input_channels": self.board_input_channels,
            "stm_input_channel": self.stm_input_channel,
            "value_target_channels": self.value_target_channels,
            "record_identity": "file-sha256-plus-logical-row-v2",
            "symmetry_rng": "processed-splitmix-v1",
        }

    def _load_uncached(
        self,
        canonical: str,
        *,
        before_heap_inflate=None,
        before_mmap_transform=None,
    ) -> dict[str, np.ndarray]:
        try:
            mapped_paths = self._mapped_shards.get(canonical)
            arrays = (
                {
                    key: np.load(path, mmap_mode="r", allow_pickle=False)
                    for key, path in mapped_paths.items()
                }
                if mapped_paths is not None
                else self._mmap_stored_arrays(canonical)
            )
            mmap_backed = arrays is not None
            if arrays is None:
                if before_heap_inflate is not None:
                    before_heap_inflate()
                with np.load(canonical, allow_pickle=False) as source:
                    arrays = {
                        key: np.ascontiguousarray(source[key])
                        for key in ("bf", "gf", "vt", "pt")
                        if key in source
                    }
            if "bf" not in arrays or "vt" not in arrays:
                missing = [key for key in ("bf", "vt") if key not in arrays]
                raise ValueError(
                    f"{self.format_id} file {canonical} is missing required key(s) {missing}"
                )
        except Exception as exc:
            if isinstance(
                exc,
                (HostMemoryBudgetExceeded, ProcessedNpzCacheAdmissionError),
            ):
                raise
            if isinstance(exc, ValueError) and self.format_id in str(exc):
                raise
            raise RuntimeError(f"failed to inspect {self.format_id} file {canonical}: {exc}") from exc
        bf = arrays["bf"]
        if bf.ndim != 4:
            raise ValueError(
                f"{self.format_id} file {canonical} key 'bf' expected (N,C,H,W), got {bf.shape}"
            )
        length = len(bf)
        if length == 0:
            raise ValueError(f"{self.format_id} file {canonical} key 'bf' is empty")
        for key, array in arrays.items():
            if len(array) != length:
                raise ValueError(
                    f"{self.format_id} file {canonical} key {key!r} has {len(array)} rows, "
                    f"expected {length}"
                )
        board_size = tuple(int(v) for v in bf.shape[-2:])
        if board_size not in self.boardsizes:
            if (
                mapped_paths is None
                and mmap_backed
                and before_mmap_transform is not None
            ):
                before_mmap_transform()
            arrays = {
                key: np.empty((0, *value.shape[1:]), dtype=value.dtype)
                for key, value in arrays.items()
            }
            for array in arrays.values():
                array.flags.writeable = False
            return arrays
        if mapped_paths is not None:
            for array in arrays.values():
                array.flags.writeable = False
            return arrays
        transforms_arrays = any(
            option is not None
            for option in (
                self.filter_stm,
                self.filter_condition,
                self.board_input_channels,
                self.stm_input_channel,
                self.value_target_channels,
            )
        )
        if (
            mmap_backed
            and before_mmap_transform is not None
            and transforms_arrays
        ):
            before_mmap_transform()
        mask = None
        if self.filter_stm is not None:
            if not isinstance(self.filter_stm, int):
                raise ValueError(f"filter_stm must be an integer, got {self.filter_stm!r}")
            if "gf" not in arrays:
                raise ValueError(
                    f"{self.format_id} file {canonical} requires key 'gf' for filter_stm"
                )
            mask = arrays["gf"][:, 0] == self.filter_stm
        if self.filter_condition is not None:
            condition_mask = evaluate_filter_condition(
                self.filter_condition,
                arrays,
            )
            mask = condition_mask if mask is None else mask & condition_mask
        if mask is not None:
            arrays = {key: value[mask] for key, value in arrays.items()}
        if self.board_input_channels is not None:
            arrays["bf"] = arrays["bf"][:, self.board_input_channels]
        if self.stm_input_channel is not None:
            if "gf" not in arrays:
                raise ValueError(
                    f"{self.format_id} file {canonical} has no 'gf' for stm_input_channel"
                )
            arrays["gf"] = arrays["gf"][:, [self.stm_input_channel]]
        if self.value_target_channels is not None:
            arrays["vt"] = arrays["vt"][:, self.value_target_channels]
        for array in arrays.values():
            array.flags.writeable = False
        return arrays

    def _inspect_compressed_headers(self, canonical: str):
        """Read NPZ schemas without inflating their array payloads."""
        with zipfile.ZipFile(canonical) as archive:
            names = set(archive.namelist())
            infos = {
                key: archive.getinfo(f"{key}.npy")
                for key in ("bf", "gf", "vt", "pt")
                if f"{key}.npy" in names
            }
            if not infos:
                return None
            all_members_compressed = all(
                info.compress_type != zipfile.ZIP_STORED
                for info in infos.values()
            )
            missing = [key for key in ("bf", "vt") if key not in infos]
            if missing:
                raise ValueError(
                    f"{self.format_id} file {canonical} is missing required key(s) {missing}"
                )
            schemas = {}
            for key, info in infos.items():
                with archive.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(member)
                        )
                    elif version in {(2, 0), (3, 0)}:
                        shape, _fortran_order, dtype = _read_npy_array_header(
                            member,
                            version,
                        )
                    else:
                        raise ValueError(
                            f"unsupported NPY header version {version!r} in {info.filename!r}"
                        )
                    payload_offset = member.tell()
                if dtype.hasobject:
                    raise ValueError(
                        f"{self.format_id} file {canonical} key {key!r} has an object dtype"
                    )
                if any(extent < 0 for extent in shape):
                    raise ValueError(
                        f"{self.format_id} file {canonical} key {key!r} has a negative extent"
                    )
                element_count = 1
                for extent in shape:
                    element_count *= int(extent)
                expected_size = payload_offset + element_count * dtype.itemsize
                if expected_size > info.file_size:
                    raise ValueError(
                        f"{self.format_id} file {canonical} key {key!r} payload size "
                        f"is {info.file_size - payload_offset} bytes, expected "
                        f"{element_count * dtype.itemsize}"
                    )
                schemas[key] = (tuple(int(v) for v in shape), dtype)
        bf_shape = schemas["bf"][0]
        if len(bf_shape) != 4:
            raise ValueError(
                f"{self.format_id} file {canonical} key 'bf' expected (N,C,H,W), got {bf_shape}"
            )
        length = bf_shape[0]
        if length == 0:
            raise ValueError(f"{self.format_id} file {canonical} key 'bf' is empty")
        for key, (shape, _) in schemas.items():
            if not shape or shape[0] != length:
                actual = 0 if not shape else shape[0]
                raise ValueError(
                    f"{self.format_id} file {canonical} key {key!r} has {actual} rows, "
                    f"expected {length}"
                )
        board_size = bf_shape[-2:]
        return (
            length if board_size in self.boardsizes else 0,
            board_size,
            schemas,
            all_members_compressed,
        )

    @staticmethod
    def _shape_element_count(shape) -> int:
        count = 1
        for extent in shape:
            count *= int(extent)
        return count

    @classmethod
    def _schema_payload_bytes(cls, schemas) -> int:
        return sum(
            cls._shape_element_count(shape) * np.dtype(dtype).itemsize
            for shape, dtype in schemas.values()
        )

    def _output_row_payload_bytes(self, schemas, board_size) -> int:
        """Estimate final core-array payload for one valid decoded row.

        Valid processed-NPZ layouts are sized exactly. For malformed layouts or
        a fixed board smaller than the stored board, use the larger plausible
        payload so inspection never understates the runtime allocation.
        """
        height, width = (int(value) for value in board_size)
        output_height, output_width = (
            (height, width)
            if self.fixed_board_size is None
            else (
                max(height, int(self.fixed_board_size[0])),
                max(width, int(self.fixed_board_size[1])),
            )
        )
        output_area = output_height * output_width

        bf_shape = schemas["bf"][0]
        board_channels = int(bf_shape[1])
        board_input_elements = board_channels * output_area

        gf_shape = schemas.get("gf", ((1, 1), np.dtype(np.float32)))[0]
        stm_input_elements = self._shape_element_count(gf_shape[1:])
        vt_shape = schemas["vt"][0]
        value_target_elements = self._shape_element_count(vt_shape[1:])

        if "pt" in schemas:
            policy_row_shape = schemas["pt"][0][1:]
        elif self.has_pass_move:
            policy_row_shape = (height * width + 1,)
        else:
            policy_row_shape = (height, width)
        policy_target_elements = self._shape_element_count(policy_row_shape)
        flat_policy_with_pass = len(policy_row_shape) == 1 and self.has_pass_move
        if len(policy_row_shape) == 1 and not self.has_pass_move:
            # decode_one/decode_batch remove the final pass entry before output.
            policy_target_elements = max(
                height * width,
                policy_target_elements - 1,
            )
        if self.fixed_board_size is not None:
            padded_policy_elements = output_area + int(flat_policy_with_pass)
            policy_target_elements = max(
                policy_target_elements,
                padded_policy_elements,
            )

        return int(
            2 * np.dtype(np.int8).itemsize
            + board_input_elements * np.dtype(np.int8).itemsize
            + stm_input_elements * np.dtype(np.float32).itemsize
            + value_target_elements * np.dtype(np.float32).itemsize
            + policy_target_elements * np.dtype(np.float32).itemsize
        )

    def _mmap_stored_arrays(self, canonical: str):
        with zipfile.ZipFile(canonical) as archive:
            names = set(archive.namelist())
            infos = {
                key: archive.getinfo(f"{key}.npy")
                for key in ("bf", "gf", "vt", "pt")
                if f"{key}.npy" in names
            }
            if not infos or any(
                info.compress_type != zipfile.ZIP_STORED
                for info in infos.values()
            ):
                return None
            if canonical not in self._validated_stored_npz_paths:
                for info in infos.values():
                    with archive.open(info) as member:
                        while member.read(1024 * 1024):
                            pass
                self._validated_stored_npz_paths.add(canonical)

        arrays = {}
        local_header = struct.Struct("<IHHHHHIIIHH")
        with open(canonical, "rb") as stream:
            for key, info in infos.items():
                stream.seek(info.header_offset)
                header = stream.read(local_header.size)
                if len(header) != local_header.size:
                    raise ValueError(
                        f"stored NPZ member {info.filename!r} has a truncated header"
                    )
                fields = local_header.unpack(header)
                if fields[0] != 0x04034B50:
                    raise ValueError(
                        f"stored NPZ member {info.filename!r} has an invalid header"
                    )
                member_offset = (
                    info.header_offset
                    + local_header.size
                    + fields[-2]
                    + fields[-1]
                )
                stream.seek(member_offset)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_1_0(stream)
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_2_0(stream)
                    )
                else:
                    return None
                if dtype.hasobject:
                    raise ValueError(
                        f"stored NPZ member {info.filename!r} has an object dtype"
                    )
                array_offset = stream.tell()
                element_count = 1
                for extent in shape:
                    element_count *= int(extent)
                if element_count == 0:
                    return None
                array_bytes = element_count * dtype.itemsize
                if array_offset + array_bytes > member_offset + info.file_size:
                    raise ValueError(
                        f"stored NPZ member {info.filename!r} has truncated array data"
                    )
                arrays[key] = np.memmap(
                    canonical,
                    mode="r",
                    dtype=dtype,
                    offset=array_offset,
                    shape=shape,
                    order="F" if fortran_order else "C",
                )
        return arrays

    @staticmethod
    def _decoded_heap_bytes(arrays) -> int:
        return sum(
            int(array.nbytes)
            for array in arrays.values()
            if not isinstance(array, np.memmap)
        )

    def _remove_cache_entry(self, canonical: str) -> None:
        self._array_cache_values.pop(canonical, None)
        reservation = self._array_cache_reservations.pop(canonical, None)
        if reservation is not None:
            reservation.retire()

    def _retire_all_cache_entries(self) -> None:
        for canonical in tuple(self._array_cache_values):
            self._remove_cache_entry(canonical)

    def _evict_idle_for_budget(
        self,
        budget: HostMemoryBudget,
        required_available_bytes: int,
    ) -> None:
        while budget.snapshot().available_bytes < required_available_bytes:
            victim = next(
                (
                    canonical
                    for canonical in self._array_cache_values
                    if (
                        self._array_cache_reservations.get(canonical) is not None
                        and self._array_cache_reservations[canonical].active_leases == 0
                    )
                ),
                None,
            )
            if victim is None:
                break
            self._remove_cache_entry(victim)

    def _cache_admission_error(
        self,
        canonical: str,
        requested_bytes: int,
        budget: HostMemoryBudget,
    ) -> ProcessedNpzCacheAdmissionError:
        snapshot = budget.snapshot()
        return ProcessedNpzCacheAdmissionError(
            f"{self.format_id} cache cannot admit decoded load {canonical!r}: "
            f"requested={requested_bytes}, available={snapshot.available_bytes}, "
            f"used={snapshot.used_bytes}, total={snapshot.total_bytes}, "
            f"active_host_leases={snapshot.active_leases}; "
            "idle decoded-cache entries "
            "were evicted before failing"
        )

    def _reserve_inflight(
        self,
        canonical: str,
        *,
        use_peak_size: bool,
    ):
        with self._array_cache_condition:
            if self._cache_closed:
                raise RuntimeError("processed NPZ cache closed during a load")
            budget = self._host_memory_budget
            if budget is None:
                return None
            sizes = self._decoded_file_size_catalog.get(canonical)
            if sizes is None:
                raise ProcessedNpzCacheAdmissionError(
                    f"{self.format_id} cache cannot admit decoded load "
                    f"{canonical!r}: no inspected size is configured; pass "
                    "inspect catalogs to configure_cache(file_size_catalog=...)"
                )
            decoded_bytes, inflight_bytes = sizes
            requested_bytes = inflight_bytes if use_peak_size else decoded_bytes
            self._evict_idle_for_budget(budget, requested_bytes)
            try:
                reservation = budget.reserve(
                    HostMemoryCategory.INFLIGHT_TRANSIENT,
                    requested_bytes,
                    label="processed NPZ decoded-file load",
                )
            except HostMemoryBudgetExceeded as exc:
                raise self._cache_admission_error(
                    canonical,
                    requested_bytes,
                    budget,
                ) from exc
            return reservation

    def _finish_inflight_reservation(
        self,
        canonical: str,
        reservation,
        actual_heap_bytes: int,
    ) -> None:
        if reservation is None:
            if self._host_memory_budget is not None and actual_heap_bytes:
                raise RuntimeError(
                    "processed NPZ heap arrays were created without an "
                    "inflight memory reservation"
                )
            return
        if actual_heap_bytes == 0:
            reservation.release()
            return
        budget = self._host_memory_budget
        if budget is None:
            reservation.release()
            raise RuntimeError("processed NPZ memory accounting changed during a load")
        growth = max(0, actual_heap_bytes - reservation.nbytes)
        with self._array_cache_condition:
            self._evict_idle_for_budget(budget, growth)
            try:
                reservation.resize(actual_heap_bytes)
            except HostMemoryBudgetExceeded as exc:
                raise self._cache_admission_error(
                    canonical,
                    actual_heap_bytes,
                    budget,
                ) from exc
            reservation.reclassify(HostMemoryCategory.DECODED_CACHE)

    def _acquire_cached_arrays(self, path: str):
        canonical = os.path.abspath(path)
        with self._array_cache_condition:
            if self._cache_closed:
                raise RuntimeError("cannot load from a closed processed NPZ cache")
            while canonical in self._array_cache_loading:
                self._array_cache_condition.wait()
                if self._cache_closed:
                    raise RuntimeError(
                        "cannot load from a closed processed NPZ cache"
                    )
            cached = self._array_cache_values.get(canonical)
            if cached is not None:
                reservation = self._array_cache_reservations.get(canonical)
                lease = (
                    None
                    if reservation is None
                    else reservation.acquire_lease()
                )
                self._array_cache_values.move_to_end(canonical)
            else:
                lease = None
                self._array_cache_loading.add(canonical)
        if cached is not None:
            return cached, lease, False

        reservation = None

        def reserve_before_inflate():
            nonlocal reservation
            reservation = self._reserve_inflight(
                canonical,
                use_peak_size=True,
            )

        try:
            arrays = self._load_uncached(
                canonical,
                before_heap_inflate=reserve_before_inflate,
                before_mmap_transform=reserve_before_inflate,
            )
            self._finish_inflight_reservation(
                canonical,
                reservation,
                self._decoded_heap_bytes(arrays),
            )
            if reservation is not None and reservation.released:
                reservation = None
        except BaseException:
            if reservation is not None:
                reservation.release()
            with self._array_cache_condition:
                self._array_cache_loading.discard(canonical)
                self._array_cache_condition.notify_all()
            raise

        lease = None
        try:
            with self._array_cache_condition:
                if self._cache_closed:
                    if reservation is not None:
                        lease = reservation.acquire_lease()
                        reservation.retire()
                else:
                    self._array_cache_values[canonical] = arrays
                    self._array_cache_reservations[canonical] = reservation
                    self._array_cache_values.move_to_end(canonical)
                    self._evict_array_cache()
                    if reservation is not None:
                        lease = reservation.acquire_lease()
                self._array_cache_loading.discard(canonical)
                self._array_cache_condition.notify_all()
        except BaseException:
            if lease is not None:
                lease.release()
            with self._array_cache_condition:
                self._remove_cache_entry(canonical)
                self._array_cache_loading.discard(canonical)
                self._array_cache_condition.notify_all()
            if reservation is not None:
                reservation.release()
            raise
        return arrays, lease, True

    @contextmanager
    def _borrow_arrays(self, path: str):
        arrays, lease, _loaded = self._acquire_cached_arrays(path)
        try:
            yield arrays
        finally:
            if lease is not None:
                lease.release()

    @staticmethod
    def _is_mmap_cache_entry(arrays) -> bool:
        return bool(arrays) and all(
            isinstance(array, np.memmap) for array in arrays.values()
        )

    def _evict_array_cache(self) -> None:

        def evict_oldest(*, mmap_entry):
            victim = None
            for path, cached in self._array_cache_values.items():
                if self._is_mmap_cache_entry(cached) == mmap_entry:
                    victim = path
                    break
            if victim is None:
                raise RuntimeError(
                    "processed NPZ cache accounting is inconsistent"
                )
            self._remove_cache_entry(victim)

        mmap_count = sum(
            self._is_mmap_cache_entry(cached)
            for cached in self._array_cache_values.values()
        )
        loaded_count = len(self._array_cache_values) - mmap_count
        while mmap_count > self._mmap_cache_capacity:
            evict_oldest(mmap_entry=True)
            mmap_count -= 1
        loaded_bytes = sum(
            sum(array.nbytes for array in cached.values())
            for cached in self._array_cache_values.values()
            if not self._is_mmap_cache_entry(cached)
        )
        while loaded_count > 1 and (
            loaded_count > self._array_cache_capacity
            or loaded_bytes > self._array_cache_byte_capacity
        ):
            for cached in self._array_cache_values.values():
                if not self._is_mmap_cache_entry(cached):
                    loaded_bytes -= sum(array.nbytes for array in cached.values())
                    break
            evict_oldest(mmap_entry=False)
            loaded_count -= 1
    def inspect(self, path: str, file_ordinal: int) -> dict:
        canonical = os.path.abspath(path)
        can_use_headers = all(
            option is None
            for option in (
                self.filter_stm,
                self.filter_condition,
                self.board_input_channels,
                self.stm_input_channel,
                self.value_target_channels,
            )
        )
        try:
            header_metadata = self._inspect_compressed_headers(canonical)
        except Exception as exc:
            if isinstance(exc, ValueError) and self.format_id in str(exc):
                raise
            raise RuntimeError(
                f"failed to inspect {self.format_id} file {canonical}: {exc}"
            ) from exc
        metadata = (
            header_metadata[:3]
            if (
                can_use_headers
                and header_metadata is not None
                and header_metadata[3]
            )
            else None
        )
        raw_payload_bytes = (
            None
            if header_metadata is None
            else self._schema_payload_bytes(header_metadata[2])
        )
        if metadata is None and raw_payload_bytes is not None:
            with self._array_cache_condition:
                self._decoded_file_size_catalog.setdefault(
                    canonical,
                    (raw_payload_bytes, raw_payload_bytes * 2),
                )
        if metadata is None:
            with self._borrow_arrays(canonical) as arrays:
                length = len(arrays["bf"])
                board_size = tuple(int(v) for v in arrays["bf"].shape[-2:])
                decoded_file_bytes = sum(
                    int(array.nbytes) for array in arrays.values()
                )
                schemas = {
                    key: (tuple(int(value) for value in array.shape), array.dtype)
                    for key, array in arrays.items()
                }
        else:
            length, board_size, schemas = metadata
            decoded_file_bytes = (
                self._schema_payload_bytes(schemas) if length else 0
            )
        if raw_payload_bytes is None:
            inflight_file_bytes = decoded_file_bytes
        elif can_use_headers:
            inflight_file_bytes = max(
                decoded_file_bytes,
                raw_payload_bytes,
            )
        else:
            inflight_file_bytes = max(
                decoded_file_bytes,
                raw_payload_bytes + decoded_file_bytes,
                raw_payload_bytes * 2,
            )
        with self._array_cache_condition:
            self._decoded_file_size_catalog[canonical] = (
                int(decoded_file_bytes),
                int(inflight_file_bytes),
            )
        output_row_bytes = self._output_row_payload_bytes(schemas, board_size)
        file_digest = _sha256_file(canonical)
        stat = os.stat(canonical)
        output_board_size = (
            board_size
            if self.fixed_board_size is None
            else self.fixed_board_size
        )
        return {
            "path": canonical,
            "file_ordinal": int(file_ordinal),
            "file_sha256": file_digest,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "logical_row_count": length,
            "board_size": output_board_size,
            "decoded_file_bytes": int(decoded_file_bytes),
            "inflight_file_bytes": int(inflight_file_bytes),
            "output_row_bytes": output_row_bytes,
        }

    def decode_one(self, ref) -> dict:
        with self._borrow_arrays(ref.path) as arrays:
            index = ref.address[0]
            if not 0 <= index < len(arrays["bf"]):
                raise RuntimeError(
                    f"{self.format_id} file {ref.path} row {index} is outside "
                    f"{len(arrays['bf'])} rows"
                )
            board_input = arrays["bf"][index].astype(np.int8, copy=True)
            height, width = board_input.shape[-2:]
            if "gf" in arrays:
                stm_input = arrays["gf"][index].astype(np.float32, copy=True)
            else:
                stm_input = np.array([0], dtype=np.float32)
            value_target = arrays["vt"][index].astype(np.float32, copy=True)
            if "pt" in arrays:
                policy_target = arrays["pt"][index].astype(np.float32, copy=True)
            else:
                policy_target = np.zeros(
                    (
                        (height * width + 1,)
                        if self.has_pass_move
                        else (height, width)
                    ),
                    dtype=np.float32,
                )
        if not self.has_pass_move and policy_target.ndim == 1:
            expected = height * width + 1
            if policy_target.shape != (expected,):
                raise ValueError(
                    f"{self.format_id} file {ref.path} row {index} policy shape "
                    f"{policy_target.shape}, expected {(expected,)}"
                )
            policy_target = policy_target[:-1].reshape(height, width)
        symmetry_index = 0
        symmetry_type = self.apply_symmetry
        if symmetry_type:
            from utils.data_utils import Symmetry

            kind = "default" if symmetry_type is True else symmetry_type
            choices = Symmetry.available_symmetries((height, width), kind)
            symmetry_index = _uniform_below_for_processed_sample_keys_v2(
                len(choices),
                self.runtime_context.seed,
                "symmetry",
                self._active_epoch,
                [ref.sample_key],
            )[0]
        return post_process_data(
            {
                "board_size": np.array([height, width], dtype=np.int8),
                "board_input": board_input,
                "stm_input": stm_input,
                "value_target": value_target,
                "policy_target": policy_target,
            },
            fixed_side_input=self.fixed_side_input,
            fixed_board_size=self.fixed_board_size,
            symmetry_type=symmetry_type,
            symmetry_index=symmetry_index,
        )

    def _post_process_vectorized(self, data: dict, refs) -> dict:
        # SourceBatchDataset validates every batched field before yielding.
        # Avoid repeating the same schema walk when no transform needs it.
        if not self.fixed_side_input and not self.apply_symmetry:
            return data

        # Validate the same field contract as the scalar post-processor before
        # applying the two transformations supported by this decoder.
        data = post_process_batch(
            data,
            fixed_side_input=False,
            symmetry_type=None,
        )
        batch_size = len(refs)
        if self.fixed_side_input:
            stm_input = np.asarray(data["stm_input"])
            white_to_move = np.reshape(stm_input > 0, (batch_size, -1))
            if white_to_move.shape[1] != 1:
                raise ValueError(
                    "fixed_side_input requires exactly one stm_input channel"
                )
            white_to_move = white_to_move[:, 0]
            if np.any(white_to_move):
                board_input = np.array(data["board_input"], copy=True)
                board_input[white_to_move] = board_input[white_to_move, ::-1]
                data["board_input"] = board_input
                value_target = np.array(data["value_target"], copy=True)
                swapped = value_target[white_to_move].copy()
                swapped[:, [0, 1]] = swapped[:, [1, 0]]
                value_target[white_to_move] = swapped
                data["value_target"] = value_target

        if not self.apply_symmetry:
            return data

        from utils.data_utils import Symmetry

        height, width = (int(v) for v in data["board_input"].shape[-2:])
        kind = "default" if self.apply_symmetry is True else self.apply_symmetry
        choices = Symmetry.available_symmetries((height, width), kind)
        symmetry_indices = np.asarray(
            _uniform_below_for_packed_processed_rows_v2(
                len(choices),
                self.runtime_context.seed,
                "symmetry",
                self._active_epoch,
                refs.content_digests,
                refs.file_indices,
                refs.rows,
            )
            if hasattr(refs, "content_digests") and not (
                len(choices) & (len(choices) - 1)
            )
            else _uniform_below_for_processed_sample_keys_v2(
                len(choices),
                self.runtime_context.seed,
                "symmetry",
                self._active_epoch,
                (
                    refs.sample_keys
                    if hasattr(refs, "sample_keys")
                    else (ref.sample_key for ref in refs)
                ),
            ),
            dtype=np.intp,
        )
        flat_policy = data["policy_target"].ndim == 2
        if flat_policy:
            expected = height * width + 1
            if data["policy_target"].shape[1] != expected:
                raise ValueError(
                    "flattened policy target does not match board_size"
                )
            source_policy = data["policy_target"][:, :-1].reshape(
                batch_size, height, width
            )
        else:
            source_policy = data["policy_target"]

        board_input = np.empty_like(data["board_input"])
        policy_board = np.empty_like(source_policy)
        for symmetry_index, symmetry in enumerate(choices):
            selected = np.flatnonzero(symmetry_indices == symmetry_index)
            if not selected.size:
                continue
            board_input[selected] = symmetry.apply_to_array(
                data["board_input"][selected]
            )
            policy_board[selected] = symmetry.apply_to_array(
                source_policy[selected]
            )

        if flat_policy:
            policy_target = np.concatenate(
                [
                    policy_board.reshape(batch_size, -1),
                    data["policy_target"][:, -1:],
                ],
                axis=1,
            )
        else:
            policy_target = policy_board
        data["board_input"] = np.ascontiguousarray(board_input)
        data["policy_target"] = np.ascontiguousarray(policy_target)
        return data

    def decode_batch(self, refs):
        if self.fixed_board_size is not None:
            return None
        if not refs:
            raise ValueError("cannot decode an empty processed NPZ batch")
        paths = tuple(dict.fromkeys(ref.path for ref in refs))
        if len(paths) == 1:
            with self._borrow_arrays(paths[0]) as arrays:
                return self._decode_loaded_batch(
                    refs,
                    {paths[0]: arrays},
                )
        return self._decode_grouped_batch(refs, paths)

    def decode_packed_batch(self, request):
        if self.fixed_board_size is not None:
            return None
        if not len(request):
            raise ValueError("cannot decode an empty packed processed NPZ batch")
        unique_indices, counts = np.unique(
            request.file_indices,
            return_counts=True,
        )
        used = tuple(
            int(index)
            for index in unique_indices[np.argsort(counts, kind="stable")]
        )
        height, width = request.board_size
        policy_shape = (
            (height * width + 1,)
            if self.has_pass_move
            else (height, width)
        )
        data = None
        output_schema = None
        for index in used:
            positions = np.flatnonzero(request.file_indices == index)
            rows = request.rows[positions]
            with self._borrow_arrays(request.paths[index]) as arrays:
                def selected(key, shape, dtype):
                    if key in arrays:
                        return arrays[key][rows]
                    return np.zeros((len(positions), *shape), dtype=dtype)

                group = {
                    "board_size": np.tile(
                        np.asarray(request.board_size, dtype=np.int8),
                        (len(positions), 1),
                    ),
                    "board_input": selected(
                        "bf", (2, height, width), np.int8
                    ).astype(np.int8, copy=False),
                    "stm_input": selected(
                        "gf", (1,), np.float32
                    ).astype(np.float32, copy=False),
                    "value_target": selected(
                        "vt", (3,), np.float32
                    ).astype(np.float32, copy=False),
                    "policy_target": selected(
                        "pt",
                        policy_shape,
                        np.float32,
                    ).astype(np.float32, copy=False),
                }
            schema = tuple(
                (key, value.shape[1:], value.dtype.str)
                for key, value in group.items()
            )
            if data is None:
                output_schema = schema
                data = {
                    key: np.empty(
                        (len(request), *value.shape[1:]), dtype=value.dtype
                    )
                    for key, value in group.items()
                }
            elif schema != output_schema:
                raise ValueError(
                    "processed NPZ files in one batch have incompatible layouts"
                )
            for key, value in group.items():
                data[key][positions] = value
        if not self.has_pass_move and data["policy_target"].ndim == 2:
            expected = height * width + 1
            if data["policy_target"].shape[1] != expected:
                raise ValueError(
                    "processed NPZ policy target does not match board_size"
                )
            data["policy_target"] = data["policy_target"][:, :-1].reshape(
                len(request), height, width
            )
        return self._post_process_vectorized(data, request)

    def decode_batches(self, ref_batches):
        if self.fixed_board_size is not None:
            return None
        if not ref_batches or any(not refs for refs in ref_batches):
            raise ValueError("cannot decode an empty processed NPZ batch chunk")
        board_sizes = {refs[0].board_size for refs in ref_batches}
        if len(board_sizes) != 1:
            return [self.decode_batch(refs) for refs in ref_batches]
        flat_refs = tuple(ref for refs in ref_batches for ref in refs)
        paths = tuple(dict.fromkeys(ref.path for ref in flat_refs))
        if len(paths) == 1:
            with self._borrow_arrays(paths[0]) as arrays:
                flat_batch = self._decode_loaded_batch(
                    flat_refs,
                    {paths[0]: arrays},
                )
        else:
            flat_batch = self._decode_grouped_batch(flat_refs, paths)
        batches = []
        offset = 0
        for refs in ref_batches:
            next_offset = offset + len(refs)
            batches.append(
                {
                    key: value[offset:next_offset]
                    for key, value in flat_batch.items()
                }
            )
            offset = next_offset
        return batches

    def _decode_grouped_batch(self, refs, paths):
        positions_by_path = {path: [] for path in paths}
        refs_by_path = {path: [] for path in paths}
        for position, ref in enumerate(refs):
            positions_by_path[ref.path].append(position)
            refs_by_path[ref.path].append(ref)

        output = None
        output_schema = None
        for path in paths:
            positions = np.asarray(positions_by_path[path], dtype=np.intp)
            with self._borrow_arrays(path) as arrays:
                group = self._decode_loaded_batch(
                    tuple(refs_by_path[path]),
                    {path: arrays},
                )
            schema = tuple(
                (key, value.shape[1:], value.dtype.str)
                for key, value in group.items()
            )
            if output is None:
                output_schema = schema
                output = {
                    key: np.empty(
                        (len(refs), *value.shape[1:]),
                        dtype=value.dtype,
                    )
                    for key, value in group.items()
                }
            elif schema != output_schema:
                raise ValueError(
                    "processed NPZ files in one batch have incompatible layouts"
                )
            for key, value in group.items():
                output[key][positions] = value
        return output

    def _decode_loaded_batch(self, refs, arrays_by_path):
        first_path = refs[0].path
        one_path = all(ref.path == first_path for ref in refs)
        selector = None
        contiguous = False
        groups = None
        restore_order = None
        if one_path:
            first_index = refs[0].address[0]
            contiguous = all(
                ref.address == (first_index + offset,)
                for offset, ref in enumerate(refs)
            )
            selector = (
                slice(first_index, first_index + len(refs))
                if contiguous
                else np.fromiter(
                    (ref.address[0] for ref in refs),
                    dtype=np.int64,
                    count=len(refs),
                )
            )
        else:
            grouped = {}
            for position, ref in enumerate(refs):
                positions, indices, board_sizes = grouped.setdefault(
                    ref.path, ([], [], [])
                )
                positions.append(position)
                indices.append(ref.address[0])
                board_sizes.append(ref.board_size)
            groups = [
                (
                    path,
                    np.asarray(positions, dtype=np.intp),
                    np.asarray(indices, dtype=np.intp),
                    board_sizes,
                )
                for path, (positions, indices, board_sizes) in grouped.items()
            ]
            grouped_positions = np.concatenate(
                [positions for _, positions, _, _ in groups]
            )
            restore_order = np.argsort(grouped_positions)

        def gather(key, default):
            if one_path:
                arrays = arrays_by_path[first_path]
                if key in arrays:
                    selected = arrays[key][selector]
                    return (
                        np.array(selected, copy=True, order="C")
                        if contiguous
                        else np.ascontiguousarray(selected)
                    )
                return np.stack(
                    [default(ref.board_size) for ref in refs], axis=0
                )
            chunks = []
            for path, _, indices, board_sizes in groups:
                arrays = arrays_by_path[path]
                chunks.append(
                    arrays[key][indices]
                    if key in arrays
                    else np.stack(
                        [default(board_size) for board_size in board_sizes],
                        axis=0,
                    )
                )
            return np.concatenate(chunks, axis=0)[restore_order]

        board_input = gather(
            "bf",
            lambda board_size: np.zeros((2, *board_size), dtype=np.int8),
        ).astype(np.int8, copy=False)
        stm_input = gather(
            "gf", lambda board_size: np.zeros((1,), dtype=np.float32)
        ).astype(np.float32, copy=False)
        value_target = gather(
            "vt", lambda board_size: np.zeros((3,), dtype=np.float32)
        ).astype(np.float32, copy=False)
        policy_target = gather(
            "pt",
            lambda board_size: np.zeros(
                (board_size[0] * board_size[1] + 1,)
                if self.has_pass_move
                else board_size,
                dtype=np.float32,
            ),
        ).astype(np.float32, copy=False)
        height, width = refs[0].board_size
        if not self.has_pass_move and policy_target.ndim == 2:
            expected = height * width + 1
            if policy_target.shape[1] != expected:
                raise ValueError(
                    f"{self.format_id} batch policy width {policy_target.shape[1]}, "
                    f"expected {expected}"
                )
            policy_target = policy_target[:, :-1].reshape(-1, height, width)
        data = {
            "board_size": np.asarray(
                [ref.board_size for ref in refs], dtype=np.int8
            ),
            "board_input": np.ascontiguousarray(board_input),
            "stm_input": np.ascontiguousarray(stm_input),
            "value_target": np.ascontiguousarray(value_target),
            "policy_target": np.ascontiguousarray(policy_target),
        }
        return self._post_process_vectorized(data, refs)


class NpzRowRecordDecoder:
    """Decode one logical NPZ row with a bounded one-file array cache."""

    schema_version = 1
    file_exts = (".npz",)

    def __init__(
        self,
        format_id,
        runtime_context,
        load_file,
        prepare_row,
        *,
        apply_symmetry=False,
        semantic_state=None,
        catalog_rows=None,
    ):
        self.format_id = format_id
        self.runtime_context = runtime_context
        self.load_file = load_file
        self.prepare_row = prepare_row
        self.apply_symmetry = apply_symmetry
        self.semantic_state = dict(semantic_state or {})
        self.catalog_rows = catalog_rows
        self._cache_path = None
        self._cache_value = None
        self._active_epoch = 0

    def set_epoch(self, epoch):
        self._active_epoch = int(epoch)

    def signature_state(self):
        return {
            "format_id": self.format_id,
            "schema_version": self.schema_version,
            "address_kind": "npz-logical-row-v1",
            "apply_symmetry": self.apply_symmetry,
            "semantic_state": self.semantic_state,
            "compact_catalog": self.catalog_rows is not None,
        }

    def inspect_compact(self, path, file_ordinal):
        canonical = os.path.abspath(path)
        data, length = self._load(canonical)
        if self.catalog_rows is None:
            indices = []
            board_sizes = []
            for index in range(length):
                sample = self.prepare_row(data, index)
                if sample is None:
                    continue
                indices.append(index)
                board_sizes.append(
                    tuple(
                        int(value)
                        for value in np.asarray(sample["board_input"]).shape[-2:]
                    )
                )
            indices = np.asarray(indices, dtype=np.int64)
            board_sizes = np.asarray(board_sizes, dtype=np.int64).reshape(-1, 2)
        else:
            indices, board_sizes = self.catalog_rows(data, length)
        if indices is None:
            logical_row_count = int(length)
        else:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.ndim != 1:
                raise ValueError(
                    f"{self.format_id} compact catalog row indices are invalid"
                )
            logical_row_count = len(indices)
        board_sizes = np.asarray(board_sizes, dtype=np.int64)
        uniform_board_size = None
        if board_sizes.shape == (2,):
            uniform_board_size = tuple(int(value) for value in board_sizes)
            board_sizes = None
        elif board_sizes.shape != (logical_row_count, 2):
            raise ValueError(
                f"{self.format_id} compact catalog has invalid row/shape arrays"
            )
        stat = os.stat(canonical)
        return {
            "path": canonical,
            "file_ordinal": int(file_ordinal),
            "file_sha256": _sha256_file(canonical),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "physical_row_count": int(length),
            "logical_row_count": logical_row_count,
            "row_indices": indices,
            "board_sizes": board_sizes,
            "uniform_board_size": uniform_board_size,
        }

    def _load(self, path):
        canonical = os.path.abspath(path)
        if canonical != self._cache_path:
            self._cache_value = self.load_file(canonical)
            self._cache_path = canonical
        return self._cache_value

    def decode_one(self, ref):
        data, length = self._load(ref.path)
        index = ref.address[0]
        if not 0 <= index < length:
            raise RuntimeError(
                f"{self.format_id} row {index} is outside {length} rows"
            )
        sample = self.prepare_row(data, index)
        if sample is None:
            raise RuntimeError(
                f"{self.format_id} record changed at row {index}"
            )
        if self.apply_symmetry:
            from utils.data_utils import Symmetry

            board_size = tuple(int(v) for v in np.asarray(sample["board_size"]))
            kind = "default" if self.apply_symmetry is True else self.apply_symmetry
            choices = Symmetry.available_symmetries(board_size, kind)
            symmetry_index, _ = uniform_below(
                len(choices),
                self.runtime_context.seed,
                "symmetry",
                (self._active_epoch, ref.sample_key, 0),
            )
            sample = post_process_data(
                copy.deepcopy(sample),
                symmetry_type=self.apply_symmetry,
                symmetry_index=symmetry_index,
            )
        return sample
