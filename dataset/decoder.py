from __future__ import annotations

import os
import copy
import threading
import struct
import zipfile
from collections import OrderedDict
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
from .stream import (
    _sha256_file,
)

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
        self._array_cache_loading = set()
        self._array_cache_condition = threading.Condition()
        self._validated_stored_npz_paths = set()
        self._active_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._active_epoch = int(epoch)

    def configure_cache(self, *, entries: int, byte_capacity: int) -> None:
        """Update decoded-array cache limits outside the decode hot path."""
        if type(entries) is not int or entries <= 0:
            raise ValueError("processed NPZ cache entries must be a positive integer")
        if type(byte_capacity) is not int or byte_capacity <= 0:
            raise ValueError("processed NPZ cache byte capacity must be a positive integer")
        with self._array_cache_condition:
            self._array_cache_capacity = entries
            self._array_cache_byte_capacity = byte_capacity
            self._evict_array_cache()

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

    def _load_uncached(self, canonical: str) -> dict[str, np.ndarray]:
        try:
            arrays = self._mmap_stored_arrays(canonical)
            if arrays is None:
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
            arrays = {
                key: np.empty((0, *value.shape[1:]), dtype=value.dtype)
                for key, value in arrays.items()
            }
            for array in arrays.values():
                array.flags.writeable = False
            return arrays
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
        """Read compressed NPZ schemas without inflating their array payloads."""
        with zipfile.ZipFile(canonical) as archive:
            names = set(archive.namelist())
            infos = {
                key: archive.getinfo(f"{key}.npy")
                for key in ("bf", "gf", "vt", "pt")
                if f"{key}.npy" in names
            }
            if not infos or any(
                info.compress_type == zipfile.ZIP_STORED
                for info in infos.values()
            ):
                return None
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
                        shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(member)
                        )
                    elif version in {(2, 0), (3, 0)}:
                        shape, fortran_order, dtype = _read_npy_array_header(
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
                schemas[key] = (tuple(int(v) for v in shape), bool(fortran_order), dtype)
        bf_shape = schemas["bf"][0]
        if len(bf_shape) != 4:
            raise ValueError(
                f"{self.format_id} file {canonical} key 'bf' expected (N,C,H,W), got {bf_shape}"
            )
        length = bf_shape[0]
        if length == 0:
            raise ValueError(f"{self.format_id} file {canonical} key 'bf' is empty")
        for key, (shape, _, _) in schemas.items():
            if not shape or shape[0] != length:
                actual = 0 if not shape else shape[0]
                raise ValueError(
                    f"{self.format_id} file {canonical} key {key!r} has {actual} rows, "
                    f"expected {length}"
                )
        board_size = bf_shape[-2:]
        return (length if board_size in self.boardsizes else 0), board_size

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

    def _load(self, path: str) -> dict[str, np.ndarray]:
        canonical = os.path.abspath(path)
        with self._array_cache_condition:
            while canonical in self._array_cache_loading:
                self._array_cache_condition.wait()
            cached = self._array_cache_values.get(canonical)
            if cached is not None:
                self._array_cache_values.move_to_end(canonical)
                return cached
            self._array_cache_loading.add(canonical)
        try:
            arrays = self._load_uncached(canonical)
        except BaseException:
            with self._array_cache_condition:
                self._array_cache_loading.remove(canonical)
                self._array_cache_condition.notify_all()
            raise
        with self._array_cache_condition:
            self._array_cache_values[canonical] = arrays
            self._array_cache_values.move_to_end(canonical)
            self._evict_array_cache()
            self._array_cache_loading.remove(canonical)
            self._array_cache_condition.notify_all()
        return arrays

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
            del self._array_cache_values[victim]

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
        metadata = None
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
        if can_use_headers:
            try:
                metadata = self._inspect_compressed_headers(canonical)
            except Exception as exc:
                if isinstance(exc, ValueError) and self.format_id in str(exc):
                    raise
                raise RuntimeError(
                    f"failed to inspect {self.format_id} file {canonical}: {exc}"
                ) from exc
        if metadata is None:
            arrays = self._load(canonical)
            length = len(arrays["bf"])
            board_size = tuple(int(v) for v in arrays["bf"].shape[-2:])
        else:
            length, board_size = metadata
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
        }

    def decode_one(self, ref) -> dict:
        arrays = self._load(ref.path)
        index = ref.address[0]
        if not 0 <= index < len(arrays["bf"]):
            raise RuntimeError(
                f"{self.format_id} file {ref.path} row {index} is outside {len(arrays['bf'])} rows"
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
                (height * width + 1,) if self.has_pass_move else (height, width),
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
            return self._decode_loaded_batch(
                refs,
                {paths[0]: self._load(paths[0])},
            )
        return self._decode_grouped_batch(refs, paths)

    def decode_packed_batch(self, request):
        if self.fixed_board_size is not None:
            return None
        if not len(request):
            raise ValueError("cannot decode an empty packed processed NPZ batch")
        used = tuple(
            sorted(
                np.unique(request.file_indices).tolist(),
                key=lambda index: np.count_nonzero(
                    request.file_indices == index
                ),
            )
        )
        height, width = request.board_size
        data = None
        output_schema = None
        for index in used:
            positions = np.flatnonzero(request.file_indices == index)
            rows = request.rows[positions]
            arrays = self._load(request.paths[index])

            def selected(key, default):
                return (
                    arrays[key][rows]
                    if key in arrays
                    else np.stack([default] * len(positions), axis=0)
                )

            group = {
                "board_size": np.tile(
                    np.asarray(request.board_size, dtype=np.int8),
                    (len(positions), 1),
                ),
                "board_input": selected(
                    "bf", np.zeros((2, height, width), dtype=np.int8)
                ).astype(np.int8, copy=False),
                "stm_input": selected(
                    "gf", np.zeros((1,), dtype=np.float32)
                ).astype(np.float32, copy=False),
                "value_target": selected(
                    "vt", np.zeros((3,), dtype=np.float32)
                ).astype(np.float32, copy=False),
                "policy_target": selected(
                    "pt",
                    np.zeros(
                        (height * width + 1,)
                        if self.has_pass_move
                        else (height, width),
                        dtype=np.float32,
                    ),
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
        flat_batch = (
            self._decode_loaded_batch(
                flat_refs,
                {paths[0]: self._load(paths[0])},
            )
            if len(paths) == 1
            else self._decode_grouped_batch(flat_refs, paths)
        )
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
            group = self._decode_loaded_batch(
                tuple(refs_by_path[path]),
                {path: self._load(path)},
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
