from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .core import (
    CORE_FIELD_SPECS,
    DatasetCapabilities,
    DatasetRuntimeContext,
    canonical_pipeline_state_bytes,
    validate_field_dict,
)


def _sha256_file(path: str) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.digest()


@dataclass(frozen=True)
class MapRecordRef:
    dataset_id: str
    index: int
    sample_key: tuple[str, bytes, str, tuple[int, ...]]
    output_shape: tuple


def reject_duplicate_physical_files(paths: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    identities: dict[tuple, list[str]] = {}
    for configured in paths:
        path = os.path.abspath(os.path.realpath(os.fspath(configured)))
        stat = os.stat(path)
        identity = (
            ("inode", int(stat.st_dev), int(stat.st_ino))
            if stat.st_ino
            else ("path", os.path.normcase(path).casefold())
        )
        aliases = identities.setdefault(identity, [])
        aliases.append(os.fspath(configured))
        normalized.append(path)
    duplicates = [origins for origins in identities.values() if len(origins) > 1]
    if duplicates:
        details = "; ".join(", ".join(group) for group in duplicates)
        raise ValueError(f"duplicate physical dataset files are forbidden: {details}")
    return normalized


@dataclass(frozen=True)
class BatchEnvelope:
    data: dict
    token: object
    is_real: np.ndarray
    sample_keys: Sequence

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()


def _cursor_digest(epoch: int, batch_index: int) -> str:
    """Digest for non-resumable map-evaluation envelopes."""
    return hashlib.sha256(
        canonical_pipeline_state_bytes(
            {"kind": "map-evaluation", "epoch": epoch, "batch_index": batch_index}
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class MapEvaluationBatchToken:
    """Minimal non-resumable identity for a map-evaluation batch."""

    epoch: int
    batch_index: int
    before_digest: str
    after_digest: str
    is_last: bool


def collate_sample_dicts(
    samples: list[dict], *, validate_core_fields: bool = False
) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    keys = samples[0].keys()
    if any(sample.keys() != keys for sample in samples[1:]):
        raise ValueError("decoded samples have inconsistent field schemas")
    specs = {spec.name: spec for spec in CORE_FIELD_SPECS}
    if validate_core_fields:
        for sample in samples:
            validate_field_dict(sample, batched=False)
        missing = sorted(
            name for name, spec in specs.items() if spec.required and name not in keys
        )
        if missing:
            raise ValueError(f"decoded samples are missing required field(s): {missing}")
        unknown = sorted(set(keys).difference(specs))
        if unknown:
            raise ValueError(
                f"decoded samples contain undeclared field(s): {unknown}"
            )
    batch = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        spec = specs.get(key)
        if spec is not None and spec.scope == "batch_shared":
            first = values[0]
            for index, value in enumerate(values[1:], start=1):
                if isinstance(first, torch.Tensor) and isinstance(
                    value, torch.Tensor
                ):
                    equal = torch.equal(first, value)
                else:
                    equal = np.array_equal(
                        np.asarray(first), np.asarray(value)
                    )
                if not equal:
                    raise ValueError(
                        f"batch-shared field {key!r} differs at row {index}"
                    )
        if spec is not None:
            for index, value in enumerate(values):
                array = (
                    value.detach().cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value)
                )
                if array.dtype.kind not in spec.dtype_kinds:
                    raise ValueError(
                        f"field {key!r} sample {index} has dtype {array.dtype}; "
                        f"expected kind in {spec.dtype_kinds}"
                    )
                if spec.spatial_axes is not None:
                    axes = tuple(
                        axis if axis >= 0 else array.ndim + axis
                        for axis in spec.spatial_axes
                    )
                    if any(not 0 <= axis < array.ndim for axis in axes):
                        raise ValueError(
                            f"field {key!r} sample {index} has invalid spatial axes "
                            f"{spec.spatial_axes} for shape {array.shape}"
                        )
                    board_size = tuple(
                        int(item) for item in np.asarray(samples[index]["board_size"])
                    )
                    spatial_shape = tuple(array.shape[axis] for axis in axes)
                    if any(
                        physical < logical
                        for physical, logical in zip(
                            spatial_shape, board_size
                        )
                    ):
                        raise ValueError(
                            f"field {key!r} sample {index} spatial shape "
                            f"{spatial_shape} does not match board_size {board_size}"
                        )
                if spec.transform == "board_policy":
                    board_size = tuple(
                        int(item)
                        for item in np.asarray(samples[index]["board_size"])
                    )
                    spatial = (
                        array.ndim == 2
                        and tuple(array.shape) == board_size
                    )
                    flat = (
                        array.ndim == 1
                        and array.shape[0]
                        in {
                            board_size[0] * board_size[1],
                            board_size[0] * board_size[1] + 1,
                            int(np.asarray(samples[index]["board_input"]).shape[-2])
                            * int(np.asarray(samples[index]["board_input"]).shape[-1]),
                            int(np.asarray(samples[index]["board_input"]).shape[-2])
                            * int(np.asarray(samples[index]["board_input"]).shape[-1])
                            + 1,
                        }
                    )
                    if not (spatial or flat):
                        raise ValueError(
                            f"field {key!r} sample {index} is neither a "
                            "board plane nor a flattened board policy"
                        )
                if spec.side_axis is not None:
                    side_axis = (
                        spec.side_axis
                        if spec.side_axis >= 0
                        else array.ndim + spec.side_axis
                    )
                    if not 0 <= side_axis < array.ndim:
                        raise ValueError(
                            f"field {key!r} sample {index} has invalid side axis "
                            f"{spec.side_axis} for shape {array.shape}"
                        )
        if all(isinstance(value, torch.Tensor) for value in values):
            batch[key] = torch.stack(values, dim=0)
        elif all(isinstance(value, (np.ndarray, np.number, int, float, bool)) for value in values):
            try:
                batch[key] = np.stack(
                    [np.ascontiguousarray(np.asarray(value)) for value in values],
                    axis=0,
                )
            except ValueError as exc:
                raise ValueError(f"field {key!r} cannot be stacked: {exc}") from exc
        else:
            batch[key] = values
    return batch


class EvaluationBatchPlannerDataset(IterableDataset):
    """Sole map-style evaluation sharding and masked-tail owner."""

    YIELDS_BATCHES = True

    def __init__(self, dataset, runtime_context: DatasetRuntimeContext):
        super().__init__()
        if not callable(getattr(dataset, "map_record_ref", None)):
            raise TypeError(
                "map evaluation datasets must expose metadata-only map_record_ref"
            )
        self.dataset = dataset
        self.runtime_context = runtime_context

    @property
    def yields_batches(self):
        return True

    @property
    def capabilities(self):
        return DatasetCapabilities(True, False, True, True)

    @property
    def is_internal_shuffleable(self):
        return True

    @property
    def is_fixed_side_input(self):
        return self.dataset.is_fixed_side_input

    def __iter__(self):
        if len(self.dataset) == 0:
            raise RuntimeError("evaluation dataset contains no real samples")
        global_size = self.runtime_context.global_batch_size
        buckets = defaultdict(list)
        for index in range(len(self.dataset)):
            ref = self.dataset.map_record_ref(index)
            if not isinstance(ref, MapRecordRef):
                raise TypeError("map_record_ref must return MapRecordRef")
            buckets[ref.output_shape].append(ref)
        global_batches = []
        for shape in buckets:
            bucket = buckets[shape]
            for start in range(0, len(bucket), global_size):
                real = bucket[start : start + global_size]
                values = list(real)
                while len(values) < global_size:
                    values.append(real[(len(values) - len(real)) % len(real)])
                global_batches.append(
                    (
                        values,
                        [True] * len(real) + [False] * (global_size - len(real)),
                    )
                )
        identity = self.runtime_context.rank_local_identity
        for batch_index, (values, mask) in enumerate(global_batches):
            local = values[identity.slice_start : identity.slice_stop]
            local_mask = np.asarray(
                mask[identity.slice_start : identity.slice_stop], dtype=np.bool_
            )
            samples = [self.dataset[ref.index] for ref in local]
            token = MapEvaluationBatchToken(
                epoch=0,
                batch_index=batch_index,
                before_digest=_cursor_digest(0, batch_index),
                after_digest=_cursor_digest(0, batch_index + 1),
                is_last=batch_index + 1 == len(global_batches),
            )
            yield BatchEnvelope(
                collate_sample_dicts(
                    samples,
                    validate_core_fields=True,
                ),
                token,
                local_mask,
                tuple(ref.sample_key for ref in local),
            )
