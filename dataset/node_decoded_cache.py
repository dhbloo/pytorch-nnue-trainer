"""Run-scoped node cache for compressed processed-NPZ arrays.

The source NPZ files remain unchanged.  One process per node expands their
fixed NPY members into a private temporary directory, then every local rank
opens those files read-only with NumPy mmap.  The directory is unique to the
current run, so publication needs neither a content hash nor persistent cache
validation.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
from typing import Iterable
import zipfile


_ARRAY_KEYS = ("bf", "gf", "vt", "pt")
_REQUIRED_KEYS = frozenset(("bf", "vt"))
_READY = "READY"
_DISABLED = "DISABLED"
_MIN_FREE_RESERVE_BYTES = 64 * 1024 * 1024


def _canonical_paths(paths: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(os.path.abspath(os.fspath(path)) for path in paths)
    if not normalized:
        raise ValueError("node decoded cache requires at least one NPZ path")
    return normalized


def _array_infos(path: str) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        members: dict[str, list[zipfile.ZipInfo]] = {}
        for info in archive.infolist():
            if info.filename.endswith(".npy"):
                members.setdefault(info.filename, []).append(info)
        duplicate = next(
            (name for name, infos in members.items() if len(infos) != 1),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"processed NPZ contains duplicate member {duplicate!r}")
        infos = {
            key: members[f"{key}.npy"][0]
            for key in _ARRAY_KEYS
            if f"{key}.npy" in members
        }
    missing = sorted(_REQUIRED_KEYS.difference(infos))
    if missing:
        raise ValueError(
            "processed NPZ is missing required member(s) " + ", ".join(missing)
        )
    return infos


def _inspect_source(
    path: str,
) -> tuple[dict[str, zipfile.ZipInfo], tuple[int, int]]:
    before = os.stat(path)
    infos = _array_infos(path)
    after = os.stat(path)
    signature = (after.st_size, after.st_mtime_ns)
    if (before.st_size, before.st_mtime_ns) != signature:
        raise RuntimeError("processed NPZ changed while inspecting the node cache")
    return infos, signature


def _expanded_bytes(source_infos: Iterable[dict[str, zipfile.ZipInfo]]) -> int:
    return sum(
        int(info.file_size)
        for infos in source_infos
        for info in infos.values()
    )


def _extract_one(
    source: str,
    destination: Path,
    infos: dict[str, zipfile.ZipInfo],
    signature: tuple[int, int],
) -> None:
    before = os.stat(source)
    if (before.st_size, before.st_mtime_ns) != signature:
        raise RuntimeError("processed NPZ changed before building the node cache")
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    try:
        with zipfile.ZipFile(source) as archive:
            for key, info in infos.items():
                with archive.open(info) as reader, open(
                    temporary / f"{key}.npy", "wb"
                ) as writer:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        after = os.stat(source)
        if (after.st_size, after.st_mtime_ns) != signature:
            raise RuntimeError("processed NPZ changed while building the node cache")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _mark_disabled(root: Path) -> None:
    try:
        (root / _DISABLED).touch()
    except OSError:
        pass


@dataclass(slots=True)
class NodeDecodedCache:
    """Ephemeral cache directory shared by the ranks on one node."""

    directory: str = field(repr=False)
    owner: bool
    _cleaned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.directory = os.path.abspath(os.fspath(self.directory))
        if type(self.owner) is not bool:
            raise TypeError("node decoded cache owner must be a boolean")
        if self.owner:
            atexit.register(self._cleanup_at_exit)

    @property
    def root(self) -> Path:
        return Path(self.directory)

    def prepare(self, paths: Iterable[str], *, workers: int) -> None:
        """Materialize the cache on the owner; peers synchronize externally."""

        if not self.owner:
            return
        if type(workers) is not int or workers <= 0:
            raise ValueError("node decoded cache workers must be a positive integer")
        sources = _canonical_paths(paths)
        root = self.root
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        inspected = tuple(_inspect_source(source) for source in sources)
        source_infos = tuple(infos for infos, _ in inspected)
        source_signatures = tuple(signature for _, signature in inspected)
        expanded_bytes = _expanded_bytes(source_infos)
        reserve = max(_MIN_FREE_RESERVE_BYTES, expanded_bytes // 20)
        try:
            free_bytes = shutil.disk_usage(root).free
        except OSError:
            _mark_disabled(root)
            return
        if free_bytes < expanded_bytes + reserve:
            _mark_disabled(root)
            return

        destinations = tuple(
            root / f"{ordinal:08d}" for ordinal in range(len(sources))
        )
        try:
            with ThreadPoolExecutor(
                max_workers=min(workers, len(sources))
            ) as executor:
                tuple(
                    executor.map(
                        _extract_one,
                        sources,
                        destinations,
                        source_infos,
                        source_signatures,
                    )
                )
        except OSError:
            for destination in destinations:
                shutil.rmtree(destination, ignore_errors=True)
            _mark_disabled(root)
            return
        try:
            (root / _READY).touch()
        except OSError:
            for destination in destinations:
                shutil.rmtree(destination, ignore_errors=True)
            try:
                (root / _READY).unlink(missing_ok=True)
            except OSError:
                pass
            _mark_disabled(root)

    def is_ready(self) -> bool:
        """Return whether this node published a complete cache."""

        root = self.root
        return not (root / _DISABLED).exists() and (root / _READY).is_file()

    def catalog(
        self,
        paths: Iterable[str],
    ) -> dict[str, dict[str, str]] | None:
        """Return source-to-NPY mappings after collective preparation."""

        sources = _canonical_paths(paths)
        root = self.root
        if (root / _DISABLED).exists():
            return None
        if not (root / _READY).is_file():
            raise RuntimeError("node decoded cache was not published")
        catalog: dict[str, dict[str, str]] = {}
        for ordinal, source in enumerate(sources):
            shard = root / f"{ordinal:08d}"
            arrays = {
                key: str(shard / f"{key}.npy")
                for key in _ARRAY_KEYS
                if (shard / f"{key}.npy").is_file()
            }
            missing = sorted(_REQUIRED_KEYS.difference(arrays))
            if missing:
                raise RuntimeError(
                    "published node cache is missing required array(s) "
                    + ", ".join(missing)
                )
            catalog[source] = arrays
        return catalog

    def cleanup(self) -> None:
        """Delete this run's exact cache directory on its owning rank."""

        if not self.owner or self._cleaned:
            return
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            self._cleaned = not self.root.exists()
        except OSError:
            # Cleanup is best-effort; retain the retry when the filesystem
            # cannot confirm whether the directory was removed.
            self._cleaned = False

    def _cleanup_at_exit(self) -> None:
        if not self.owner or self._cleaned:
            return
        try:
            shutil.rmtree(self.root)
        except OSError:
            return
        self._cleaned = True

__all__ = ["NodeDecodedCache"]
