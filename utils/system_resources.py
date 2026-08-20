"""Portable, best-effort system resource probes.

The values returned here are observations, not allocation policy.  Callers can
use the accompanying source and confidence metadata when resolving their own
memory and worker limits.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Generic, TypeVar


T = TypeVar("T", int, float)


class Confidence(str, Enum):
    """Confidence in a probed or derived value."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ResourceValue(Generic[T]):
    """A resource value together with its provenance."""

    value: T | None
    source: str
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class MemoryResources:
    """Host, process, and optional control-group memory observations."""

    host_total_bytes: ResourceValue[int]
    host_available_bytes: ResourceValue[int]
    effective_total_bytes: ResourceValue[int]
    effective_available_bytes: ResourceValue[int]
    process_rss_bytes: ResourceValue[int]
    process_peak_rss_bytes: ResourceValue[int]
    cgroup_limit_bytes: ResourceValue[int]
    cgroup_usage_bytes: ResourceValue[int]
    cgroup_headroom_bytes: ResourceValue[int]


@dataclass(frozen=True, slots=True)
class CpuResources:
    """Logical, affinity, quota, and effective CPU observations."""

    logical_count: ResourceValue[int]
    affinity_count: ResourceValue[int]
    quota_count: ResourceValue[float]
    effective_count: ResourceValue[float]


@dataclass(frozen=True, slots=True)
class SystemResources:
    """An immutable snapshot of resources visible to the current process."""

    platform: str
    memory: MemoryResources
    cpu: CpuResources


@dataclass(frozen=True, slots=True)
class ProbeEnvironment:
    """Injectable operating-system dependencies used by the probes.

    Tests can construct this class directly to simulate a platform.  Production
    callers normally use :meth:`current` indirectly through
    :func:`probe_system_resources`.
    """

    platform: str
    cpu_count: Callable[[], int | None] | None = None
    sysconf: Callable[[str], int] | None = None
    affinity_count: Callable[[], int | None] | None = None
    read_text: Callable[[str], str | None] | None = None
    windows_memory_status: Callable[[], tuple[int, int]] | None = None
    windows_process_rss: Callable[[], int | None] | None = None
    resource_max_rss: Callable[[], int | float | None] | None = None

    @classmethod
    def current(cls) -> "ProbeEnvironment":
        """Build an environment backed only by the Python standard library."""

        affinity_count: Callable[[], int | None] | None = None
        process_cpu_count = getattr(os, "process_cpu_count", None)
        if callable(process_cpu_count):
            affinity_count = process_cpu_count
        elif hasattr(os, "sched_getaffinity"):
            affinity_count = lambda: len(os.sched_getaffinity(0))
        elif sys.platform.startswith("win"):
            affinity_count = _windows_affinity_count

        is_windows = sys.platform.startswith("win")
        return cls(
            platform=sys.platform,
            cpu_count=os.cpu_count,
            sysconf=getattr(os, "sysconf", None),
            affinity_count=affinity_count,
            read_text=_read_text_file if sys.platform.startswith("linux") else None,
            windows_memory_status=_windows_memory_status if is_windows else None,
            windows_process_rss=_windows_process_rss if is_windows else None,
            resource_max_rss=None if is_windows else _resource_max_rss,
        )


@dataclass(frozen=True, slots=True)
class _HostMemory:
    total: ResourceValue[int]
    available: ResourceValue[int]


@dataclass(frozen=True, slots=True)
class _CgroupMemory:
    limit: ResourceValue[int]
    usage: ResourceValue[int]
    headroom: ResourceValue[int]
    detected: bool = False


@dataclass(frozen=True, slots=True)
class _CgroupLocation:
    leaf: PurePosixPath
    root: PurePosixPath


_CONFIDENCE_ORDER = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}
_CGROUP_V1_UNLIMITED_MIN = 1 << 60


def _unavailable(source: str = "unavailable") -> ResourceValue:
    return ResourceValue(None, source, Confidence.LOW)


def _weaker_confidence(values: Iterable[ResourceValue]) -> Confidence:
    return min((value.confidence for value in values), key=_CONFIDENCE_ORDER.__getitem__)


def _minimum_value(*values: ResourceValue, source: str = "unavailable") -> ResourceValue:
    available = [item for item in values if item.value is not None]
    if not available:
        return _unavailable(source)
    if len(available) == 1:
        return available[0]
    sources = ",".join(dict.fromkeys(item.source for item in available))
    return ResourceValue(
        min(item.value for item in available),
        f"minimum({sources})",
        _weaker_confidence(available),
    )


def _safe_call(function: Callable | None, *args):
    if function is None:
        return None
    try:
        return function(*args)
    except Exception:
        return None


def _nonnegative_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_int(value) -> int | None:
    value = _nonnegative_int(value)
    return value if value is not None and value > 0 else None


def _read_text_file(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _read_text(environment: ProbeEnvironment, path: str) -> str | None:
    value = _safe_call(environment.read_text, path)
    return value if isinstance(value, str) else None


def _sysconf_value(environment: ProbeEnvironment, *names: str) -> int | None:
    for name in names:
        value = _positive_int(_safe_call(environment.sysconf, name))
        if value is not None:
            return value
    return None


def _probe_posix_host_memory(environment: ProbeEnvironment) -> _HostMemory:
    page_size = _sysconf_value(environment, "SC_PAGE_SIZE", "SC_PAGESIZE")
    total_pages = _sysconf_value(environment, "SC_PHYS_PAGES")
    available_pages = _safe_call(environment.sysconf, "SC_AVPHYS_PAGES")
    available_pages = _nonnegative_int(available_pages)

    total_bytes = page_size * total_pages if page_size is not None and total_pages is not None else None
    available_bytes = (
        page_size * available_pages
        if page_size is not None and available_pages is not None
        else None
    )
    if total_bytes is not None and available_bytes is not None:
        available_bytes = min(total_bytes, available_bytes)

    return _HostMemory(
        ResourceValue(total_bytes, "posix.sysconf", Confidence.HIGH)
        if total_bytes is not None
        else _unavailable("posix.sysconf"),
        ResourceValue(available_bytes, "posix.sysconf", Confidence.MEDIUM)
        if available_bytes is not None
        else _unavailable("posix.sysconf"),
    )


def _parse_linux_meminfo(text: str | None) -> tuple[int | None, int | None]:
    if text is None:
        return None, None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.split()
        if not fields or (len(fields) > 1 and fields[1].lower() != "kb"):
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if value >= 0:
            values[key] = value * 1024

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if available is None:
        estimate_fields = ("MemFree", "Buffers", "Cached", "SReclaimable")
        if all(name in values for name in estimate_fields):
            available = sum(values[name] for name in estimate_fields) - values.get("Shmem", 0)
            available = max(0, available)
    if total is not None and available is not None:
        available = min(total, available)
    return total, available


def _probe_linux_host_memory(environment: ProbeEnvironment) -> _HostMemory:
    portable = _probe_posix_host_memory(environment)
    if portable.total.value is not None and portable.available.value is not None:
        return portable

    total, available = _parse_linux_meminfo(_read_text(environment, "/proc/meminfo"))
    return _HostMemory(
        portable.total
        if portable.total.value is not None
        else (
            ResourceValue(total, "linux.procfs.meminfo", Confidence.HIGH)
            if total is not None
            else portable.total
        ),
        portable.available
        if portable.available.value is not None
        else (
            ResourceValue(available, "linux.procfs.meminfo", Confidence.HIGH)
            if available is not None
            else portable.available
        ),
    )


def _probe_windows_host_memory(environment: ProbeEnvironment) -> _HostMemory:
    status = _safe_call(environment.windows_memory_status)
    if not isinstance(status, tuple) or len(status) != 2:
        return _HostMemory(
            _unavailable("windows.GlobalMemoryStatusEx"),
            _unavailable("windows.GlobalMemoryStatusEx"),
        )
    total = _positive_int(status[0])
    available = _nonnegative_int(status[1])
    if total is None or available is None:
        return _HostMemory(
            _unavailable("windows.GlobalMemoryStatusEx"),
            _unavailable("windows.GlobalMemoryStatusEx"),
        )
    available = min(total, available)
    return _HostMemory(
        ResourceValue(total, "windows.GlobalMemoryStatusEx", Confidence.HIGH),
        ResourceValue(available, "windows.GlobalMemoryStatusEx", Confidence.HIGH),
    )


def _probe_process_rss(environment: ProbeEnvironment, platform_name: str) -> ResourceValue[int]:
    if platform_name == "windows":
        rss = _nonnegative_int(_safe_call(environment.windows_process_rss))
        return (
            ResourceValue(rss, "windows.GetProcessMemoryInfo", Confidence.HIGH)
            if rss is not None
            else _unavailable("windows.GetProcessMemoryInfo")
        )
    if platform_name != "linux":
        return _unavailable("current process RSS unsupported")

    statm = _read_text(environment, "/proc/self/statm")
    fields = statm.split() if statm is not None else []
    if len(fields) < 2:
        return _unavailable("linux.procfs.statm")
    try:
        resident_pages = int(fields[1])
    except ValueError:
        return _unavailable("linux.procfs.statm")
    page_size = _sysconf_value(environment, "SC_PAGE_SIZE", "SC_PAGESIZE")
    if resident_pages < 0 or page_size is None:
        return _unavailable("linux.procfs.statm")
    return ResourceValue(resident_pages * page_size, "linux.procfs.statm", Confidence.HIGH)


def _probe_process_peak_rss(environment: ProbeEnvironment, platform_name: str) -> ResourceValue[int]:
    raw_rss = _safe_call(environment.resource_max_rss)
    if isinstance(raw_rss, bool) or not isinstance(raw_rss, (int, float)) or raw_rss < 0:
        return _unavailable("resource.getrusage")
    if platform_name == "darwin":
        rss = int(raw_rss)
    elif platform_name == "linux" or platform_name in {"freebsd", "openbsd", "netbsd"}:
        rss = int(raw_rss * 1024)
    else:
        return _unavailable("resource.getrusage units unknown")
    return ResourceValue(rss, "resource.getrusage", Confidence.HIGH)


def _probe_logical_cpus(environment: ProbeEnvironment) -> ResourceValue[int]:
    count = _positive_int(_safe_call(environment.cpu_count))
    return (
        ResourceValue(count, "os.cpu_count", Confidence.HIGH)
        if count is not None
        else _unavailable("os.cpu_count")
    )


def _probe_affinity_cpus(environment: ProbeEnvironment) -> ResourceValue[int]:
    count = _positive_int(_safe_call(environment.affinity_count))
    return (
        ResourceValue(count, "process affinity", Confidence.HIGH)
        if count is not None
        else _unavailable("process affinity")
    )


def _unescape_mount_path(value: str) -> str:
    for encoded, decoded in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
        value = value.replace(encoded, decoded)
    return value


def _absolute_posix_path(value: str) -> PurePosixPath | None:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        return None
    return PurePosixPath("/", *(part for part in path.parts if part not in {"/", "."}))


def _parse_cgroup_membership(text: str | None) -> tuple[str | None, dict[str, str]]:
    unified: str | None = None
    controllers: dict[str, str] = {}
    if text is None:
        return unified, controllers
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or _absolute_posix_path(fields[2]) is None:
            continue
        if fields[0] == "0" and not fields[1]:
            unified = fields[2]
        else:
            for controller in fields[1].split(","):
                if controller:
                    controllers[controller] = fields[2]
    return unified, controllers


def _mount_locations(
    text: str | None,
    filesystem: str,
    controller: str | None,
    membership: str | None,
) -> list[_CgroupLocation]:
    if text is None or membership is None:
        return []
    group_path = _absolute_posix_path(membership)
    if group_path is None:
        return []
    locations: list[_CgroupLocation] = []
    for line in text.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 5 or len(fields) <= separator + 3 or fields[separator + 1] != filesystem:
            continue
        if controller is not None:
            options = set(fields[5].split(",")) | set(fields[separator + 3].split(","))
            if controller not in options:
                continue
        mount_root = _absolute_posix_path(_unescape_mount_path(fields[3]))
        mount_point = _absolute_posix_path(_unescape_mount_path(fields[4]))
        if mount_root is None or mount_point is None:
            continue
        try:
            relative = group_path.relative_to(mount_root)
        except ValueError:
            continue
        locations.append(_CgroupLocation(mount_point / relative, mount_point))
    return locations


def _cgroup_locations(
    environment: ProbeEnvironment,
    version: int,
    controller: str,
) -> list[_CgroupLocation]:
    membership_text = _read_text(environment, "/proc/self/cgroup")
    unified, controllers = _parse_cgroup_membership(membership_text)
    mountinfo = _read_text(environment, "/proc/self/mountinfo")
    if version == 2:
        locations = _mount_locations(mountinfo, "cgroup2", None, unified)
        fallback_root = PurePosixPath("/sys/fs/cgroup")
        membership = unified
    else:
        membership = controllers.get(controller)
        locations = _mount_locations(mountinfo, "cgroup", controller, membership)
        fallback_root = PurePosixPath("/sys/fs/cgroup") / controller

    if membership is not None:
        group_path = _absolute_posix_path(membership)
        if group_path is not None:
            locations.append(_CgroupLocation(fallback_root / group_path.relative_to("/"), fallback_root))
    locations.append(_CgroupLocation(fallback_root, fallback_root))

    unique: list[_CgroupLocation] = []
    seen: set[tuple[str, str]] = set()
    for location in locations:
        key = (str(location.leaf), str(location.root))
        if key not in seen:
            unique.append(location)
            seen.add(key)
    return unique


def _cgroup_levels(location: _CgroupLocation) -> Iterable[PurePosixPath]:
    try:
        location.leaf.relative_to(location.root)
    except ValueError:
        return
    current = location.leaf
    for _ in range(128):
        yield current
        if current == location.root:
            return
        current = current.parent


def _parse_cgroup_integer(text: str | None) -> int | None:
    if text is None:
        return None
    fields = text.split()
    if len(fields) != 1:
        return None
    try:
        value = int(fields[0])
    except ValueError:
        return None
    return value if value >= 0 else None


def _parse_memory_limit(text: str | None, version: int) -> tuple[bool, int | None]:
    if text is None:
        return False, None
    fields = text.split()
    if len(fields) != 1:
        return False, None
    if version == 2 and fields[0] == "max":
        return True, None
    try:
        value = int(fields[0])
    except ValueError:
        return False, None
    if version == 1 and (value < 0 or value >= _CGROUP_V1_UNLIMITED_MIN):
        return True, None
    if value < 0:
        return False, None
    return True, value


def _probe_cgroup_memory_version(environment: ProbeEnvironment, version: int) -> _CgroupMemory:
    limit_file = "memory.max" if version == 2 else "memory.limit_in_bytes"
    usage_file = "memory.current" if version == 2 else "memory.usage_in_bytes"
    source_prefix = f"cgroup-v{version}.memory"

    for location in _cgroup_locations(environment, version, "memory"):
        finite_limits: list[int] = []
        headrooms: list[int] = []
        leaf_usage: int | None = None
        detected = False
        unlimited = False
        incomplete_finite_headroom = False
        for index, level in enumerate(_cgroup_levels(location)):
            limit_valid, limit = _parse_memory_limit(
                _read_text(environment, str(level / limit_file)), version
            )
            usage = _parse_cgroup_integer(_read_text(environment, str(level / usage_file)))
            detected = detected or limit_valid or usage is not None
            unlimited = unlimited or (limit_valid and limit is None)
            if index == 0 and usage is not None:
                leaf_usage = usage
            if limit is not None:
                finite_limits.append(limit)
                if usage is not None:
                    headrooms.append(max(0, limit - usage))
                else:
                    incomplete_finite_headroom = True
        if not detected:
            continue
        limit_value = min(finite_limits) if finite_limits else None
        # A parent's usage includes work outside this leaf cgroup. A readable
        # child usage cannot safely stand in for a missing finite ancestor
        # usage because the real ancestor headroom may already be zero.
        headroom_value = (
            min(headrooms)
            if headrooms and not incomplete_finite_headroom
            else None
        )
        return _CgroupMemory(
            ResourceValue(limit_value, f"{source_prefix}.limit", Confidence.HIGH)
            if limit_value is not None
            else ResourceValue(
                None,
                f"{source_prefix}.limit(unlimited)" if unlimited else f"{source_prefix}.limit",
                Confidence.HIGH if unlimited else Confidence.LOW,
            ),
            ResourceValue(leaf_usage, f"{source_prefix}.current", Confidence.HIGH)
            if leaf_usage is not None
            else _unavailable(f"{source_prefix}.current"),
            ResourceValue(headroom_value, f"{source_prefix}.headroom", Confidence.HIGH)
            if headroom_value is not None
            else _unavailable(
                f"{source_prefix}.headroom(incomplete)"
                if incomplete_finite_headroom
                else f"{source_prefix}.headroom"
            ),
            True,
        )
    return _CgroupMemory(_unavailable(), _unavailable(), _unavailable(), False)


def _probe_linux_cgroup_memory(environment: ProbeEnvironment) -> _CgroupMemory:
    membership = _read_text(environment, "/proc/self/cgroup")
    unified, controllers = _parse_cgroup_membership(membership)
    versions = (2, 1) if unified is not None else ((1, 2) if "memory" in controllers else (2, 1))
    for version in versions:
        result = _probe_cgroup_memory_version(environment, version)
        if result.detected:
            return result
    return _CgroupMemory(_unavailable(), _unavailable(), _unavailable(), False)


def _parse_cpu_quota(
    text: str | None,
    version: int,
    period_text: str | None = None,
) -> tuple[bool, float | None]:
    if version == 2:
        fields = text.split() if text is not None else []
        if len(fields) != 2:
            return False, None
        if fields[0] == "max":
            try:
                return (int(fields[1]) > 0), None
            except ValueError:
                return False, None
        quota_text, period = fields
    else:
        quota_fields = text.split() if text is not None else []
        period_fields = period_text.split() if period_text is not None else []
        if len(quota_fields) != 1 or len(period_fields) != 1:
            return False, None
        quota_text, period = quota_fields[0], period_fields[0]
    try:
        quota = int(quota_text)
        period_value = int(period)
    except ValueError:
        return False, None
    if version == 1 and quota == -1 and period_value > 0:
        return True, None
    if quota <= 0 or period_value <= 0:
        return False, None
    return True, quota / period_value


def _probe_cgroup_cpu_version(
    environment: ProbeEnvironment,
    version: int,
) -> tuple[ResourceValue[float], bool]:
    source = f"cgroup-v{version}.cpu.quota"
    for location in _cgroup_locations(environment, version, "cpu"):
        quotas: list[float] = []
        detected = False
        unlimited = False
        for level in _cgroup_levels(location):
            if version == 2:
                valid, quota = _parse_cpu_quota(_read_text(environment, str(level / "cpu.max")), 2)
            else:
                valid, quota = _parse_cpu_quota(
                    _read_text(environment, str(level / "cpu.cfs_quota_us")),
                    1,
                    _read_text(environment, str(level / "cpu.cfs_period_us")),
                )
            detected = detected or valid
            unlimited = unlimited or (valid and quota is None)
            if quota is not None:
                quotas.append(quota)
        if not detected:
            continue
        if quotas:
            return ResourceValue(min(quotas), source, Confidence.HIGH), True
        return ResourceValue(
            None,
            f"{source}(unlimited)" if unlimited else source,
            Confidence.HIGH if unlimited else Confidence.LOW,
        ), True
    return _unavailable(source), False


def _probe_linux_cpu_quota(environment: ProbeEnvironment) -> ResourceValue[float]:
    membership = _read_text(environment, "/proc/self/cgroup")
    unified, controllers = _parse_cgroup_membership(membership)
    versions = (2, 1) if unified is not None else ((1, 2) if "cpu" in controllers else (2, 1))
    for version in versions:
        quota, detected = _probe_cgroup_cpu_version(environment, version)
        if detected:
            return quota
    return _unavailable("cgroup CPU quota")


def _normalize_platform(platform_name: str) -> str:
    lowered = platform_name.lower()
    if lowered.startswith("linux"):
        return "linux"
    if lowered.startswith("win"):
        return "windows"
    if lowered in {"darwin", "macos"}:
        return "darwin"
    return lowered or "unknown"


def probe_process_resident_bytes(
    environment: ProbeEnvironment | None = None,
) -> ResourceValue[int]:
    """Probe process RSS, falling back to peak RSS when necessary."""

    environment = environment or ProbeEnvironment.current()
    platform_name = _normalize_platform(environment.platform)
    current = _probe_process_rss(environment, platform_name)
    if current.value is not None:
        return current
    return _probe_process_peak_rss(environment, platform_name)


def probe_system_resources(environment: ProbeEnvironment | None = None) -> SystemResources:
    """Probe current system resources without applying any resource policy."""

    environment = environment or ProbeEnvironment.current()
    platform_name = _normalize_platform(environment.platform)

    if platform_name == "windows":
        host_memory = _probe_windows_host_memory(environment)
    elif platform_name == "linux":
        host_memory = _probe_linux_host_memory(environment)
    else:
        host_memory = _probe_posix_host_memory(environment)

    process_rss = _probe_process_rss(environment, platform_name)
    process_peak_rss = _probe_process_peak_rss(environment, platform_name)
    logical_cpus = _probe_logical_cpus(environment)
    affinity_cpus = _probe_affinity_cpus(environment)

    if platform_name == "linux":
        cgroup_memory = _probe_linux_cgroup_memory(environment)
        cpu_quota = _probe_linux_cpu_quota(environment)
    else:
        cgroup_memory = _CgroupMemory(_unavailable(), _unavailable(), _unavailable(), False)
        cpu_quota = _unavailable("cgroup CPU quota unsupported")

    effective_total = _minimum_value(
        host_memory.total,
        cgroup_memory.limit,
        source="effective memory capacity unavailable",
    )
    if (
        cgroup_memory.limit.value is not None
        and cgroup_memory.headroom.value is None
    ):
        # A finite control-group limit without matching hierarchy usage has no
        # safe available-memory estimate. Do not fall back to host availability,
        # which can be much larger than the remaining job cap.
        effective_available = _unavailable(
            "effective memory headroom unavailable: incomplete cgroup usage"
        )
    else:
        effective_available = _minimum_value(
            host_memory.available,
            cgroup_memory.headroom,
            source="effective memory headroom unavailable",
        )
    if effective_available.value is not None and effective_total.value is not None:
        effective_available = _minimum_value(effective_available, effective_total)

    effective_cpus = _minimum_value(
        logical_cpus,
        affinity_cpus,
        cpu_quota,
        source="effective CPU count unavailable",
    )
    effective_cpus = ResourceValue(
        float(effective_cpus.value) if effective_cpus.value is not None else None,
        effective_cpus.source,
        effective_cpus.confidence,
    )

    return SystemResources(
        platform=platform_name,
        memory=MemoryResources(
            host_total_bytes=host_memory.total,
            host_available_bytes=host_memory.available,
            effective_total_bytes=effective_total,
            effective_available_bytes=effective_available,
            process_rss_bytes=process_rss,
            process_peak_rss_bytes=process_peak_rss,
            cgroup_limit_bytes=cgroup_memory.limit,
            cgroup_usage_bytes=cgroup_memory.usage,
            cgroup_headroom_bytes=cgroup_memory.headroom,
        ),
        cpu=CpuResources(
            logical_count=logical_cpus,
            affinity_count=affinity_cpus,
            quota_count=cpu_quota,
            effective_count=effective_cpus,
        ),
    )


def _resource_max_rss() -> int | float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None


def _windows_memory_status() -> tuple[int, int]:
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    function.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _windows_process_rss() -> int:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()
    try:
        function = kernel32.K32GetProcessMemoryInfo
    except AttributeError:
        function = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    function.restype = wintypes.BOOL
    if not function(process, ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def _windows_affinity_count() -> int:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    function = kernel32.GetProcessAffinityMask
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    function.restype = wintypes.BOOL
    if not function(kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
    return int(process_mask.value).bit_count()


__all__ = [
    "Confidence",
    "CpuResources",
    "MemoryResources",
    "ProbeEnvironment",
    "ResourceValue",
    "SystemResources",
    "probe_process_resident_bytes",
    "probe_system_resources",
]
