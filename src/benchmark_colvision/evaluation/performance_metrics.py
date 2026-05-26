"""Wall-clock timing and GPU memory tracking for benchmark runs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    label: str
    duration_s: float


@dataclass
class GPUMemorySample:
    timestamp: float
    used_mb: float
    total_mb: float


@dataclass
class GPUMemoryReport:
    device_index: int
    samples: list[GPUMemorySample] = field(default_factory=list)

    @property
    def peak_used_mb(self) -> float:
        return max((s.used_mb for s in self.samples), default=0.0)

    @property
    def mean_used_mb(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.used_mb for s in self.samples) / len(self.samples)


@contextmanager
def timed(label: str) -> Iterator[TimingResult]:
    """Context manager measuring the wall-clock duration of a block."""
    start = time.perf_counter()
    result = TimingResult(label=label, duration_s=0.0)
    try:
        yield result
    finally:
        result.duration_s = time.perf_counter() - start


class GPUMemoryTracker:
    """Poll-based GPU memory tracker using nvidia-ml-py.

    Call `sample()` on a regular cadence (e.g. from a background thread or
    between iterations) and read `report` at the end.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.report = GPUMemoryReport(device_index=device_index)
        self._handle = None
        self._pynvml = None

    def __enter__(self) -> GPUMemoryTracker:
        import pynvml  # type: ignore[import-not-found]

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pynvml is not None:
            self._pynvml.nvmlShutdown()

    def sample(self) -> GPUMemorySample:
        if self._handle is None or self._pynvml is None:
            raise RuntimeError("GPUMemoryTracker must be used as a context manager")
        info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        sample = GPUMemorySample(
            timestamp=time.time(),
            used_mb=info.used / (1024 * 1024),
            total_mb=info.total / (1024 * 1024),
        )
        self.report.samples.append(sample)
        return sample


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy-compatible, avoids the import)."""
    if not values:
        return float("nan")
    if not 0 <= p <= 100:
        raise ValueError("percentile p must be in [0, 100]")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


@dataclass
class LatencySummary:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_seconds(cls, durations_s: list[float]) -> LatencySummary:
        ms = [d * 1000 for d in durations_s]
        if not ms:
            return cls(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        return cls(
            n=len(ms),
            p50_ms=percentile(ms, 50),
            p95_ms=percentile(ms, 95),
            p99_ms=percentile(ms, 99),
            mean_ms=sum(ms) / len(ms),
            min_ms=min(ms),
            max_ms=max(ms),
        )
