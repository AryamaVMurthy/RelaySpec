from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager


class CudaRegionRecorder:
    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._pairs = defaultdict(list)
        self._active: list[str] = []

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        if self._active:
            raise RuntimeError("nested CUDA regions are not supported")
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        self._active.append(name)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._active.pop()
            self._pairs[name].append((start, end))

    def totals(self, request_seconds: float) -> dict[str, float]:
        self._torch.cuda.synchronize()
        totals = {
            name: sum(start.elapsed_time(end) for start, end in pairs)
            for name, pairs in self._pairs.items()
        }
        totals["unattributed_runtime"] = max(
            0.0,
            request_seconds * 1_000 - sum(totals.values()),
        )
        return dict(sorted(totals.items()))
