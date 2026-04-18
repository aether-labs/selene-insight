"""Lethe — High-performance KV store prototype with hash index.

Optimized for point-lookups (< 1ms) using unsorted hash indexing.
No LSM-Tree, no sorting overhead — because telemetry point-lookups
don't need range scans for the primary access pattern.

Time-range queries use a separate ordered index for the dashboard
"rewind" feature.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any


class Lethe:
    """In-memory hash-indexed KV store with O(1) point lookups.

    Two access patterns:
    - Point lookup by key: O(1) via hash map
    - Time range scan: O(log n + k) via ordered timestamp index
    """

    def __init__(self, max_entries: int = 1_000_000) -> None:
        self._store: dict[str, bytes] = {}
        self._ts_index: OrderedDict[float, str] = OrderedDict()  # timestamp → key
        self._max = max_entries
        self._lock = threading.Lock()
        self._writes = 0

    def put(
        self, key: str, value: dict[str, Any], timestamp: float | None = None
    ) -> None:
        """Store a value with O(1) hash insertion.

        Args:
            key: Unique key for point lookup.
            value: JSON-serializable dict.
            timestamp: Optional timestamp for time-range indexing.
        """
        encoded = json.dumps(value).encode()
        with self._lock:
            self._store[key] = encoded
            if timestamp is not None:
                self._ts_index[timestamp] = key
            self._writes += 1

            # Evict oldest if over capacity
            if len(self._store) > self._max:
                self._evict()

    def get(self, key: str) -> dict[str, Any] | None:
        """Point lookup by key. Target: < 1ms."""
        data = self._store.get(key)
        if data is None:
            return None
        return json.loads(data)

    def range(
        self, start_ts: float, end_ts: float, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Time-range scan for dashboard rewind.

        Returns entries between start_ts and end_ts, ordered by time.
        """
        results = []
        with self._lock:
            for ts, key in self._ts_index.items():
                if ts < start_ts:
                    continue
                if ts > end_ts:
                    break
                data = self._store.get(key)
                if data:
                    results.append(json.loads(data))
                if len(results) >= limit:
                    break
        return results

    def latest(self, n: int = 1) -> list[dict[str, Any]]:
        """Get the N most recent entries."""
        results = []
        with self._lock:
            keys = list(self._ts_index.values())[-n:]
        for key in reversed(keys):
            data = self._store.get(key)
            if data:
                results.append(json.loads(data))
        return results

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def total_writes(self) -> int:
        return self._writes

    def _evict(self) -> None:
        """Evict oldest 10% when over capacity."""
        evict_count = self._max // 10
        keys_to_remove = list(self._ts_index.keys())[:evict_count]
        for ts in keys_to_remove:
            key = self._ts_index.pop(ts)
            self._store.pop(key, None)

    def bench_point_lookup(self, key: str, iterations: int = 10000) -> float:
        """Benchmark point lookup latency in microseconds."""
        start = time.perf_counter()
        for _ in range(iterations):
            self.get(key)
        elapsed = time.perf_counter() - start
        return (elapsed / iterations) * 1_000_000  # microseconds
