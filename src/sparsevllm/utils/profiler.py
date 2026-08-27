import time
import os
from collections import defaultdict
from contextlib import contextmanager

import torch
import sparsevllm.platforms as platforms
from sparsevllm.utils.log import logger

class Profiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.samples = defaultdict(list)
        self.enabled = False
        self.rank = 0
        # 通过环境变量开启设备同步，以准确测量设备耗时；保留旧 CUDA 名称兼容。
        self.device_sync = (
            os.environ.get("SPARSEVLLM_SYNC_DEVICE", "0") == "1"
            or os.environ.get("CUDA_SYNC_SVLLM", "0") == "1"
        )

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def set_rank(self, rank: int):
        self.rank = rank

    @contextmanager
    def record(self, name):
        if not self.enabled:
            yield
            return
        
        platform = platforms.current_platform
        capturing = platform.is_stream_capturing()
        if self.device_sync and not capturing:
            platform.synchronize()
        t1 = time.perf_counter()
        yield
        capturing = platform.is_stream_capturing()
        if self.device_sync and not capturing:
            platform.synchronize()
        t2 = time.perf_counter()
        
        elapsed = t2 - t1
        self.times[name] += elapsed
        self.counts[name] += 1
        self.samples[name].append(elapsed)

    def reset(self):
        self.times.clear()
        self.counts.clear()
        self.samples.clear()

    @staticmethod
    def _percentile_ms(values, quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = (len(ordered) - 1) * float(quantile)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
        return float(value) * 1000.0

    def snapshot(self):
        return {
            name: self._snapshot_entry(name, total_s)
            for name, total_s in sorted(self.times.items())
        }

    def _snapshot_entry(self, name, total_s):
        count = int(self.counts[name])
        values = self.samples[name]
        if not values and count:
            values = [float(total_s) / count]
        return {
            "calls": count,
            "total_s": float(total_s),
            "avg_ms": float(total_s) * 1000.0 / count if count else 0.0,
            "p50_ms": self._percentile_ms(values, 0.50),
            "p95_ms": self._percentile_ms(values, 0.95),
            "p99_ms": self._percentile_ms(values, 0.99),
        }

    def print_stats(self):
        if not self.enabled or not self.times:
            return

        logger.info(f"\n=== Sparse-vLLM Profiler Report (Rank {self.rank}) ===")
        # 按照总耗时降序排列
        sorted_keys = sorted(self.times.keys(), key=lambda x: self.times[x], reverse=True)
        
        # 尝试找出总耗时作为基准 (通常是 step)
        total_time = self.times.get("step", sum(self.times.values()))
        if total_time == 0: total_time = 1e-9

        print(
            f"{'Category':<30} {'Calls':<10} {'Avg (ms)':<12} {'P50 (ms)':<12} "
            f"{'P95 (ms)':<12} {'P99 (ms)':<12} {'Total (s)':<12} {'Percentage':<10}"
        )
        print("-" * 120)
        for key in sorted_keys:
            t = self.times[key]
            c = self.counts[key]
            avg = (t / c) * 1000 if c > 0 else 0
            values = self.samples[key]
            p50 = self._percentile_ms(values, 0.50)
            p95 = self._percentile_ms(values, 0.95)
            p99 = self._percentile_ms(values, 0.99)
            pct = (t / total_time) * 100
            print(
                f"{key:<30} {c:<10} {avg:<12.4f} {p50:<12.4f} {p95:<12.4f} "
                f"{p99:<12.4f} {t:<12.4f} {pct:<10.2f}%"
            )
        print("-" * 120)

# 全局单例
profiler = Profiler()
