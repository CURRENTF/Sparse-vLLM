import time
import os
from collections import defaultdict
from contextlib import contextmanager

import torch
from sparsevllm.utils.log import logger

class Profiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.enabled = False
        self.rank = 0
        # 通过环境变量 CUDA_SYNC_SVLLM=1 开启 CUDA 同步，以准确测量 GPU 耗时
        self.cuda_sync = os.environ.get("CUDA_SYNC_SVLLM", "0") == "1"

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def set_rank(self, rank: int):
        self.rank = rank

    @contextmanager
    def record(self, name):
        if not self.enabled:
            yield
            return
        
        if self.cuda_sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        yield
        if self.cuda_sync:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        self.times[name] += (t2 - t1)
        self.counts[name] += 1

    def reset(self):
        self.times.clear()
        self.counts.clear()

    def print_stats(self):
        if not self.enabled or not self.times:
            return

        logger.info(f"\n=== Sparse-vLLM Profiler Report (Rank {self.rank}) ===")
        # 按照总耗时降序排列
        sorted_keys = sorted(self.times.keys(), key=lambda x: self.times[x], reverse=True)
        
        # 尝试找出总耗时作为基准 (通常是 step)
        total_time = self.times.get("step", sum(self.times.values()))
        if total_time == 0: total_time = 1e-9

        print(f"{'Category':<30} {'Calls':<10} {'Avg (ms)':<15} {'Total (s)':<15} {'Percentage':<10}")
        print("-" * 80)
        for key in sorted_keys:
            t = self.times[key]
            c = self.counts[key]
            avg = (t / c) * 1000 if c > 0 else 0
            pct = (t / total_time) * 100
            print(f"{key:<30} {c:<10} {avg:<15.4f} {t:<15.4f} {pct:<10.2f}%")
        print("-" * 80)

# 全局单例
profiler = Profiler()
