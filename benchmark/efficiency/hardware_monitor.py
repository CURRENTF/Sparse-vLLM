# SPDX-License-Identifier: Apache-2.0
"""High-resolution GPU hardware timeline monitor and duty-cycle/bubble analyzer."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_STOP = False


def _sig_handler(signum, frame):
    global _STOP
    _STOP = True


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class GPUHardwareMonitor:
    """Threaded or standalone GPU hardware lifecycle monitor."""

    def __init__(
        self,
        gpus: list[int],
        interval_ms: int = 200,
        output_file: str | Path | None = None,
        watch_pid: int | None = None,
    ):
        self.gpus = sorted(gpus)
        self.interval_s = max(0.05, interval_ms / 1000.0)
        self.interval_ms = interval_ms
        self.output_file = Path(output_file) if output_file else None
        self.watch_pid = watch_pid
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self) -> None:
        self.samples.clear()
        self._stop_event.clear()
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.end_time = time.time()
        summary = self.analyze_and_save()
        return summary

    def _monitor_loop(self) -> None:
        gpu_str = ",".join(map(str, self.gpus))
        cmd = [
            "nvidia-smi",
            "-i",
            gpu_str,
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set() and not _STOP:
            if self.watch_pid and not is_pid_alive(self.watch_pid):
                break

            t_now = time.time() - self.start_time
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines = res.stdout.strip().split("\n")
                row: dict[str, Any] = {"time_s": round(t_now, 3)}
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        idx = int(parts[0])
                        row[f"gpu{idx}_util"] = float(parts[1])
                        row[f"gpu{idx}_mem_util"] = float(parts[2])
                        row[f"gpu{idx}_mem_mb"] = float(parts[3])
                        row[f"gpu{idx}_power_w"] = float(parts[4])
                        row[f"gpu{idx}_temp_c"] = float(parts[5])
                self.samples.append(row)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def analyze_and_save(self) -> dict[str, Any]:
        total_duration = max(0.001, (self.end_time or time.time()) - self.start_time)
        summary: dict[str, Any] = {
            "duration_seconds": round(total_duration, 3),
            "total_samples": len(self.samples),
            "sampling_interval_ms": self.interval_ms,
            "gpus": {},
            "aggregate": {},
        }

        all_gpu_utils = []
        all_gpu_mem_utils = []
        all_powers = []
        total_energy_joules = 0.0

        for idx in self.gpus:
            utils = [s.get(f"gpu{idx}_util", 0.0) for s in self.samples]
            mem_utils = [s.get(f"gpu{idx}_mem_util", 0.0) for s in self.samples]
            powers = [s.get(f"gpu{idx}_power_w", 0.0) for s in self.samples]
            mems = [s.get(f"gpu{idx}_mem_mb", 0.0) for s in self.samples]

            avg_util = sum(utils) / len(utils) if utils else 0.0
            max_util = max(utils) if utils else 0.0
            avg_mem_util = sum(mem_utils) / len(mem_utils) if mem_utils else 0.0
            avg_power = sum(powers) / len(powers) if powers else 0.0
            max_power = max(powers) if powers else 0.0
            peak_mem_gb = (max(mems) / 1024.0) if mems else 0.0

            # Active Duty Cycle (>10% compute utilization)
            busy_pct = (sum(1 for u in utils if u > 10.0) / len(utils) * 100.0) if utils else 0.0
            bubble_pct = 100.0 - busy_pct

            # Utilization Distribution Buckets
            buckets = [0] * 5
            for u in utils:
                b = min(4, int(u // 20))
                buckets[b] += 1
            bucket_pcts = [(c / len(utils) * 100.0) if utils else 0.0 for c in buckets]

            energy_j = avg_power * total_duration
            total_energy_joules += energy_j

            summary["gpus"][f"gpu_{idx}"] = {
                "avg_compute_util_pct": round(avg_util, 2),
                "max_compute_util_pct": round(max_util, 2),
                "avg_memory_bus_util_pct": round(avg_mem_util, 2),
                "avg_power_w": round(avg_power, 2),
                "max_power_w": round(max_power, 2),
                "energy_joules": round(energy_j, 2),
                "peak_vram_gb": round(peak_mem_gb, 2),
                "active_duty_cycle_pct": round(busy_pct, 2),
                "host_launch_bubble_pct": round(bubble_pct, 2),
                "utilization_distribution_pct": {
                    "0-20%": round(bucket_pcts[0], 2),
                    "20-40%": round(bucket_pcts[1], 2),
                    "40-60%": round(bucket_pcts[2], 2),
                    "60-80%": round(bucket_pcts[3], 2),
                    "80-100%": round(bucket_pcts[4], 2),
                },
            }

            all_gpu_utils.extend(utils)
            all_gpu_mem_utils.extend(mem_utils)
            all_powers.extend(powers)

        # Aggregate cross-GPU summary
        if self.gpus:
            n_gpus = len(self.gpus)
            avg_all_util = sum(all_gpu_utils) / len(all_gpu_utils) if all_gpu_utils else 0.0
            avg_all_mem_util = sum(all_gpu_mem_utils) / len(all_gpu_mem_utils) if all_gpu_mem_utils else 0.0
            avg_total_power = sum(s.get(f"gpu{idx}_power_w", 0.0) for s in self.samples for idx in self.gpus) / len(self.samples) if self.samples else 0.0
            mean_duty_cycle = sum(summary["gpus"][f"gpu_{idx}"]["active_duty_cycle_pct"] for idx in self.gpus) / n_gpus
            mean_bubble = sum(summary["gpus"][f"gpu_{idx}"]["host_launch_bubble_pct"] for idx in self.gpus) / n_gpus

            summary["aggregate"] = {
                "num_gpus": n_gpus,
                "mean_compute_util_pct": round(avg_all_util, 2),
                "mean_memory_bus_util_pct": round(avg_all_mem_util, 2),
                "avg_total_power_w": round(avg_total_power, 2),
                "total_energy_joules": round(total_energy_joules, 2),
                "mean_active_duty_cycle_pct": round(mean_duty_cycle, 2),
                "mean_host_launch_bubble_pct": round(mean_bubble, 2),
            }

        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "timeline": self.samples}, f, indent=2)

            summary_file = self.output_file.with_name(
                self.output_file.stem + "_summary.json"
            )
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="High-resolution GPU hardware timeline monitor.")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU indices")
    parser.add_argument("--interval_ms", type=int, default=200, help="Sampling interval in milliseconds")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output json file")
    parser.add_argument("--watch_pid", type=int, default=None, help="Stop monitoring when this process exits")
    return parser.parse_args()


def main():
    args = parse_args()
    target_gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    monitor = GPUHardwareMonitor(
        gpus=target_gpus,
        interval_ms=args.interval_ms,
        output_file=args.output_file,
        watch_pid=args.watch_pid,
    )
    print(f"[Hardware Monitor] Started monitoring GPUs {target_gpus} every {args.interval_ms}ms...")
    monitor.start()

    try:
        while not _STOP:
            if args.watch_pid and not is_pid_alive(args.watch_pid):
                print(f"[Hardware Monitor] Target PID {args.watch_pid} has exited. Stopping monitor.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        summary = monitor.stop()
        print("\n" + "=" * 80)
        print(f"GPU HARDWARE PROFILE SUMMARY ({summary['duration_seconds']}s RUNTIME, {summary['total_samples']} SAMPLES)")
        print("=" * 80)
        for idx in target_gpus:
            g_data = summary["gpus"].get(f"gpu_{idx}", {})
            print(f"GPU {idx}:")
            print(f"  • Compute Util: Avg = {g_data.get('avg_compute_util_pct', 0):5.1f}% | Peak = {g_data.get('max_compute_util_pct', 0):5.1f}%")
            print(f"  • Duty Cycle: {g_data.get('active_duty_cycle_pct', 0):5.1f}% (Launch Bubble: {g_data.get('host_launch_bubble_pct', 0):5.1f}%)")
            print(f"  • Memory Bus Util: Avg = {g_data.get('avg_memory_bus_util_pct', 0):5.1f}% | Peak VRAM = {g_data.get('peak_vram_gb', 0):.2f} GB")
            print(f"  • Power: Avg = {g_data.get('avg_power_w', 0):5.1f} W | Energy = {g_data.get('energy_joules', 0):.1f} J")
        print("=" * 80 + "\n")
        if args.output_file:
            print(f"[Hardware Monitor] Timeline saved to: {args.output_file}")


if __name__ == "__main__":
    main()
