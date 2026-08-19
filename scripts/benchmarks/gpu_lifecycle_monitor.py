# SPDX-License-Identifier: Apache-2.0
"""Full-lifecycle continuous GPU timeline monitor and curve recorder."""
import time
import subprocess
import json
import os
import sys
import argparse
import signal

_STOP = False

def sig_handler(signum, frame):
    global _STOP
    _STOP = True

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Full-lifecycle continuous GPU timeline monitor.")
    parser.add_argument("--gpus", type=str, default="2,3", help="Comma-separated GPU indices")
    parser.add_argument("--interval_ms", type=int, default=500, help="Sampling interval in milliseconds")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output json file")
    parser.add_argument("--watch_pid", type=int, default=None, help="Stop monitoring when this process exits")
    return parser.parse_args()

def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def main():
    args = parse_args()
    target_gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    gpu_str = ",".join(map(str, target_gpus))
    interval_s = args.interval_ms / 1000.0
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    samples = []
    start_time = time.time()
    print(f"[GPU Monitor] Started monitoring GPUs {target_gpus} every {args.interval_ms}ms...")
    if args.watch_pid:
        print(f"[GPU Monitor] Watching target PID {args.watch_pid}...")
    
    while not _STOP:
        if args.watch_pid and not is_pid_alive(args.watch_pid):
            print(f"[GPU Monitor] Target PID {args.watch_pid} has exited. Stopping monitor.")
            break
            
        t_now = time.time() - start_time
        cmd = [
            "nvidia-smi",
            f"-i", gpu_str,
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = res.stdout.strip().split("\n")
            row = {"time_s": round(t_now, 2)}
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    idx = int(parts[0])
                    row[f"gpu{idx}_util"] = float(parts[1])
                    row[f"gpu{idx}_mem_util"] = float(parts[2])
                    row[f"gpu{idx}_mem_mb"] = float(parts[3])
                    row[f"gpu{idx}_power_w"] = float(parts[4])
                    row[f"gpu{idx}_temp_c"] = float(parts[5])
            samples.append(row)
        except Exception:
            pass
        time.sleep(interval_s)
    
    total_duration = time.time() - start_time
    print(f"\n[GPU Monitor] Finished monitoring. Total duration: {total_duration:.1f}s ({len(samples)} samples collected).")
    
    # Statistical analysis
    summary = {
        "duration_seconds": round(total_duration, 2),
        "total_samples": len(samples),
        "sampling_interval_ms": args.interval_ms,
        "gpus": {}
    }
    
    print("\n" + "=" * 85)
    print(f"FULL-LIFECYCLE GPU UTILIZATION PROFILE SUMMARY ({total_duration:.1f}s TOTAL RUNTIME)")
    print("=" * 85)
    
    for idx in target_gpus:
        utils = [s.get(f"gpu{idx}_util", 0) for s in samples]
        mem_utils = [s.get(f"gpu{idx}_mem_util", 0) for s in samples]
        powers = [s.get(f"gpu{idx}_power_w", 0) for s in samples]
        mems = [s.get(f"gpu{idx}_mem_mb", 0) for s in samples]
        
        avg_util = sum(utils) / len(utils) if utils else 0
        max_util = max(utils) if utils else 0
        avg_mem_util = sum(mem_utils) / len(mem_utils) if mem_utils else 0
        avg_power = sum(powers) / len(powers) if powers else 0
        max_power = max(powers) if powers else 0
        peak_mem_gb = (max(mems) / 1024.0) if mems else 0
        
        busy_pct = (sum(1 for u in utils if u > 10) / len(utils) * 100) if utils else 0
        
        # Buckets: 0-20, 20-40, 40-60, 60-80, 80-100
        buckets = [0] * 5
        for u in utils:
            b = min(4, int(u // 20))
            buckets[b] += 1
        bucket_pcts = [(c / len(utils) * 100) if utils else 0 for c in buckets]
        
        summary["gpus"][f"gpu_{idx}"] = {
            "avg_compute_util_pct": round(avg_util, 2),
            "max_compute_util_pct": round(max_util, 2),
            "avg_memory_bus_util_pct": round(avg_mem_util, 2),
            "avg_power_w": round(avg_power, 2),
            "max_power_w": round(max_power, 2),
            "peak_vram_gb": round(peak_mem_gb, 2),
            "active_duty_cycle_pct": round(busy_pct, 2),
            "utilization_distribution_pct": {
                "0-20%": round(bucket_pcts[0], 2),
                "20-40%": round(bucket_pcts[1], 2),
                "40-60%": round(bucket_pcts[2], 2),
                "60-80%": round(bucket_pcts[3], 2),
                "80-100%": round(bucket_pcts[4], 2),
            }
        }
        
        print(f"GPU {idx}:")
        print(f"  • Compute Util: Avg = {avg_util:5.1f}% | Peak = {max_util:5.1f}% | Active Duty (>10% util) = {busy_pct:5.1f}%")
        print(f"  • Memory Bus Util: Avg = {avg_mem_util:5.1f}% | Peak VRAM Allocated = {peak_mem_gb:.2f} GB")
        print(f"  • Power Draw: Avg = {avg_power:5.1f} W | Peak = {max_power:5.1f} W / 700W TDP")
        print("  • Compute Utilization Distribution:")
        labels = [" 0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        for lbl, pct in zip(labels, bucket_pcts):
            bar = "#" * int(pct / 2.5)
            print(f"    {lbl}: |{bar:<40}| {pct:5.1f}%")
    print("=" * 85 + "\n")
    
    # Save files
    with open(args.output_file, "w") as f:
        json.dump({"summary": summary, "timeline": samples}, f, indent=2)
    
    summary_file = args.output_file.replace(".json", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[GPU Monitor] Full timeline saved to: {args.output_file}")
    print(f"[GPU Monitor] Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
