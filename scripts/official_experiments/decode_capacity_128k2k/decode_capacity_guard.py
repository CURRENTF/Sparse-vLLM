"""Keep a visible reservation between cases and reject external contention."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def descendants(pid, parent):
    for _ in range(64):
        if pid == parent:
            return True
        if pid <= 1:
            return False
        try:
            fields = Path(f"/proc/{pid}/status").read_text().splitlines()
        except FileNotFoundError:
            return True
        pid = int(next(line.split()[1] for line in fields if line.startswith("PPid:")))
    raise RuntimeError("Process ancestry exceeded 64 levels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=172800)
    args = parser.parse_args()

    def check():
        text = subprocess.check_output([
            "nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
        return [int(pid) for pid in text.split() if not descendants(int(pid), args.parent)]

    if check():
        raise RuntimeError("GPU busy before reservation")
    import torch
    allocations = [torch.zeros(1024, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]
    for i in range(len(allocations)):
        torch.cuda.synchronize(i)
    if check():
        raise RuntimeError("GPU ownership changed while attaching reservation")
    args.ready.write_text(json.dumps({"pid": os.getpid(), "devices": os.environ["CUDA_VISIBLE_DEVICES"]}))
    deadline = time.monotonic() + args.max_seconds
    while time.monotonic() < deadline:
        os.kill(args.parent, 0)
        foreign = check()
        if foreign:
            args.ready.with_suffix(".contention.json").write_text(json.dumps({"foreign_pids": foreign}))
            os.kill(args.parent, signal.SIGTERM)
            raise RuntimeError(f"External GPU contention: {foreign}")
        time.sleep(5)
    os.kill(args.parent, signal.SIGTERM)
    raise RuntimeError("Reservation lifetime exceeded")


if __name__ == "__main__":
    main()
