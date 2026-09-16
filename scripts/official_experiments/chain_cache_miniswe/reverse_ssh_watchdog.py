"""Probe dedicated loopback reverse-SSH listeners and rebuild stalled sessions.

Run on the relay. Only listeners explicitly supplied by the operator are eligible;
a session sharing any other listening socket is never signalled.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time


def dedicated_pid(listeners: str, port: int) -> int:
    rows = listeners.splitlines()
    matches = [row for row in rows if len(row.split()) > 3
               and row.split()[3] == f"127.0.0.1:{port}"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one loopback listener on {port}: {matches}")
    pids = set(re.findall(r"pid=(\d+),", matches[0]))
    if len(pids) != 1 or '"sshd"' not in matches[0]:
        raise RuntimeError(f"Listener is not one identifiable sshd: {matches[0]}")
    pid = int(pids.pop())
    if sum(bool(re.search(rf"pid={pid},", row)) for row in rows) != 1:
        raise RuntimeError(f"Refusing to stop sshd {pid}: it owns other listeners")
    return pid


def probe(port: int, timeout: float) -> None:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as stream:
        banner = stream.recv(255)
        if not banner.startswith(b"SSH-2.0-"):
            raise RuntimeError(f"Invalid SSH banner on {port}: {banner!r}")


def reset_listener(port: int, expected_uid: int) -> int:
    def listeners():
        return subprocess.check_output(["ss", "-Hlntp"], text=True, timeout=5)
    pid = dedicated_pid(listeners(), port)
    # Pin process identity before validating and signalling; PID reuse is unsafe.
    fd = os.pidfd_open(pid)
    try:
        proc = Path(f"/proc/{pid}")
        if proc.stat().st_uid != expected_uid or (proc / "comm").read_text().strip() != "sshd":
            raise RuntimeError(f"Unexpected owner or executable for PID {pid}")
        if dedicated_pid(listeners(), port) != pid:
            raise RuntimeError("Listener changed while validating its owner")
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    return pid


def check_port(port, state, *, timeout, failures, max_rebuilds, expected_uid):
    try:
        probe(port, timeout)
    except (OSError, RuntimeError) as exc:
        state["failures"] = state.get("failures", 0) + 1
        state["last_error"] = str(exc)
        state["healthy"] = False
        if state["failures"] >= failures and state.get("rebuilds", 0) < max_rebuilds:
            # A missing listener is left to the worker's own reconnect service.
            try:
                pid = reset_listener(port, expected_uid)
            except (OSError, RuntimeError, subprocess.SubprocessError) as reset_error:
                state["reset_error"] = str(reset_error)
            else:
                state["last_reset_pid"] = pid
                state["rebuilds"] = state.get("rebuilds", 0) + 1
                state["failures"] = 0
                state.pop("reset_error", None)
        state["exhausted"] = state.get("rebuilds", 0) >= max_rebuilds
    else:
        state.clear()
        state.update(healthy=True, failures=0, rebuilds=0, exhausted=False)
    state["checked_at"] = time.time()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-uid", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=4)
    parser.add_argument("--failures", type=int, default=3)
    parser.add_argument("--max-rebuilds", type=int, default=30)
    args = parser.parse_args()
    if (any(not 1 <= p <= 65535 for p in args.ports)
            or min(args.timeout, args.failures, args.max_rebuilds) <= 0):
        parser.error("ports and bounds must be positive and valid")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        states = json.loads(args.state.read_text()) if args.state.exists() else {}
        for port in args.ports:
            states[str(port)] = check_port(
                port, states.get(str(port), {}), timeout=args.timeout,
                failures=args.failures, max_rebuilds=args.max_rebuilds,
                expected_uid=args.expected_uid,
            )
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(states, indent=2) + "\n")
        temporary.replace(args.state)
        print(json.dumps(states), flush=True)
    if any(not states[str(port)]["healthy"] for port in args.ports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
