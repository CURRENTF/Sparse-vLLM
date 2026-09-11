"""Inspect a stable middle slice of an Nsight decode trace, not paper throughput."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--offset", type=int, help="Explicit graph-launch offset after inspecting startup replays")
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        raise ValueError("Require positive steps and a fresh output path")
    db = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    processes = db.execute("""SELECT globalPid, COUNT(DISTINCT correlationId)
        FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId != 0
        GROUP BY globalPid ORDER BY COUNT(DISTINCT correlationId) DESC""").fetchall()
    if not processes or processes[0][1] < args.steps * 2:
        raise ValueError("Missing a long-running decode graph process")
    pid = processes[0][0]
    launches = db.execute("""SELECT correlationId, COUNT(*), MIN(start), MAX(end)
        FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? AND graphNodeId != 0
        GROUP BY correlationId ORDER BY MIN(start)""", (pid,)).fetchall()
    offset = (len(launches) - args.steps) // 2 if args.offset is None else args.offset
    if offset < 0 or offset + args.steps > len(launches):
        raise ValueError("Selected graph slice lies outside the recorded trace")
    selected = launches[offset:offset + args.steps]
    if len({row[1] for row in selected}) != 1:
        raise ValueError("Middle slice changes graph kernel count; inspect manually")
    ids = [row[0] for row in selected]
    markers = ",".join("?" for _ in ids)
    sql = f"""SELECT s.value, COUNT(*), SUM(k.end-k.start),
        MIN(k.end-k.start), MAX(k.end-k.start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
        WHERE k.globalPid=? AND k.graphNodeId != 0 AND k.correlationId IN ({markers})
        GROUP BY s.id ORDER BY SUM(k.end-k.start) DESC"""
    rows = db.execute(sql, (pid, *ids)).fetchall()
    total_ns = sum(row[2] for row in rows)
    result = dict(
        scope="diagnostic_decode_graph_slice_not_paper_window",
        caveat="Node tracing perturbs timing. Kernel durations are cumulative, not a critical-path attribution or unprofiled throughput.",
        source=str(args.sqlite.resolve()),
        sqlite_sha256=hashlib.file_digest(args.sqlite.open("rb"), "sha256").hexdigest(),
        process=db.execute("SELECT pid,name FROM PROCESSES WHERE globalPid=?", (pid,)).fetchone(),
        graph_launches=len(launches), selected_steps=args.steps, selected_offset=offset,
        node_count_histogram=dict(Counter(row[1] for row in launches)),
        selected_correlation_ids=ids,
        kernel_sum_ms_per_step=total_ns / args.steps / 1e6,
        mean_graph_gpu_span_ms=sum(row[3]-row[2] for row in selected) / args.steps / 1e6,
        kernels=[dict(name=name, calls=count, ms_per_step=ns / args.steps / 1e6,
                      share_percent=100 * ns / total_ns, mean_us=ns / count / 1e3,
                      min_us=low / 1e3, max_us=high / 1e3)
                 for name, count, ns, low, high in rows],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("kernels", "selected_correlation_ids")}, indent=2))


if __name__ == "__main__":
    main()
