#!/usr/bin/env python3
"""Simulate DeltaKV dynamic-stride reference-token counts with a simple loop."""

from __future__ import annotations

import argparse


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def compressed_history_len(ctx_len: int, sink: int, recent: int) -> int:
    buffer_len = max(0, ctx_len - sink)
    if buffer_len < 2 * recent:
        return 0
    return ((buffer_len - recent) // recent) * recent


def ref_token_count(ctx_len: int, alpha: float, base_stride: int, sink: int, recent: int) -> int:
    hist_len = compressed_history_len(ctx_len, sink=sink, recent=recent)
    if hist_len <= 0:
        return 0

    if alpha <= 0:
        return (hist_len + base_stride - 1) // base_stride

    pos = sink
    end = sink + hist_len
    count = 0
    while pos < end:
        count += 1
        step = base_stride + int(alpha * (pos - sink))
        pos += max(step, 1)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ctx-lens",
        default="4096,8192,16384,32768,65536,131072,262144",
        help="Comma-separated context lengths.",
    )
    parser.add_argument(
        "--alphas",
        default="0,0.001,0.02,0.05,0.1,0.2",
        help="Comma-separated stride_alpha values.",
    )
    parser.add_argument("--cluster-ratio", type=float, default=0.1)
    parser.add_argument("--num-sink-tokens", type=int, default=8)
    parser.add_argument("--num-recent-tokens", type=int, default=128)
    parser.add_argument(
        "--show-total-retained",
        action="store_true",
        help="Also print sink + recent + ref counts.",
    )
    args = parser.parse_args()

    ctx_lens = parse_csv_ints(args.ctx_lens)
    alphas = parse_csv_floats(args.alphas)
    base_stride = max(1, int(1 / args.cluster_ratio))

    print(
        f"base_stride={base_stride}, sink={args.num_sink_tokens}, recent={args.num_recent_tokens}"
    )
    print()

    header = ["alpha \\ ctx_len"] + [str(x) for x in ctx_lens]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for alpha in alphas:
        row = [str(alpha)]
        for ctx_len in ctx_lens:
            row.append(
                str(
                    ref_token_count(
                        ctx_len,
                        alpha,
                        base_stride=base_stride,
                        sink=args.num_sink_tokens,
                        recent=args.num_recent_tokens,
                    )
                )
            )
        print("| " + " | ".join(row) + " |")

    if not args.show_total_retained:
        return

    print()
    header = ["alpha \\ ctx_len"] + [str(x) for x in ctx_lens]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    retained_prefix = args.num_sink_tokens + args.num_recent_tokens
    for alpha in alphas:
        row = [str(alpha)]
        for ctx_len in ctx_lens:
            row.append(
                str(
                    retained_prefix
                    + ref_token_count(
                        ctx_len,
                        alpha,
                        base_stride=base_stride,
                        sink=args.num_sink_tokens,
                        recent=args.num_recent_tokens,
                    )
                )
            )
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
