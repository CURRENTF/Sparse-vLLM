#!/usr/bin/env python3
"""Legacy script-name entrypoint for the LLaVA-OneVision benchmark.

Use the explicitly named visual-cache benchmark for new runs. It now separates
standard checkpoint-backed DeltaKV, no-compressor delta quantization, and the
visual-token uniform-pruning baseline.
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    print(
        "[deprecated] scripts/bench_llava_onevision_deltakv.py now delegates to "
        "scripts/bench_llava_onevision_visual_prune.py. "
        "Select --methods deltakv, deltakv_delta_quant, or visual_uniform_keep "
        "explicitly.",
        flush=True,
    )
    runpy.run_path(
        str(Path(__file__).with_name("bench_llava_onevision_visual_prune.py")),
        run_name="__main__",
    )
