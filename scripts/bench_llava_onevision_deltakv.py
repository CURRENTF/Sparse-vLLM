#!/usr/bin/env python3
"""Compatibility entrypoint for the old LLaVA-OneVision benchmark name.

The no-compressor path in this benchmark is visual-token uniform pruning, not
DeltaKV clustering or learned DeltaKV compression. Keep this wrapper so older
nohup commands still resolve, but use the explicitly named script for new runs.
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    print(
        "[deprecated] scripts/bench_llava_onevision_deltakv.py now delegates to "
        "scripts/bench_llava_onevision_visual_prune.py. "
        "With --compressor_path none this is visual-token uniform pruning, not "
        "DeltaKV cluster/compressor inference.",
        flush=True,
    )
    runpy.run_path(
        str(Path(__file__).with_name("bench_llava_onevision_visual_prune.py")),
        run_name="__main__",
    )
