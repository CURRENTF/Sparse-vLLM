# Multimodal Benchmarks

This package contains multimodal benchmark entrypoints and model-specific
adapters.

## Layout

```text
benchmark/multimodal/
  common/          shared artifact, parsing, video, and adapter helpers
  model_adapters/  model-specific loading and generation adapters
  video_qa/        video QA benchmarks and utilities
  image_qa/        image QA benchmarks
```

## Current Entry Points

| Task | Entry point |
| --- | --- |
| Unified video QA for MVBench, LongVideoBench, MLVU, Video-MME | `benchmark/multimodal/video_qa/evaluate.py` |
| StreamingBench compatibility path | `benchmark/multimodal/video_qa/streamingbench.py` |
| QA-Ego4D/ReKV-style compatibility path | `benchmark/multimodal/video_qa/qaego4d.py` |
| StreamingBench frame cache | `benchmark/multimodal/video_qa/frame_cache.py` |
| LiveVLM Table 4 audit | `benchmark/multimodal/video_qa/audit_livevlm_table4.py` |
| AI2D | `benchmark/multimodal/image_qa/ai2d.py` |
| VQAv2 | `benchmark/multimodal/image_qa/vqav2.py` |
| ScienceQA-IMG / POPE / MMBench_EN / MME / MMMU | `benchmark/multimodal/image_qa/small_image_bench.py` |

## Adapter Boundary

Dataset/task code should stay under `video_qa/` or `image_qa/`. Model-specific
loading and generation glue belongs under `model_adapters/`.

Current model families:

- `llava_onevision`: `vanilla` and visual-token pruning baselines.
- `qwen3_vl`: `vanilla` and visual-token pruning baselines. Requires a
  Transformers build with Qwen3-VL support.
