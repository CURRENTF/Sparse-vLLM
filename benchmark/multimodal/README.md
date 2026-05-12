# Multimodal Benchmarks

This package contains model-agnostic multimodal benchmark entrypoints and
model-specific adapters.

## Layout

```text
benchmark/multimodal/
  common/          shared artifact, parsing, video, and adapter interfaces
  model_adapters/  model-specific loading/generation adapters
  video_qa/        video QA benchmarks and utilities
  image_qa/        image QA benchmarks
  visual_cache/    visual-cache ablation benchmarks
```

## Current Entry Points

| Task | Entry point |
| --- | --- |
| StreamingBench | `benchmark/multimodal/video_qa/streamingbench.py` |
| Video-MME | `benchmark/multimodal/video_qa/videomme.py` |
| QA-Ego4D/ReKV-style video QA | `benchmark/multimodal/video_qa/qaego4d.py` |
| StreamingBench frame cache | `benchmark/multimodal/video_qa/frame_cache.py` |
| LiveVLM Table 4 audit | `benchmark/multimodal/video_qa/audit_livevlm_table4.py` |
| AI2D | `benchmark/multimodal/image_qa/ai2d.py` |
| Visual-cache ablations | `benchmark/multimodal/visual_cache/run_visual_cache.py` |

## Adapter Boundary

Dataset/task code should stay under `video_qa/`, `image_qa/`, or
`visual_cache/`. Model-specific loading and generation glue belongs under
`model_adapters/`.

The current implemented adapter is `model_adapters/llava_onevision.py`. The
`model_adapters/qwen3_vl.py` file is intentionally a fail-fast placeholder so
Qwen3-VL support can be added without copying StreamingBench or Video-MME
dataset logic.
