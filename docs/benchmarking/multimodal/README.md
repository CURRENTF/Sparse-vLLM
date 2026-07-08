# Multimodal Benchmarks

This page is the stable entry point for multimodal benchmark commands. Keep
local run histories and old paper-comparison notes out of this repo doc; cite
the actual run artifacts when reporting a result.

## Current Entry Points

| Scope | Preferred entry point | Notes |
| --- | --- | --- |
| Video QA | `benchmark/multimodal/video_qa/evaluate.py` | Unified evaluator for `mvbench`, `longvideobench`, `mlvu`, and `videomme`. Prefer this for new video runs. |
| Image QA suite | `benchmark/multimodal/image_qa/small_image_bench.py` | ScienceQA-IMG, POPE, MMBench_EN, MME, and MMMU. |
| AI2D | `benchmark/multimodal/image_qa/ai2d.py` | LLaVA-OneVision `vanilla`/`deltakv` only. |
| VQAv2 | `benchmark/multimodal/image_qa/vqav2.py` | LLaVA-OneVision `vanilla`/`deltakv` only. |
| Visual-cache ablation | `benchmark/multimodal/visual_cache/run_visual_cache.py` | LLaVA-OneVision visual-token uniform-pruning baseline and DeltaKV comparison. |

Dataset-specific scripts such as `streamingbench.py`, `videomme.py`, and
`qaego4d.py` still exist for compatibility with older workflows, but do not
maintain separate runbooks for them unless a current task requires one. For new
Video-MME runs, use the unified video evaluator.

## Method Support

| Entry point | Model family | Supported methods |
| --- | --- | --- |
| `video_qa/evaluate.py` | `llava_onevision` | `vanilla`, `deltakv`, `snapkv`, `omnikv`, `divprune`, `divprune_official`, `fastv`, `visionzip`, `fastvid_official_repo`, `pact_official_repo` |
| `video_qa/evaluate.py` | `qwen3_vl` | `vanilla`, `deltakv`, `divprune`, `divprune_official`, `fastv`, `fastvid` |
| `image_qa/small_image_bench.py` | `llava_onevision` | Same LLaVA adapter as above, except `fastvid_official_repo` is video-only. |
| `image_qa/small_image_bench.py` | `qwen3_vl` | Same Qwen3-VL adapter as above. |
| `image_qa/ai2d.py`, `image_qa/vqav2.py` | LLaVA-OneVision | `vanilla`, `deltakv` |
| `visual_cache/run_visual_cache.py` | LLaVA-OneVision | `vanilla`, `deltakv`, `visual_uniform_keep` |

Method constraints:

- `deltakv` requires a real `--deltakv_checkpoint_path` trained for the same
  base model.
- `visual_uniform_keep` requires `--deltakv_checkpoint_path none`; it is a
  uniform visual-token pruning baseline, not DeltaKV compressor inference.
- `snapkv`, `omnikv`, and `visual_uniform_keep` do not use learned compressor
  checkpoints.
- `pact_official_repo` must run alone in a fresh evaluator process; do not mix
  it with HF methods in one command.
- Qwen3-VL evaluation requires a Transformers build that provides
  `Qwen3VLForConditionalGeneration`; the adapter runs with batch size 1.

## Example Commands

Video-MME through the unified evaluator:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=$PWD:$PWD/src \
python benchmark/multimodal/video_qa/evaluate.py \
  --benchmark videomme \
  --model_family llava_onevision \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir <DATA_ROOT>/Video-MME_modelscope \
  --output_dir <OUTPUT_ROOT>/videomme_llava_vanilla_smoke \
  --methods vanilla \
  --num_samples 8 \
  --batch_size 1 \
  --num_video_frames 32 \
  --frame_sampling_backend decord
```

Image QA suite smoke:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=$PWD:$PWD/src \
python benchmark/multimodal/image_qa/small_image_bench.py \
  --benchmark scienceqa_img \
  --model_family llava_onevision \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir <DATA_ROOT>/ScienceQA \
  --output_dir <OUTPUT_ROOT>/scienceqa_llava_vanilla_smoke \
  --methods vanilla \
  --num_samples 16 \
  --batch_size 1
```

Visual uniform baseline:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=$PWD:$PWD/src \
python benchmark/multimodal/visual_cache/run_visual_cache.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-7b-ov-hf \
  --deltakv_checkpoint_path none \
  --source_vqa_dir <DATA_ROOT>/VQAv2 \
  --output_dir <OUTPUT_ROOT>/vqav2_visual_uniform_keep_smoke \
  --methods vanilla,visual_uniform_keep \
  --visual_keep_ratio 0.1 \
  --num_samples 16 \
  --batch_size 1
```

## Artifacts And Reporting

Most current multimodal evaluators write:

- `<method>_raw_outputs.jsonl`
- `<method>_parsed_outputs.jsonl`
- `<method>_per_sample_results.jsonl`
- `<method>_aggregate_metrics.json`
- `run_info.json`
- a `last_*_result.json` summary for the specific entry point

Before reporting a number, verify the aggregate metric and the per-sample
status counts from the run artifact. Do not mix incompatible metric scales:
for MME, keep official-style score separate from local yes/no accuracy percent.
