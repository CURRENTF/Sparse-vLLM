# 多模态 Benchmark

本页是多模态 benchmark 命令的稳定入口。不要在仓库文档中保存本地 run history 和旧论文对比说明；报告结果时应引用实际 run artifact。

## 当前入口

| 范围 | 推荐入口 | 说明 |
| --- | --- | --- |
| Video QA | `benchmark/multimodal/video_qa/evaluate.py` | `mvbench`、`longvideobench`、`mlvu` 和 `videomme` 的统一 evaluator。新 video run 优先使用此入口。 |
| Image QA suite | `benchmark/multimodal/image_qa/small_image_bench.py` | ScienceQA-IMG、POPE、MMBench_EN、MME 和 MMMU。 |
| AI2D | `benchmark/multimodal/image_qa/ai2d.py` | LLaVA-OneVision 和受支持的 visual-token pruning 方法。 |
| VQAv2 | `benchmark/multimodal/image_qa/vqav2.py` | LLaVA-OneVision 和受支持的 visual-token pruning 方法。 |

`streamingbench.py`、`videomme.py` 和 `qaego4d.py` 等 dataset-specific script 仍保留，以兼容旧 workflow；除非当前任务需要，否则不要为它们维护独立 runbook。新的 Video-MME run 应使用统一 video evaluator。

## 方法支持

| 入口 | 模型类别 | 支持的方法 |
| --- | --- | --- |
| `video_qa/evaluate.py` | `llava_onevision` | `vanilla`, `divprune`, `divprune_official`, `fastv`, `visionzip`, `fastvid_official_repo`, `pact_official_repo` |
| `video_qa/evaluate.py` | `qwen3_vl` | `vanilla`, `divprune`, `divprune_official`, `fastv`, `fastvid` |
| `image_qa/small_image_bench.py` | `llava_onevision` | 与上面的 LLaVA adapter 相同，但 `fastvid_official_repo` 仅支持 video。 |
| `image_qa/small_image_bench.py` | `qwen3_vl` | 与上面的 Qwen3-VL adapter 相同。 |
| `image_qa/ai2d.py`, `image_qa/vqav2.py` | LLaVA-OneVision | 与 image suite 使用相同 adapter。 |

方法限制：

- `pact_official_repo` 必须在新的 evaluator process 中单独运行；不要在一条命令中与其他方法混合。
- Qwen3-VL 评估要求 Transformers build 提供 `Qwen3VLForConditionalGeneration`；adapter 使用 batch size 1。

## 示例命令

通过统一 evaluator 运行 Video-MME：

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

Image QA suite smoke test：

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

## Artifact 与报告

当前大多数多模态 evaluator 会写入：

- `<method>_raw_outputs.jsonl`
- `<method>_parsed_outputs.jsonl`
- `<method>_per_sample_results.jsonl`
- `<method>_aggregate_metrics.json`
- `run_info.json`
- 特定入口的 `last_*_result.json` summary

报告数值前，应检查 run artifact 中的 aggregate metric 和 per-sample status count。不要混用不兼容的 metric scale：对于 MME，应将 official-style score 与本地 yes/no accuracy percentage 分开。
