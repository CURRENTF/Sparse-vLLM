# LLaVA-OneVision ReKV-Style QA-Ego4D Evaluation

This page is the stable runbook for
`benchmark/multimodal/video_qa/qaego4d.py`. Historical full-run commands,
result tables, and KR candidate notes are archived in
[`dev_docs/benchmark-run-history/multimodal/llava-onevision-rekv-qaego4d.md`](../../../dev_docs/benchmark-run-history/multimodal/llava-onevision-rekv-qaego4d.md).

## Scope

The adapter evaluates LLaVA-OneVision `vanilla` and no-checkpoint
`deltakv_delta_quant` on `QAEGO4Dtest-mc` using the ReKV paper's multiple-choice
protocol:

- metric: multiple-choice `qa_acc`;
- video sampling: `sample_fps=0.5`;
- context budget: 64 video context frames;
- prompt: ReKV MC prompt ending with `Best option: (`;
- optional evaluator:
  `<REKV_ROOT>/video_qa/eval/eval_multiple_choice.py`.

This is not a reproduction of ReKV retrieval. It uses the ReKV dataset, prompt,
frame budget, and evaluator to compare this repo's LLaVA-OneVision paths.

## Data

Expected local layout:

```text
<DATA_ROOT>/rekv_qaego4d/test_mc.json
<DATA_ROOT>/rekv_qaego4d/videos.zip
<DATA_ROOT>/rekv_qaego4d/videos/*.mp4
```

For repeated full runs, reuse a frame cache:

```text
<DATA_ROOT>/rekv_qaego4d_frame_cache_fps05_64
```

## Command Template

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=$PWD/src \
<SVLLM_PYTHON> -u benchmark/multimodal/video_qa/qaego4d.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir <DATA_ROOT>/rekv_qaego4d \
  --output_dir <OUTPUT_ROOT>/llava_onevision_rekv_qaego4d_7b \
  --methods vanilla,deltakv_delta_quant \
  --num_samples -1 \
  --batch_size 1 \
  --sample_fps 0.5 \
  --max_context_frames 64 \
  --cuda_device 0 \
  --reuse_frame_cache \
  --frame_cache_dir <DATA_ROOT>/rekv_qaego4d_frame_cache_fps05_64 \
  --log_every 25
```

For the no-checkpoint DeltaKV delta-quant path, keep
`--deltakv_checkpoint_path none` and set the keep budgets, full-attention
layers, `--deltakv_center_ratio`, and `--delta_quant_bits` explicitly when
departing from script defaults.

## Official Evaluator

```bash
<SVLLM_PYTHON> <REKV_ROOT>/video_qa/eval/eval_multiple_choice.py \
  --results_path <OUTPUT_ROOT>/llava_onevision_rekv_qaego4d_7b/vanilla_results.csv

<SVLLM_PYTHON> <REKV_ROOT>/video_qa/eval/eval_multiple_choice.py \
  --results_path <OUTPUT_ROOT>/llava_onevision_rekv_qaego4d_7b/llava_deltakv_delta_quant_results.csv
```

Record evaluator errors separately from model accuracy. Do not report a CSV
metric unless the evaluator completed and the benchmark artifacts are preserved.
