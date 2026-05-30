# LLaVA-OneVision StreamingBench Evaluation

This page is the stable runbook for
`benchmark/multimodal/video_qa/streamingbench.py`. Historical local commands,
result tables, and one-off sweeps are archived in
[`dev_docs/benchmark-run-history/multimodal/llava-onevision-streamingbench.md`](../../../dev_docs/benchmark-run-history/multimodal/llava-onevision-streamingbench.md).

## Scope

The adapter evaluates the multiple-choice QA portions of StreamingBench:

- `real`: Real-Time Visual Understanding.
- `omni`: Omni-Source Understanding.
- `contextual`: Contextual Understanding.
- `sqa`: Sequential Question Answering with ground-truth previous QA context.

It does not implement the proactive-output timing protocol because that
requires stream polling and trigger-time scoring. Use this script for accuracy,
throughput, and memory comparisons on MCQA tasks.

## Profiles

| Profile | Purpose | Forced settings |
| --- | --- | --- |
| `official_60s` | StreamingBench leaderboard-style dense baseline. | 32 frames, 60-second context, decord sampling. |
| `official_all_context` | Full available context before the query timestamp. | 32 frames, all context, decord sampling. |
| `livevlm_table4` | LiveVLM Table 4 alignment. | `real,omni,contextual`, 32 frames, all context. |

For paper-aligned dense baselines, use `--methods vanilla`,
`--torch_dtype float16`, `--attn_implementation flash_attention_2`, and
`--choice_parse_mode official_first_char`.

## Data

Download the CSV annotations:

```bash
<HF_BIN> download \
  mjuicem/StreamingBench \
  --repo-type dataset \
  --include 'StreamingBench/*.csv' \
  --local-dir <DATA_ROOT>/StreamingBench_hf
```

Download only the video shards needed for the run. The script indexes video
files recursively under `--video_dir`, parses `sample_N` from `question_id`,
and fails fast on missing videos unless `--allow_missing_videos` is explicitly
set for an intentional partial-shard smoke test.

## Methods

| Method | Behavior |
| --- | --- |
| `vanilla` | Standard `LlavaOnevisionForConditionalGeneration`. |
| `deltakv_delta_quant` | No-checkpoint DeltaKV-style cluster/ref path with direct token-space residual int4 quantization. |

For `deltakv_delta_quant`, use `--deltakv_checkpoint_path none` and set the
DeltaKV knobs explicitly: `--delta_quant_bits`, `--deltakv_center_ratio`,
`--deltakv_neighbor_count`, `--recent_keep_tokens`, `--sink_keep_tokens`,
`--decode_keep_tokens`, and `--prefill_keep_tokens`.

## Smoke Command

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=$PWD/src \
<SVLLM_PYTHON> -u benchmark/multimodal/video_qa/streamingbench.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-0.5b-ov-hf \
  --dataset_dir <DATA_ROOT>/StreamingBench_hf \
  --video_dir <DATA_ROOT>/StreamingBench_hf/videos \
  --output_dir <OUTPUT_ROOT>/llava_onevision_streamingbench_real_smoke \
  --tasks real \
  --methods vanilla,deltakv_delta_quant \
  --deltakv_checkpoint_path none \
  --num_samples 16 \
  --batch_size 1 \
  --streamingbench_profile official_60s \
  --frame_sampling_backend decord \
  --allow_missing_videos \
  --cuda_device 0
```

Omit `--allow_missing_videos` for full runs so missing media fails fast.

## Outputs

The script writes:

- `last_streamingbench_result.json`
- `<method>_raw_outputs.jsonl`
- `<method>_parsed_outputs.jsonl`
- `<method>_per_sample_results.jsonl`
- `<method>_aggregate_metrics.json`
- `run_info.json`
- `frame_cache/`

For reported results, keep the raw, parsed, per-sample, aggregate, and run-info
files together with the exact command.
