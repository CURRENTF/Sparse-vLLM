# LLaVA-OneVision StreamingBench Evaluation

This document tracks the local StreamingBench adapter used for LLaVA-OneVision
vanilla and DeltaKV visual-cache experiments.

## Scope

`scripts/bench_llava_onevision_streamingbench.py` evaluates the
multiple-choice video QA portions of StreamingBench:

- `real`: Real-Time Visual Understanding.
- `omni`: Omni-Source Understanding.
- `contextual`: Contextual Understanding.
- `sqa`: Sequential Question Answering, with the official-style ground-truth
  previous QA context included in the prompt.

The proactive-output timing protocol is not included in this script because it
requires polling a video stream and scoring both trigger time and generated
content. Use this script for accuracy, throughput, and memory comparisons on
the multiple-choice QA tasks.

LiveVLM Table 4 displays the `real` and four `omni` MCQA subitems. Its
`Overall=58.85` is consistent with the older StreamingBench 4000-row scope:
all `real` rows plus the full Omni-Source split, where ACU/MCU now live in the
current dataset's `contextual` CSV. Use `--streamingbench_profile
livevlm_table4` for that paper-aligned scope.

## ReKV on StreamingBench

The original ReKV paper evaluates streaming video QA on RVS-Ego and RVS-Movie,
not on the dataset named StreamingBench.

A later paper, StreamKV, does evaluate ReKV on StreamingBench. Its Table 1
reports `ReKV-7B` at `0.5fps` on StreamingBench with an overall score of
`53.5`, using LLaVA-OneVision-Qwen2-7B as the backbone and the same broad
streaming-KV retrieval setup as ReKV.

References:

- ReKV paper: <https://arxiv.org/abs/2503.00540>
- StreamKV paper: <https://arxiv.org/abs/2511.07278>
- StreamingBench leaderboard: <https://streamingbench.github.io/>
- StreamingBench code: <https://github.com/THUNLP-MT/StreamingBench>

## Dataset Layout

Download the small CSV annotation files:

```bash
source /etc/network_turbo
/home/haojitai/miniconda3/envs/svllm/bin/hf download \
  mjuicem/StreamingBench \
  --repo-type dataset \
  --include 'StreamingBench/*.csv' \
  --local-dir /data2/haojitai/datasets/StreamingBench_hf
```

Download and unzip the video shards needed for the task you want to evaluate.
For example, the first 50 real-time visual understanding videos:

```bash
source /etc/network_turbo
/home/haojitai/miniconda3/envs/svllm/bin/hf download \
  mjuicem/StreamingBench \
  'Real-Time Visual Understanding_1-50.zip' \
  --repo-type dataset \
  --local-dir /data2/haojitai/datasets/StreamingBench_hf

mkdir -p /data2/haojitai/datasets/StreamingBench_hf/videos/real_1_50
unzip -o \
  '/data2/haojitai/datasets/StreamingBench_hf/Real-Time Visual Understanding_1-50.zip' \
  -d /data2/haojitai/datasets/StreamingBench_hf/videos/real_1_50
```

The script indexes videos recursively under `--video_dir`. It parses
`sample_N` from the CSV `question_id` and selects the matching local video file.
Missing videos fail fast by default. If only one shard is downloaded and a
partial-shard run is intentional, pass `--allow_missing_videos` explicitly.

## Methods

`vanilla` loads `LlavaOnevisionForConditionalGeneration`.

`deltakv_delta_quant` loads the LLaVA DeltaKV wrapper with no learned
compressor checkpoint:

- `--deltakv_checkpoint_path none`
- `--delta_quant_bits 4`
- `--deltakv_center_ratio`
- `--deltakv_neighbor_count`
- `--recent_keep_tokens`, `--sink_keep_tokens`, `--decode_keep_tokens`,
  `--prefill_keep_tokens`

The method uses cluster/ref reconstruction and direct token-space residual int4
quantization.

## Full-Attention Baseline Protocol

LiveVLM Table 4 reports LLaVA-OneVision-7B with `32` frames on the Real-Time
Visual Understanding and Omni-Source Understanding MCQA tasks. The paper's
LLaVA-OneVision-7B row is:

```text
OP 80.38 | CR 74.22 | CS 76.03 | ATP 80.72 | EU 72.67 | TR 71.65 |
PR 67.59 | SU 65.45 | ACP 65.72 | CT 45.08 | ER 40.80 | SCU 37.20 |
SD 33.60 | MA 44.80 | Overall 58.85
```

For this repo, the LiveVLM Table 4 dense/full-attention baseline is:

```bash
--methods vanilla
--streamingbench_profile livevlm_table4
--tasks livevlm_table4
--frame_sampling_backend decord
--torch_dtype float16
--attn_implementation flash_attention_2
--choice_parse_mode official_first_char
```

`livevlm_table4` forces:

```text
tasks = real,omni,contextual
num_video_frames = 32
context_seconds = -1
frame_sampling_backend = decord
```

The script reports `livevlm_table4_stats` with the 14 subitems and the expected
LLaVA-OneVision-7B Table 4 values for direct comparison. The default
`--choice_parse_mode official_first_char` matches StreamingBench's official
multiple-choice counter by reading the first non-whitespace generated character;
non-`A/B/C/D` predictions are marked `parse_failed` and counted as incorrect.
The `overall_extra_subtasks` field records the ACU/MCU rows that are not printed
as Table 4 subitems but are included in the paper's `Overall` denominator.
Each subtask record also stores `expected_rows` and `matches_expected_rows`.
For full `livevlm_table4` runs, the loader fails unless the row counts are:

```text
OP 369 | CR 128 | CS 317 | ATP 312 | EU 159 | TR 321 | PR 108 |
SU 246 | ACP 352 | CT 188 | ER 250 | SCU 250 | SD 250 | MA 250 |
ACU 250 | MCU 250 | Overall 4000
```

The StreamingBench leaderboard also reports LLaVA-OneVision-7B with `32`
frames. The main leaderboard setting uses 60 seconds of video context before
the query.

For this repo, the aligned dense/full-attention baseline is:

```bash
--methods vanilla
--streamingbench_profile official_60s
--frame_sampling_backend decord
--torch_dtype float16
--attn_implementation flash_attention_2
```

`official_60s` forces:

```text
num_video_frames = 32
context_seconds = 60
```

The script samples frames with decord uniform frame indices to match the
official LLaVA-OneVision adapter. `vanilla` then runs standard dense attention
over those 32 selected frames plus the text prompt.

For the official all-context variant, use:

```bash
--streamingbench_profile official_all_context
```

This forces `num_video_frames=32` and `context_seconds=-1`, meaning the clip
starts at video time 0 and ends at the query timestamp.

## Example Commands

Run a small real-task comparison on GPU 7:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
  scripts/bench_llava_onevision_streamingbench.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-0.5b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/StreamingBench_hf \
  --video_dir /data2/haojitai/datasets/StreamingBench_hf/videos \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_real_smoke \
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

When not using `CUDA_VISIBLE_DEVICES`, pass the physical GPU id directly, for
example `--cuda_device 7`.

The script writes:

- `last_streamingbench_result.json`: method summaries and per-question records.
- `<method>_raw_outputs.jsonl`: raw model generations.
- `<method>_parsed_outputs.jsonl`: parsed answers, explicit status, and labels.
- `<method>_per_sample_results.jsonl`: per-sample records.
- `<method>_aggregate_metrics.json`: aggregate metrics, including Table 4
  subitem stats when applicable.
- `run_info.json`: command, config, model, dataset, prompt/decoding settings,
  seed, and sample count.
- `frame_cache/`: extracted frames keyed by video path, time window, and frame
  count.

`--frame_load_workers N` loads cached frame images for each batch with a thread
pool. `--preprocess_prefetch_batches 1` additionally prepares the next batch on
one background thread while the current batch is generating. The prefetch path
keeps processor calls ordered and still fails fast if frame loading or processor
conversion fails.

When exactly two methods are run and one is `vanilla`, the candidate
`<method>_aggregate_metrics.json` includes accuracy, speed, and memory deltas
against the vanilla run.

LiveVLM Table 4 baseline command on physical GPU 6 after all video shards have
been downloaded:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
  scripts/bench_llava_onevision_streamingbench.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/StreamingBench_hf \
  --video_dir /data2/haojitai/datasets/StreamingBench_hf/videos \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla \
  --methods vanilla \
  --num_samples -1 \
  --batch_size 8 \
  --streamingbench_profile livevlm_table4 \
  --torch_dtype float16 \
  --attn_implementation flash_attention_2 \
  --max_new_tokens 8 \
  --choice_parse_mode official_first_char \
  --cuda_device 0 \
  --seed 0 \
  --log_every 200 \
  --reuse_frame_cache \
  --frame_cache_dir /data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla/frame_cache
```

For the background runner, `LIVEVLM_TABLE4_GPU_IDS` can list allowed physical
GPUs. It defaults to `6,7` in this repo so the runner still obeys the current
card restriction while using whichever allowed GPU becomes free first:

```bash
LIVEVLM_TABLE4_GPU_IDS=6,7 \
bash scripts/nohup_run_livevlm_table4_vanilla_after_download.sh
```

Audit the completed baseline with:

```bash
/home/haojitai/miniconda3/envs/svllm/bin/python \
  scripts/audit_livevlm_table4_result.py \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla
```

The audit fails fast if the metrics file is missing, if the run is not the
4000-row Table 4 scope, if any visible/overall-only subtask is missing, or if
any subtask row count differs from the expected counts above. It also checks
that `run_info.json`, `last_streamingbench_result.json`,
`vanilla_raw_outputs.jsonl`, `vanilla_parsed_outputs.jsonl`, and
`vanilla_per_sample_results.jsonl` exist, contain 4000 rows where applicable,
and record the baseline settings: 7B model path, `vanilla`, `livevlm_table4`,
32 frames, all prior context, decord sampling, `float16`,
`flash_attention_2`, greedy 8-token decoding, `official_first_char`, and seed
0. To enforce a numeric
tolerance against the paper's `Overall=58.85`, add for example
`--require_overall_delta_within_pct 1.0`.

## Local Results

### 7B, Full 4000-Row LiveVLM/StreamingBench Baselines

These are the full local LLaVA-OneVision-7B dense/full-attention runs used to
check alignment against LiveVLM Table 4 and the StreamingBench 60-second main
setting.

Common settings:

```text
model_path = /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf
dataset_dir = /data2/haojitai/datasets/StreamingBench_hf
video_dir = /data2/haojitai/datasets/StreamingBench_hf/videos
methods = vanilla
tasks = livevlm_table4
num_video_frames = 32
batch_size = 8
torch_dtype = float16
attn_implementation = flash_attention_2
choice_parse_mode = official_first_char
max_new_tokens = 8
seed = 0
```

Result paths:

```text
/data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla
/data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla_ctx60
```

| Run | Profile | Context | Samples | Success | Correct | Accuracy | Delta vs LiveVLM 58.85 | New tok/s | Examples/s | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all prior context | `livevlm_table4` | `-1` | 4000 | 4000 | 2284 | `57.10%` | `-1.75` | `6.580` | `3.244` | `29.77 GB` |
| official 60s | `official_60s` | `60s` | 4000 | 4000 | 2406 | `60.15%` | `+1.30` | `6.567` | `3.238` | `29.77 GB` |

The official 60s run is the main comparison point for StreamingBench-style
evaluation. It is aligned with LiveVLM's LLaVA-OneVision-7B `Overall=58.85`
within `+1.30` points. The all-context run is lower because 32 frames are spread
over the full prior video history instead of being concentrated in the
60-second window before the query.

Official 60s subtask results:

| Subtask | Correct / Total | Accuracy |
| --- | ---: | ---: |
| OP | 304 / 369 | `82.38%` |
| CR | 100 / 128 | `78.12%` |
| CS | 262 / 317 | `82.65%` |
| ATP | 263 / 312 | `84.29%` |
| EU | 112 / 159 | `70.44%` |
| TR | 244 / 321 | `76.01%` |
| PR | 78 / 108 | `72.22%` |
| SU | 157 / 246 | `63.82%` |
| ACP | 241 / 352 | `68.47%` |
| CT | 77 / 188 | `40.96%` |
| ER | 102 / 250 | `40.80%` |
| SCU | 64 / 250 | `25.60%` |
| SD | 103 / 250 | `41.20%` |
| MA | 138 / 250 | `55.20%` |
| ACU | 82 / 250 | `32.80%` |
| MCU | 79 / 250 | `31.60%` |

Audit artifact:

```text
/data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla_ctx60/livevlm_table4_audit.json
```

The audit checks the 4000-row scope, all expected subtask row counts, raw output
count, parsed output count, per-sample result count, `official_60s`, 32 frames,
`flash_attention_2`, greedy 8-token decoding, and `official_first_char` parsing.
Two rows for `sample_332` used an explicitly recorded frame-cache fallback
because the local video file is corrupt after roughly 250 seconds; this can
affect at most `0.05` percentage points.

Small 0.5B prefetch correctness/speed smoke on the first 64 official-60s rows,
GPU 7, batch size 8, `frame_load_workers=4`, and the same cached frames:

| Mode | Samples | Status | Accuracy | E2E seconds | E2E examples/s | Output diff |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `preprocess_prefetch_batches=0` | 64 | `success: 64` | `68.75%` | `38.477` | `1.663` | reference |
| `preprocess_prefetch_batches=1` | 64 | `success: 64` | `68.75%` | `31.817` | `2.011` | 0 mismatches |

The per-sample comparison checks question id, prediction, answer, correctness,
status, raw prediction, input-token count, padded length, and video-token count.
The prefetch run gives `1.21x` higher end-to-end examples/s on this small smoke.

### 7B, Full 4000-Row DeltaKV KR/CR Sweep

This sweep uses the same full 4000-row `official_60s` StreamingBench scope as
the dense baseline above:

```text
model_path = /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf
dataset_dir = /data2/haojitai/datasets/StreamingBench_hf
video_dir = /data2/haojitai/datasets/StreamingBench_hf/videos
methods = deltakv_delta_quant
deltakv_checkpoint_path = none
tasks = livevlm_table4
streamingbench_profile = official_60s
num_video_frames = 32
frame_sampling_backend = decord
batch_size = 8
torch_dtype = float16
attn_implementation = flash_attention_2
choice_parse_mode = official_first_char
max_new_tokens = 8
seed = 0
frame_cache_dir = /data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla_ctx60/frame_cache
```

The commands were launched on physical GPUs 6 and 7 with
`CUDA_VISIBLE_DEVICES=6` or `CUDA_VISIBLE_DEVICES=7`; `run_info.json` records
`cuda_device=0` after that remapping. All runs reused the same official 60s
decord frame cache rather than re-extracting frames.

For the no-compressor `deltakv_delta_quant` path, KR is estimated with the same
formula used in the QAEGO4D note:

```text
KR = Lfull / L + Lsparse / L * (deltakv_center_ratio + 0.25)
```

where `0.25` is direct int4 residual storage relative to fp16 KV. Approximate CR
below uses the measured full-run mean input length `6445.82625` tokens and
counts `decode_keep_tokens + sink_keep_tokens + recent_keep_tokens` for sparse
layers.

| Config | Full layers | Center ratio | Decode keep | KR | Approx CR | Correct | Accuracy | New tok/s | Examples/s | E2E examples/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kr30_cr1024_full` | `0` | `0.03` | 1024 | `30.57%` | `20.92%` | 2405 / 4000 | `60.125%` | `4.314` | `2.121` | `1.168` |
| `kr30_cr2048_full` | `0` | `0.03` | 2048 | `30.57%` | `36.24%` | 2405 / 4000 | `60.125%` | `4.292` | `2.123` | `1.146` |
| `kr33_center005_full` | `0` | `0.05` | 1024 | `32.50%` | `20.92%` | 2405 / 4000 | `60.125%` | `4.313` | `2.120` | `1.151` |
| `kr33_full01_full` | `0,1` | `0.03` | 1024 | `33.14%` | `23.85%` | 2405 / 4000 | `60.125%` | `4.122` | `2.035` | `1.130` |

The best accuracy among these full runs is tied across all four configurations.
Use the lowest-budget tied configuration by default:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
  scripts/bench_llava_onevision_streamingbench.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/StreamingBench_hf \
  --video_dir /data2/haojitai/datasets/StreamingBench_hf/videos \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_deltakv_7b_official60_kr30_cr1024_full \
  --methods deltakv_delta_quant \
  --deltakv_checkpoint_path none \
  --num_samples -1 \
  --batch_size 8 \
  --tasks livevlm_table4 \
  --streamingbench_profile official_60s \
  --frame_sampling_backend decord \
  --torch_dtype float16 \
  --attn_implementation flash_attention_2 \
  --max_new_tokens 8 \
  --choice_parse_mode official_first_char \
  --cuda_device 0 \
  --seed 0 \
  --log_every 400 \
  --reuse_frame_cache \
  --frame_cache_dir /data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla_ctx60/frame_cache \
  --frame_load_workers 4 \
  --full_attention_layers 0 \
  --deltakv_center_ratio 0.03 \
  --deltakv_neighbor_count 1 \
  --decode_keep_tokens 1024 \
  --prefill_keep_tokens 4096
```

Result paths:

```text
/data2/haojitai/datasets/llava_onevision_streamingbench_deltakv_7b_official60_kr30_cr1024_full
/data2/haojitai/datasets/llava_onevision_streamingbench_deltakv_7b_official60_kr30_cr2048_full
/data2/haojitai/datasets/llava_onevision_streamingbench_deltakv_7b_official60_kr33_center005_full
/data2/haojitai/datasets/llava_onevision_streamingbench_deltakv_7b_official60_kr33_full01_full
```

For each full run, `raw_outputs.jsonl`, `parsed_outputs.jsonl`, and
`per_sample_results.jsonl` contain 4000 rows, and `status_counts` is
`{"success": 4000}`. Against the dense official 60s baseline, the recommended
DeltaKV configuration keeps the same row order and has 3994 identical
predictions, 1 `false -> true` transition, and 2 `true -> false` transitions.
The net difference is therefore `-1 / 4000`, or `-0.025` accuracy points:

```text
vanilla official_60s: 2406 / 4000 = 60.150%
DeltaKV recommended: 2405 / 4000 = 60.125%
```

Task-level comparison for the recommended DeltaKV run:

| Task group | Vanilla | DeltaKV | Delta |
| --- | ---: | ---: | ---: |
| real | `73.52%` | `73.48%` | `-0.04` |
| omni | `40.70%` | `40.70%` | `+0.00` |
| contextual | `32.20%` | `32.20%` | `+0.00` |

Complete subtask matrix read from the saved `*_aggregate_metrics.json` files:

| Subtask | Rows | Paper LLaVA-OV-7B | Vanilla official_60s | DeltaKV kr30 cr1024 | DeltaKV kr30 cr2048 | DeltaKV kr33 center005 | DeltaKV kr33 full01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OP | 369 | `80.38%` | 304/369 `82.38%` | 304/369 `82.38%` | 304/369 `82.38%` | 304/369 `82.38%` | 304/369 `82.38%` |
| CR | 128 | `74.22%` | 100/128 `78.12%` | 100/128 `78.12%` | 100/128 `78.12%` | 100/128 `78.12%` | 100/128 `78.12%` |
| CS | 317 | `76.03%` | 262/317 `82.65%` | 262/317 `82.65%` | 262/317 `82.65%` | 262/317 `82.65%` | 262/317 `82.65%` |
| ATP | 312 | `80.72%` | 263/312 `84.29%` | 262/312 `83.97%` | 262/312 `83.97%` | 262/312 `83.97%` | 262/312 `83.97%` |
| EU | 159 | `72.67%` | 112/159 `70.44%` | 112/159 `70.44%` | 112/159 `70.44%` | 112/159 `70.44%` | 112/159 `70.44%` |
| TR | 321 | `71.65%` | 244/321 `76.01%` | 245/321 `76.32%` | 245/321 `76.32%` | 245/321 `76.32%` | 245/321 `76.32%` |
| PR | 108 | `67.59%` | 78/108 `72.22%` | 78/108 `72.22%` | 78/108 `72.22%` | 78/108 `72.22%` | 78/108 `72.22%` |
| SU | 246 | `65.45%` | 157/246 `63.82%` | 157/246 `63.82%` | 157/246 `63.82%` | 157/246 `63.82%` | 157/246 `63.82%` |
| ACP | 352 | `65.72%` | 241/352 `68.47%` | 240/352 `68.18%` | 240/352 `68.18%` | 240/352 `68.18%` | 240/352 `68.18%` |
| CT | 188 | `45.08%` | 77/188 `40.96%` | 77/188 `40.96%` | 77/188 `40.96%` | 77/188 `40.96%` | 77/188 `40.96%` |
| ER | 250 | `40.80%` | 102/250 `40.80%` | 102/250 `40.80%` | 102/250 `40.80%` | 102/250 `40.80%` | 102/250 `40.80%` |
| SCU | 250 | `37.20%` | 64/250 `25.60%` | 64/250 `25.60%` | 64/250 `25.60%` | 64/250 `25.60%` | 64/250 `25.60%` |
| SD | 250 | `33.60%` | 103/250 `41.20%` | 103/250 `41.20%` | 103/250 `41.20%` | 103/250 `41.20%` | 103/250 `41.20%` |
| MA | 250 | `44.80%` | 138/250 `55.20%` | 138/250 `55.20%` | 138/250 `55.20%` | 138/250 `55.20%` | 138/250 `55.20%` |
| ACU | 250 | n/a | 82/250 `32.80%` | 82/250 `32.80%` | 82/250 `32.80%` | 82/250 `32.80%` | 82/250 `32.80%` |
| MCU | 250 | n/a | 79/250 `31.60%` | 79/250 `31.60%` | 79/250 `31.60%` | 79/250 `31.60%` | 79/250 `31.60%` |
| Overall | 4000 | `58.85%` | 2406/4000 `60.15%` | 2405/4000 `60.12%` | 2405/4000 `60.12%` | 2405/4000 `60.12%` | 2405/4000 `60.12%` |

`ACU` and `MCU` are included in the 4000-row overall denominator but are not
printed as visible subtasks in LiveVLM Table 4. The four full DeltaKV
configurations in the KR/CR sweep produced the same correct counts for every
subtask.

### 7B, Official 60s/32-Frame Real-Time Visual Understanding, sample 201-250 shard

Local data currently contains only this shard:

```text
/data2/haojitai/datasets/StreamingBench_hf/videos/real_201_250
```

That is 50 videos and 250 QA rows. The full StreamingBench dataset is larger,
so this is a shard-level check, not a complete leaderboard reproduction.

Vanilla full-attention command on physical GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
  scripts/bench_llava_onevision_streamingbench.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/StreamingBench_hf \
  --video_dir /data2/haojitai/datasets/StreamingBench_hf/videos \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_real_7b_official60_fullattn32_vanilla \
  --tasks real \
  --methods vanilla \
  --num_samples -1 \
  --batch_size 1 \
  --streamingbench_profile official_60s \
  --frame_sampling_backend decord \
  --allow_missing_videos \
  --torch_dtype float16 \
  --attn_implementation sdpa \
  --max_new_tokens 8 \
  --cuda_device 0 \
  --log_every 25
```

DeltaKV KR30 direct delta-quant command on physical GPU 7:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
  scripts/bench_llava_onevision_streamingbench.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/StreamingBench_hf \
  --video_dir /data2/haojitai/datasets/StreamingBench_hf/videos \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_real_7b_official60_kr30_delta_quant \
  --tasks real \
  --methods deltakv_delta_quant \
  --deltakv_checkpoint_path none \
  --num_samples -1 \
  --batch_size 1 \
  --streamingbench_profile official_60s \
  --frame_sampling_backend decord \
  --allow_missing_videos \
  --torch_dtype float16 \
  --attn_implementation sdpa \
  --max_new_tokens 8 \
  --cuda_device 0 \
  --full_attention_layers 0 \
  --deltakv_center_ratio 0.03 \
  --decode_keep_tokens 1024 \
  --prefill_keep_tokens 4096 \
  --delta_quant_bits 4 \
  --deltakv_neighbor_count 1 \
  --log_every 25
```

Result files:

```text
/data2/haojitai/datasets/llava_onevision_streamingbench_real_7b_official60_fullattn32_vanilla/last_streamingbench_result.json
/data2/haojitai/datasets/llava_onevision_streamingbench_real_7b_official60_kr30_delta_quant/last_streamingbench_result.json
```

| Method | Frames | Context | Samples | Accuracy | New tok/s | Examples/s | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vanilla` | 32 | 60s | 250 | `0.7120` | `5.880` | `2.940` | `16.856 GB` |
| `llava_deltakv_delta_quant` KR30 | 32 | 60s | 250 | `0.7120` | `3.625` | `1.813` | `16.856 GB` |

The two output JSON files have the same 250 question ids in the same order, and
the predictions are identical for all 250 rows.

### 7B, Older 8-Frame Real-Time Visual Understanding, sample 201-250 shard

Command:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
  scripts/bench_llava_onevision_streamingbench.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/StreamingBench_hf \
  --video_dir /data2/haojitai/datasets/StreamingBench_hf/videos \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_real_7b_shard201_250 \
  --tasks real \
  --methods vanilla,deltakv_delta_quant \
  --deltakv_checkpoint_path none \
  --num_samples -1 \
  --batch_size 1 \
  --num_video_frames 8 \
  --context_seconds 60 \
  --allow_missing_videos \
  --max_new_tokens 8 \
  --cuda_device 0 \
  --reuse_frame_cache
```

Result file:

```text
/data2/haojitai/datasets/llava_onevision_streamingbench_real_7b_shard201_250/last_streamingbench_result.json
```

| Method | Accuracy | New tok/s | Examples/s | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| `vanilla` | `0.6840` | `19.90` | `9.95` | `15.45 GB` |
| `llava_deltakv_delta_quant` | `0.6800` | `10.76` | `5.38` | `15.46 GB` |

DeltaKV quant is `-0.0040` accuracy versus vanilla on this 250-question shard
and runs at `0.540x` vanilla generation throughput. The short 8-frame video
prompts do not show a memory reduction because model weights dominate the peak
memory at this sequence length.

An earlier 32-question prefix run is saved at:

```text
/data2/haojitai/datasets/llava_onevision_streamingbench_real_7b_n32/last_streamingbench_result.json
```

### 0.5B Smoke

The 0.5B smoke run used the same task/shard with `--num_samples 8`:

| Method | Accuracy | New tok/s | Examples/s | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| `vanilla` | `0.3750` | `19.62` | `8.26` | `2.16 GB` |
| `llava_deltakv_delta_quant` | `0.5000` | `12.09` | `4.03` | `2.16 GB` |

Result file:

```text
/data2/haojitai/datasets/llava_onevision_streamingbench_real_05b_smoke/last_streamingbench_result.json
```
