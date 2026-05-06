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
--attn_implementation sdpa
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

The StreamingBench leaderboard also reports LLaVA-OneVision-7B with `32`
frames. The main leaderboard setting uses 60 seconds of video context before
the query.

For this repo, the aligned dense/full-attention baseline is:

```bash
--methods vanilla
--streamingbench_profile official_60s
--frame_sampling_backend decord
--torch_dtype float16
--attn_implementation sdpa
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
  --batch_size 1 \
  --streamingbench_profile livevlm_table4 \
  --torch_dtype float16 \
  --attn_implementation sdpa \
  --max_new_tokens 8 \
  --choice_parse_mode official_first_char \
  --cuda_device 0 \
  --seed 0 \
  --log_every 50
```

Audit the completed baseline with:

```bash
/home/haojitai/miniconda3/envs/svllm/bin/python \
  scripts/audit_livevlm_table4_result.py \
  --output_dir /data2/haojitai/datasets/llava_onevision_streamingbench_livevlm_table4_7b_vanilla
```

The audit fails fast if the metrics file is missing, if the run is not the
4000-row Table 4 scope, or if the visible/overall-only subtask structure is
incomplete. To enforce a numeric tolerance against the paper's `Overall=58.85`,
add for example `--require_overall_delta_within_pct 1.0`.

## Local Results

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
