# LLaVA-OneVision ReKV-Style QAEGO4D Evaluation

This benchmark evaluates LLaVA-OneVision vanilla and DeltaKV delta-only quantization on
QAEGO4D-test-mc under the ReKV paper's multiple-choice evaluation protocol.

It aligns these ReKV evaluation details:

- Dataset: `QAEGO4Dtest-mc`
- Metric: multiple-choice `qa_acc`
- Video sampling: `sample_fps=0.5`
- Context budget: 64 video context frames, matching ReKV `retrieve_size=64`
- Prompt: ReKV MC prompt ending with `Best option: (`
- Evaluator: `/tmp/ReKV/video_qa/eval/eval_multiple_choice.py`

This is not a reproduction of ReKV retrieval itself. It evaluates our vanilla and
DeltaKV paths using the ReKV dataset, prompt, frame budget, and official CSV metric.

## Data

Downloaded under `/data2/haojitai/datasets`:

```bash
/data2/haojitai/datasets/rekv_qaego4d/test_mc.json
/data2/haojitai/datasets/rekv_qaego4d/videos.zip
/data2/haojitai/datasets/rekv_qaego4d/videos/*.mp4
```

The archive contains 148 videos and the annotation contains 500 QA pairs.

Frame cache used by the full run:

```bash
/data2/haojitai/datasets/rekv_qaego4d_frame_cache_fps05_64
```

## Command

The full 7B run used physical GPU 7 only:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
scripts/bench_llava_onevision_rekv_qaego4d.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/rekv_qaego4d \
  --output_dir /data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_full \
  --methods vanilla,deltakv_delta_quant \
  --num_samples -1 \
  --batch_size 1 \
  --sample_fps 0.5 \
  --max_context_frames 64 \
  --cuda_device 0 \
  --reuse_frame_cache \
  --frame_cache_dir /data2/haojitai/datasets/rekv_qaego4d_frame_cache_fps05_64 \
  --log_every 25
```

Official ReKV evaluator checks:

```bash
/home/haojitai/miniconda3/envs/svllm/bin/python \
/tmp/ReKV/video_qa/eval/eval_multiple_choice.py \
  --results_path /data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_full/vanilla_results.csv

/home/haojitai/miniconda3/envs/svllm/bin/python \
/tmp/ReKV/video_qa/eval/eval_multiple_choice.py \
  --results_path /data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_full/llava_deltakv_delta_quant_results.csv
```

## Result

Output files:

```bash
/data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_full/last_rekv_qaego4d_result.json
/data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_full/vanilla_results.csv
/data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_full/llava_deltakv_delta_quant_results.csv
```

| Method | Samples | QA Acc | New tok/s | Examples/s | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vanilla` | 500 | 52.6 | 7.035 | 1.436 | 18.681 GB |
| `llava_deltakv_delta_quant` | 500 | 52.8 | 3.640 | 0.750 | 18.682 GB |

DeltaKV delta-only quantization:

- Accuracy delta vs vanilla: `+0.2`
- Speed ratio vs vanilla: `0.517x`
- Official evaluator reported `%Errors: 0.00` for both CSV files.

## KR 30 Candidate

For the current no-compressor `llava_deltakv_delta_quant` path, the cached
residual is token-space int4 rather than learned low-dimensional DeltaKV. Under
the DeltaKV paper's KR formula, this means:

```text
comp_ratio = 4bit / fp16 = 0.25
KR = Lfull / L + Lsparse / L * (cluster_ratio + comp_ratio)
```

The default 7B configuration used above has `L=28`, `Lfull=7`,
`cluster_ratio=0.1`, so:

```text
KR = 7/28 + 21/28 * (0.1 + 0.25) = 51.25%
```

To reduce KR to about 30% while keeping a valid OmniKV-style observation anchor,
use one full/observation layer and a lower reference ratio:

```bash
--full_attention_layers 0
--deltakv_center_ratio 0.03
--decode_keep_tokens 1024
--prefill_keep_tokens 4096
```

For LLaVA-OV-7B (`L=28`):

```text
KR = 1/28 + 27/28 * (0.03 + 0.25) = 30.57%
CR ~= 1/28 + 27/28 * (1024 / 12605.216) = 11.40%
CR ~= 12.45% if sink/recent/current tokens are counted.
```

Layer `0` is kept because in the HF DeltaKV wrapper full-attention layers also
serve as observation anchors for subsequent sparse layers. With no full layer, or
with only a later full layer, early sparse layers do not have a prior selector and
fall back to reconstructing broader history.

Full QAEGO4D-test-mc sanity check on LLaVA-OV-0.5B:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
/home/haojitai/miniconda3/envs/svllm/bin/python -u \
scripts/bench_llava_onevision_rekv_qaego4d.py \
  --model_path /data2/haojitai/models/llava-onevision-qwen2-0.5b-ov-hf \
  --dataset_dir /data2/haojitai/datasets/rekv_qaego4d \
  --output_dir /data2/haojitai/datasets/llava_onevision_rekv_qaego4d_05b_kr30_full \
  --methods vanilla,deltakv_delta_quant \
  --num_samples -1 \
  --batch_size 1 \
  --sample_fps 0.5 \
  --max_context_frames 64 \
  --cuda_device 0 \
  --reuse_frame_cache \
  --frame_cache_dir /data2/haojitai/datasets/rekv_qaego4d_frame_cache_fps05_64 \
  --full_attention_layers 0 \
  --deltakv_center_ratio 0.03 \
  --decode_keep_tokens 1024 \
  --prefill_keep_tokens 4096
```

| Model | Method | KR | Samples | QA Acc | New tok/s | Examples/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-OV-0.5B | `vanilla` | 100.00 | 500 | 48.4 | 17.792 | 3.616 |
| LLaVA-OV-0.5B | `llava_deltakv_delta_quant` | 31.00 | 500 | 48.4 | 8.168 | 1.657 |
| LLaVA-OV-7B | `llava_deltakv_delta_quant` | 30.57 | 500 | 52.8 | 3.655 | 0.754 |

The 7B candidate output is:

```bash
/data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_kr30_candidate/last_rekv_qaego4d_result.json
/data2/haojitai/datasets/llava_onevision_rekv_qaego4d_7b_kr30_candidate/llava_deltakv_delta_quant_results.csv
```

Official ReKV evaluator reported `%Errors: 0.00` for the 0.5B CSV files and
for the 7B KR30 candidate CSV. The 7B run used physical GPU 1 because GPUs 6
and 7 were occupied; it completed within the requested 30-minute limit.
