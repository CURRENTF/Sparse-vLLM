# LLaVA-OneVision Vanilla Batch Benchmarks

This page is the stable runbook for vanilla LLaVA-OneVision batch benchmarks.
Historical local run results are archived in
[`dev_docs/benchmark-run-history/multimodal/llava-onevision-vanilla-batch-benchmarks.md`](../../../dev_docs/benchmark-run-history/multimodal/llava-onevision-vanilla-batch-benchmarks.md).

## Scope

Implemented scripts:

- `benchmark/multimodal/image_qa/ai2d.py`: vanilla HF LLaVA-OneVision AI2D
  benchmark with batch generation.
- `benchmark/multimodal/visual_cache/run_visual_cache.py`: VQAv2 visual-cache
  benchmark where the `vanilla` method accepts `--batch_size`.

Batch behavior:

- tokenizer padding side is set to left;
- generation uses `padding=True`, `attention_mask`, greedy decoding, and
  `use_cache=True`;
- generated tokens are sliced after the common padded prompt length;
- per-sample records are saved to JSON.

## AI2D Alignment

The LLaVA-OneVision paper reports AI2D as a single-image task and follows the
LMMs-Eval `ai2d` settings: `lmms-lab/ai2d`, `max_new_tokens=16`,
`do_sample=False`, and the direct-answer post prompt.

Paper targets encoded in the script:

| Model | Paper AI2D |
| --- | ---: |
| `llava-onevision-qwen2-0.5b-ov-hf` | 57.1 |
| `llava-onevision-qwen2-7b-ov-hf` | 81.4 |

## Dataset

```bash
<HF_BIN> download lmms-lab/ai2d \
  --repo-type dataset \
  --local-dir <DATA_ROOT>/lmms-lab_ai2d \
  --cache-dir <HF_CACHE_ROOT> \
  --max-workers 8
```

## AI2D Command Template

```bash
PYTHONPATH=$PWD/src <SVLLM_PYTHON> -u \
  benchmark/multimodal/image_qa/ai2d.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-0.5b-ov-hf \
  --dataset_dir <DATA_ROOT>/lmms-lab_ai2d \
  --dataset_cache_dir <HF_CACHE_ROOT> \
  --output_dir <OUTPUT_ROOT>/llava_onevision_ai2d_vanilla_05b \
  --num_samples -1 \
  --batch_size 16 \
  --max_new_tokens 16 \
  --cuda_device 0 \
  --attn_implementation flash_attention_2 \
  --log_every 200
```

## VQAv2 Vanilla Smoke

```bash
PYTHONPATH=$PWD/src <SVLLM_PYTHON> -u \
  benchmark/multimodal/visual_cache/run_visual_cache.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-0.5b-ov-hf \
  --deltakv_checkpoint_path none \
  --dataset_dir <OUTPUT_ROOT>/llava_onevision_vanilla_batch_smoke \
  --source_vqa_dir <DATA_ROOT>/VQAv2 \
  --num_samples 2 \
  --max_new_tokens 4 \
  --batch_size 2 \
  --cuda_device 0 \
  --methods vanilla \
  --attn_implementation flash_attention_2 \
  --log_every 1
```

Treat tiny VQAv2 runs as batch-path smoke tests only. Use full AI2D or VQAv2
runs before reporting accuracy.
