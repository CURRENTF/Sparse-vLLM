# LLaVA-OneVision Vanilla Batch Benchmarks

This note records the vanilla HF LLaVA-OneVision batch benchmark path and the
paper-alignment run.

## Scope

Implemented scripts:

- `benchmark/multimodal/visual_cache/run_visual_cache.py`: the existing VQAv2 visual-cache
  benchmark now accepts `--batch_size` for the `vanilla` method. Non-vanilla
  visual-cache methods still run with effective batch size 1 because the current
  visual-token-prune wrapper is not batch-enabled.
- `benchmark/multimodal/image_qa/ai2d.py`: a dedicated vanilla HF
  LLaVA-OneVision AI2D benchmark with batch generation.

Batch behavior:

- tokenizer padding side is set to left;
- generation uses `padding=True`, `attention_mask`, greedy decoding, and
  `use_cache=True`;
- generated tokens are sliced after the common padded prompt length returned by
  HF `generate`;
- per-sample records are saved to JSON, while terminal summaries hide full
  records unless `--print_records` is set.

## AI2D Paper Alignment

The LLaVA-OneVision paper reports AI2D as one of its single-image tasks and says
the evaluation follows LMMs-Eval task settings. The LMMs-Eval `ai2d` task uses
the `lmms-lab/ai2d` dataset, `max_new_tokens=16`, `do_sample=False`, and the
default post prompt:

```text
Answer with the option's letter from the given choices directly.
```

Sources:

- LLaVA-OneVision paper: <https://arxiv.org/abs/2408.03326>
- LMMs-Eval AI2D task config: <https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/ai2d/ai2d.yaml>

Paper targets encoded in the script:

| Model | Paper AI2D |
| --- | ---: |
| `llava-onevision-qwen2-0.5b-ov-hf` | 57.1 |
| `llava-onevision-qwen2-7b-ov-hf` | 81.4 |

## Dataset Download

```bash
<HF_BIN> download lmms-lab/ai2d \
  --repo-type dataset \
  --local-dir <DATA_ROOT>/lmms-lab_ai2d \
  --cache-dir <DATA_ROOT>/hf_cache \
  --max-workers 8
```

Downloaded location:

```text
<DATA_ROOT>/lmms-lab_ai2d
```

## Full 0.5B AI2D Run

Command:

```bash
PYTHONPATH=$PWD/src <SVLLM_PYTHON> -u \
  benchmark/multimodal/image_qa/ai2d.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-0.5b-ov-hf \
  --dataset_dir <DATA_ROOT>/lmms-lab_ai2d \
  --dataset_cache_dir <DATA_ROOT>/hf_cache \
  --output_dir <DATA_ROOT>/llava_onevision_ai2d_vanilla_05b_full_bs16 \
  --num_samples -1 \
  --batch_size 16 \
  --max_new_tokens 16 \
  --cuda_device 7 \
  --attn_implementation flash_attention_2 \
  --log_every 200
```

Result:

| Metric | Value |
| --- | ---: |
| Samples | 3088 |
| Batch size | 16 |
| Accuracy | 56.8329 |
| Paper target | 57.1 |
| Delta vs paper | -0.2671 |
| Examples/s | 34.9331 |
| New tokens/s | 70.4092 |
| Peak memory | 15.6454 GB |

Result file:

```text
<DATA_ROOT>/llava_onevision_ai2d_vanilla_05b_full_bs16/last_ai2d_vanilla_result.json
```

The 0.5B full AI2D result is within 0.27 points of the reported paper score,
which is close enough for this local HF vanilla benchmark path to be treated as
aligned on AI2D.

## 7B AI2D Batch Sanity

The 7B model was smoke-tested to confirm the same vanilla batch path runs:

```bash
PYTHONPATH=$PWD/src <SVLLM_PYTHON> -u \
  benchmark/multimodal/image_qa/ai2d.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-7b-ov-hf \
  --dataset_dir <DATA_ROOT>/lmms-lab_ai2d \
  --dataset_cache_dir <DATA_ROOT>/hf_cache \
  --output_dir <DATA_ROOT>/llava_onevision_ai2d_vanilla_7b_sanity_bs2 \
  --num_samples 8 \
  --batch_size 2 \
  --max_new_tokens 4 \
  --cuda_device 7 \
  --attn_implementation flash_attention_2 \
  --log_every 2
```

Result:

| Metric | Value |
| --- | ---: |
| Samples | 8 |
| Batch size | 2 |
| Accuracy | 100.0 |
| Examples/s | 5.5630 |
| New tokens/s | 11.1259 |
| Peak memory | 16.1491 GB |

Result file:

```text
<DATA_ROOT>/llava_onevision_ai2d_vanilla_7b_sanity_bs2/last_ai2d_vanilla_result.json
```

This is a batch-path sanity check only; it is not a full 7B paper-alignment run.

## VQAv2 Vanilla Batch Smoke

The existing visual-cache benchmark's `vanilla` method was smoke-tested with
the 0.5B model:

```bash
PYTHONPATH=$PWD/src <SVLLM_PYTHON> -u \
  benchmark/multimodal/visual_cache/run_visual_cache.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-0.5b-ov-hf \
  --deltakv_checkpoint_path none \
  --dataset_dir <DATA_ROOT>/llava_onevision_vanilla_batch_smoke \
  --source_vqa_dir <DATA_ROOT>/VQAv2 \
  --num_samples 2 \
  --max_new_tokens 4 \
  --batch_size 2 \
  --cuda_device 7 \
  --methods vanilla \
  --attn_implementation flash_attention_2 \
  --log_every 1
```

Result:

| Batch size | Samples | Mean VQA score | Contains-answer acc | Examples/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 1.0 | 1.0 | 3.0280 |
| 2 | 2 | 1.0 | 1.0 | 3.0714 |

The two decoded predictions matched between batch size 1 and batch size 2 on
this smoke set.
