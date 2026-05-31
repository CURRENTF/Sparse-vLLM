# Reproducibility

Use this page as the stable checklist for reproducing Sparse-vLLM experiments.
For exact historical runs, see
[dev_docs/experiment-records.md](../../dev_docs/experiment-records.md).

## Environment

The README contains the current install command. The expected baseline is:

- Python 3.10.
- PyTorch 2.8.0.
- `transformers[torch]==4.53.3`.
- `flash-attn` installed with `MAX_JOBS=8 pip install flash-attn --no-build-isolation`.
- Editable install from the repository root with `pip install -e .`.

Record CUDA version, GPU type/count, visible GPU ids, branch, commit, and any
relevant uncommitted changes with every reported benchmark.

## Models And Checkpoints

Base models and DeltaKV compressor checkpoints must match. Public compressor
checkpoints are listed in the README section
[Download DeltaKV compressor checkpoints](README.md#deltakv-checkpoints).

Pass the downloaded local directory as `deltakv_checkpoint_path`. Current
loaders read local `model.safetensors` files; do not assume a Hugging Face repo
id can be passed directly everywhere.

## Data Paths

LongBench and MathBench read data roots from environment variables:

- `DELTAKV_OUTPUT_DIR`: output root for benchmark predictions and logs.
- `DELTAKV_DATA_DIR`: general benchmark dataset root.
- `DELTAKV_LONGBENCH_DATA_DIR`: LongBench root containing `data/*.jsonl`.

If a command uses local placeholders such as `<DATA_ROOT>`, `<MODEL_ROOT>`, or
`<OUTPUT_ROOT>`, rewrite them for the target machine and record the final paths
in the run record.

## Parameter Rules

Use canonical public parameter names:

- `sparse_method`
- `deltakv_checkpoint_path`
- `decode_keep_tokens`
- `prefill_keep_tokens`
- `sink_keep_tokens`
- `recent_keep_tokens`
- `full_attention_layers`
- `hf_prefill_chunk_size` for HF wrappers
- `engine_prefill_chunk_size` for Sparse-vLLM

Do not use legacy public keys such as `chunk_prefill_size`,
`vllm_sparse_method`, `model_cls`, `compressor_path`, `deltakv_path`,
`num_top_tokens`, or `seq_chunk_size` in new commands. See
[runtime-parameter-semantics.md](../configuration/runtime-parameter-semantics.md) for the full
alias map and backend-specific behavior.

Sparse-vLLM requires explicit integer keep budgets. HF paths may accept ratios
such as `decode_keep_tokens=0.17`; convert ratios to token counts before moving
the same policy to Sparse-vLLM.

## Smoke Checks

Start with small commands before long benchmarks:

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 1 \
  --methods vanilla \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.8,"engine_prefill_chunk_size":512}'
```

For a no-checkpoint DeltaKV-style Sparse-vLLM smoke:

```bash
PYTHONPATH=$PWD/src python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <LOCAL_BASE_MODEL> \
  --lengths 1024 \
  --batch_sizes 2 \
  --methods deltakv-delta-quant \
  --output_len 4 \
  --hyper_params '{"gpu_memory_utilization":0.9,"engine_prefill_chunk_size":512,"max_num_seqs_in_batch":2,"max_decoding_seqs":2,"max_num_batched_tokens":2048,"full_attention_layers":"0,1","sink_keep_tokens":4,"recent_keep_tokens":32,"decode_keep_tokens":64,"prefill_keep_tokens":64,"deltakv_center_ratio":0.1,"deltakv_neighbor_count":1,"deltakv_latent_quant_bits":4,"deltakv_full_pool_reserve_ratio":0.2}'
```

For compressor-backed DeltaKV, add a matching local
`deltakv_checkpoint_path` and verify the loader logs that compressor weights
were loaded.

## Artifact Expectations

For reported results, save or record:

- Exact command and working directory.
- Runtime config and canonical sparse parameters.
- Model, tokenizer, checkpoint, precision, backend, and quantization settings.
- Dataset path, split, sample count, filtering/truncation, and seed.
- Raw outputs, parsed outputs, per-sample records, aggregate metrics, and
  run info when the benchmark supports them.
- Log paths and result file paths.
- Failure status and key error lines for failed or inconclusive runs.

Do not report a metric without a source log or result artifact.
