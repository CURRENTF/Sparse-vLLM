# MLA prefill kernels

The repository has two BF16 partial-attention implementations for equal
Q/K/V head dimensions of 256:

- `prefill_pipelined.attention_partial`: MMA-v2 with asynchronous K/V copies.
- `prefill_hopper.attention_partial`: Hopper TMA copies and WGMMA. K/V pointers
  and token/head strides must be aligned to 16 bytes.

Both return BF16 output and FP32 natural-log LSE, accept packed ragged metadata,
and use bottom-right causal alignment. Fully masked rows return zero output and
negative-infinity LSE. The implementations preserve strided projection views
for V and use 64-bit storage offsets.

`split_kv=True` partitions K when the query grid supplies too little parallel
work. Partial outputs and LSE are stored in FP32, then combined by stable
LSE-weighted reduction. The unsplit and split paths share the attention core.
The prepared Triton partial-attention registry enables split-KV and accounts for
its scratch before the layer checks the prefill workspace budget. For S>1
partitions, T query tokens and H heads, scratch requires
`S * H * T * (256 + 1) * 4` bytes, in addition to final output and LSE. Sequential
history chunks contribute their maximum scratch requirement, not their sum.

Repository prefill selection is prepared independently of decode selection:
Hopper uses TMA/WGMMA when projection storage satisfies alignment; other
Ampere-or-newer CUDA devices use MMA-v2. Neither default requires a measured
shape, batch, head count, GPU name, TP count, or toolchain version profile.
Actual BF16/D=256 and hardware constraints still apply. The baseline remains
for compatible contracts outside that domain. Upstream FA3 selection at the
full MLA provider remains upstream-first. Execution failures propagate instead
of triggering a runtime switch.

Binding reports expose the selected prefill provider and kernel path. These
algorithmic defaults generalize the execution strategy; they do not claim
measured speedups on untested configurations. Standalone kernel calls keep
`split_kv=False` by default for callers that do not account for split scratch.

## Reproduce a comparison

Activate the project environment, select an idle GPU, and provide a new output
directory. The canonical runner includes allocation, partial attention and
split reduction inside each measured callable:

```bash
PYTHONPATH=src python scripts/profiling/kernel_bench/benchmark_mla_prefill.py \
  --vllm --candidate-kernel pipelined-split --include-old --strided-v \
  --suite shortquery --warmup 10 --rounds 20 --output-dir "$RUN_DIR"
```

Use `--candidate-kernel hopper` on Hopper, or `pipelined` for the unsplit MMA
candidate. `provider` exercises the current production provider selection.
`--candidate-module module:callable` explicitly selects an experimental callable.

- `multibatch`: fixed total Q=8192, batch 1/2/4/8/16, uniform and ragged lengths.
- `shortquery`: batch 1/8, per-request Q=1/32/128/256, K=16K/64K.
- `longcontext`: Q=8192 with K=128K/256K/512K, plus Q=32/128 with K=512K.

The runner validates against FP32 Torch before timing and records its installed
vLLM FlashAttention version. Long-Q validation samples boundary query rows;
Q<=256 checks every query row, sequence and head. `--validate-only` performs the
numerical checks without timing. Timing uses interleaved eager CUDA events and
does not measure model latency or a complete 512K prefill.

Run `tests/test_mla_pipelined_prefill.py` for independent ragged, empty-KV,
mutable-metadata CUDA Graph and large-storage-offset checks. Provider binding
tests and direct kernel tests have different scopes; direct kernel success does
not establish production dispatch or model-level correctness.
