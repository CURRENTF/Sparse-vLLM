# Qwen3MoE Expert Parallelism

Sparse-vLLM supports unquantized Qwen3MoE inference with expert parallelism
(EP). Attention, KV cache, embeddings, norms, router weights, and the LM head
are replicated on every EP rank. Each rank loads only its contiguous expert
shard, computes its local routed contribution, and participates in an EP
all-reduce before the decoder layer continues.

## Supported Scope

The first version intentionally has a narrow, fail-fast contract:

| Capability | Supported |
| --- | --- |
| Model type | Hugging Face `qwen3_moe` with MoE in every decoder layer |
| Expert weights | Unquantized BF16 or FP16 |
| Parallel topology | `TP=1`, `DP=1`, positive EP dividing `num_experts` |
| Execution | Eager only; `decode_cuda_graph=False` |
| MoE backends | `triton` (default) and `pytorch` (correctness oracle) |
| Sparse methods | See the validated method list below |
| Prefix cache | `vanilla`, `omnikv`, and `quest` |
| Hardware-validated EP sizes | EP=1 and EP=2 |

The implementation is topology-generic for valid expert divisors, but EP=4
and EP=8 have not been hardware-validated in the initial delivery because the
validation host exposed two H20 GPUs. Do not treat them as measured topologies.

The validated sparse methods are `vanilla`, `streamingllm`, `snapkv`,
`pyramidkv`, `omnikv`, `quest`, and `rkv`.

The following combinations are rejected before execution:

- TP+EP, DP+EP, or `DP>1`;
- CUDA graph decode;
- quantized MoE or shared experts;
- Qwen3MoE with DeltaKV;
- Qwen3MoE with SkipKV when no model-matched steering asset is registered;
- prefix caching with methods outside `vanilla`, `omnikv`, and `quest`.

## Basic Usage

Expose one GPU per EP rank. `LLM` starts and coordinates the additional world
workers internally.

```python
from sparsevllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-30B-A3B-Instruct-2507",
    sparse_method="vanilla",
    tensor_parallel_size=1,
    expert_parallel_size=2,
    data_parallel_size=1,
    moe_backend="triton",
    enforce_eager=True,
    decode_cuda_graph=False,
    gpu_memory_utilization=0.72,
    max_model_len=4096,
)

outputs = llm.generate(
    ["Explain expert parallelism in one sentence."],
    SamplingParams(temperature=0.0, max_tokens=64),
)
llm.exit()
```

Run with two visible devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python your_script.py
```

Select the retained PyTorch oracle explicitly when validating a new checkpoint
or kernel change:

```python
llm = LLM(
    model_path,
    sparse_method="vanilla",
    expert_parallel_size=2,
    tensor_parallel_size=1,
    data_parallel_size=1,
    moe_backend="pytorch",
    enforce_eager=True,
    decode_cuda_graph=False,
)
```

There is no automatic fallback between the Triton and PyTorch backends. A
backend error is surfaced to the caller.

## Prefix Cache

Prefix caching is supported for the replicated-KV `vanilla`, `omnikv`, and
`quest` paths:

```python
llm = LLM(
    model_path,
    sparse_method="quest",
    expert_parallel_size=2,
    tensor_parallel_size=1,
    data_parallel_size=1,
    enforce_eager=True,
    decode_cuda_graph=False,
    enable_prefix_caching=True,
    prefix_cache_block_size=16,
    prefix_cache_max_blocks=1024,
)
```

Every EP rank owns an equivalent prefix index and complete-head KV payload.
Lookup, attach, commit, reference release, eviction, and control operations run
on all world workers. A lookup compares its hit metadata across ranks before
the scheduler proceeds; divergence is a fatal error rather than a rank-0-only
warning.

## Runtime Design

### Parallel groups

`ParallelContext` models DP, EP, and TP as separate dimensions. The world-rank
mapping is DP-major, then EP, then TP:

```text
world_rank = ((dp_rank * ep_size) + ep_rank) * tp_size + tp_rank
```

Dense tensor-parallel operations use the TP group. Routed expert output uses
the EP group. Cache managers derive attention head ownership from TP only, so
increasing EP never slices KV heads.

### Expert layout and loading

For `E` experts and `P` EP ranks, rank `r` owns:

```text
experts_per_rank = E / P
local_start = r * experts_per_rank
local_end = local_start + experts_per_rank
```

Each layer packs local expert weights as:

```text
w13_weight: [local_experts, 2 * moe_intermediate_size, hidden_size]
w2_weight:  [local_experts, hidden_size, moe_intermediate_size]
```

The loader requires every replicated tensor and every local expert tensor
exactly once. It only permits skips for experts outside the local interval and
reports unexpected skips, missing weights, duplicate weights, and shape
mismatches as errors.

### Routed execution

The router is replicated and computes softmax, TopK, and normalized routing
weights on every rank. The Triton path then:

1. counts and aligns local assignments;
2. runs the packed W13 routed GEMM;
3. applies SiLU-and-mul;
4. runs the packed W2 routed GEMM with routing weights;
5. sums local TopK contributions;
6. all-reduces those contributions across the EP group;
7. casts once to the model activation dtype.

The final local sum and EP reduction use FP64. This avoids topology-dependent
rounding from regrouping expert contributions between EP=1 and EP>1. Expert
GEMMs continue to use BF16/FP16 inputs with FP32 dot-product accumulation.

Replicated post-attention hidden and residual tensors are broadcast from EP
rank 0 before routing. This prevents tiny independent-attention differences
from changing TopK ties on different ranks.

## Validation Runbooks

The validation programs save run configuration, raw outputs, parsed outputs,
per-step or per-sample status, and aggregate metrics separately.

Run the sparse and prefix EP1-versus-EP2 matrix:

```bash
PYTHONPATH=src python scripts/validation/run_qwen3_moe_sparse_ep_matrix.py \
  --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --output-root /path/to/results/qwen3-moe-ep-matrix
```

Replay an EP=1 engine artifact through Hugging Face SDPA:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/validation/validate_qwen3_moe_hf_reference.py \
  --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --engine-reference /path/to/vanilla-ep1/raw_outputs.pt \
  --output-dir /path/to/results/hf-reference \
  --attention-implementation sdpa \
  --atol 0.5 --rtol 0.05
```

Benchmark the exact-reduction MoE kernel:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/validation/benchmark_moe_kernels.py \
  --output-dir /path/to/results/moe-microbench \
  --tokens 1,16,128,512 \
  --num-experts 128 --top-k 8 \
  --hidden-size 2048 --intermediate-size 768 \
  --output-dtype float64
```

Run the fixed manual-QA smoke set on EP=1, then compare EP=2 exactly:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/validation/validate_qwen3_moe_manual_qa.py \
  --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --output-dir /path/to/results/manual-qa-ep1 \
  --expert-parallel-size 1

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
python scripts/validation/validate_qwen3_moe_manual_qa.py \
  --model /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --output-dir /path/to/results/manual-qa-ep2 \
  --expert-parallel-size 2 \
  --reference /path/to/results/manual-qa-ep1/raw_outputs.json
```
