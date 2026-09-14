# DP Attention with Expert Parallelism

GLM-4.7-Flash supports single-node DP attention: each GPU processes its own
requests and KV cache, while routed experts are partitioned across GPUs. Enable
it with `tensor_parallel_size=1` and equal `data_parallel_size` and
`expert_parallel_size`. DP and EP share the same rank axis; DP=EP=2 uses two GPUs.
The existing TP/EP execution remains available with `data_parallel_size=1`.

Attention layout and expert layout are separate choices. For GLM on two GPUs:

| Layout | `tensor_parallel_size` | `expert_parallel_size` | `data_parallel_size` | Attention work |
| --- | ---: | ---: | ---: | --- |
| DP attention + EP | 1 | 2 | 2 | Each GPU processes different requests. |
| TP attention + EP | 2 | 2 | 1 | Both GPUs shard attention for the same requests. |
| Replicated attention + EP | 1 | 2 | 1 | Both GPUs repeat full attention for the same requests. |

All three shard routed experts across the same two GPUs. The relevant comparison
for choosing attention parallelism is **DP attention + EP versus TP attention +
EP**. A replicated-attention baseline must be labeled explicitly. The layouts
apply to both prefill and decode.

```python
import os
from sparsevllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        os.environ["MODEL_PATH"],
        tensor_parallel_size=1,
        expert_parallel_size=2,
        data_parallel_size=2,
        decode_graph=True,
        max_num_batched_tokens=2048,
        max_num_seqs_in_batch=8,
        max_decoding_seqs=8,
    )
    try:
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question in ["What is 2 + 3?", "中国的首都是哪里？"]
        ]
        print(llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32)))
    finally:
        llm.exit()
```

Set `CUDA_VISIBLE_DEVICES` to the participating GPUs before starting Python.
Use a guarded entrypoint because workers are spawned processes. The same runtime
parameters can be supplied through the serving engine configuration.

## Capacity and request ownership

`max_num_batched_tokens`, `max_num_seqs_in_batch`, `max_decoding_seqs`, KV-cache
limits, and prefix-cache budgets apply **per replica**. For example, two replicas
with `max_decoding_seqs=8` admit up to 16 simultaneous decoding requests, subject
to local memory and scheduler constraints. KV capacity is not pooled between
replicas. Use `worker_load()` for aggregate counters and its `replicas` entries
for individual cache/load records.

New requests prefer the least loaded replica; radix prefix hits break load ties.
An admitted request stays on its owner, including cancellation and chain-cache
continuations. A cached prefix on one replica is not automatically copied to
another. The frontend supports `generate`, request admission/abort, `step`, and
the asynchronous serving dispatcher. It does not expose a single `scheduler` or
`model_runner`; integrations must use public engine methods.

## Communication and decode graphs

The DP path defaults to All-Gather / Reduce-Scatter (AG/RS). Each MoE
block gathers token activations, executes the existing router and local expert
operator, and reduces outputs back to their token owners. Shared experts run on
the owner. TP/EP retains its existing All-Reduce transport.

Compatible single-host 2/4/8-rank configurations prefer FlashInfer mixed
communication for prefill and decode, without device-name or batch-size tuning
restrictions. It requires SM90/SM100, FP16/BF16 rows aligned to 16 bytes, and
multicast with peer access across the group. The operator report identifies
the selected implementation. This optional path requires `cuda-python` and
NVIDIA NVSHMEM, including for single-node execution,
and compiles kernels on first use. All replicas select communication from the
agreed token capacity, including mixed prefill/decode steps. Capacities outside
the prepared range use NCCL. Missing optional dependencies or unsupported
hardware select NCCL at startup; an installed provider's initialization or
execution failure is reported without silently switching implementations.

Set `moe_communication_backend="all2all"` to select DeepEP V1 normal NVLink
transport. `"auto"` keeps AG/RS for DP, and `"agrs"` selects it explicitly.
The all-to-all path retains the existing local expert operator and requires
single-host EP=DP in {2, 4, 8}, attention TP=1, MoE TP=1, BF16 activations,
a hidden size divisible by 256, and NVLink peer access between every EP device.
It does not use RDMA, low-latency mode, or shared-expert overlap.

DeepEP is optional. Install the base runtime first, then build the optional extra
in the same Torch/CUDA environment:

```bash
python -m pip install --no-build-isolation -e '.[all2all]'
```

The extra pins the official DeepEP V1.2.1 source and includes `nvidia-ml-py`
for topology validation. A compatible CUDA toolkit/compiler is required; follow
[DeepEP's build instructions](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)
for the target GPU architecture. The runtime accepts DeepEP >=1.2.1,<2 with the
required public API. Dependency, extension compatibility and NVLink checks run
at startup only when `all2all` is selected. Missing or incompatible installations
fail explicitly; runtime errors never silently switch the transport to AG/RS.
The operator runtime report identifies the transport provider and version.
Normal mode can have higher latency than AG/RS at small batches; choose the
transport using matched workload measurements.

Decode CUDA Graphs are captured at startup. Replicas agree on a batch capacity
for communication and may replay different local short/long attention paths.
An idle replica participates in experts without creating a request or KV state.
When one replica prefills while another decodes, that step uses eager execution;
normal decode graph replay resumes on subsequent pure-decode steps. No runtime
recapture is needed. Sampling runs outside the graph;
`decode_graph_capture_sampling=True` is rejected for DP attention.

## Sparse methods and prefix cache

The GLM DP path supports Vanilla, StreamingLLM, SnapKV, H2O, OmniKV, QuEST, and
R-KV with their existing model-specific constraints. Prefix caching supports
radix mode for Vanilla/OmniKV and chain mode for StreamingLLM/SnapKV/H2O/R-KV.
GLM QuEST prefix caching remains unsupported. Each replica owns its sparse
selection and physical cache lifecycle.

Prefix inspection, subtree deletion, and priority updates return results under
`replicas`. Score-based prefix-prune jobs are unsupported in DP mode. The request
mode of the efficiency probe supports DP; continuous decode windows and tools
that directly access a single scheduler require a separate DP integration.

This implementation requires attention TP=1 and MoE TP=1. It does not provide
multi-node execution, DeepEP V2, or DeepSeek V4 support. AG/RS
replicates token dispatch; benefits depend on workload balance and attention
cost. Compare matched global request and token budgets before choosing DP+EP over
TP+EP. BF16 results across parallel layouts or transports need not be bitwise identical.
