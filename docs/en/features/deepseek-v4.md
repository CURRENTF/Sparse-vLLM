# DeepSeek-V4-Flash-0731 (experimental)

Sparse-vLLM provides an experimental native runtime for the original
DeepSeek-V4-Flash-0731 checkpoint. It runs the complete decoder, chunked
prefill, decode and continuous batching, with decode CUDA Graphs and GPU
radix prefix caching. Independent short- and 4K-context comparisons show
numerical error comparable to vLLM against the checkpoint reference. Cross-engine
logits and some greedy predictions differ; exact output equality is not guaranteed.

## Installation

Use Python 3.12 and the CUDA 13.0 environment described in
[Getting Started](../getting_started/README.md). Install the model extra
from the repository root:

```bash
uv sync --extra cu130 --extra deepseek-v4
```

The extra supplies the required FlashInfer version and Hadamard-transform
package. A CUDA toolkit and C++ compiler must be available to build native
extensions. Keep the original checkpoint configuration, tokenizer, safetensors
index and all weight shards together. The chat example also uses the
checkpoint's `encoding/encoding_dsv4.py` formatter.

Optionally, reuse routing and clipped SwiGLU kernels from a separate vLLM 0.29 installation
by setting `SPARSEVLLM_VLLM_MOE_LIBRARY` to that installation's
`vllm/_moe_C_stable_libtorch.abi3.so`. Keep its sibling
`_C_stable_libtorch.abi3.so` and wheel metadata in place.
This loads the kernel libraries without importing the vLLM engine or changing
Sparse-vLLM's Torch installation. Without this optional library, the portable
router and activation implementations remain available; an incompatible configured library fails at startup.

## Four-GPU example

This configuration has been exercised on four H100 80 GB GPUs. Set the model
directory and choose four available GPUs before launching the script:

```bash
export MODEL_ROOT="<MODEL_ROOT>"
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run --no-sync python run_deepseek_v4.py
```

Save the following as `run_deepseek_v4.py`. Formatting the conversation and
passing token IDs avoids relying on a generic chat template.

```python
import os
from pathlib import Path
import runpy

from transformers import AutoTokenizer
from sparsevllm import LLM, SamplingParams


def main():
    model_root = Path(os.environ["MODEL_ROOT"])
    encode_messages = runpy.run_path(
        str(model_root / "encoding" / "encoding_dsv4.py")
    )["encode_messages"]
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    conversation = encode_messages(
        [{"role": "user", "content": "Explain why the sky is blue."}],
        thinking_mode="chat",
    )
    prompt = tokenizer.encode(conversation, add_special_tokens=False)
    engine = LLM(
        str(model_root),
        tensor_parallel_size=1,
        data_parallel_size=4,
        expert_parallel_size=4,
        moe_backend="agrs",
        max_model_len=4352,
        max_num_batched_tokens=128,
        engine_prefill_chunk_size=128,
        mlp_chunk_size=128,
        max_num_seqs_in_batch=2,
        max_decoding_seqs=2,
        gpu_memory_utilization=0.75,
        decode_graph=True,
        enable_prefix_caching=True,
        prefix_cache_max_blocks=256,
    )
    try:
        outputs = engine.generate(
            [prompt], SamplingParams(temperature=0.0, max_tokens=32)
        )
        print(tokenizer.decode(outputs[0]["token_ids"], skip_special_tokens=True))
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
```

`sparse_method` is selected automatically from the checkpoint. Explicitly
setting `sparse_method="deepseek_v4"` is equivalent. Other sparse methods
cannot replace the model's native attention. The example uses the synchronous
Python API; incremental requests use the usual `add_request`, `step` and
`is_finished` interfaces.

## Runtime constraints

- Preserve the original FP8 weights and UE8M0 scales, MXFP4 expert weights,
  compression configuration and positional encoding. Generic BF16-converted
  checkpoints are not accepted by this runtime.
- Attention TP must be 1. Expert TP must also be 1, so `EP=world_size=DP`
  for the DP deployment above. Experts must divide evenly across EP ranks.
  Use `moe_backend="agrs"` with DP; DeepEP V1 is incompatible with this
  model's required FP32 expert reduction.
- Prefix caching is local to each DP replica. A cache hit restores the
  native compression state as well as reusable KV data. Prefix offload and
  chain cache are unsupported. Disable reuse with
  `enable_prefix_caching=False` when comparing cold runs.
- `max_model_len` includes prompt and generated tokens. KV capacity must
  cover window storage, compressed history and retained prefix state; a
  request can be deferred even when a request slot is available. Reduce
  context length, concurrency or retained prefix capacity if startup or
  admission reports insufficient memory.
- Set `decode_graph=False` for eager diagnostics. Speculative decoding and
  the checkpoint's next-token prediction layers are outside this integration.

The example is a bounded starting configuration. Numerical validation covers
short and 4K inputs on H100; the checkpoint's maximum context length has not
been validated.
