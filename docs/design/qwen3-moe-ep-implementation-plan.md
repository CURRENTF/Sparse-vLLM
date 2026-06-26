# Qwen3-MoE Expert Parallel Implementation Plan

This plan adds first-stage Qwen3-MoE inference and expert parallelism to Sparse-VLLM. It is written for architecture review before implementation.

## Implementation and validation status

Status as of 2026-06-26:

- Implemented `ParallelContext`, EP config guardrails, process-world vs TP-world separation, and TP-context usage in dense linear/embed/model paths.
- Implemented `model_type=qwen3_moe` with Qwen3 attention reuse, Qwen3-MoE router, local expert execution, and EP all-reduce combine.
- Implemented model-owned expert checkpoint loading before generic packed-module mapping.
- The local `Qwen3-30B-A3B-Instruct-2507` checkpoint uses named expert tensors (`experts.<id>.gate_proj.weight`, `up_proj.weight`, `down_proj.weight`), so the loader supports both this named layout and fused expert tensors.
- Added `scripts/validation/qwen3_moe_ep_smoke.py` for file-based multiprocessing smoke runs.
- Unit validation covers tiny MoE math parity, named/fused expert loading, fail-fast missing expert parts, dense/cache EP isolation, decode graph alias rejection, and CPU/gloo EP all-reduce parity.
- Real checkpoint smoke used `/data2/haojitai/models/Qwen3-30B-A3B-Instruct-2507` with greedy `max_tokens=1`, prompt `The capital of France is`.
- Real EP=1 on GPU 7 loaded 18867 weights and generated token ids `[12095]`, text `' Paris'`.
- Real EP=2 on GPUs 5,6 loaded 9651 weights per rank and generated token ids `[12095]`, text `' Paris'`.
- Machine-readable artifacts are under `/data2/haojitai/outputs/Sparse-vLLM-qwen3-moe-ep/qwen3_moe_ep_smoke_20260626_191423/`.
- EP=4/8 real checkpoint runs were not part of this smoke because the available validation window exposed only three idle H100s. No throughput or memory-efficiency claim is made from these runs.

## Goals

- Support text-only `model_type=qwen3_moe` inference.
- Support Expert Parallel size greater than 1 with Tensor Parallel size fixed to 1.
- Keep the cache manager EP-unaware.
- Treat EP as an intra-replica MoE execution strategy; do not claim native DP scheduling in v1.
- Validate EP correctness against a single-rank Qwen3-MoE oracle.
- Keep the first backend simple enough to debug before adding all-to-all or DeepEP.

## Non-goals

- No TP+EP hybrid support in v1.
- No Qwen3-VL or multimodal MoE support.
- No native data-parallel scheduler implementation.
- No DeepEP or all-to-all dispatch in the first correctness milestone.
- No decode CUDA graph support for EP v1.

## Current constraints to address

- `src/sparsevllm/engine/llm_engine.py` currently spawns workers only from `config.tensor_parallel_size`.
- `src/sparsevllm/engine/model_runner.py` currently sets runner `world_size` from `config.tensor_parallel_size`, then uses it for process-group setup, model loading, and cache-manager construction.
- Dense parallel layers currently use `dist.get_world_size()` as TP size. EP must not make dense layers shard by EP world size.
- The cache manager sizes KV heads and memory using the runner world size. EP must not make KV cache appear TP-sharded.
- The generic loader currently uses substring-based `packed_modules_mapping` before any MoE-specific key classification.
- Qwen3 support is currently dense-only; Qwen3-MoE needs a separate model path.
- The model loader currently has no expert-aware loading or local-expert filtering.
- Decode CUDA graph capture currently captures the full decode forward, so EP cannot simply skip the MoE section without a piecewise graph path.

## Architecture decisions

### Parallel topology

Introduce a `ParallelContext` before implementing Qwen3-MoE or EP execution. It must separate process-world identity from tensor-parallel identity:

- `global_rank`: rank in the launched distributed process world.
- `global_world_size`: number of launched model-runner processes.
- `tp_size`: tensor-parallel size used by dense layers, attention, LM head, and cache manager.
- `tp_rank`: rank inside the TP group.
- `tp_group`: process group used by TP collectives.
- `ep_size`: expert-parallel size used only by MoE executor and expert loading.
- `ep_rank`: rank inside the EP group.
- `ep_group`: process group used by MoE collectives.
- `dp_rank`: future outer replica rank; not required for v1 scheduler work.

For v1:

- Require `tp_size == 1` when `ep_size > 1`.
- Use `parallel_world_size = tp_size * ep_size`; under v1 EP this is `ep_size`.
- `LLMEngine` spawns `parallel_world_size` model-runner processes, not just `tensor_parallel_size`.
- `ModelRunner` initializes the process group with `global_rank/global_world_size`.
- Dense layers, attention, LM head, loader TP sharding, and cache manager use `tp_size/tp_rank`, never `global_world_size/global_rank`.
- MoE executor and expert-aware loader use `ep_group/ep_rank`.
- All dense modules and cache-manager code observe `tp_size == 1`.
- Every EP rank loads replicated non-expert parameters.
- Every EP rank owns a full KV cache.
- Expert parameters are partitioned by expert id across EP ranks.

Do not patch EP locally into loader or executor before this topology split exists. Without this split, EP=2/4/8 is indistinguishable from TP=2/4/8 to the current dense layers and cache manager.

### Config surface

Use `expert_parallel_size: int = 1` as the primary EP switch:

- `expert_parallel_size == 1` means EP is off.
- `expert_parallel_size > 1` means EP is on.
- `expert_parallel_backend: str = "all_reduce"` selects the MoE communication backend.
- `expert_placement_policy: str = "contiguous"` selects expert ownership.

Do not require a separate `enable_expert_parallel` flag. If a compatibility alias is added, validate it strictly:

- `enable_expert_parallel=True` requires `expert_parallel_size > 1`.
- `enable_expert_parallel=False` requires `expert_parallel_size == 1`.

EP v1 rejects `tensor_parallel_size > 1 && expert_parallel_size > 1`.

### Qwen3-MoE model path

Add `src/sparsevllm/models/qwen3_moe.py` rather than overloading dense Qwen3.

The Qwen3-MoE model should mirror dense Qwen3 where possible:

- reuse Qwen3 RMSNorm behavior;
- reuse Qwen3 attention behavior with TP size 1;
- replace dense MLP with Qwen3-MoE block;
- expose a `packed_modules_mapping` or loader adapter for QKV and MoE weights;
- register `model_type=qwen3_moe` in model runner and config validation.

The model adapter owns Qwen3-specific details:

- `num_experts`;
- `num_experts_per_tok`;
- `moe_intermediate_size`;
- `decoder_sparse_step`;
- `mlp_only_layers`, if present;
- `norm_topk_prob`;
- Qwen3-MoE checkpoint parameter names and tensor shapes.

The implementation must follow the generic Qwen3-MoE layer-selection logic:

- use MoE when `layer_idx not in mlp_only_layers`, `num_experts > 0`, and `(layer_idx + 1) % decoder_sparse_step == 0`;
- use dense MLP otherwise.

For the target `Qwen/Qwen3-30B-A3B-Instruct-2507`, the config checked on 2026-06-26 reports `model_type="qwen3_moe"`, 48 layers, 128 experts, top-8 routing, `moe_intermediate_size=768`, `hidden_size=2048`, `mlp_only_layers=[]`, and `decoder_sparse_step=1`.
The model repository is large, so tiny synthetic parity tests are required before full-model smoke tests.

### MoE executor abstraction

Split MoE code into model math and distributed execution:

- `src/sparsevllm/layers/moe.py` or `src/sparsevllm/layers/moe/` owns pure model components: router, local expert module, shape/dtype assertions, and an EP=1 reference executor.
- `src/sparsevllm/engine/expert_parallel/` owns distributed context, expert ownership mapping, backend interfaces, and all-reduce combine.

Suggested interfaces:

- `MoEParallelContext`: `ep_size`, `ep_rank`, `ep_group`, expert ownership policy.
- `MoERouter`: computes top-k expert ids and weights from router logits.
- `LocalExpertStore`: owns local expert weights and maps global expert ids to local tensors.
- `MoEExecutor`: executes routed experts and combines local contributions.
- `MoEBackend`: backend strategy name, initially `all_reduce`.

Keep the abstraction narrow:

- It should support Qwen3-MoE first.
- It should not hide model-specific routing differences behind broad fallback logic.
- Unsupported router or expert layouts should fail fast with clear errors.

### EP backend v1: local experts plus all-reduce

For each MoE layer:

1. Compute router logits on every EP rank.
2. Select top-k expert ids and weights on every EP rank.
3. Each rank computes only tokens routed to its local experts.
4. Each rank writes local weighted expert output into a `[tokens, hidden]` contribution tensor.
5. All ranks call `dist.all_reduce(SUM)` over that contribution tensor.
6. The reduced tensor is the MoE output for the layer.

This backend is not the final efficiency target, but it is the lowest-risk correctness target.
Do not make throughput claims from this backend beyond measured smoke results.
DeepEP and all-to-all are explicit follow-up backends, not prerequisites for calling the correctness path complete.

Expert ownership is deterministic and defaults to contiguous shards:

- `expert_placement_policy="contiguous"` is the only advertised v1 policy.
- `round_robin` may exist later, but should not be the default.
- ownership metadata lives in the EP context: `global_expert_ids`, `expert_id_to_owner_rank`, and `global_to_local_expert_id`.
- loader and executor must read the same ownership object.

Contiguous ownership matches the target model's expert count for EP=2/4/8 and allows efficient safetensors slicing along expert dim0 for fused expert tensors.

### Loader changes

Make the loader expert-aware before applying generic packed-module substring rules.

Add a `Qwen3MoeCheckpointAdapter` or equivalent model-owned loader adapter. For each checkpoint key, classify it first as:

- local expert parameter;
- non-local expert parameter;
- router parameter;
- non-expert dense parameter;
- unsupported or unexpected parameter.

Only non-expert dense parameters should fall through to the generic `packed_modules_mapping` loader. Expert keys must not be matched accidentally by dense `gate_proj` / `up_proj` / `down_proj` packed-module rules.

The adapter must support or explicitly reject each observed expert layout:

- non-expert parameters are loaded on all EP ranks;
- router parameters are loaded on all EP ranks;
- local expert parameters are loaded only by their owner EP rank;
- non-local expert parameters are skipped explicitly, not ignored accidentally;
- expert tensor shape mismatches fail immediately;
- missing expected expert tensors fail immediately.

For fused expert tensors such as `experts.gate_up_proj[num_experts, 2 * moe_intermediate_size, hidden_size]` and `experts.down_proj[num_experts, hidden_size, moe_intermediate_size]`, the loader must prefer safetensors slice reads over full-tensor `get_tensor()` followed by local slicing. If owned-expert slice loading is not possible for a layout, fail clearly or mark that layout unsupported; do not silently read every expert on every EP rank.

For named expert layouts, missing local expert tensors must fail, and non-local expert tensors must be reported as explicit skips.

### CUDA graph policy

EP v1 disables decode CUDA graph capture and runs decode eager.

Reject EP with decode graph only after all aliases and implicit graph-enabling paths are normalized. This includes `decode_cuda_graph`, `decode_graph`, `omnikv_decode_cuda_graph`, `omnikv_decode_graph`, and legacy sparse-method graph forcing. The error should be explicit:

`expert_parallel_size > 1 disables decode_cuda_graph in EP v1; set decode_cuda_graph=False or run without EP.`

Follow-up piecewise graph work:

1. Mark MoE boundaries in decode execution.
2. Capture larger graph-safe dense segments between MoE layers where possible.
3. Keep MoE eager.
4. Validate static tensor addresses and sparse-state references across eager holes.
5. Re-enable graph only when parity and perf gates pass.

This preserves the existing full-forward CUDA graph path for non-EP runs.

## Implementation phases

### Phase 0: config and hard guardrails

- Add `expert_parallel_size`, `expert_parallel_backend`, and `expert_placement_policy`.
- Keep EP default-off with `expert_parallel_size=1`.
- Add validation that rejects `ep_size > 1 && tp_size > 1`.
- Normalize decode graph aliases and legacy forced graph settings before rejecting EP with decode CUDA graph in v1.
- Add clear user-facing error messages for unsupported model or parallel combinations.

Review gate:

- Config naming is accepted.
- Failure modes are explicit.
- `decode_graph` and `omnikv_decode_graph` aliases cannot bypass EP graph rejection.
- No current non-EP behavior changes.

### Phase 1: parallel topology hard prerequisite

- Introduce `ParallelContext` with process-world, TP, and EP fields.
- Make `LLMEngine` spawn `parallel_world_size = tp_size * ep_size`.
- Make `ModelRunner` initialize distributed using `global_rank/global_world_size`.
- Create TP and EP process groups from the context.
- Bind device using global/local process rank while keeping TP identity separate.
- Replace direct dense-layer dependence on global `dist.get_world_size()` and `dist.get_rank()` where EP would be misinterpreted as TP.
- Pass TP size/rank into dense linear layers, embedding/head, loader TP sharding, and cache manager.
- Ensure cache manager receives TP size, not EP size or global world size.
- Keep existing TP behavior unchanged for non-EP runs.

Review gate:

- Dense Qwen3 and existing sparse methods still run with TP=1.
- `linear.py` does not shard dense weights when `global_world_size == ep_size` and `tp_size == 1`.
- `CacheManager` keeps `num_kv_heads` unchanged when EP is enabled.
- `LLMEngine` spawns EP workers from `expert_parallel_size`.
- Existing TP code paths are not broadened beyond the needed EP guard.

### Phase 2: Qwen3-MoE single-rank support

- Add `qwen3_moe.py`.
- Implement Qwen3-MoE block with EP size 1.
- Add model registration and config validation.
- Add loader support for Qwen3-MoE checkpoint names.
- Add tiny synthetic Qwen3-MoE config and hand-built or random state-dict parity tests before relying on the full target model.
- Run single-rank smoke generation when hardware and model files are available.

Review gate:

- Tiny synthetic Qwen3-MoE router/expert path matches an eager reference.
- `Qwen/Qwen3-30B-A3B-Instruct-2507` loads on one GPU when hardware and model files are available.
- Single-rank output is stable under greedy decode.

### Phase 3: MoE executor and EP all-reduce backend

- Add generic MoE executor interfaces.
- Add expert ownership mapping.
- Load only local expert weights on each EP rank.
- Implement local-expert routed execution.
- Add all-reduce combine.
- Add shape and dtype assertions around routed tensors and contribution tensors.

Review gate:

- Synthetic tiny MoE test matches single-rank reference.
- EP=2 and EP=4 produce matching generated tokens against EP=1 on short greedy prompts.
- Any mismatch saves selected experts, routing weights, and per-rank contribution stats.

### Phase 4: integration validation

Use `Qwen/Qwen3-30B-A3B-Instruct-2507` as the correctness oracle:

- Compare EP=1 against EP=2, EP=4, and EP=8 when hardware allows.
- Use identical prompts, max tokens, seed, dtype, and greedy decoding.
- Record first decode-step logits difference.
- Record generated token ids.
- Record final text.
- Run both short prompts and at least one longer context prompt.
- Do not make full-model validation the only correctness signal; tiny synthetic tests remain the required fast regression gate.

Acceptance thresholds:

- Generated token ids match for deterministic greedy smoke prompts.
- First decode-step logits match within a documented fp tolerance, with max/mean/std/top-token diff recorded.
- Any mismatch includes saved raw logits where practical, or at minimum logits slices, top-k token ids, selected experts, routing weights, and per-rank contribution stats.
- EP=8 is hardware-available coverage, not a blocker for all environments.

### Phase 5: sparse-method compatibility check

After dense Qwen3-MoE EP is correct:

- run vanilla attention first;
- run the target Sparse-VLLM sparse methods without changing method semantics;
- verify cache manager state remains independent of EP size;
- compare EP=1 and EP>1 under the same sparse-method settings.

Review gate:

- Sparse-method failures are reported as method compatibility issues, not hidden by EP fallbacks.

### Phase 6: piecewise CUDA graph follow-up

Only after phases 1-5 pass:

- add a piecewise decode executor;
- capture graph-safe non-MoE segments;
- keep MoE eager;
- validate graph active/inactive state explicitly;
- compare eager EP and piecewise-graph EP outputs.

Review gate:

- CUDA graph support is opt-in and fails explicitly on unsupported backends.
- Existing non-EP full-forward graph capture remains unchanged.

### Phase 7: efficient EP backend follow-up

After correctness is stable:

- add an all-to-all dispatch backend;
- evaluate DeepEP integration if dependency and deployment constraints are acceptable;
- compare all-reduce, all-to-all, and DeepEP backends on throughput and memory;
- keep backend selection explicit in config.

Review gate:

- Correctness parity against all-reduce EP is preserved.
- Performance results include batch size, prompt length, output length, EP size, dtype, and GPUs.

## Test plan

Unit tests:

- `ParallelContext` rank/group mapping;
- router top-k and weight normalization;
- expert ownership mapping;
- local expert id to local tensor mapping;
- synthetic all-reduce MoE equivalence;
- loader skip/load behavior for local and non-local experts;
- fused expert tensor owned-slice loading;
- named expert layout missing-local-parameter failure;
- config rejection for TP+EP and EP+decode-graph.

Smoke tests:

- dense Qwen3 unchanged;
- Qwen3-MoE EP=1 load and greedy generation;
- Qwen3-MoE EP=2 short greedy generation;
- Qwen3-MoE EP=4 or EP=8 when GPUs are available.

Regression checks:

- existing Sparse-VLLM regression tests still pass for non-EP models;
- dense layers do not shard when global world size comes from EP and TP size is 1;
- cache manager allocation is unchanged when EP is disabled;
- cache manager KV-head count is unchanged when EP is enabled with TP size 1;
- `LLMEngine` spawns `expert_parallel_size` workers for EP v1;
- loader records explicit non-local expert skips;
- decode graph aliases and OmniKV graph aliases fail under EP;
- decode CUDA graph remains active for supported non-EP runs.

Artifacts to save for validation:

- exact command;
- git commit;
- model path;
- dtype;
- TP size;
- EP size;
- backend;
- prompt file;
- generated token ids;
- generated text;
- first-step logits diff summary;
- error logs if any.

## Expert-review questions

- Is `all_reduce` an acceptable first EP backend if the plan explicitly defers throughput claims?
- Is contiguous expert ownership sufficient as the only advertised v1 placement policy?
- Is the proposed split between `layers/moe.py` and `engine/expert_parallel/` the right ownership boundary?
- Are `expert_parallel_size`, `expert_parallel_backend`, and `expert_placement_policy` the right config surface?
- Is the proposed piecewise CUDA graph follow-up, with MoE boundary markers and larger dense segments, the right second-stage design?
- Is single-DP-replica EP enough to call v1 correctness complete if the docs avoid production-scale DP claims?
