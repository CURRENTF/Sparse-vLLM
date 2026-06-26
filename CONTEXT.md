# DeltaKV Research Context

This context defines project-specific language used when discussing DeltaKV research evaluation workflows.

## Language

**Unified Video QA Evaluator**:
A single command-line entry point that normalizes supported video question-answering benchmarks into a shared evaluation and artifact shape while keeping benchmark-specific dataset loading separate. It is not a full replacement for lmms-eval or official benchmark submission tooling.
_Avoid_: full benchmark framework, lmms-eval replacement

**Video QA Metadata Download**:
A download scope that fetches annotations, dataset cards, and small metadata files for loader/schema dry-runs only. It is not sufficient for real video QA evaluation, which requires the referenced video files.
_Avoid_: benchmark download, evaluation-ready download

**Video QA Full Download**:
A download scope that fetches both metadata and video media needed for real local benchmark evaluation.
_Avoid_: metadata-only setup

**Sparse-VLLM LongBench Baseline**:
A LongBench comparison run that uses the Sparse-VLLM backend and a sparse method supported by the Sparse-VLLM runtime. For current DeltaKV comparisons, full-attention/vanilla, SnapKV, PyramidKV, and Quest are default Sparse-VLLM baselines; OmniKV and DeltaKV variants are full-layer-dependent baselines and should be run only after the full-layer policy for the target model is specified. It does not imply running StreamingLLM unless it is explicitly requested.
_Avoid_: all historical baselines, HF-only baselines, StreamingLLM by default, full-layer-dependent baselines before full-layer policy is specified

**Sparse-VLLM Qwen3 DeltaKV Support**:
A text-only Sparse-VLLM inference capability for running DeltaKV-family methods on a Qwen3 language model with a fixed full-layer policy. First-stage support includes both compressor-backed DeltaKV with a matching compressor and no-checkpoint DeltaKV Delta-Quant; it does not include Qwen3-VL, Tensor Parallel support, or Thinking-model validation unless those are named separately.
_Avoid_: Qwen3-VL support, all Qwen3 variants, TP-ready Qwen3 DeltaKV

**TP-Local Sparse Selection**:
A tensor-parallel sparse decode semantics where ranks share execution schedule and cache lifecycle, while each rank makes sparse-token decisions from its own local attention heads or KV heads without cross-rank sparse-index aggregation. It is not algorithmically equivalent to a single-rank or global-head sparse selection unless separately validated.
_Avoid_: global sparse-index aggregation, TP-equivalent sparse selection

**HF DeltaKV Sanity Alignment**:
A small validation check that compares a Sparse-VLLM DeltaKV run against the repository's HF DeltaKV path on the same text-only model, compressor, prompt, and sparse-method settings. It is a smoke-level correctness gate, not a full benchmark or a guarantee of exact token-by-token equivalence across all tasks.
_Avoid_: full LongBench validation, exact parity proof, benchmark score

**Qwen3 Delta-Quant First-Stage Check**:
The first-stage no-checkpoint Qwen3 DeltaKV Delta-Quant validation uses 4-bit residual or full-layer quantization settings and compares against the HF origin-codec DeltaKV path with `use_compression=false`. It is not a 2-bit quantization claim.
_Avoid_: Qwen3 2-bit validation, compressor-backed DeltaKV validation

**Full-Layer Policy**:
The chosen set of layers that act as full-attention layers or observation anchors for a target model and sparse method. A policy is fixed before a comparison run so results can be reproduced and interpreted.
_Avoid_: layer list, full layer config

**Qwen3-4B Instruct DeltaKV Full-Layer Policy**:
The fixed full-layer policy used for first-stage text-only Qwen3-4B Instruct DeltaKV validation in this repository: layers 0, 1, 2, 3, 8, 16, and 22. It is a reused validation policy, not a new offline calibration result.
_Avoid_: calibrated Qwen3 policy, dynamic full-layer selection

**Offline Full-Layer Calibration**:
A calibration workflow that chooses a fixed full-layer policy from representative prompts before evaluation. It is distinct from changing the full-layer policy dynamically per evaluated sample.
_Avoid_: runtime layer selection, per-sample full-layer routing

**Token Coverage Score**:
A calibration metric that measures how well an anchor layer's selected token-index set covers the selected token-index sets of sparse layers that depend on that anchor. It is used to compare candidate full-layer policies, not as a benchmark task metric.
_Avoid_: task accuracy, LongBench score

**Decode-Style Calibration Point**:
A prompt position used during offline full-layer calibration where token selection is scored with a single-query decode-style attention step. It can be sampled from within a prompt or placed at the answer boundary to approximate the first real generation step.
_Avoid_: prefill score point, generated sample

**Asymmetric DeltaKV Compressor**:
A learned DeltaKV compressor whose compression path and reconstruction path intentionally use different architecture choices or capacity. It does not merely mean that the two paths have separate, untied weights.
_Avoid_: untied compressor weights

**Prefix Cache Block**:
A block-aligned segment of request context tokens that is eligible for prefix-cache reuse by later requests. Prefix-cache matching happens at this block granularity; it is a shared prefix-caching concept, not a method-specific storage representation or a current-step decode cache.
_Avoid_: prompt-only block, decode cache, Quest page, token slot payload, method payload

**Prefix Cache Control Plane**:
A management surface for inspecting and influencing prefix-cache residency without exposing tree nodes, tensors, or method-specific payloads. It is separate from the runtime attention path.
_Avoid_: tree node API, payload API, debug-only object access

**Prefix Cache Subtree**:
The cached descendants rooted at a matched prefix-cache block path. Control-plane deletion and eviction-priority changes target subtrees, while preserving blocks that are still needed by active or locked runtime state.
_Avoid_: single-block deletion, arbitrary token range deletion, forced live-prefix deletion

**Prefix Cache Eviction Priority**:
A control-plane value on prefix-cache blocks that determines eviction preference. Negative values mean protected from safe deletion and eviction, zero is default, and positive values prefer eviction.
_Avoid_: soft lock, LRU score, cache hit score

**TP Prefix-Graph Support**:
A Sparse-VLLM capability where tensor-parallel execution, prefix KV reuse, and decode CUDA graph execution are validated together for a sparse method. It means the method has compatible cache lifecycle semantics across ranks, not merely that three configuration flags can be parsed together.
_Avoid_: allowlist-only support, config-toggle compatibility

**Rank-Local Mirrored Prefix Cache**:
A tensor-parallel prefix-cache model where every rank owns a local prefix index with the same stable block identities while storing rank-local KV payloads. It is not a shared rank-0 physical cache or a cross-rank payload transfer mechanism.
_Avoid_: central prefix cache, shared physical KV cache, rank-0-owned prefix cache

**Rank-0 Prefix Cache View**:
The user-visible tensor-parallel prefix-cache control view returned by rank 0 after all ranks have executed the control action successfully. It represents logical prefix-cache state, not every rank's physical KV slot or page layout.
_Avoid_: aggregated physical cache view, all-rank payload dump

**Prefix-Cache Stress Run**:
A stress validation run that repeatedly exercises real prefix-cache hits while the target runtime mode is enabled. A generic long-context or high-concurrency decode run is not a prefix-cache stress run unless reused prefix blocks are observed.
_Avoid_: graph-only stress, decode-only stress, no-hit prefix configuration
