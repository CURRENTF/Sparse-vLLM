# TP CUDA graph captures full decode forward

## Status

Accepted. Current compatibility checks live in `configs/cuda_graph.py` and
`method_registry.py`; this ADR records the execution boundary, not a frozen
method or hardware support list.

## Decision

Each TP rank captures and replays its own full decode forward, including the
collective operations used by that forward. This preserves model-layer
boundaries. A capture or execution failure must remain explicit rather than
silently selecting eager execution.

Sparse selection is TP-local: each rank selects tokens from its local heads
or KV heads without cross-rank sparse-index aggregation. Graph correctness
therefore means equivalence to the same TP eager/static path, not equivalence
to TP=1 or global-head selection. Sparse TP graph configurations emit a warning
about this distinction.

Sampling stays outside TP graph capture because worker ranks do not materialize
rank-0 gathered logits. Setting `decode_graph_capture_sampling=True` with TP
is rejected during configuration validation; the option is not silently reset.

Validation must distinguish graph capture/replay evidence, numerical correctness
against the same TP path, and matched performance measurements. Configuration
eligibility alone is not evidence that a model/topology has been validated.
