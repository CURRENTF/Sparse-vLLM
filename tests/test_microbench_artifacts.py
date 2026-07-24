from types import SimpleNamespace

from benchmark.microbench import _resolved_engine_config


def test_resolved_engine_config_records_backend_and_jsonable_values():
    llm = SimpleNamespace(
        config=SimpleNamespace(
            vllm_sparse_method="deltakv",
            prefill_schedule_policy="long_bs1full_short_batch",
            chunk_prefill_size=4096,
            decode_cuda_graph=True,
            decode_cuda_graph_capture_sampling=False,
            deltakv_sparse_decode_backend="fa2",
            deltakv_triton_materialize_block_tokens=16,
            deltakv_triton_gather_heads_per_program=4,
            deltakv_triton_reconstruct_heads_per_program=2,
            full_layer_kv_quant_bits=4,
            kv_quant_bits=0,
            kv_quant_group_size=64,
            full_attn_layers=(0, 1, 2, 8),
            obs_layer_ids=[2, 8],
        )
    )

    resolved = _resolved_engine_config(llm)

    assert resolved["deltakv_sparse_decode_backend"] == "fa2"
    assert resolved["full_attn_layers"] == [0, 1, 2, 8]
    assert resolved["obs_layer_ids"] == [2, 8]
