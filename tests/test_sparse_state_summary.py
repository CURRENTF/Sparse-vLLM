from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.engine.cache_manager.base import _debug_tensor_summary
from sparsevllm.engine.model_runner import ModelRunner
from sparsevllm.engine.sparse_controller import LayerBatchSparseState, SparseController


def test_debug_tensor_summary_is_order_sensitive_and_deterministic():
    first = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    same = first.clone()
    reordered = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32)

    assert _debug_tensor_summary(first) == _debug_tensor_summary(same)
    assert _debug_tensor_summary(first)["sha256"] != _debug_tensor_summary(reordered)[
        "sha256"
    ]


def test_debug_tensor_summary_supports_bfloat16_scalars():
    summary = _debug_tensor_summary(torch.tensor(1.5, dtype=torch.bfloat16))

    assert summary["shape"] == []
    assert summary["dtype"] == "torch.bfloat16"
    assert summary["numel"] == 1
    assert len(summary["sha256"]) == 64


def test_sparse_controller_summary_captures_selection_and_cache_state():
    controller = object.__new__(SparseController)
    controller.sparse_method = "quest"
    controller.layer_batch_sparse_states = {
        0: LayerBatchSparseState(
            active_indices=torch.tensor([[0, 2]], dtype=torch.int64),
            active_slots=torch.tensor([[7, 9]], dtype=torch.int32),
            context_lens=torch.tensor([2], dtype=torch.int32),
            max_context_len=2,
        ),
        1: LayerBatchSparseState(),
    }
    controller.debug_dynamic_selection = {"quest": {"0": {"calls": 1}}}
    controller.cache_manager = SimpleNamespace(
        debug_state_summary=lambda: {
            "cache_manager_class": "FakeCacheManager",
            "free_slot_stats": {"free_slots": 12},
        }
    )

    summary = controller.debug_state_summary()

    assert summary["sparse_method"] == "quest"
    assert list(summary["layers"]) == ["0"]
    assert summary["layers"]["0"]["tensors"]["active_indices"]["shape"] == [1, 2]
    assert summary["dynamic_selection"]["quest"]["0"]["calls"] == 1
    assert summary["cache"]["free_slot_stats"] == {"free_slots": 12}


def test_model_runner_gathers_one_debug_summary_per_world_rank():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.rank = 0
    runner.parallel_context = SimpleNamespace(
        world_rank=0,
        ep_rank=0,
        world=SimpleNamespace(process_group=object()),
    )
    runner.sparse_controller = SimpleNamespace(
        debug_state_summary=lambda: {"sparse_method": "", "layers": {}}
    )

    def gather(output, local, group):
        assert group is runner.parallel_context.world.process_group
        output[:] = [local, {"world_rank": 1, "ep_rank": 1, "state": local["state"]}]

    with (
        patch.object(runner, "_sync_tp_rpc_status") as sync_status,
        patch("sparsevllm.engine.model_runner.dist.all_gather_object", side_effect=gather),
    ):
        summaries = runner.debug_sparse_state_summaries()

    assert [summary["world_rank"] for summary in summaries] == [0, 1]
    assert summaries[0]["state"] == summaries[1]["state"]
    sync_status.assert_called_once_with("debug_sparse_state_summaries", None)
