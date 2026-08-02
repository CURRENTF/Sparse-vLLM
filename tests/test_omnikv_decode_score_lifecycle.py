from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.utils.context import reset_context, set_context


class _Manager:
    device = torch.device("cpu")

    def __init__(self, context_len: int = 6):
        self.context_len = int(context_len)

    def get_layer_batch_states(self, layer_idx):
        del layer_idx
        return SimpleNamespace(
            context_lens=torch.tensor([self.context_len], dtype=torch.int32),
            max_context_len=self.context_len,
            req_indices=torch.tensor([0], dtype=torch.int32),
        )


def _make_controller():
    layers = 4
    config = SimpleNamespace(
        vllm_sparse_method="omnikv",
        obs_layer_ids=[0, 2],
        full_attn_layers=[0, 2],
        runtime_layout=SimpleNamespace(
            is_full_attention=lambda layer_idx: 0 <= int(layer_idx) < layers,
            kv_layer_index=lambda layer_idx: int(layer_idx),
        ),
        hf_config=SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=8,
            num_attention_heads=2,
            head_dim=4,
            torch_dtype=torch.float32,
        ),
        tensor_parallel_size=1,
        num_sink_tokens=0,
        num_recent_tokens=1,
        decode_keep_tokens=2,
        sparse_attn_score_dtype="float32",
        decode_cuda_graph=True,
    )
    manager = _Manager()
    controller = SparseController(config, manager)
    seqs = [Sequence([1])]
    set_context(
        False,
        cache_manager=manager,
        is_long_text=True,
        seqs=seqs,
    )
    controller.prepare_forward(seqs, is_prefill=False)
    return controller


def teardown_function():
    reset_context()


def test_omnikv_observation_layers_share_one_decode_score_workspace():
    controller = _make_controller()
    score0 = controller.layer_batch_sparse_states[0].attn_score
    score2 = controller.layer_batch_sparse_states[2].attn_score

    assert score0 is not None and score2 is not None
    assert score0.untyped_storage().data_ptr() == score2.untyped_storage().data_ptr()
    assert score0.data_ptr() == score2.data_ptr()
    assert controller.layer_batch_sparse_states[1].attn_score is None
    assert controller.layer_batch_sparse_states[3].attn_score is None
    assert controller._decode_attn_score_buffers == {}
    assert controller._omnikv_decode_attn_score_buffer is not None


def test_omnikv_consumes_shared_raw_scores_before_the_next_observation_layer():
    controller = _make_controller()
    controller._update_dynamic_omnikv_indices = MagicMock()
    states = controller.layer_batch_sparse_states
    raw0 = states[0].attn_score
    raw2 = states[2].attn_score
    assert raw0 is not None and raw2 is not None

    layer0_logits = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, -7.0], [2.0, 1.0, 4.0, 3.0, 6.0, -8.0]]]
    )
    raw0.copy_(layer0_logits)
    controller.on_layer_end(0, SimpleNamespace(is_prefill=False))

    layer0_scores = states[0].attn_score
    expected = torch.full((1, 6), torch.finfo(torch.float32).min)
    expected[:, :5] = torch.softmax(layer0_logits[:, :, :5] * 0.5, dim=-1).amax(dim=1)
    torch.testing.assert_close(layer0_scores, expected)
    assert layer0_scores is not None and layer0_scores.dim() == 2
    assert raw2.dim() == 3

    raw2.fill_(9.0)
    torch.testing.assert_close(states[0].attn_score, expected)
    controller.on_layer_end(2, SimpleNamespace(is_prefill=False))
    assert states[2].attn_score is not None and states[2].attn_score.dim() == 2
    assert controller._update_dynamic_omnikv_indices.call_count == 2


def test_omnikv_graph_reset_and_keepalive_cover_the_shared_workspace():
    controller = _make_controller()
    shared = controller._omnikv_decode_attn_score_buffer
    assert shared is not None
    refs = {
        layer_idx: {"attn_score": state.attn_score}
        for layer_idx, state in controller.layer_batch_sparse_states.items()
    }
    shared.fill_(1.0)

    assert controller.reset_decode_attn_scores_for_graph(refs)
    assert torch.all(shared == -1e20)
    assert any(
        tensor.untyped_storage().data_ptr() == shared.untyped_storage().data_ptr()
        for tensor in controller.decode_cuda_graph_keepalive_tensors()
    )

    controller.clear_decode_attn_score_buffers()
    assert controller._omnikv_decode_attn_score_buffer is None
