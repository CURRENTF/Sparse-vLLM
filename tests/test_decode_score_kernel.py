from types import SimpleNamespace

import pytest
import torch


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton decode-score tests.",
)


def _reference(
    scores: torch.Tensor,
    candidate_lens: torch.Tensor,
    *,
    candidate_start: int,
    softmax_scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    candidate_scores = scores[:, :, candidate_start:]
    lengths = candidate_lens.long().clamp(
        min=0,
        max=candidate_scores.shape[-1],
    )
    positions = torch.arange(candidate_scores.shape[-1], device=scores.device)
    mask = positions[None, :] < lengths[:, None]
    logits = candidate_scores.float() * softmax_scale
    logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
    reduced = torch.softmax(logits, dim=-1).amax(dim=1).to(output_dtype)
    reduced = reduced.masked_fill(~mask, torch.finfo(output_dtype).min)
    output = torch.full(
        (scores.shape[0], scores.shape[-1]),
        torch.finfo(output_dtype).min,
        dtype=output_dtype,
        device=scores.device,
    )
    output[:, candidate_start:] = reduced
    return output


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("score_dtype", "output_dtype", "lens_dtype", "candidate_start"),
    [
        (torch.float32, torch.bfloat16, torch.int32, 5),
        (torch.bfloat16, torch.bfloat16, torch.int64, 0),
        (torch.float16, torch.float16, torch.int32, 5),
        (torch.float32, torch.float16, torch.int64, 5),
        (torch.bfloat16, torch.float32, torch.int32, 5),
    ],
)
def test_decode_score_normalization_matches_ragged_strided_reference(
    score_dtype: torch.dtype,
    output_dtype: torch.dtype,
    lens_dtype: torch.dtype,
    candidate_start: int,
) -> None:
    from sparsevllm.kernels.triton.decode_score import (
        decode_softmax_token_scores,
    )

    torch.manual_seed(31)
    batch, heads, width = 3, 7, 773
    score_storage = torch.randn(
        batch,
        heads * 2,
        width,
        dtype=score_dtype,
        device="cuda",
    )
    scores = score_storage[:, ::2, :]
    candidate_lens = torch.tensor(
        [width + 17, 257, -7],
        dtype=lens_dtype,
        device="cuda",
    )
    lse_storage = torch.empty(
        batch,
        heads + 3,
        dtype=torch.float32,
        device="cuda",
    )
    lse = lse_storage[:, :heads]
    output = torch.empty(batch, width, dtype=output_dtype, device="cuda")
    scale = 128**-0.5

    actual = decode_softmax_token_scores(
        scores,
        candidate_lens,
        candidate_start=candidate_start,
        softmax_scale=scale,
        output_dtype=output_dtype,
        lse_workspace=lse,
        output=output,
    )
    expected = _reference(
        scores,
        candidate_lens,
        candidate_start=candidate_start,
        softmax_scale=scale,
        output_dtype=output_dtype,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    assert torch.equal(
        actual[2],
        torch.full_like(actual[2], torch.finfo(output_dtype).min),
    )


@CUDA_REQUIRED
def test_decode_score_normalization_replays_cuda_graph_with_new_inputs() -> None:
    from sparsevllm.kernels.triton.decode_score import (
        decode_softmax_token_scores,
    )

    batch, heads, width = 2, 7, 773
    candidate_start = 5
    scale = 128**-0.5
    scores = torch.empty(
        batch, heads, width, dtype=torch.float32, device="cuda"
    )
    candidate_lens = torch.empty(batch, dtype=torch.int32, device="cuda")
    lse = torch.empty(batch, heads, dtype=torch.float32, device="cuda")
    output = torch.empty(batch, width, dtype=torch.bfloat16, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        scores.normal_()
        candidate_lens.copy_(
            torch.tensor([width - candidate_start, 257], device="cuda")
        )
        for _ in range(3):
            decode_softmax_token_scores(
                scores,
                candidate_lens,
                candidate_start=candidate_start,
                softmax_scale=scale,
                output_dtype=torch.bfloat16,
                lse_workspace=lse,
                output=output,
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_softmax_token_scores(
            scores,
            candidate_lens,
            candidate_start=candidate_start,
            softmax_scale=scale,
            output_dtype=torch.bfloat16,
            lse_workspace=lse,
            output=output,
        )

    torch.manual_seed(47)
    replay_scores = torch.randn_like(scores)
    replay_lens = torch.tensor([511, 0], dtype=torch.int32, device="cuda")
    scores.copy_(replay_scores)
    candidate_lens.copy_(replay_lens)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(
        replay_scores,
        replay_lens,
        candidate_start=candidate_start,
        softmax_scale=scale,
        output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-3)


@CUDA_REQUIRED
def test_runtime_reuses_decode_score_workspaces_during_graph_replay() -> None:
    from sparsevllm.engine.sparse_methods.base import SparseMethodRuntime

    class Runtime(SparseMethodRuntime):
        def needs_attention_score(self, layer_idx, step):
            del layer_idx, step
            return True

        def build_prefill_selection(self, request):
            raise NotImplementedError

        def build_decode_selection(self, request):
            raise NotImplementedError

    runtime = object.__new__(Runtime)
    runtime.config = SimpleNamespace(
        hf_config=SimpleNamespace(torch_dtype=torch.bfloat16)
    )
    runtime.attn_softmax_scale = 128**-0.5
    runtime._decode_attn_score_buffers = {}
    runtime._decode_score_lse_workspace = None
    runtime._decode_score_output_workspace = None

    batch, heads, width = 2, 7, 773
    candidate_start = 4
    scores = torch.randn(batch, heads, width, dtype=torch.float32, device="cuda")
    candidate_lens = torch.tensor([511, 257], dtype=torch.int32, device="cuda")
    output = runtime._decode_softmax_token_scores(
        scores,
        candidate_start=candidate_start,
        candidate_lens=candidate_lens,
    )
    output_ptr = output.data_ptr()
    keepalive_ptrs = {
        tensor.data_ptr() for tensor in runtime.decode_graph_keepalive_tensors()
    }
    assert len(keepalive_ptrs) == 2
    assert output_ptr in keepalive_ptrs

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = runtime._decode_softmax_token_scores(
            scores,
            candidate_start=candidate_start,
            candidate_lens=candidate_lens,
        )
    assert replay_output.data_ptr() == output_ptr

    torch.manual_seed(53)
    replay_scores = torch.randn_like(scores)
    replay_lens = torch.tensor([width - candidate_start, 0], device="cuda")
    scores.copy_(replay_scores)
    candidate_lens.copy_(replay_lens)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        replay_scores,
        replay_lens,
        candidate_start=candidate_start,
        softmax_scale=runtime.attn_softmax_scale,
        output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(replay_output, expected, rtol=2e-2, atol=2e-3)

    del graph
    runtime.clear_decode_attn_score_buffers()
    assert runtime.decode_graph_keepalive_tensors() == []
