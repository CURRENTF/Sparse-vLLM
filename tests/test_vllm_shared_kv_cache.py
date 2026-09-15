"""External packed-cache ABI, dependency isolation and graph mutation contracts."""

import sys
from types import ModuleType

import pytest
import torch

from sparsevllm.kernels.external.vllm_cache import _source_module, shared_kv_cache_ops


def test_source_module_rejects_conflicting_installation(monkeypatch, tmp_path):
    name = "sparsevllm_cache_dependency_test"
    existing = ModuleType(name)
    existing.__file__ = str(tmp_path / "original.py")
    monkeypatch.setitem(sys.modules, name, existing)
    with pytest.raises(ValueError, match="different installation"):
        _source_module(name, tmp_path / "replacement.py")
    assert sys.modules[name] is existing


def test_failed_source_load_does_not_leave_partial_module(tmp_path):
    name = "sparsevllm_cache_failed_dependency_test"
    source = tmp_path / "broken.py"
    source.write_text("raise ImportError('missing dependency')\n")
    with pytest.raises(ImportError, match="missing dependency"):
        _source_module(name, source)
    assert name not in sys.modules


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_external_packed_writer_and_gather_replay_live_slots():
    # Protect page-boundary scale placement, padded writes, input updates and
    # the real compiler's dependency imports, which import-only tests miss.
    roots_before = {name: sys.modules.get(name) for name in ("vllm", "quack")}
    ops = shared_kv_cache_ops()
    if ops is None:
        pytest.skip("optional vLLM wheel is unavailable")
    writer, gather = ops
    torch.manual_seed(731)
    x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([0, 63, 64, 127, 128, 255, -1], device="cuda", dtype=torch.int64)
    cache = torch.zeros(4, 37440, device="cuda", dtype=torch.uint8)
    view = torch.as_strided(cache, (4, 64, 584), (37440, 584, 1))
    query = torch.zeros(7, 1, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(7, device="cuda", dtype=torch.int64)
    identity_rope = torch.cat([torch.ones(1, 32, device="cuda"), torch.zeros(1, 32, device="cuda")], -1)
    out = torch.empty(1, 256, 512, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([256], device="cuda", dtype=torch.int32)
    pages = torch.arange(4, device="cuda", dtype=torch.int32)[None]

    def run():
        writer(query, x, cache, slots, positions, identity_rope, 8, 1e-6, 64)
        gather(out, view, lengths, None, pages, 64, 0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(3):
        cache.zero_()
        if iteration == 1:
            slots.copy_(slots.roll(2))
            x.mul_(.5)
        elif iteration == 2:
            slots.fill_(-1)
        graph.replay()
        valid = slots >= 0
        groups = x[valid, :448].float().reshape(-1, 7, 64)
        scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)))
        rounded = (groups / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]
        expected = torch.zeros_like(out)
        expected[0, slots[valid]] = torch.cat([rounded.flatten(1), x[valid, 448:].float()], -1).bfloat16()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    graph.reset()
    assert {name: sys.modules.get(name) for name in roots_before} == roots_before
