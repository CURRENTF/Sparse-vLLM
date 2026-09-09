"""Regression guards for paired external-engine input and server identity."""
import io
import json
from types import SimpleNamespace

import pytest

from benchmark.long_bench_v2 import external


def test_http_adapter_preserves_token_ids_and_sampling(monkeypatch, tmp_path):
    requests = []
    def open_response(request, timeout):
        requests.append(request)
        if isinstance(request, str):
            result = {"server_args": {"model_path": str(tmp_path), "disable_radix_cache": True}}
        else:
            result = [{"text": "The correct answer is C"}]
        return io.BytesIO(json.dumps(result).encode())
    monkeypatch.setattr(external, "urlopen", open_response)
    generate, _ = external.get_generate_api(
        engine="sglang-http", model_path=str(tmp_path), max_model_len=128, seed=7,
        engine_kwargs={}, server_url="http://localhost:12345")
    tokens = [[17, 31, 47]]
    assert generate(tokens, temperature=0, top_p=1, top_k=1, max_new_tokens=13,
                    eos_token_id=[2, 9]) == ["The correct answer is C"]
    payload = json.loads(requests[-1].data)
    assert payload["input_ids"] == tokens
    assert "text" not in payload
    assert payload["sampling_params"]["stop_token_ids"] == [2, 9]
    assert payload["sampling_params"]["max_new_tokens"] == 13


def test_http_adapter_rejects_wrong_model_before_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(external, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(
        {"server_args": {"model_path": str(tmp_path / "wrong"), "disable_radix_cache": True}}
    ).encode()))
    with pytest.raises(ValueError, match="model mismatch"):
        external.get_generate_api(engine="sglang-http", model_path=str(tmp_path),
                                  max_model_len=128, seed=7, engine_kwargs={},
                                  server_url="http://localhost:12345")


def test_vllm_adapter_does_not_retokenize_paired_inputs(monkeypatch, tmp_path):
    observed = {}
    class LLM:
        def __init__(self, **kwargs):
            observed["config"] = kwargs
        def generate(self, prompts, params, use_tqdm):
            observed.update(prompts=prompts, params=params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="The correct answer is A")])]
    fake = SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw,
                           __version__="test", __file__=str(tmp_path / "vllm.py"))
    monkeypatch.setitem(__import__("sys").modules, "vllm", fake)
    generate, _ = external.get_generate_api(engine="vllm", model_path=str(tmp_path),
        max_model_len=128, seed=7, engine_kwargs={}, server_url=None)
    generate([[3, 8, 21]], temperature=0, top_p=1, top_k=1, max_new_tokens=9, eos_token_id=[2])
    assert observed["prompts"] == [{"prompt_token_ids": [3, 8, 21]}]
    assert observed["params"]["stop_token_ids"] == [2]
    with pytest.raises(ValueError, match="may not override"):
        external.get_generate_api(engine="vllm", model_path=str(tmp_path),
            max_model_len=128, seed=7, engine_kwargs={"model": "wrong"}, server_url=None)


@pytest.mark.parametrize("method,options", [
    ("snapkv", {}),
    ("vanilla", {"compression_scorer": "snapkv", "compression_budget_tokens": 64}),
    ("h2o", {}),
])
def test_external_method_label_cannot_silently_describe_a_different_algorithm(method, options):
    with pytest.raises(ValueError, match="method label"):
        external.get_generate_api(engine="vllm", model_path="unused", max_model_len=128,
            seed=7, engine_kwargs=options, server_url=None, sparse_method=method)
