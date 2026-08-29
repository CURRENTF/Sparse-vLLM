from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest


class FakeFormatError(Exception):
    pass


def _response(
    chain_id,
    *,
    content="answer",
    tool_calls=None,
    query_error=None,
    finish_reason="stop",
    chain_status="created",
):
    message = SimpleNamespace(
        model_dump=lambda mode="json": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            "provider_specific_fields": {"chain_id": chain_id},
        }
    )
    return SimpleNamespace(
        chain_id=chain_id,
        chain_status=chain_status,
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ],
        query_error=query_error,
    )


def _load_model_module(monkeypatch, responses):
    class FakeLitellmModel:
        def __init__(self, **kwargs):
            del kwargs
            self.config = SimpleNamespace(
                model_name="openai/test-model",
                model_kwargs={
                    "api_base": "http://127.0.0.1:18000/v1",
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
            )
            self.calls = []

        def query(self, messages, **kwargs):
            response = self._query(
                self._prepare_messages_for_api(messages),
                **kwargs,
            )
            query_error = getattr(response, "query_error", None)
            if query_error is not None:
                raise query_error
            return {
                "role": "assistant",
                "content": "answer",
                "extra": {},
            }

        def _query(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return responses.pop(0)

        def _prepare_messages_for_api(self, messages):
            return messages

    minisweagent = ModuleType("minisweagent")
    models = ModuleType("minisweagent.models")
    exceptions = ModuleType("minisweagent.exceptions")
    litellm_model = ModuleType("minisweagent.models.litellm_model")
    exceptions.FormatError = FakeFormatError
    litellm_model.BASH_TOOL = {
        "type": "function",
        "function": {"name": "bash", "parameters": {"type": "object"}},
    }
    litellm_model.LitellmModel = FakeLitellmModel
    monkeypatch.setitem(sys.modules, "minisweagent", minisweagent)
    monkeypatch.setitem(sys.modules, "minisweagent.models", models)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.models.litellm_model",
        litellm_model,
    )
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmark"
        / "swe_bench_lite"
        / "model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_sparsevllm_swe_model",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chain_model_creates_then_resumes_one_chain(monkeypatch):
    responses = [
        _response(
            "chain-a",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"true"}',
                    },
                }
            ],
        ),
        _response("chain-a", content="done"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEVLLM_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()

    first_messages = [{"role": "user", "content": "first"}]
    assistant = {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"true"}',
                },
            }
        ],
        "provider_specific_fields": {"chain_id": "chain-a"},
    }
    model.query(first_messages)
    model.query(
        [
            *first_messages,
            assistant,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "ok",
            },
        ]
    )

    first_extra = model.calls[0][1]["extra_body"]
    second_extra = model.calls[1][1]["extra_body"]
    assert first_extra == {
        "thinking": {"type": "disabled"},
        "chain_id": None,
        "preserve_thinking": True,
    }
    assert second_extra == {
        "thinking": {"type": "disabled"},
        "chain_id": "chain-a",
        "preserve_thinking": True,
        "chain_append_start": 2,
    }


def test_chain_model_fails_when_server_omits_chain_id(monkeypatch):
    module = _load_model_module(
        monkeypatch,
        [SimpleNamespace(chain_id=None, model_extra={}, choices=[])],
    )
    monkeypatch.setenv("SPARSEVLLM_CHAIN_CACHE", "true")
    model = module.SparseVLLMLitellmModel()

    with pytest.raises(RuntimeError, match="without a chain_id"):
        model.query([{"role": "user", "content": "first"}])

def test_non_chain_model_does_not_send_chain_id(monkeypatch):
    module = _load_model_module(
        monkeypatch,
        [SimpleNamespace(chain_id=None, choices=[])],
    )
    monkeypatch.delenv("SPARSEVLLM_CHAIN_CACHE", raising=False)
    model = module.SparseVLLMLitellmModel()

    model.query([{"role": "user", "content": "first"}])

    assert "extra_body" not in model.calls[0][1]


def test_non_chain_model_prunes_once_and_verifies_next_turn_reuse(
    monkeypatch,
    tmp_path,
):
    module = _load_model_module(
        monkeypatch,
        [SimpleNamespace(chain_id=None, choices=[]), SimpleNamespace(chain_id=None, choices=[])],
    )
    events = tmp_path / "prune.jsonl"
    monkeypatch.setenv("SPARSEVLLM_PREFIX_PRUNE_POLICY", "snapkv_global")
    monkeypatch.setenv("SPARSEVLLM_PREFIX_PRUNE_TRIGGER_TOKENS", "4096")
    monkeypatch.setenv("SPARSEVLLM_PREFIX_PRUNE_RANGE_START", "512")
    monkeypatch.setenv("SPARSEVLLM_PREFIX_PRUNE_RANGE_END", "4096")
    monkeypatch.setenv("SPARSEVLLM_PREFIX_PRUNE_KEEP_TOKENS", "1792")
    monkeypatch.setenv("SPARSEVLLM_PREFIX_PRUNE_EVENTS", str(events))
    model = module.SparseVLLMLitellmModel()
    matches = iter(
        [
            {"usable_tokens": 4096, "matched_tokens": 4096, "resident_kv_tokens": 4096},
            {"usable_tokens": 4096, "matched_tokens": 4096, "resident_kv_tokens": 2304},
            {"usable_tokens": 4200, "matched_tokens": 4096, "resident_kv_tokens": 2304},
        ]
    )
    monkeypatch.setattr(model, "_match_prefix", lambda _chat: next(matches))

    def request(method, path, body=None):
        if method == "POST":
            assert path == "/prefix_cache/prune"
            assert body["chat"]["model"] == "test-model"
            return {"prune_id": "job-1", "status": "queued"}
        assert path == "/prefix_cache/prune/job-1"
        return {
            "prune_id": "job-1",
            "status": "completed",
            "result": {"freed_device_slots": 1792, "quality_degraded": True},
        }

    monkeypatch.setattr(model, "_prefix_cache_request", request)
    first = [{"role": "user", "content": "first"}]
    model.query(first)
    model.query(
        [
            *first,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
    )

    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["prune_completed", "reuse_verified"]
    assert all(row["freed_device_slots"] == 1792 for row in rows)


def test_chain_model_rejects_rewritten_history(monkeypatch):
    responses = [
        _response("chain-a", content="invalid response"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEVLLM_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()

    model.query([{"role": "user", "content": "first"}])
    with pytest.raises(
        RuntimeError,
        match="previous assistant response changed",
    ):
        model.query(
            [
                {"role": "user", "content": "first"},
                {
                    "role": "user",
                    "content": "The prior response was invalid; retry.",
                },
            ]
        )

    assert model.calls[0][1]["extra_body"]["chain_id"] is None
    assert len(model.calls) == 1
    assert model._chain_id == "chain-a"


def test_chain_model_full_rerenders_after_length_finish(monkeypatch):
    responses = [
        _response("chain-a", finish_reason="length"),
        _response("chain-a", chain_status="resumed"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEVLLM_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    first_messages = [{"role": "user", "content": "first"}]

    model.query(first_messages)
    model.query(
        [
            *first_messages,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "continue"},
        ]
    )

    second_extra = model.calls[1][1]["extra_body"]
    assert second_extra["chain_id"] == "chain-a"
    assert "chain_append_start" not in second_extra


def test_chain_model_starts_new_chain_after_invalidation(monkeypatch):
    responses = [
        _response("chain-a", chain_status="invalidated"),
        _response("chain-b"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEVLLM_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()
    first_messages = [{"role": "user", "content": "first"}]

    model.query(first_messages)
    assert model._chain_id is None
    model.query(
        [
            *first_messages,
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "continue"},
        ]
    )

    assert model.calls[1][1]["extra_body"]["chain_id"] is None
    assert "chain_append_start" not in model.calls[1][1]["extra_body"]
    assert model._chain_id == "chain-b"


def test_chain_model_commits_state_only_after_successful_query(monkeypatch):
    responses = [
        _response(
            "chain-a",
            content=None,
            query_error=FakeFormatError("missing tool call"),
        ),
        _response("chain-b", content="recovered"),
    ]
    module = _load_model_module(monkeypatch, responses)
    monkeypatch.setenv("SPARSEVLLM_CHAIN_CACHE", "1")
    model = module.SparseVLLMLitellmModel()

    first_messages = [{"role": "user", "content": "first"}]
    with pytest.raises(FakeFormatError, match="missing tool call"):
        model.query(first_messages)

    assert model._chain_id is None
    assert model._last_request_messages is None
    assert model._last_response_message is None
    assert model._force_new_chain_reason == "format_error"
    assert model._recovery_chain_id == "chain-a"

    recovered = model.query(
        [
            *first_messages,
            {
                "role": "user",
                "content": "The prior response had no tool call; retry.",
            },
        ]
    )

    assert model.calls[0][1]["extra_body"]["chain_id"] is None
    assert model.calls[1][1]["extra_body"] == {
        "thinking": {"type": "disabled"},
        "chain_id": "chain-a",
        "preserve_thinking": True,
    }
    assert model._chain_id == "chain-b"
    assert model._recovery_chain_id is None
    assert model._force_new_chain_reason is None
    assert recovered["extra"]["chain_reset_reason"] == "format_error"
