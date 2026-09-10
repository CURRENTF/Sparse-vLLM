"""Protect official truncation parity and prepared-input reproducibility."""

import json
from functools import partial
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from benchmark.long_bench_v2.contracts import prepare_official_samples, render_prompt
from benchmark.long_bench_v2.preparation import prepare_chat, prepare_official_samples_parallel


UPSTREAM = Path(__file__).resolve().parents[1] / "benchmark/long_bench_v2/upstream"
TEMPLATE = "$DOC$\n$Q$\n$C_A$\n$C_B$\n$C_C$\n$C_D$"


@pytest.fixture
def local_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "[BOS]": 1, "x": 2, "y": 3, "z": 4}, unk_token="[UNK]",
    ))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]")
    tokenizer.chat_template = "{{ bos_token }} USER {{ messages[0]['content'] }} ASSISTANT"
    tokenizer.save_pretrained(tmp_path)
    return tokenizer, str(tmp_path)


def test_parallel_preparation_matches_serial_with_exact_source_order(local_tokenizer):
    # Real spawned workers must preserve inputs and metadata despite different sample costs.
    tokenizer, path = local_tokenizer
    rows = [{**sample(), "_id": str(i), "context": "x y z " * size}
            for i, size in enumerate([500, 1, 10, 300, 2, 40])]
    serial = prepare_official_samples(
        rows, template=TEMPLATE, tokenizer=tokenizer,
        prepare_chat=partial(prepare_chat, tokenizer, no_chat_template=False),
        truncate_max_tokens=31, max_prompt_tokens=100,
    )
    parallel = prepare_official_samples_parallel(
        rows, tokenizer_path=path, template=TEMPLATE, no_chat_template=False,
        truncate_max_tokens=31, max_prompt_tokens=100, workers=2,
    )
    assert parallel == serial


def test_parallel_worker_failure_is_propagated(local_tokenizer):
    # A worker failure must invalidate the run instead of dropping the failing sample.
    _, path = local_tokenizer
    with pytest.raises(ValueError, match="does not truncate again"):
        prepare_official_samples_parallel(
            [sample(), {**sample(), "_id": "two"}], tokenizer_path=path,
            template=TEMPLATE, no_chat_template=False,
            truncate_max_tokens=31, max_prompt_tokens=1, workers=2,
        )


class CharacterTokenizer:
    bos_token = "<bos>"

    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=False):
        assert skip_special_tokens
        return "".join(chr(token) for token in tokens if token != 0)


def sample():
    return dict(_id="one", domain="test", sub_domain="test", difficulty="easy",
                length="short", context="abcdefghijklmnopqrstuvwxyz", question="Q?",
                choice_A="a", choice_B="b", choice_C="c", choice_D="d", answer="A")


def test_inline_json_exceeding_filename_limit_never_accesses_filesystem(monkeypatch):
    # Observed H2O failure: stat() on a long inline object raises ENAMETOOLONG.
    from benchmark.long_bench_v2.pred import _load_json_object

    config = {"description": "x" * 4096, "decode_graph": True}

    def unexpected_stat(*args, **kwargs):
        pytest.fail("Inline JSON must not be treated as a filename")

    with monkeypatch.context() as context:
        context.setattr(Path, "stat", unexpected_stat)
        assert _load_json_object(json.dumps(config, indent=2)) == config
        with pytest.raises(ValueError, match="Invalid inline JSON"):
            _load_json_object('{"broken":' + ' ' * 4096)


def test_json_file_loading_and_object_validation(tmp_path):
    from benchmark.long_bench_v2.pred import _load_json_object

    path = tmp_path / "config.json"
    path.write_text('{"tensor_parallel_size": 1}')
    assert _load_json_object(str(path)) == {"tensor_parallel_size": 1}
    with pytest.raises(ValueError, match="object"):
        _load_json_object('[1, 2]')


def test_invalid_runtime_config_fails_before_dataset_loading(monkeypatch, tmp_path):
    # Bad configs should fail before expensive tokenization of the full dataset.
    from benchmark.long_bench_v2 import pred

    monkeypatch.setattr(sys, "argv", [
        "pred.py", "--model-path", "unused", "--data-path", str(tmp_path / "missing.json"),
        "--output-dir", str(tmp_path / "out"), "--all-samples",
        "--overflow-policy", "official-middle", "--truncate-max-tokens", "120000",
        "--sparse-method", "h2o", "--hyper-param-json", '{"sparse_method": "vanilla"}',
    ])
    with pytest.raises(ValueError, match="conflicting sparse_method"):
        pred.main()
    status = json.loads((tmp_path / "out/run_status.json").read_text())
    assert status["phase"] == "input"


@pytest.mark.parametrize("limit", [1, 8, 9, 38, 100])
def test_official_middle_matches_executed_upstream_query(monkeypatch, limit):
    # Execute the pinned oracle: catches stage-order, special-token and odd-budget drift.
    monkeypatch.chdir(UPSTREAM)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=object))
    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace())
    upstream = runpy.run_path(str(UPSTREAM / "pred.py"))
    query = upstream["query_llm"]
    query.__globals__["model_map"] = {"test": "test"}
    query.__globals__["maxlen_map"] = {"test": limit}
    observed = {}

    def create(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A"))])

    tokenizer = CharacterTokenizer()
    query(render_prompt(TEMPLATE, sample()), "test", tokenizer,
          client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    expected_content = observed["messages"][0]["content"]

    def chat(text):
        wrapped = "USER:" + text + ":ASSISTANT"
        return wrapped, tokenizer.encode(wrapped)

    result = prepare_official_samples(
        [sample()], template=TEMPLATE, tokenizer=tokenizer, prepare_chat=chat,
        truncate_max_tokens=limit, max_prompt_tokens=200,
    )[0]
    assert result["prompt"] == chat(expected_content)[0]
    assert result["prompt_token_ids"] == chat(expected_content)[1]
    assert result["truncated"] == (len(tokenizer.encode(render_prompt(TEMPLATE, sample()))) > limit)


def test_official_middle_rejects_chat_overflow_without_second_truncation():
    # Chat wrapping/retokenization can exceed runtime capacity despite pre-chat truncation.
    with pytest.raises(ValueError, match="does not truncate again"):
        prepare_official_samples(
            [sample()], template=TEMPLATE, tokenizer=CharacterTokenizer(),
            prepare_chat=lambda text: (text, list(range(20))),
            truncate_max_tokens=8, max_prompt_tokens=10,
        )


def test_full_preparation_reuse_binds_official_truncation_budget(monkeypatch, tmp_path):
    # A prepared export must not silently override a changed experimental input budget.
    from benchmark.long_bench_v2 import pred

    monkeypatch.setattr(pred.AutoTokenizer, "from_pretrained", lambda *a, **kw: CharacterTokenizer())
    from benchmark.long_bench_v2 import preparation
    monkeypatch.setattr(preparation, "build_chat", lambda tokenizer, prompt, *a, **kw: "USER:" + prompt)
    data = tmp_path / "data.json"
    rows = [sample(), {**sample(), "_id": "two"}]
    data.write_text(json.dumps(rows))
    template = tmp_path / "prompt.txt"
    template.write_text(TEMPLATE)
    common = ["pred.py", "--model-path", str(tmp_path), "--data-path", str(data),
              "--prompt-template", str(template), "--all-samples", "--prepare-only",
              "--preprocess-workers", "1",
              "--overflow-policy", "official-middle", "--max-model-len", "200",
              "--max-new-tokens", "10"]
    original = tmp_path / "original"
    monkeypatch.setattr(sys, "argv", common + ["--truncate-max-tokens", "8", "--output-dir", str(original)])
    assert pred.main() == 0
    config = json.loads((original / "resolved_config.json").read_text())
    assert config["source_samples"] == config["selected_samples"] == len(rows)
    assert config["truncated_samples"] == len(rows)
    prepared = original / "prepared_samples.json"
    exported = json.loads(prepared.read_text())
    assert [item["sample"]["_id"] for item in exported["samples"]] == [row["_id"] for row in rows]
    for budget, expected_status in [(8, 0), (9, 1)]:
        output = tmp_path / f"reuse-{budget}"
        monkeypatch.setattr(sys, "argv", common + ["--truncate-max-tokens", str(budget),
                            "--prepared-samples", str(prepared), "--output-dir", str(output)])
        if expected_status:
            with pytest.raises(ValueError, match="truncate_max_tokens"):
                pred.main()
            status = json.loads((output / "run_status.json").read_text())
            assert "truncate_max_tokens" in status["error"]
        else:
            assert pred.main() == 0
