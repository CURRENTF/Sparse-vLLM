from __future__ import annotations

from collections import Counter

import pytest

from benchmark.ruler_vt.tasks import (
    canonical_task,
    generate_non_vt_samples,
    resolve_task_config,
)


class WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


@pytest.mark.parametrize(
    "task",
    ["niah_single_1", "niah_multikey_2", "cwe", "fwe"],
)
def test_self_contained_ruler_tasks_are_deterministic_and_fill_context(task: str):
    tokenizer = WhitespaceTokenizer()
    kwargs = {
        "task": task,
        "tokenizer": tokenizer,
        "context_lengths": [1024],
        "samples_per_length": 2,
        "seed": 20260901,
        "config": resolve_task_config(task, None),
    }

    first = generate_non_vt_samples(**kwargs)
    second = generate_non_vt_samples(**kwargs)

    assert first == second
    assert [sample.index for sample in first] == [0, 1]
    for sample in first:
        assert sample.task == task
        assert 0.9 <= sample.length / sample.context_length <= 1.0
        assert sample.outputs
        assert all(answer.lower() in sample.input.lower() for answer in sample.outputs)
        if task.startswith("niah_"):
            assert sample.input.count(sample.outputs[0]) == 1


def test_cwe_and_fwe_answers_follow_independent_frequency_oracle():
    tokenizer = WhitespaceTokenizer()
    generated = {}
    for task in ("cwe", "fwe"):
        generated[task] = generate_non_vt_samples(
            task=task,
            tokenizer=tokenizer,
            context_lengths=[1024],
            samples_per_length=1,
            seed=7,
            config=resolve_task_config(task, None),
        )[0]

    cwe = generated["cwe"]
    cwe_context = cwe.input.split("Question:", 1)[0]
    cwe_counts = Counter(
        token.rstrip(".")
        for token in cwe_context.split()
        if token.startswith("ruler-")
    )
    assert {cwe_counts[word] for word in cwe.outputs} == {30}
    assert max(
        count for word, count in cwe_counts.items() if word not in cwe.outputs
    ) == 3

    fwe = generated["fwe"]
    fwe_context = fwe.input.split("Question:", 1)[0]
    fwe_counts = Counter(fwe_context.split())
    expected = [word for word, _count in fwe_counts.most_common(4) if word != "..."][:3]
    assert fwe.outputs == expected


def test_ruler_task_boundary_rejects_external_or_invalid_configs():
    assert canonical_task("ruler_vt") == "vt"
    with pytest.raises(ValueError, match="Unsupported RULER task"):
        canonical_task("qa_1")
    with pytest.raises(ValueError, match="freq_cw > freq_ucw"):
        resolve_task_config("cwe", {"freq_cw": 3, "freq_ucw": 3})
