import pytest

from scripts.validation.validate_qwen3_moe_hf_reference import _step_input_tokens


def test_hf_replay_chunks_prefill_and_advances_offset():
    tokens, offset = _step_input_tokens(
        stage="prefill",
        prompt_token_ids=[1, 2, 3, 4, 5],
        prompt_offset=0,
        chunk_size=3,
        next_decode_token=None,
    )

    assert tokens == [1, 2, 3]
    assert offset == 3


def test_hf_replay_uses_previous_sample_for_decode():
    tokens, offset = _step_input_tokens(
        stage="decode",
        prompt_token_ids=[1, 2, 3],
        prompt_offset=3,
        chunk_size=2,
        next_decode_token=99,
    )

    assert tokens == [99]
    assert offset == 3


def test_hf_replay_rejects_decode_without_sample():
    with pytest.raises(ValueError, match="no preceding sampled token"):
        _step_input_tokens(
            stage="decode",
            prompt_token_ids=[1],
            prompt_offset=1,
            chunk_size=1,
            next_decode_token=None,
        )
