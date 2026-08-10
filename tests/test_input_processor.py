from unittest.mock import Mock

import pytest

from sparsevllm.engine.input_processor import tokenize_text_prompt


def test_tokenize_text_prompt_preserves_token_ids():
    tokenizer = Mock()

    assert tokenize_text_prompt(tokenizer, [1, 2, 3]) == [1, 2, 3]
    tokenizer.encode.assert_not_called()


@pytest.mark.parametrize(
    "prompt",
    [
        {"image": "example.png", "text": "describe"},
        {"video": "example.mp4", "text": "describe"},
        {"mtp": True, "text": "continue"},
        [1, "2"],
    ],
)
def test_tokenize_text_prompt_rejects_structured_inputs(prompt):
    with pytest.raises(TypeError, match="text prompts only"):
        tokenize_text_prompt(Mock(), prompt)


def test_tokenize_text_prompt_rejects_empty_token_ids():
    with pytest.raises(ValueError, match="must not be empty"):
        tokenize_text_prompt(Mock(), [])


@pytest.mark.parametrize(
    ("bos_token", "prompt", "add_special_tokens"),
    [
        ("<s>", "hello", True),
        ("<s>", "<s>hello", False),
        (None, "hello", False),
    ],
)
def test_tokenize_text_prompt_handles_bos_once(
    bos_token,
    prompt,
    add_special_tokens,
):
    tokenizer = Mock(bos_token=bos_token)
    tokenizer.encode.return_value = [1]

    assert tokenize_text_prompt(tokenizer, prompt) == [1]
    tokenizer.encode.assert_called_once_with(
        prompt,
        add_special_tokens=add_special_tokens,
    )
