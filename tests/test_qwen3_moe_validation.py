from unittest.mock import Mock, patch

from scripts.validation.validate_qwen3_moe_pure_tp import _generate_sample


def test_generate_sample_records_model_failure():
    llm = Mock()
    llm.generate.side_effect = RuntimeError("generation failed")

    with patch(
        "scripts.validation.validate_qwen3_moe_pure_tp.SamplingParams"
    ) as sampling_params:
        row = _generate_sample(
            llm,
            sample_id="sample-1",
            prompt="prompt",
            max_tokens=8,
        )

    assert row["status"] == "model_failed"
    assert row["sample_id"] == "sample-1"
    assert row["prompt"] == "prompt"
    assert row["text"] == ""
    assert row["token_ids"] == []
    assert "generation failed" in row["error"]
    assert "RuntimeError: generation failed" in row["traceback"]
    sampling_params.assert_called_once_with(temperature=0.0, max_tokens=8)
