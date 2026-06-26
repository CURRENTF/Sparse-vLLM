import os
import unittest
from unittest.mock import patch

from deltakv.get_chat_api import (
    _pop_sparsevllm_generate_micro_batch_size,
    _sparsevllm_generate_outputs,
)


class SparseVllmGenerateMicroBatchTest(unittest.TestCase):
    def test_micro_batching_preserves_output_order(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def generate(self, prompts, sampling_params, use_tqdm=False):
                self.calls.append((list(prompts), sampling_params, use_tqdm))
                return [{"text": f"out:{prompt}"} for prompt in prompts]

        llm = FakeLLM()
        outputs = _sparsevllm_generate_outputs(
            llm,
            ["a", "b", "c", "d", "e"],
            sampling_params=object(),
            micro_batch_size=2,
        )

        self.assertEqual([out["text"] for out in outputs], ["out:a", "out:b", "out:c", "out:d", "out:e"])
        self.assertEqual([call[0] for call in llm.calls], [["a", "b"], ["c", "d"], ["e"]])
        self.assertTrue(all(call[2] is False for call in llm.calls))

    def test_micro_batch_config_is_popped_and_validated(self):
        config = {"sparsevllm_generate_micro_batch_size": "64", "tensor_parallel_size": 1}
        self.assertEqual(_pop_sparsevllm_generate_micro_batch_size(config), 64)
        self.assertEqual(config, {"tensor_parallel_size": 1})

        with patch.dict(os.environ, {"SPARSEVLLM_GENERATE_MICRO_BATCH_SIZE": "32"}):
            self.assertEqual(_pop_sparsevllm_generate_micro_batch_size({}), 32)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            _pop_sparsevllm_generate_micro_batch_size({"sparsevllm_generate_micro_batch_size": 0})


if __name__ == "__main__":
    unittest.main()
