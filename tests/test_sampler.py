import pickle
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sparsevllm.engine.sequence import Sequence
from sparsevllm.layers.sampler import Sampler
from sparsevllm.sampling_params import SamplingParams
from sparsevllm.sampling_params import resolve_eos_token_ids


class SamplerTest(unittest.TestCase):
    def test_all_greedy_skips_sampling_path(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])

        out = sampler(logits, temperatures=None, all_greedy=True)

        self.assertEqual(out.tolist(), [1, 2])

    def test_non_greedy_requires_temperatures(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 3.0, 2.0]])

        with self.assertRaises(ValueError):
            sampler(logits, temperatures=None, all_greedy=False)

    def test_non_greedy_requires_top_p(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        temperatures = torch.tensor([1.0])

        with self.assertRaises(ValueError):
            sampler(logits, temperatures=temperatures, top_ps=None, all_greedy=False)

    def test_top_p_can_keep_only_top_token(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        temperatures = torch.tensor([1.0])
        top_ps = torch.tensor([0.01])

        out = sampler(logits, temperatures=temperatures, top_ps=top_ps, all_greedy=False)

        self.assertEqual(out.tolist(), [1])

    def test_top_k_limits_sampling_candidates(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 5.0, 4.0]])
        temperatures = torch.tensor([1.0])
        top_ps = torch.tensor([1.0])
        top_ks = torch.tensor([1])

        out = sampler(logits, temperatures=temperatures, top_ps=top_ps, top_ks=top_ks, all_greedy=False)

        self.assertEqual(out.tolist(), [1])

    def test_sampling_params_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            SamplingParams(top_k=-1)
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)
        for presence_penalty in (-2.01, 2.01, float("nan")):
            with self.subTest(presence_penalty=presence_penalty):
                with self.assertRaises(ValueError):
                    SamplingParams(presence_penalty=presence_penalty)
        for repetition_penalty in (0.0, -1.0, float("nan")):
            with self.subTest(repetition_penalty=repetition_penalty):
                with self.assertRaises(ValueError):
                    SamplingParams(repetition_penalty=repetition_penalty)

    def test_sampling_penalty_defaults_are_neutral(self):
        params = SamplingParams()
        self.assertEqual(params.presence_penalty, 0.0)
        self.assertEqual(params.repetition_penalty, 1.0)

        sampler = Sampler()
        logits = torch.tensor([[1.0, 2.0]])
        output = sampler.apply_penalties(
            logits,
            presence_penalties=[0.0],
            repetition_penalties=[1.0],
            presence_token_ids=[None],
            repetition_token_ids=[None],
        )
        self.assertIs(output, logits)

    def test_resolve_eos_token_ids_uses_request_precedence(self):
        self.assertEqual(
            resolve_eos_token_ids((7, 8), (9,), fallback_eos_token_id=10),
            frozenset({7, 8}),
        )
        self.assertEqual(
            resolve_eos_token_ids((), (9,), fallback_eos_token_id=10),
            frozenset({9}),
        )
        self.assertEqual(
            resolve_eos_token_ids((), (), fallback_eos_token_id=10),
            frozenset({10}),
        )

    def test_presence_and_repetition_penalty_formulas(self):
        sampler = Sampler()
        logits = torch.tensor(
            [
                [4.0, -3.0, 2.0, 1.0],
                [4.0, -3.0, 2.0, 1.0],
            ]
        )

        output = sampler.apply_penalties(
            logits,
            presence_penalties=[0.5, -1.0],
            repetition_penalties=[2.0, 0.5],
            presence_token_ids=[torch.tensor([1, 2]), torch.tensor([2])],
            repetition_token_ids=[torch.tensor([0, 1]), torch.tensor([0, 1])],
        )

        torch.testing.assert_close(
            output,
            torch.tensor(
                [
                    [2.0, -6.5, 1.5, 1.0],
                    [8.0, -1.5, 3.0, 1.0],
                ]
            ),
        )
        torch.testing.assert_close(
            logits,
            torch.tensor(
                [
                    [4.0, -3.0, 2.0, 1.0],
                    [4.0, -3.0, 2.0, 1.0],
                ]
            ),
        )

    def test_penalties_own_fp32_output_for_low_precision_inputs(self):
        sampler = Sampler()
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[4.0, -3.0, 2.0]], dtype=dtype)
                original = logits.clone()

                output = sampler.apply_penalties(
                    logits,
                    presence_penalties=[0.5],
                    repetition_penalties=[2.0],
                    presence_token_ids=[torch.tensor([2])],
                    repetition_token_ids=[torch.tensor([0, 1])],
                )

                self.assertEqual(output.dtype, torch.float32)
                self.assertNotEqual(output.data_ptr(), logits.data_ptr())
                torch.testing.assert_close(output, torch.tensor([[2.0, -6.0, 1.5]]))
                torch.testing.assert_close(logits, original)

    def test_low_precision_penalty_conversion_uses_no_clone(self):
        class OperationRecorder(TorchDispatchMode):
            def __init__(self):
                self.operations = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.operations.append(func)
                return func(*args, **(kwargs or {}))

        sampler = Sampler()
        presence_ids = torch.tensor([2])
        repetition_ids = torch.tensor([0, 1])
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[4.0, -3.0, 2.0]], dtype=dtype)
                recorder = OperationRecorder()

                with recorder:
                    sampler.apply_penalties(
                        logits,
                        presence_penalties=[0.5],
                        repetition_penalties=[2.0],
                        presence_token_ids=[presence_ids],
                        repetition_token_ids=[repetition_ids],
                    )

                self.assertEqual(
                    recorder.operations.count(torch.ops.aten.clone.default),
                    0,
                )
                self.assertEqual(
                    recorder.operations.count(torch.ops.aten._to_copy.default),
                    1,
                )

    def test_penalty_token_scope_and_incremental_cache(self):
        seq = Sequence(
            [0, 0, 1],
            SamplingParams(
                max_tokens=4,
                presence_penalty=0.5,
                repetition_penalty=2.0,
            ),
        )

        repetition_ids = seq.repetition_penalty_token_ids_tensor(
            device=torch.device("cpu"),
            vocab_size=8,
        )
        self.assertEqual(repetition_ids.tolist(), [0, 1])
        self.assertIsNone(
            seq.presence_penalty_token_ids_tensor(
                device=torch.device("cpu"),
                vocab_size=8,
            )
        )

        repetition_data_ptr = repetition_ids.data_ptr()
        seq.append_token(2)
        seq.append_token(2)
        seq.append_token(0)

        repetition_ids = seq.repetition_penalty_token_ids_tensor(
            device=torch.device("cpu"),
            vocab_size=8,
        )
        presence_ids = seq.presence_penalty_token_ids_tensor(
            device=torch.device("cpu"),
            vocab_size=8,
        )
        self.assertEqual(repetition_ids.tolist(), [0, 1, 2])
        self.assertEqual(presence_ids.tolist(), [2, 0])
        self.assertEqual(repetition_ids.data_ptr(), repetition_data_ptr)

    def test_singleton_penalty_updates_reuse_buffers_without_torch_tensor(self):
        seq = Sequence(
            [0, 1],
            SamplingParams(
                max_tokens=4,
                presence_penalty=0.5,
                repetition_penalty=2.0,
            ),
        )
        repetition_ids = seq.repetition_penalty_token_ids_tensor(
            device=torch.device("cpu"),
            vocab_size=8,
        )
        repetition_data_ptr = repetition_ids.data_ptr()

        seq.append_token(2)
        with patch(
            "sparsevllm.engine.sequence.torch.tensor",
            side_effect=AssertionError("singleton update must not allocate a tensor"),
        ):
            repetition_ids = seq.repetition_penalty_token_ids_tensor(
                device=torch.device("cpu"),
                vocab_size=8,
            )
            presence_ids = seq.presence_penalty_token_ids_tensor(
                device=torch.device("cpu"),
                vocab_size=8,
            )
        presence_data_ptr = presence_ids.data_ptr()

        seq.append_token(3)
        with patch(
            "sparsevllm.engine.sequence.torch.tensor",
            side_effect=AssertionError("singleton update must not allocate a tensor"),
        ):
            repetition_ids = seq.repetition_penalty_token_ids_tensor(
                device=torch.device("cpu"),
                vocab_size=8,
            )
            presence_ids = seq.presence_penalty_token_ids_tensor(
                device=torch.device("cpu"),
                vocab_size=8,
            )

        self.assertEqual(repetition_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(presence_ids.tolist(), [2, 3])
        self.assertEqual(repetition_ids.data_ptr(), repetition_data_ptr)
        self.assertEqual(presence_ids.data_ptr(), presence_data_ptr)

    def test_greedy_sampling_uses_penalized_logits(self):
        sampler = Sampler()
        logits = sampler.apply_penalties(
            torch.tensor([[4.0, 3.0]]),
            presence_penalties=[0.0],
            repetition_penalties=[2.0],
            presence_token_ids=[None],
            repetition_token_ids=[torch.tensor([0])],
        )

        output = sampler(logits, temperatures=None, all_greedy=True)

        self.assertEqual(output.tolist(), [1])

    def test_sequence_ipc_preserves_penalty_scalars_without_history(self):
        seq = Sequence(
            [1, 2, 3],
            SamplingParams(
                max_tokens=2,
                presence_penalty=0.25,
                repetition_penalty=1.1,
            ),
        )
        seq.current_chunk_size = 1

        restored = pickle.loads(pickle.dumps(seq))

        self.assertEqual(restored.presence_penalty, 0.25)
        self.assertEqual(restored.repetition_penalty, 1.1)
        self.assertEqual(restored.token_ids, [1])
        self.assertIsNone(restored._presence_penalty_tokens)
        self.assertIsNone(restored._repetition_penalty_tokens)

    def test_top_k_zero_and_large_values_are_unlimited(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 5.0, 4.0], [1.0, 5.0, 4.0]])
        temperatures = torch.tensor([1.0, 1.0])
        top_ps = torch.tensor([0.01, 0.01])
        top_ks = torch.tensor([0, 99])

        out = sampler(logits, temperatures=temperatures, top_ps=top_ps, top_ks=top_ks, all_greedy=False)

        self.assertEqual(out.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
