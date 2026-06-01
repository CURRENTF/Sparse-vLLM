import unittest

import torch

from sparsevllm.layers.sampler import Sampler
from sparsevllm.sampling_params import SamplingParams


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
