import unittest

import torch

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.qwen2.qwen2_with_compress_inference import Qwen2KVCompress


class Qwen2LeftPaddingBatchTest(unittest.TestCase):
    def test_left_padded_batch_matches_single_prompt_logits(self):
        torch.manual_seed(0)
        config = KVQwen2Config(
            vocab_size=80,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            num_sink_tokens=1,
            num_recent_tokens=8,
            full_attn_layers="0,1",
            use_cluster=False,
            use_compression=False,
            chunk_prefill_size=3,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "eager"
        model = Qwen2KVCompress(config).eval()

        input_ids = torch.tensor(
            [
                [0, 0, 5, 6, 7],
                [8, 9, 10, 11, 12],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )

        with torch.no_grad():
            batch_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            ).logits[:, -1, :]
            single_0 = model(
                input_ids=torch.tensor([[5, 6, 7]], dtype=torch.long),
                attention_mask=torch.ones((1, 3), dtype=torch.long),
                use_cache=True,
            ).logits[0, -1, :]
            single_1 = model(
                input_ids=torch.tensor([[8, 9, 10, 11, 12]], dtype=torch.long),
                attention_mask=torch.ones((1, 5), dtype=torch.long),
                use_cache=True,
            ).logits[0, -1, :]

        self.assertTrue(torch.allclose(batch_logits[0], single_0, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(batch_logits[1], single_1, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
