import unittest

import torch

from sparsevllm.engine.cache_manager.raw_kv_offload import RawKVOffloadBuffer


class RawKVOffloadBufferTest(unittest.TestCase):
    def test_chunked_mode_is_default(self):
        buffer = RawKVOffloadBuffer(pin_memory=False)

        self.assertEqual(buffer.mode, "chunked")

    def test_put_and_restore_prefix_for_all_modes(self):
        for mode in ("chunked", "contiguous"):
            with self.subTest(mode=mode):
                buffer = RawKVOffloadBuffer(pin_memory=False, mode=mode)
                buffer.ensure_entry(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    total_len=5,
                    k_shape_tail=(2, 4),
                    v_shape_tail=(2, 4),
                    dtype=torch.float32,
                )
                k0 = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
                v0 = k0 + 100
                k1 = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) + 24
                v1 = k1 + 100

                buffer.put_range(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    start=0,
                    k=k0,
                    v=v0,
                )
                buffer.put_range(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    start=3,
                    k=k1,
                    v=v1,
                )

                k_out, v_out = buffer.restore_prefix(
                    layer_idx=3,
                    row_idx=1,
                    kind="sparse_pre_rope",
                    end=5,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

                self.assertTrue(torch.equal(k_out, torch.cat([k0, k1], dim=0)))
                self.assertTrue(torch.equal(v_out, torch.cat([v0, v1], dim=0)))

    def test_chunked_mode_rejects_gaps(self):
        buffer = RawKVOffloadBuffer(pin_memory=False, mode="chunked")
        buffer.ensure_entry(
            layer_idx=0,
            row_idx=0,
            kind="full_post_rope",
            total_len=4,
            k_shape_tail=(1, 2),
            v_shape_tail=(1, 2),
            dtype=torch.float32,
        )

        with self.assertRaisesRegex(RuntimeError, "cannot leave a gap"):
            buffer.put_range(
                layer_idx=0,
                row_idx=0,
                kind="full_post_rope",
                start=2,
                k=torch.zeros((1, 1, 2), dtype=torch.float32),
                v=torch.zeros((1, 1, 2), dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
