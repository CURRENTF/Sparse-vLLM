import unittest

from sparsevllm.configs.runtime_params import normalize_runtime_params


class RuntimeParamNormalizationTest(unittest.TestCase):
    def test_sparsevllm_normalizes_canonical_runtime_params(self):
        normalized = normalize_runtime_params(
            {
                "sparse_method": "deltakv",
                "deltakv_checkpoint_path": "/tmp/compressor",
                "engine_prefill_chunk_size": 512,
            },
            backend="sparsevllm",
        )

        self.assertEqual(
            normalized.infer_config,
            {
                "vllm_sparse_method": "deltakv",
                "deltakv_path": "/tmp/compressor",
                "chunk_prefill_size": 512,
            },
        )

    def test_legacy_runtime_names_raise(self):
        with self.assertRaisesRegex(ValueError, "Legacy runtime parameter"):
            normalize_runtime_params({"vllm_sparse_method": "x"}, backend="sparsevllm")

    def test_sparsevllm_vanilla_alias_maps_to_empty_method(self):
        normalized = normalize_runtime_params({"sparse_method": "vanilla"}, backend="sparsevllm")
        self.assertEqual(normalized.infer_config["vllm_sparse_method"], "")

    def test_sparsevllm_rkv_alias_maps_to_canonical_method(self):
        normalized = normalize_runtime_params({"sparse_method": "r-kv"}, backend="sparsevllm")
        self.assertEqual(normalized.infer_config["vllm_sparse_method"], "rkv")

        normalized = normalize_runtime_params({"sparse_method": "skip-kv"}, backend="sparsevllm")
        self.assertEqual(normalized.infer_config["vllm_sparse_method"], "skipkv")

if __name__ == "__main__":
    unittest.main()
