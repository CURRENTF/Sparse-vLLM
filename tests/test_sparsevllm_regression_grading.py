import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmark.sparsevllm_regression.grading import (
    grade_logits,
    grade_memory,
    grade_perf,
    grade_quality,
    grade_stress,
    grade_stress_v2,
)
from benchmark.sparsevllm_regression.longbench_mini import select_longbench_mini_samples
from benchmark.long_bench.pred import build_chat, strip_thinking_content
from benchmark.sparsevllm_regression.manifest import (
    ManifestError,
    REQUIRED_METHODS,
    REQUIRED_MODELS,
    compressor_path_for,
    load_manifest,
    missing_runtime_inputs,
    resolve_manifest_paths,
    validate_manifest,
)
from benchmark.sparsevllm_regression.run_suite import _perf_command, _scbench_command, _stress_command, _stress_v2_command
from benchmark.sparsevllm_regression.run_suite import _quality_command
from sparsevllm.engine.cache_manager.base import CacheManager


class FakeTokenizer:
    bos_token = None
    chat_template = None

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(range(len(str(text).split())))


class FakeCacheManager(CacheManager):
    def __init__(self):
        hf_config = types.SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=4,
            hidden_size=4,
            num_attention_heads=1,
            torch_dtype=torch.float16,
        )
        config = types.SimpleNamespace(
            hf_config=hf_config,
            max_model_len=10,
            max_num_seqs_in_batch=2,
            num_kvcache_slots=16,
        )
        super().__init__(config, rank=0, world_size=1)
        self.kv_cache = torch.empty((2, 2, 16, 1, 4), dtype=torch.float16)
        self.buffer_req_to_token_slots = torch.empty((2, 10), dtype=torch.int32)
        self.latent_scales = torch.empty((2, 16, 1), dtype=torch.float16)
        self.row_seq_lens = np.array([3, 2], dtype=np.int32)
        self._num_free_slots = 8

    @property
    def num_free_slots(self):
        return self._num_free_slots

    def allocate_kv_cache(self):
        raise NotImplementedError

    def get_layer_batch_states(self, layer_idx):
        raise NotImplementedError

    def get_layer_kv_cache(self, layer_idx):
        raise NotImplementedError

    def get_layer_store_view(self, layer_idx):
        raise NotImplementedError

    def get_layer_compute_tensors(self, layer_idx, selection=None):
        del selection
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        raise NotImplementedError

    def free_seq(self, seq_id):
        raise NotImplementedError

    def free_part_slots(self, layer_idx, seq, keep_indices):
        raise NotImplementedError

    def _prepare_prefill(self, seqs):
        raise NotImplementedError

    def _prepare_decode(self, seqs):
        raise NotImplementedError


class SparseVLLMRegressionGradingTest(unittest.TestCase):
    def test_manifest_covers_required_models_methods_and_artifacts(self):
        manifest = load_manifest()
        self.assertLessEqual(REQUIRED_MODELS, set(manifest["models"]))
        self.assertLessEqual(REQUIRED_METHODS, set(manifest["methods"]))

        broken = copy.deepcopy(manifest)
        broken["methods"].pop("quest")
        with self.assertRaises(ManifestError):
            validate_manifest(broken)

    def test_manifest_requires_absolute_vanilla_quality_floor(self):
        manifest = copy.deepcopy(load_manifest())
        manifest["quality"].pop("minimum_vanilla_score")

        with self.assertRaisesRegex(ManifestError, "minimum_vanilla_score"):
            validate_manifest(manifest)

    def test_model_specific_compressor_path_resolution(self):
        manifest = copy.deepcopy(load_manifest())
        method = manifest["methods"]["deltakv"]
        method.pop("compressor_path_env", None)
        validate_manifest(manifest)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {}
            for model_id, model in manifest["models"].items():
                model_path = root / f"{model_id}-model"
                model_path.mkdir()
                env[model["model_path_env"]] = str(model_path)

            qwen3_compressor = root / "qwen3-compressor"
            qwen3_compressor.mkdir()
            global_compressor = root / "global-compressor"
            global_compressor.mkdir()
            env["DELTAKV_COMPRESSOR_QWEN3_4B"] = str(qwen3_compressor)
            env["DELTAKV_COMPRESSOR_PATH"] = str(global_compressor)

            with patch.dict(os.environ, env, clear=False):
                resolved = resolve_manifest_paths(manifest)

        resolved_method = resolved["methods"]["deltakv"]
        qwen3_model = resolved["models"]["qwen3_4b"]
        qwen25_model = resolved["models"]["qwen25_7b"]

        self.assertEqual(compressor_path_for(qwen3_model, resolved_method), str(qwen3_compressor))
        self.assertIsNone(compressor_path_for(qwen25_model, resolved_method))
        self.assertIn(
            "DELTAKV_COMPRESSOR_QWEN25_7B",
            missing_runtime_inputs(resolved, "qwen25_7b", "deltakv"),
        )

    def test_manifest_perf_and_stress_policy(self):
        manifest = load_manifest()

        self.assertEqual(manifest["performance"]["output_len"], 256)
        self.assertEqual(manifest["performance"]["lengths"], [32000, 64000])
        self.assertEqual(manifest["performance"]["batch_sizes"], [4, 8])
        self.assertEqual(manifest["stress"]["request_counts"], [80])
        self.assertEqual(manifest["stress"]["max_num_seqs_in_batch"], 80)
        self.assertEqual(manifest["stress"]["max_decoding_seqs"], 80)
        self.assertEqual(manifest["stress_v2"]["workloads"], "shared_prefix,multiturn")
        self.assertEqual(manifest["stress_v2"]["sessions"], 8)
        self.assertEqual(manifest["stress_v2"]["max_active_requests"], 8)
        self.assertLess(manifest["stress_v2"]["user_min_len"], manifest["stress_v2"]["user_len"])
        self.assertEqual(manifest["scbench"]["model"], "qwen3_4b")
        self.assertEqual(manifest["scbench"]["methods"], ["vanilla", "omnikv", "quest"])
        self.assertEqual(manifest["scbench"]["tasks"], ["scbench_kv", "scbench_qa_eng"])
        self.assertEqual(manifest["scbench"]["batch_size"], 4)

    def test_omnikv_and_deltakv_full_layers_are_model_specific(self):
        manifest = load_manifest()
        model = {"model_path": "/tmp/model", "tokenizer_path": "/tmp/model"}
        expected = {
            "qwen25_7b": "0,2,4,11,16,22",
            "qwen3_4b": "0,1,3,9,13,16,21,28",
            "llama31_8b": "0,2,7,13,16,26",
        }

        for method_id in ("omnikv", "deltakv", "deltakv-less-memory", "deltakv-less-memory-cudagraph"):
            method = manifest["methods"][method_id]
            for model_id, full_layers in expected.items():
                with self.subTest(method_id=method_id, model_id=model_id):
                    cmd = _quality_command(
                        model_id=model_id,
                        method_id=method_id,
                        model=model,
                        method=method,
                        quality=manifest["quality"],
                        output_root=Path("/tmp/out"),
                    )
                    hyper_params = json.loads(cmd[cmd.index("--hyper_param") + 1])
                    self.assertEqual(hyper_params["full_attention_layers"], full_layers)

    def test_benchmark_commands_disable_decode_graph_for_unsupported_methods(self):
        manifest = load_manifest()
        model = {"model_path": "/tmp/model", "tokenizer_path": "/tmp/model"}
        performance = {
            "lengths": [16],
            "batch_sizes": [1],
            "output_len": 1,
            "decode_cuda_graph": True,
            "enforce_eager": False,
        }
        stress = {
            "length": 16,
            "request_counts": [1],
            "output_len": 1,
            "max_decode_steps_after_full": 1,
        }

        for command_builder in (_perf_command, _stress_command):
            for method_id, expected in (
                ("deltakv", True),
                ("deltakv-less-memory", True),
                ("deltakv-less-memory-cudagraph", True),
            ):
                with self.subTest(command_builder=command_builder.__name__, method_id=method_id):
                    kwargs = {
                        "model_id": "qwen25_7b",
                        "model": model,
                        "method_id": method_id,
                        "method": manifest["methods"][method_id],
                        "performance": performance,
                        "output_jsonl": Path("/tmp/out.jsonl"),
                    }
                    if command_builder is _stress_command:
                        kwargs["stress"] = stress
                    cmd = command_builder(**kwargs)
                    hyper_params = json.loads(cmd[cmd.index("--hyper_params") + 1])
                    self.assertIs(hyper_params["decode_cuda_graph"], expected)

    def test_scbench_regression_command_uses_batched_multiturn_subset(self):
        manifest = load_manifest()
        cmd = _scbench_command(
            manifest_path=Path("/tmp/manifest.json"),
            model_id="qwen3_4b",
            method_ids=["vanilla", "omnikv", "quest"],
            scbench=manifest["scbench"],
            output_dir=Path("/tmp/out"),
        )

        self.assertIn("scripts/benchmarks/run_scbench_sparsevllm_methods.py", cmd)
        self.assertEqual(cmd[cmd.index("--model_id") + 1], "qwen3_4b")
        self.assertEqual(cmd[cmd.index("--methods") + 1], "vanilla,omnikv,quest")
        self.assertEqual(cmd[cmd.index("--tasks") + 1], "scbench_kv,scbench_qa_eng")
        self.assertEqual(cmd[cmd.index("--batch_size") + 1], "4")
        self.assertEqual(cmd[cmd.index("--max_turns") + 1], "2")
        self.assertIn("--trust_remote_code", cmd)

    def test_stress_v2_command_uses_prefix_cache_serving_trace(self):
        manifest = load_manifest()
        cmd = _stress_v2_command(
            model_id="qwen3_4b",
            model={"model_path": "/tmp/model", "tokenizer_path": "/tmp/model"},
            method_id="omnikv",
            method=manifest["methods"]["omnikv"],
            stress_v2={**manifest["stress_v2"], "sessions": 2, "shared_prompts": 2},
            output_dir=Path("/tmp/stress-v2"),
        )

        self.assertIn("scripts/benchmarks/bench_prefix_cache.py", cmd)
        self.assertEqual(cmd[cmd.index("--cases") + 1], "prefix_omnikv")
        self.assertEqual(cmd[cmd.index("--workloads") + 1], "shared_prefix,multiturn")
        self.assertEqual(cmd[cmd.index("--session_prefix_min_len") + 1], "1024")
        self.assertEqual(cmd[cmd.index("--user_min_len") + 1], "128")
        self.assertEqual(cmd[cmd.index("--shared_suffix_min_len") + 1], "512")
        self.assertEqual(cmd[cmd.index("--max_active_requests") + 1], "8")
        self.assertEqual(cmd[cmd.index("--output_dir") + 1], "/tmp/stress-v2")

    def test_quality_grade_thresholds(self):
        self.assertEqual(grade_quality(50.0, 50.0, minimum_vanilla_score=25.0).grade, "A")
        self.assertEqual(grade_quality(50.0, 49.6, minimum_vanilla_score=25.0).grade, "B")
        self.assertEqual(grade_quality(50.0, 49.1, minimum_vanilla_score=25.0).grade, "C")
        self.assertEqual(grade_quality(50.0, 48.9, minimum_vanilla_score=25.0).grade, "D")
        self.assertEqual(grade_quality(50.0, 51.0, minimum_vanilla_score=25.0).grade, "A")

    def test_quality_grade_rejects_broken_vanilla_baseline(self):
        grade = grade_quality(20.37, 20.37, minimum_vanilla_score=25.0)

        self.assertEqual(grade.grade, "D")
        self.assertEqual(grade.status, "failed")
        self.assertIn("below minimum", grade.reason)

    def test_thinking_off_applies_chat_template_to_legacy_raw_prompt_tasks(self):
        class QwenTokenizer:
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "templated<think>\n"

        tokenizer = QwenTokenizer()

        prompt = build_chat(tokenizer, "classify this", "trec", thinking_mode="off")

        self.assertEqual(prompt, "templated<think>\n</think>\n")
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertEqual(
            build_chat(tokenizer, "classify this", "trec", no_chat_template=True, thinking_mode="off"),
            "classify this",
        )

    def test_strip_thinking_content_requires_complete_reasoning(self):
        self.assertEqual(
            strip_thinking_content("reasoning\n</think>\nfinal answer"),
            "final answer",
        )
        with self.assertRaisesRegex(ValueError, "ended before </think>"):
            strip_thinking_content("truncated reasoning")

    def test_logits_perf_memory_and_stress_grades(self):
        metrics = {
            "decode_steps": [
                {
                    "argmax_match": True,
                    "topk_overlap": {"5": {"ratio": 0.8}, "10": {"ratio": 0.9}},
                    "p99_abs_diff": 0.01,
                }
            ]
        }
        self.assertEqual(grade_logits(metrics, p99_threshold=0.1).grade, "A")
        self.assertEqual(grade_logits(None).grade, "N/A")
        self.assertEqual(grade_perf(1.1, graph_expected=True, graph_active=True).grade, "C")
        self.assertEqual(grade_perf(0.8, graph_expected=True, graph_active=True, require_speedup=False).grade, "A")
        self.assertEqual(grade_perf(2.1, graph_expected=True, graph_active=False).grade, "D")
        self.assertEqual(grade_memory(expected_savings=0.3, observed_savings=0.21).grade, "B")
        self.assertEqual(grade_memory(expected_savings=0.3, observed_savings=0.05).grade, "D")
        self.assertEqual(
            grade_stress(
                completed=True,
                crashed=False,
                preemptions=0,
                full_admission_window=True,
                utilization_ok=True,
            ).grade,
            "A",
        )
        self.assertEqual(
            grade_stress(
                completed=True,
                crashed=False,
                preemptions=2,
                full_admission_window=False,
                utilization_ok=False,
            ).grade,
            "C",
        )
        stress_v2_summary = {
            "status": "success",
            "cases": [
                {
                    "case": "prefix_full",
                    "status": "success",
                    "enable_prefix_caching": True,
                    "hit_requests": 3,
                    "total_cached_tokens": 1024,
                    "total_eligible_cache_tokens": 1024,
                    "eligible_cache_hit_rate": 1.0,
                    "min_prompt_tokens": 1024,
                    "max_prompt_tokens": 2048,
                    "unique_prompt_lengths": 2,
                }
            ],
        }
        self.assertEqual(grade_stress_v2(stress_v2_summary).grade, "A")
        no_hits = copy.deepcopy(stress_v2_summary)
        no_hits["cases"][0]["hit_requests"] = 0
        self.assertEqual(grade_stress_v2(no_hits).grade, "D")
        fixed_lengths = copy.deepcopy(stress_v2_summary)
        fixed_lengths["cases"][0]["min_prompt_tokens"] = 1024
        fixed_lengths["cases"][0]["max_prompt_tokens"] = 1024
        fixed_lengths["cases"][0]["unique_prompt_lengths"] = 1
        self.assertEqual(grade_stress_v2(fixed_lengths).grade, "D")

    def test_longbench_mini_selects_fixed_long_samples(self):
        data = [
            {"context": "short"},
            {"context": "one two three four five"},
            {"context": "one two three four five six"},
            {"context": "tiny"},
        ]
        selected, meta = select_longbench_mini_samples(
            data=data,
            tokenizer=FakeTokenizer(),
            dataset="lcc",
            prompt_format="{context}",
            min_prompt_tokens=5,
            samples_per_task=2,
            min_required_samples=2,
            no_chat_template=True,
        )
        self.assertEqual(meta["status"], "success")
        self.assertEqual([item.source_idx for item in selected], [1, 2])

        selected, meta = select_longbench_mini_samples(
            data=data,
            tokenizer=FakeTokenizer(),
            dataset="lcc",
            prompt_format="{context}",
            min_prompt_tokens=10,
            samples_per_task=2,
            min_required_samples=1,
            no_chat_template=True,
        )
        self.assertEqual(selected, [])
        self.assertEqual(meta["status"], "skipped_by_policy")

    def test_cache_manager_memory_accounting_fake_tensors(self):
        manager = FakeCacheManager()
        accounting = manager.memory_accounting()
        self.assertEqual(accounting["status"], "success")
        self.assertEqual(accounting["dense_baseline_bytes"], 512)
        self.assertEqual(accounting["kv_or_latent_tensor_bytes"], 512)
        self.assertEqual(accounting["slot_map_bytes"], 80)
        self.assertEqual(accounting["scale_min_metadata_bytes"], 64)
        self.assertEqual(accounting["logical_live_kv_bytes"], 160)
        self.assertEqual(accounting["allocated_tensor_bytes"], 656)
        self.assertLess(accounting["observed_savings"], 0)


if __name__ == "__main__":
    unittest.main()
