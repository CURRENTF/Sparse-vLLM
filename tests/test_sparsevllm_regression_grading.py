import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmark.sparsevllm_regression.grading import (
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
    runtime_support_reason,
    validate_manifest,
)
from benchmark.sparsevllm_regression.run_suite import (
    _perf_command,
    _require_successful_perf_matrix,
    _require_synchronized_step_timing,
    _scbench_command,
    _stress_command,
    _stress_v2_command,
    main as run_suite_main,
)
from benchmark.sparsevllm_regression.run_suite import _quality_command
from sparsevllm.config import RuntimeLayout
from sparsevllm.engine.cache_manager.base import CacheManager
from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.method_registry import (
    H2O_SUPPORTED_MODEL_TYPES,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    get_default_prefill_schedule_policy,
)


def _single_process_parallel_context() -> ParallelContext:
    group = ParallelGroup(process_group=None, ranks=(0,), rank=0, size=1)
    return ParallelContext(world=group, tensor=group, expert=group, data=group)


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
            runtime_layout=RuntimeLayout.dense(2),
            max_model_len=10,
            max_num_seqs_in_gpu=2,
            max_num_seqs_in_batch=2,
            num_kvcache_slots=16,
        )
        super().__init__(config, _single_process_parallel_context())
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

    def test_manifest_uses_policy_specific_prefill_controls(self):
        manifest = load_manifest()
        self.assertEqual(
            manifest["methods"]["vanilla"]["config"]["engine_prefill_chunk_size"],
            96 * 1024,
        )

        for method_id, method in manifest["methods"].items():
            config = method["config"]
            policy = get_default_prefill_schedule_policy(method["sparse_method"])
            with self.subTest(method=method_id, policy=policy):
                if policy == PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH:
                    self.assertEqual(
                        config["long_prefill_offload_threshold"],
                        64 * 1024,
                    )
                    if "engine_prefill_chunk_size" in config:
                        self.assertLessEqual(
                            config["engine_prefill_chunk_size"],
                            config["long_prefill_offload_threshold"],
                        )
                else:
                    self.assertGreater(config["engine_prefill_chunk_size"], 0)

    def test_manifest_accepts_chunk_size_below_long_prefill_boundary(self):
        manifest = copy.deepcopy(load_manifest())
        pyramid_config = manifest["methods"]["pyramidkv"]["config"]
        pyramid_config["engine_prefill_chunk_size"] = 4096

        validate_manifest(manifest)

    def test_manifest_rejects_chunk_size_above_long_prefill_boundary(self):
        manifest = copy.deepcopy(load_manifest())
        pyramid_config = manifest["methods"]["pyramidkv"]["config"]
        pyramid_config["engine_prefill_chunk_size"] = (
            pyramid_config["long_prefill_offload_threshold"] + 1
        )

        with self.assertRaisesRegex(
            ManifestError,
            "engine_prefill_chunk_size must be <= long_prefill_offload_threshold",
        ):
            validate_manifest(manifest)

    def test_h2o_manifest_declares_supported_models_tp_runtime_matrix(self):
        manifest = load_manifest()
        method = manifest["methods"]["h2o"]
        self.assertEqual(
            method["supported_model_families"],
            ["qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "llama", "minimax_m2"],
        )
        self.assertEqual(
            set(method["supported_model_families"]),
            set(H2O_SUPPORTED_MODEL_TYPES),
        )
        self.assertEqual(method["supported_tensor_parallel_sizes"], [1, 2])
        self.assertEqual(method["performance"]["minimum_prefill_speedup"], 1.0)
        self.assertIsNone(
            runtime_support_reason(
                manifest,
                "qwen25_7b",
                "h2o",
                tensor_parallel_sizes=(1,),
            )
        )
        self.assertIsNone(
            runtime_support_reason(
                manifest, "qwen3_4b", "h2o", tensor_parallel_sizes=(1,)
            )
        )
        self.assertIsNone(
            runtime_support_reason(
                manifest,
                "qwen25_7b",
                "h2o",
                tensor_parallel_sizes=(2,),
            ),
        )
        self.assertIn(
            "tensor_parallel_size",
            runtime_support_reason(
                manifest,
                "qwen25_7b",
                "h2o",
                tensor_parallel_sizes=(4,),
            ),
        )

    def test_manifest_rejects_invalid_method_prefill_performance_floor(self):
        for invalid_value in (0.0, float("nan"), True):
            manifest = copy.deepcopy(load_manifest())
            manifest["methods"]["h2o"]["performance"]["minimum_prefill_speedup"] = (
                invalid_value
            )

            with self.subTest(invalid_value=invalid_value), self.assertRaisesRegex(
                ManifestError, "minimum_prefill_speedup"
            ):
                validate_manifest(manifest)

        manifest = copy.deepcopy(load_manifest())
        manifest["methods"]["h2o"]["performance"]["minimum_prefill_speedups"] = 1.0
        with self.assertRaisesRegex(ManifestError, "unknown keys"):
            validate_manifest(manifest)

    def test_h2o_supported_qwen3_dry_run_emits_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "qwen3-model"
            model_path.mkdir()
            argv = [
                "run_suite.py",
                "--layer",
                "quality",
                "--models",
                "qwen3_4b",
                "--methods",
                "h2o",
                "--dry_run",
                "--run_id",
                "h2o-unsupported-dry-run",
                "--output_root",
                str(root),
            ]
            with patch.object(sys, "argv", argv), patch.dict(
                os.environ,
                {"DELTAKV_MODEL_QWEN3_4B": str(model_path)},
                clear=False,
            ):
                self.assertEqual(run_suite_main(), 0)

            summary = json.loads(
                (
                    root
                    / "sparsevllm_regression"
                    / "h2o-unsupported-dry-run"
                    / "grade_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary["commands"]), 1)
            self.assertEqual(summary["commands"][0]["status"], "skipped_by_policy")
            self.assertEqual(summary["skipped"], [])

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
        self.assertTrue(manifest["models"]["qwen36"]["mixed_attention"])
        self.assertEqual(
            manifest["methods"]["omnikv"]["model_configs"]["qwen36"]["full_attention_layers"],
            "3,11,23,31,35,47,59",
        )

    def test_manifest_decode_keep_tokens_follow_eviction_policy(self):
        manifest = load_manifest()
        static_eviction_methods = {
            "streamingllm",
            "snapkv",
            "pyramidkv",
            "rkv",
            "skipkv",
        }
        dynamic_sparse_view_methods = {
            "omnikv",
            "quest",
            "deltakv",
            "deltakv-less-memory",
            "deltakv-less-memory-cudagraph",
        }
        expected_decode_keep = {
            **dict.fromkeys(static_eviction_methods, 4096),
            **dict.fromkeys(dynamic_sparse_view_methods, 2048),
        }
        configured_methods = {
            method_id
            for method_id, method in manifest["methods"].items()
            if "decode_keep_tokens" in method["config"]
        }
        self.assertEqual(configured_methods, set(expected_decode_keep))

        for method_id, expected in expected_decode_keep.items():
            method = manifest["methods"][method_id]
            configs = [("config", method["config"])]
            configs.extend(
                (f"model_configs.{model_id}", override)
                for model_id, override in (method.get("model_configs") or {}).items()
            )
            for config_name, config in configs:
                if "decode_keep_tokens" not in config:
                    continue
                self.assertEqual(
                    config["decode_keep_tokens"],
                    expected,
                    f"{method_id}.{config_name} does not match its eviction policy.",
                )

        quest = manifest["methods"]["quest"]["config"]
        self.assertNotIn("quest_token_budget", quest)
        self.assertGreater(
            quest["decode_keep_tokens"]
            + quest["sink_keep_tokens"]
            + quest["recent_keep_tokens"],
            0,
        )
        self.assertNotIn("quest_token_budget", manifest["stress_v2"])

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
                    self.assertIn("--synchronize_step_timing", cmd)

    def test_regression_rejects_unsynchronized_success_throughput(self):
        with self.assertRaisesRegex(RuntimeError, "requires synchronized"):
            _require_synchronized_step_timing(
                [
                    {
                        "status": "SUCCESS",
                        "method": "h2o",
                        "synchronize_step_timing": False,
                    }
                ],
                artifact=Path("/tmp/perf.jsonl"),
            )

    def test_regression_requires_complete_successful_perf_matrix(self):
        rows = [
            {
                "status": "SUCCESS",
                "method": method,
                "length": length,
                "batch_size": batch_size,
            }
            for method in ("vanilla", "h2o")
            for length in (32000, 64000)
            for batch_size in (4, 8)
        ]

        _require_successful_perf_matrix(
            rows,
            methods=("vanilla", "h2o"),
            lengths=(32000, 64000),
            batch_sizes=(4, 8),
            artifact=Path("/tmp/perf.jsonl"),
        )

        with self.assertRaisesRegex(RuntimeError, "missing"):
            _require_successful_perf_matrix(
                rows[:-1],
                methods=("vanilla", "h2o"),
                lengths=(32000, 64000),
                batch_sizes=(4, 8),
                artifact=Path("/tmp/perf.jsonl"),
            )

        failed_rows = copy.deepcopy(rows)
        failed_rows[-1]["status"] = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "non-success"):
            _require_successful_perf_matrix(
                failed_rows,
                methods=("vanilla", "h2o"),
                lengths=(32000, 64000),
                batch_sizes=(4, 8),
                artifact=Path("/tmp/perf.jsonl"),
            )

    def test_regression_rejects_duplicate_or_unexpected_perf_cases(self):
        row = {
            "status": "SUCCESS",
            "method": "vanilla",
            "length": 32000,
            "batch_size": 4,
        }
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _require_successful_perf_matrix(
                [row, dict(row)],
                methods=("vanilla",),
                lengths=(32000,),
                batch_sizes=(4,),
                artifact=Path("/tmp/perf.jsonl"),
            )

        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            _require_successful_perf_matrix(
                [row, {**row, "method": "h2o"}],
                methods=("vanilla",),
                lengths=(32000,),
                batch_sizes=(4,),
                artifact=Path("/tmp/perf.jsonl"),
            )

    def test_perf_layer_applies_h2o_prefill_non_regression_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "qwen25-model"
            model_path.mkdir()

            def fake_run_and_record(summary, cmd, **kwargs):
                del kwargs
                output_path = Path(cmd[cmd.index("--output_jsonl") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                rows = [
                    {
                        "status": "SUCCESS",
                        "method": "vanilla",
                        "length": length,
                        "batch_size": batch_size,
                        "prefill_tp": 100.0,
                        "decode_tp": 100.0,
                        "decode_cuda_graph_expected": True,
                        "decode_cuda_graph_active": True,
                        "synchronize_step_timing": True,
                    }
                    for length in (32000, 64000)
                    for batch_size in (4, 8)
                ]
                rows.extend(
                    {
                        "status": "SUCCESS",
                        "method": "h2o",
                        "length": length,
                        "batch_size": batch_size,
                        "prefill_tp": 99.0,
                        "decode_tp": 110.0,
                        "decode_cuda_graph_expected": True,
                        "decode_cuda_graph_active": True,
                        "synchronize_step_timing": True,
                        "memory_accounting": {"observed_savings": 0.3},
                    }
                    for length in (32000, 64000)
                    for batch_size in (4, 8)
                )
                output_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                summary["commands"].append({"status": "success", "cmd": cmd})

            argv = [
                "run_suite.py",
                "--layer",
                "perf",
                "--models",
                "qwen25_7b",
                "--methods",
                "h2o",
                "--run_id",
                "h2o-prefill-gate",
                "--output_root",
                str(root),
            ]
            with patch.object(sys, "argv", argv), patch.dict(
                os.environ,
                {
                    "DELTAKV_MODEL_QWEN25_7B": str(model_path),
                    "DELTAKV_TOKENIZER_QWEN25_7B": str(model_path),
                },
                clear=False,
            ), patch(
                "benchmark.sparsevllm_regression.run_suite._run_and_record",
                side_effect=fake_run_and_record,
            ):
                with self.assertRaisesRegex(RuntimeError, "Required regression gates failed"):
                    run_suite_main()

            summary = json.loads(
                (
                    root
                    / "sparsevllm_regression"
                    / "h2o-prefill-gate"
                    / "grade_summary.json"
                ).read_text(encoding="utf-8")
            )
            perf_grades = [
                grade
                for grade in summary["grades"]
                if grade["name"] == "performance" and grade["method"] == "h2o"
            ]
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["worst_required_grade"], "D")
            self.assertEqual(len(perf_grades), 4)
            self.assertTrue(all(grade["grade"] == "D" for grade in perf_grades))
            self.assertTrue(
                all(grade["metrics"]["prefill_speedup"] == 0.99 for grade in perf_grades)
            )

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

    def test_legacy_raw_prompt_tasks_skip_chat_template(self):
        class QwenTokenizer:
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                raise AssertionError("legacy raw-prompt tasks must not apply a chat template")

        tokenizer = QwenTokenizer()

        prompt = build_chat(tokenizer, "classify this", "trec", thinking_mode="off")

        self.assertEqual(prompt, "classify this")
        self.assertFalse(hasattr(tokenizer, "kwargs"))
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

    def test_perf_memory_and_stress_grades(self):
        self.assertEqual(grade_perf(1.1, graph_expected=True, graph_active=True).grade, "C")
        self.assertEqual(grade_perf(0.8, graph_expected=True, graph_active=True, require_speedup=False).grade, "A")
        self.assertEqual(grade_perf(2.1, graph_expected=True, graph_active=False).grade, "D")
        self.assertEqual(
            grade_perf(
                1.1,
                graph_expected=True,
                graph_active=True,
                prefill_speedup=1.0,
                minimum_prefill_speedup=1.0,
            ).grade,
            "C",
        )
        prefill_regression = grade_perf(
            1.1,
            graph_expected=True,
            graph_active=True,
            prefill_speedup=0.99,
            minimum_prefill_speedup=1.0,
        )
        self.assertEqual(prefill_regression.grade, "D")
        self.assertIn("prefill", prefill_regression.reason)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            grade_perf(
                1.1,
                prefill_speedup=1.0,
                minimum_prefill_speedup=0.0,
            )
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
