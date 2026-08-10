import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.long_bench import pred as longbench_pred
from benchmark.long_bench.metrics import classification_score, qa_f1_score


class LongBenchDeltaKVContractsTest(unittest.TestCase):
    def test_no_chat_datasets_remain_raw_for_every_thinking_mode(self):
        for dataset in longbench_pred.NO_CHAT_TEMPLATE_DATASETS:
            for thinking_mode in ("off", "on", "on_strip"):
                with self.subTest(dataset=dataset, thinking_mode=thinking_mode):
                    self.assertFalse(
                        longbench_pred.should_use_chat_template(dataset, thinking_mode=thinking_mode)
                    )

    def test_chat_template_policy_matches_regular_prompt_paths(self):
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="off")
        )
        self.assertTrue(
            longbench_pred.should_use_chat_template("hotpotqa", thinking_mode="on_strip")
        )
        self.assertFalse(
            longbench_pred.should_use_chat_template("hotpotqa", no_chat_template=True)
        )

    def test_hotpotqa_and_trec_metric_contracts(self):
        self.assertEqual(qa_f1_score("Paris", "Paris"), 1.0)
        self.assertEqual(qa_f1_score("Paris", "London"), 0)
        self.assertEqual(
            classification_score(
                "DESC",
                "DESC",
                all_classes=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"],
            ),
            1.0,
        )

    def test_longbench_data_validation_fails_fast_for_missing_hotpotqa_and_trec(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = longbench_pred.DATA_PREFIX_PATH
            longbench_pred.DATA_PREFIX_PATH = str(Path(tmp) / "missing")
            try:
                with self.assertRaisesRegex(FileNotFoundError, "LongBench data root"):
                    longbench_pred.validate_longbench_data_paths(["hotpotqa", "trec"], use_longbench_e=False)
            finally:
                longbench_pred.DATA_PREFIX_PATH = old_root

    def test_longbench_data_validation_requires_explicit_root(self):
        old_root = longbench_pred.DATA_PREFIX_PATH
        longbench_pred.DATA_PREFIX_PATH = None
        try:
            with self.assertRaisesRegex(FileNotFoundError, "SPARSEVLLM_LONGBENCH_DATA_DIR"):
                longbench_pred.validate_longbench_data_paths(["hotpotqa"], use_longbench_e=False)
        finally:
            longbench_pred.DATA_PREFIX_PATH = old_root

    def test_longbench_writes_prompt_and_canonical_per_sample_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_path = str(Path(tmp) / "qasper.jsonl")
            record = {
                "dataset": "qasper",
                "sample_idx": 0,
                "source_idx": 12,
                "status": "success",
                "prompt_tokens": 16001,
                "rendered_prompt": "fixed rendered prompt",
                "rendered_prompt_sha256": longbench_pred._sha256_text(
                    "fixed rendered prompt"
                ),
                "pred": "answer",
                "raw_pred": "answer",
                "answers": ["answer"],
                "all_classes": [],
                "length": 16001,
            }

            longbench_pred._write_sample_record(
                out_root=tmp,
                task_out_path=task_path,
                record=record,
            )

            canonical = (Path(tmp) / "per_sample_results.jsonl").read_text(
                encoding="utf-8"
            )
            historical = (Path(tmp) / "sample_results.jsonl").read_text(
                encoding="utf-8"
            )
            raw = json.loads(
                (Path(tmp) / "raw_outputs.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(canonical, historical)
            self.assertEqual(raw["rendered_prompt"], "fixed rendered prompt")
            self.assertEqual(
                raw["rendered_prompt_sha256"],
                longbench_pred._sha256_text("fixed rendered prompt"),
            )

    def test_sparsevllm_data_workers_receive_distinct_master_ports(self):
        launched = []

        class Process:
            def wait(self):
                return 0

        def fake_popen(command, *, env, cwd):
            launched.append((command, env, cwd))
            return Process()

        worker_args = SimpleNamespace(ws=4)
        with (
            patch.dict(
                "os.environ",
                {
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "SPARSEVLLM_MASTER_PORT": "24300",
                },
                clear=False,
            ),
            patch.object(longbench_pred.subprocess, "Popen", side_effect=fake_popen),
        ):
            longbench_pred.launch_single_gpu_workers(worker_args, "/tmp/longbench-output")

        self.assertEqual(
            [env["CUDA_VISIBLE_DEVICES"] for _command, env, _cwd in launched],
            ["0", "1", "2", "3"],
        )
        self.assertEqual(
            [env["SPARSEVLLM_MASTER_PORT"] for _command, env, _cwd in launched],
            ["24300", "24301", "24302", "24303"],
        )

    def test_longbench_records_actual_decode_cuda_graph_state(self):
        worker_statuses = [
            {
                "world_rank": 0,
                "decode_cuda_graph_configured": True,
                "decode_cuda_graph_state_count": 2,
                "decode_cuda_graph_graph_count": 1,
                "decode_cuda_graph_active": True,
            },
            {
                "world_rank": 1,
                "decode_cuda_graph_configured": True,
                "decode_cuda_graph_state_count": 2,
                "decode_cuda_graph_graph_count": 1,
                "decode_cuda_graph_active": True,
            },
        ]
        generate_fn = SimpleNamespace(
            _sparsevllm_llm=SimpleNamespace(
                config=SimpleNamespace(world_size=2),
                model_runner=SimpleNamespace(
                    call=lambda method: worker_statuses
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            status = longbench_pred._write_decode_cuda_graph_status(
                generate_fn=generate_fn,
                out_root=tmp,
                rank=2,
            )
            path = Path(tmp) / "decode_cuda_graph_status_rank2.json"

            self.assertEqual(status["launcher_rank"], 2)
            self.assertTrue(status["configured_on_all_workers"])
            self.assertTrue(status["active_on_all_workers"])
            self.assertEqual(status["workers"], worker_statuses)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                status,
            )

    def test_longbench_fails_if_any_requested_graph_worker_is_inactive(self):
        generate_fn = SimpleNamespace(
            _sparsevllm_llm=SimpleNamespace(
                config=SimpleNamespace(world_size=2),
                model_runner=SimpleNamespace(
                    call=lambda method: [
                        {
                            "world_rank": 0,
                            "decode_cuda_graph_configured": True,
                            "decode_cuda_graph_active": True,
                        },
                        {
                            "world_rank": 1,
                            "decode_cuda_graph_configured": True,
                            "decode_cuda_graph_active": False,
                        },
                    ]
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "not active on every"):
                longbench_pred._write_decode_cuda_graph_status(
                    generate_fn=generate_fn,
                    out_root=tmp,
                    rank=0,
                )

    def test_longbench_fails_if_requested_graph_is_unconfigured(self):
        generate_fn = SimpleNamespace(
            _sparsevllm_llm=SimpleNamespace(
                config=SimpleNamespace(
                    world_size=1,
                    decode_cuda_graph=True,
                ),
                model_runner=SimpleNamespace(
                    call=lambda method: [
                        {
                            "world_rank": 0,
                            "decode_cuda_graph_configured": False,
                            "decode_cuda_graph_active": False,
                        }
                    ]
                ),
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "not configured on every"):
                longbench_pred._write_decode_cuda_graph_status(
                    generate_fn=generate_fn,
                    out_root=tmp,
                    rank=0,
                )

    def test_longbench_fails_if_sparsevllm_graph_state_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "_sparsevllm_llm"):
                longbench_pred._write_decode_cuda_graph_status(
                    generate_fn=object(),
                    out_root=tmp,
                    rank=0,
                )

if __name__ == "__main__":
    unittest.main()
