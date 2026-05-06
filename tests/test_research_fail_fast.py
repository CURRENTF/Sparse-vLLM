import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import bench_llava_onevision_streamingbench as streamingbench
from sparsevllm.config import Config


class ResearchFailFastTest(unittest.TestCase):
    def test_streamingbench_missing_videos_fail_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_dir = root / "StreamingBench"
            csv_dir.mkdir()
            video_dir = root / "videos"
            video_dir.mkdir()
            csv_path = csv_dir / "Real_Time_Visual_Understanding.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "question_id",
                        "task_type",
                        "question",
                        "time_stamp",
                        "answer",
                        "options",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "question_id": "Real-Time Visual Understanding_sample_1_1",
                        "task_type": "Real-Time Visual Understanding",
                        "question": "What is shown?",
                        "time_stamp": "00:00:05",
                        "answer": "A",
                        "options": "['one', 'two', 'three', 'four']",
                    }
                )

            args = SimpleNamespace(
                dataset_dir=str(root),
                csv_dir="",
                video_dir=str(video_dir),
                tasks="real",
                allow_missing_videos=False,
                sample_start=0,
                num_samples=-1,
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing videos"):
                streamingbench.load_streamingbench_rows(args)

            args.allow_missing_videos = True
            rows, info = streamingbench.load_streamingbench_rows(args)
            self.assertEqual(rows, [])
            self.assertEqual(info["missing_video_rows"], 1)

    def test_streamingbench_rejects_silent_row_defaults(self):
        with self.assertRaisesRegex(ValueError, "empty timestamp"):
            streamingbench.parse_timestamp("")
        with self.assertRaisesRegex(ValueError, "exactly 4 choices"):
            streamingbench.parse_options("['one', 'two']")

    def test_streamingbench_livevlm_table4_scope_and_stats(self):
        args = SimpleNamespace(
            streamingbench_profile="livevlm_table4",
            tasks="real",
            num_video_frames=8,
            context_seconds=60.0,
            frame_sampling_backend="ffmpeg",
        )
        streamingbench.apply_streamingbench_profile(args)
        self.assertEqual(args.tasks, "livevlm_table4")
        self.assertEqual(streamingbench.list_tasks(args.tasks), ["real", "omni", "contextual"])
        self.assertEqual(args.num_video_frames, 32)
        self.assertEqual(args.context_seconds, -1.0)
        self.assertEqual(args.frame_sampling_backend, "decord")

        records = [
            {
                "task_type": "Object Perception",
                "status": "success",
                "correct": True,
            },
            {
                "task_type": "Causal Reasoning",
                "status": "parse_failed",
                "correct": False,
            },
            {
                "task_type": "Anomaly Context Understanding",
                "status": "success",
                "correct": False,
            },
        ]
        stats = streamingbench.compute_livevlm_table4_stats(records)
        self.assertEqual(stats["overall"]["total"], 3)
        self.assertEqual(stats["overall"]["correct"], 1)
        self.assertEqual(stats["overall"]["status_counts"], {"success": 2, "parse_failed": 1})
        self.assertEqual(stats["expected_llava_onevision_7b_overall_pct"], 58.85)
        self.assertEqual(stats["expected_overall_row_count"], 4000)
        self.assertEqual(stats["subtasks"][0]["abbr"], "OP")
        self.assertEqual(stats["subtasks"][0]["expected_llava_onevision_7b_pct"], 80.38)
        self.assertEqual(stats["subtasks"][1]["status_counts"], {"parse_failed": 1})
        self.assertEqual(stats["overall_extra_subtasks"][0]["abbr"], "ACU")
        self.assertTrue(stats["overall_extra_subtasks"][0]["used_for_paper_overall"])

    def test_streamingbench_choice_parse_modes(self):
        self.assertEqual(streamingbench.extract_choice(" A", "official_first_char"), "A")
        self.assertEqual(streamingbench.extract_choice("The answer is A", "official_first_char"), "T")
        self.assertEqual(streamingbench.extract_choice("The answer is A", "robust"), "A")

    def test_streamingbench_video_resolution_uses_task_type_hints(self):
        video_index = {
            1: [
                Path("/tmp/videos/Emotion Recognition/sample_1.mp4"),
                Path("/tmp/videos/Source Discrimination/sample_1.mp4"),
                Path("/tmp/videos/Anomaly Context Understanding/sample_1.mp4"),
                Path("/tmp/videos/Misleading Context Understanding/sample_1.mp4"),
            ]
        }
        self.assertEqual(
            streamingbench.resolve_video_path(video_index, "omni", "Source Discrimination", 1),
            Path("/tmp/videos/Source Discrimination/sample_1.mp4"),
        )
        self.assertEqual(
            streamingbench.resolve_video_path(video_index, "contextual", "Misleading Context Recognition", 1),
            Path("/tmp/videos/Misleading Context Understanding/sample_1.mp4"),
        )

    def test_sparsevllm_raw_config_fallback_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(
                '{"model_type": "qwen2", "torch_dtype": "float16", "max_position_embeddings": 32768}\n',
                encoding="utf-8",
            )
            with patch("sparsevllm.config.AutoConfig.from_pretrained", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "Refusing to silently fall back"):
                    Config(model=str(model_dir))

    def test_sparsevllm_deltakv_requires_checkpoint_path(self):
        hf_config = SimpleNamespace(
            model_type="qwen2",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=hf_config):
                with self.assertRaisesRegex(ValueError, "requires deltakv_path"):
                    Config(model=tmp, vllm_sparse_method="deltakv")
                with self.assertRaisesRegex(ValueError, "requires deltakv_path"):
                    Config(model=tmp, vllm_sparse_method="deltakv", deltakv_path="none")

    def test_sparsevllm_missing_model_dir_has_clear_error(self):
        missing = "/tmp/sparsevllm-definitely-missing-model-dir"
        with self.assertRaisesRegex(FileNotFoundError, "Model directory does not exist"):
            Config(model=missing)


if __name__ == "__main__":
    unittest.main()
