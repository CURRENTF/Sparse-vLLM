"""Protect result trust boundaries, not a table of preferred tuning constants."""
import json
from pathlib import Path
import tempfile
import unittest

from summarize import load_run


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.manifest = {"status": "completed", "cases": [{"id": "synthetic"}],
                         "config": {"abba_rounds": 1, "legal_splits": [1, 2, 4, 8, 16, 32]},
                         "gpu": {"sm_count": 8}}
        # Independent synthetic timing samples; these are not benchmark winners.
        self.row = {"id": "synthetic", "status": "success", "oracle": {"status": "success"},
                    "context": 256, "actual_lengths": [256] * 4, "base_ctas": 4,
                    "raw_splits": 4, "identical_config": False,
                    "configs": {"legacy": {"num_split": 2, "block_h": 16},
                                "formula": {"num_split": 4, "block_h": 16}},
                    "samples_us": {"legacy": [2.0, 2.2], "formula": [1.0, 1.2]},
                    "median_us": {"legacy": 2.1, "formula": 1.1}, "speedup": 2.1 / 1.1}
        self.raw = [{"case_id": "synthetic", "arm": arm, "latency_us": value, "status": "success"}
                    for arm, values in self.row["samples_us"].items() for value in values]

    def save(self):
        (self.root / "run_manifest.json").write_text(json.dumps(self.manifest))
        (self.root / "cases.jsonl").write_text(json.dumps(self.row) + "\n")
        (self.root / "raw_samples.jsonl").write_text("\n".join(map(json.dumps, self.raw)))

    def test_reconstruct_real_sample_medians(self):
        self.save()
        self.assertEqual(load_run(self.root)[1][0]["speedup"], 2.1 / 1.1)

    def test_tampered_aggregate_is_rejected(self):
        self.row["median_us"]["formula"] = 0.1
        self.save()
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            load_run(self.root)

    def test_partial_raw_run_is_not_complete(self):
        self.raw.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "sample count"):
            load_run(self.root)

    def test_failed_oracle_cannot_be_scored(self):
        self.row["oracle"]["status"] = "failed"
        self.save()
        with self.assertRaisesRegex(ValueError, "numerical"):
            load_run(self.root)

    def test_second_tuning_dimension_cannot_be_mixed_in(self):
        self.row["configs"]["formula"]["block_h"] = 32
        self.save()
        with self.assertRaisesRegex(ValueError, "more than splits"):
            load_run(self.root)

    def test_missing_case_is_not_silently_excluded(self):
        self.manifest["cases"].append({"id": "missing"})
        self.save()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_run(self.root)


if __name__ == "__main__":
    unittest.main()
