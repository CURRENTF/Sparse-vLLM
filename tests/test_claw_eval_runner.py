import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from string import Template

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "benchmark" / "claw_eval" / "run_sparsevllm_claw_eval.sh"
CONFIG = REPO_ROOT / "benchmark" / "claw_eval" / "sparsevllm_config.yaml"


class ClawEvalRunnerTest(unittest.TestCase):
    def test_runner_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_sandbox_image_override_is_resolved_from_final_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "python-env"
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "python").symlink_to(sys.executable)
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "RUN_NAME": "unit-test",
                    "CLAW_EVAL_CONDA_ENV": str(env_dir),
                    "CLAW_EVAL_ARGS": (
                        "batch --config config.yaml --sandbox "
                        "--sandbox-image custom-agent:test --no-judge"
                    ),
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_effective_sandbox_image',
                    "bash",
                    str(RUNNER),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.stdout.strip(), "custom-agent:test")

    def test_config_routes_model_and_sandbox_without_secrets_in_runner(self):
        rendered = Template(CONFIG.read_text(encoding="utf-8")).substitute(
            {
                "SPARSEVLLM_OPENAI_API_KEY": "local-test-key",
                "SPARSEVLLM_OPENAI_BASE_URL": "http://127.0.0.1:18000/v1",
                "SPARSEVLLM_CLAW_MODEL_ID": "test-model",
                "SPARSEVLLM_CONTEXT_WINDOW": "32768",
                "OPENROUTER_API_KEY": "judge-test-key",
                "CLAW_EVAL_JUDGE_BASE_URL": "https://openrouter.ai/api/v1",
                "CLAW_EVAL_JUDGE_MODEL": "judge-model",
                "CLAW_EVAL_TRACE_DIR": "/tmp/traces",
                "CLAW_EVAL_SANDBOX_IMAGE": "claw-eval-agent:test",
            }
        )
        config = yaml.safe_load(rendered)

        self.assertEqual(config["model"]["base_url"], "http://127.0.0.1:18000/v1")
        self.assertEqual(config["model"]["model_id"], "test-model")
        self.assertFalse(config["sandbox"]["enabled"])
        self.assertEqual(config["sandbox"]["image"], "claw-eval-agent:test")
        self.assertNotIn("sk-", RUNNER.read_text(encoding="utf-8").lower())

    def test_manifest_records_external_server_and_docker_identity(self):
        script = RUNNER.read_text(encoding="utf-8")

        for field in (
            '"claw_eval_commit"',
            '"start_sparsevllm_server"',
            '"server_health_url"',
            '"sandbox_image_id"',
            '"sandbox_image_size_bytes"',
        ):
            self.assertIn(field, script)
        self.assertIn('START_SPARSEVLLM_SERVER="${START_SPARSEVLLM_SERVER:-1}"', script)
        self.assertIn("Starting sandbox preflight container", script)
        self.assertIn("Set CLAW_EVAL_BUILD_SANDBOX_IMAGE=1", script)


if __name__ == "__main__":
    unittest.main()
