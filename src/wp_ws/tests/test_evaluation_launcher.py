"""Launch script checks use --dry-run only; no evaluation/GPU access."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

from src.wp_ws.evaluate_act import build_parser

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/wp_ws/evaluate_flip_the_pillow_act.sh"


class EvaluationLauncherTests(unittest.TestCase):
    def launch(self, *arguments, **environment):
        env = dict(os.environ, PYTHON_BIN=sys.executable, GPU_ID="0")
        env.update(environment)
        return subprocess.run(["bash", str(SCRIPT), *arguments], env=env, cwd="/tmp", capture_output=True, text=True)

    def parse(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(x for x in result.stdout.splitlines() if x.startswith("Command: "))
        tokens = shlex.split(line.removeprefix("Command: "))
        self.assertEqual(tokens[3:5], ["-m", "src.wp_ws.evaluate_act"])
        return tokens[0], build_parser().parse_args(tokens[5:])

    def test_preflight_and_evaluate_default_match_checkpoint_not_new_model_settings(self):
        _, pre = self.parse(self.launch("preflight", "--dry-run"))
        _, evaluation = self.parse(self.launch("--dry-run"))
        self.assertEqual((pre.command, evaluation.command), ("preflight", "evaluate"))
        self.assertNotEqual(pre.run_name, evaluation.run_name)
        self.assertEqual(pre.source_run, "act_mix8_b16_10k_v1")
        self.assertEqual(evaluation.checkpoint, "best")
        self.assertIsNone(evaluation.batch_size)
        self.assertEqual(evaluation.examples_per_episode, 2)

    def test_server_gpu_overrides_and_no_python_on_dry_run(self):
        gpu, args = self.parse(self.launch("evaluate", "--dry-run", "--source-run", "server_run", "--run-name", "eval2",
                                          "--checkpoint", "last", "--batch-size", "8", GPU_ID="1"))
        self.assertEqual(gpu, "CUDA_VISIBLE_DEVICES=1")
        self.assertEqual((args.source_run, args.run_name, args.checkpoint, args.batch_size), ("server_run", "eval2", "last", 8))
        self.assertEqual(self.launch("--dry-run", PYTHON_BIN="/bin/false").returncode, 0)
        self.assertEqual(self.launch("--help", PYTHON_BIN="/bin/false").returncode, 0)
        self.assertEqual(self.launch("--dry-run", GPU_ID="1,2").returncode, 2)
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
