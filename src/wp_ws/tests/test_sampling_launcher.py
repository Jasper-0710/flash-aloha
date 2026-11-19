"""Dry runs only: scripts never invoke training in these tests."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

from src.wp_ws.train_act import build_parser, policy_configuration

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh"


class SamplingLauncherTests(unittest.TestCase):
    def launch(self, *arguments, **environment):
        env = dict(os.environ, PYTHON_BIN=sys.executable, GPU_ID="0")
        env.update(environment)
        return subprocess.run(["bash", str(SCRIPT), *arguments], env=env, cwd="/tmp",
                              capture_output=True, text=True, check=False)

    def parsed(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("Command: "))
        tokens = shlex.split(line.removeprefix("Command: "))
        self.assertEqual(tokens[3:5], ["-m", "src.wp_ws.train_act"])
        return tokens[0], build_parser().parse_args(tokens[5:])

    def test_mixed_control_differ_only_in_sampler_and_run_identity(self):
        _, mixed = self.parsed(self.launch("mixed", "--dry-run"))
        _, control = self.parsed(self.launch("control", "--dry-run"))
        self.assertEqual(mixed.sampler, "episode_window")
        self.assertEqual(control.sampler, "episode_block")
        self.assertEqual(policy_configuration(mixed), policy_configuration(control))
        self.assertNotEqual(mixed.run_name, control.run_name)
        a, b = vars(mixed).copy(), vars(control).copy()
        for key in ("sampler", "run_name"):
            a.pop(key); b.pop(key)
        self.assertEqual(a, b)
        self.assertEqual((mixed.steps, mixed.batch_size, mixed.chunk_size, mixed.workers), (10000, 16, 50, 0))
        self.assertEqual((mixed.val_max_batches, mixed.validate_every, mixed.train_probe_per_episode), (0, 1000, 2))
        self.assertTrue(mixed.eval_at_start and mixed.acknowledge_audit_warnings)

    def test_cpu_preflight_and_overrides_and_server_gpu(self):
        _, args = self.parsed(self.launch("preflight", "--dry-run"))
        self.assertEqual(args.command, "preflight")
        self.assertEqual(args.sampler, "episode_window")
        gpu, args = self.parsed(self.launch("mixed", "--dry-run", "--run-name", "server_mix_v2", "--steps", "2000", GPU_ID="1"))
        self.assertEqual(gpu, "CUDA_VISIBLE_DEVICES=1")
        self.assertEqual((args.run_name, args.steps), ("server_mix_v2", 2000))

    def test_syntax_help_no_python_and_multiple_gpus_rejected(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)
        result = self.launch("mixed", "--dry-run", PYTHON_BIN="/bin/false")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN", result.stdout)
        result = self.launch("--help", PYTHON_BIN="/bin/false")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("Command:", result.stdout)
        self.assertEqual(self.launch("mixed", "--dry-run", GPU_ID="1,2").returncode, 2)


if __name__ == "__main__":
    unittest.main()
