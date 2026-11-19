"""Shell dry-run checks only: never launch Python training, CUDA or hardware."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / 'scripts' / 'wp_ws' / 'train_flip_the_pillow_act.sh'


class TrainingLauncherTests(unittest.TestCase):
    def launch(self, *arguments, **environment):
        env = dict(os.environ, PYTHON_BIN=sys.executable, GPU_ID='0')
        env.update(environment)
        return subprocess.run(['bash', str(SCRIPT), *arguments], env=env,
                              cwd='/tmp', capture_output=True, text=True, check=False)

    def arguments(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith('Command: '))
        tokens = shlex.split(line.removeprefix('Command: '))
        self.assertEqual(tokens[1], 'PYTHONUNBUFFERED=1')
        return tokens

    def test_shell_syntax(self):
        result = subprocess.run(['bash', '-n', str(SCRIPT)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_smoke_defaults(self):
        result = self.launch('smoke', '--dry-run')
        args = self.arguments(result)
        self.assertEqual(args[0], 'CUDA_VISIBLE_DEVICES=0')
        for key, value in (('--steps', '100'), ('--batch-size', '16'), ('--val-max-batches', '10')):
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('src.wp_ws.train_act', args)
        self.assertIn('DRY RUN', result.stdout)

    def test_formal_defaults(self):
        args = self.arguments(self.launch('train', '--dry-run'))
        for key, value in (('--steps', '100000'), ('--val-max-batches', '0'),
                           ('--validate-every', '5000'), ('--save-every', '5000'),
                           ('--height', '240'), ('--width', '320'), ('--backbone-init', 'imagenet')):
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--acknowledge-audit-warnings', args)

    def test_trainer_overrides_and_server_gpu(self):
        args = self.arguments(self.launch('train', '--dry-run', '--steps', '30000',
                                         '--run-name', 'another_run', '--batch-size', '4', GPU_ID='1'))
        self.assertEqual(args[0], 'CUDA_VISIBLE_DEVICES=1')
        from src.wp_ws.train_act import build_parser
        parsed = build_parser().parse_args(args[5:])
        self.assertEqual((parsed.steps, parsed.batch_size, parsed.run_name), (30000, 4, 'another_run'))

    def test_dry_run_does_not_execute_interpreter(self):
        result = self.launch('train', '--dry-run', PYTHON_BIN='/bin/false')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_multiple_devices(self):
        result = self.launch('train', '--dry-run', GPU_ID='1,2,3')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not DDP', result.stderr)

    def test_help_is_not_training(self):
        result = self.launch('--help', PYTHON_BIN='/bin/false')
        self.assertEqual(result.returncode, 0)
        self.assertIn('no resume', result.stdout)
        self.assertNotIn('Command:', result.stdout)


if __name__ == '__main__':
    unittest.main()
