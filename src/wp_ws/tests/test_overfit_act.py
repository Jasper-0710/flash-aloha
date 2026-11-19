"""Synthetic CPU-only tests: no real recordings, native GPU training or robot access."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wp_ws import overfit_act as module
from src.wp_ws.act_dataset import ActDataset, fit_training_statistics
from src.wp_ws.audit_core import AuditConfig, audit_dataset, hash_file
from src.wp_ws.audit_dataset import write_artifacts
from src.wp_ws.baseline_data_config import load_config
from src.wp_ws.tests.test_audit_dataset import fixture, edit_column
from src.wp_ws.verified_data import _verify_at


class FixedSubsetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.parent = Path(temp.name)
        self.root, self.audit = self.parent / "source", self.parent / "audit"
        fixture(self.root, count=5, length=6)
        for eid in range(5):
            def varied(values, eid=eid):
                for row, value in enumerate(values):
                    value[0] = row + eid * 10
            edit_column(self.root, "action", varied, eid=eid)
            edit_column(self.root, "observation.state", varied, eid=eid)
        write_artifacts(self.audit, audit_dataset(self.root, AuditConfig(val_episodes=2)))
        self.data = _verify_at(self.root, self.audit, load_config())
        self.stats = fit_training_statistics(self.data, progress=None)
        self.ds = ActDataset(self.data, self.stats, {"height": 6, "width": 8, "num_queries": 5}, partition="train")

    def cache(self, count=3):
        return module.FixedSamples(self.ds, module.select_samples(self.ds, samples_per_episode=count))

    def test_selection_exact_stable_no_duplicates_train_only(self):
        selected = module.select_samples(self.ds, samples_per_episode=3)
        self.assertEqual(selected, module.select_samples(self.ds, samples_per_episode=3))
        ids = sorted(self.ds.episode_ids)[:2]
        self.assertEqual([self.ds.locate(i) for i in selected], [(eid, row) for eid in ids for row in [0, 2, 5]])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertFalse({self.ds.locate(i)[0] for i in selected} & set(self.data.split["val_episode_indices"]))
        self.assertEqual(module.select_samples(self.ds, ids[::-1], 3), selected)

    def test_rejects_val_unknown_duplicate_and_oversized_selection(self):
        for ids, count in [([self.data.split["val_episode_indices"][0]], 3), ([999], 3),
                           ([], 3), ([self.ds.episode_ids[0]] * 2, 3), (None, 7), (None, 0)]:
            with self.assertRaises(ValueError):
                module.select_samples(self.ds, ids, count)
        val = ActDataset(self.data, self.stats, {"height": 6, "width": 8, "num_queries": 5}, partition="val")
        with self.assertRaisesRegex(ValueError, "training partition"):
            module.select_samples(val, samples_per_episode=2)

    def test_cache_exact_adapter_parity_original_grippers_and_read_only(self):
        before = {str(p): hash_file(p) for p in self.parent.rglob("*") if p.is_file()}
        cache = self.cache()
        self.assertIsNone(self.ds._cache_table)
        for i, entry in enumerate(cache.metadata):
            original = self.ds[entry["flat_index"]]
            for key in module.ARRAY_KEYS:
                np.testing.assert_array_equal(cache.arrays[key][i], original[key])
                self.assertFalse(cache.arrays[key].flags.writeable)
            raw = cache.arrays["actions"][i] * self.ds.std["action"] + self.ds.mean["action"]
            self.assertGreater(raw[0, 6], 1.0)
            self.assertLess(raw[0, 13], 0.0)
        batch = cache.batch([0, 1])
        self.assertTrue(batch["images"].flags.writeable)
        batch["images"][:] = 0
        self.assertGreater(float(cache.arrays["images"].sum()), 0)
        self.data.assert_unchanged()
        self.assertEqual(before, {str(p): hash_file(p) for p in self.parent.rglob("*") if p.is_file()})
        self.assertEqual(cache.manifest(), self.cache().manifest())

    def test_cache_limit_fails_before_decoding(self):
        with patch.object(self.ds, "_table", side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ValueError, "cache-limit-mib"):
                module.FixedSamples(self.ds, [0], limit_mib=0)

    def test_baselines_use_exact_masked_subset(self):
        cache = self.cache()
        baselines = module.constant_baselines(cache, self.stats["action"]["std"])
        valid = cache.arrays["actions"][~cache.arrays["is_pad"]].astype(np.float64)
        self.assertAlmostEqual(baselines["train_mean_constant"]["normalized_l1"], np.abs(valid).mean())
        expected = np.abs(valid - np.median(valid, axis=0)).mean()
        self.assertAlmostEqual(baselines["subset_median_constant"]["normalized_l1"], expected)
        self.assertLessEqual(expected, baselines["train_mean_constant"]["normalized_l1"])

    def test_eval_mode_no_grad_and_restore_including_exception(self):
        cache = self.cache()
        class ZeroPolicy(torch.nn.Module):
            def forward(self, qpos, images):
                self.assertions(qpos, images)
                return torch.zeros((len(qpos), 5, 14))
            def assertions(self, qpos, images):
                assert not self.training and torch.is_inference_mode_enabled()
                assert not torch.is_grad_enabled()
        policy = ZeroPolicy()
        result = module.evaluate_policy(policy, cache, self.stats["action"]["std"], 4, "cpu")
        self.assertTrue(policy.training)
        self.assertEqual(result["evaluation_scope"], module.EVAL_SCOPE)
        baseline = module.constant_baselines(cache, self.stats["action"]["std"])["train_mean_constant"]
        for key in ["normalized_l1", "per_episode", "per_horizon", "per_dimension", "valid_action_elements"]:
            self.assertEqual(result[key], baseline[key])
        policy.eval()
        module.evaluate_policy(policy, cache, self.stats["action"]["std"], 1, "cpu")
        self.assertFalse(policy.training)
        policy.train()
        with patch.object(policy, "forward", side_effect=RuntimeError("simulated failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                module.evaluate_policy(policy, cache, self.stats["action"]["std"], 4, "cpu")
        self.assertTrue(policy.training)

    def test_synthetic_preflight_artifacts_and_no_overwrite(self):
        logs, ckpts = self.parent / "logs", self.parent / "ckpts"
        arguments = ["preflight", "--run-name", "test", "--height", "6", "--width", "8",
                     "--samples-per-episode", "3", "--chunk-size", "5"]
        with patch.object(module.train_act, "LOG_DIR", logs), patch.object(module.train_act, "CKPT_DIR", ckpts), \
             patch("src.wp_ws.verified_data.load_verified_data", return_value=self.data), \
             patch.object(module, "run_training", side_effect=AssertionError("GPU training not allowed")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA not allowed")), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(module.main(arguments), 0)
            run = logs / "wp_ws/flip_the_pillow/test"
            manifest = json.loads((run / "run_config.json").read_text())
            self.assertEqual(manifest["format"], "wp_ws.act_overfit.v1")
            self.assertFalse(manifest["held_out_validation_performed"])
            self.assertEqual(manifest["selection"]["sample_count"], 6)
            self.assertEqual(json.loads((run / "split.json").read_text()), self.data.split)
            self.assertEqual(json.loads((run / "normalization_stats.json").read_text()), self.stats)
            before = {p.name: hash_file(p) for p in run.iterdir()}
            self.assertEqual(module.main(arguments), 1)
            self.assertEqual(before, {p.name: hash_file(p) for p in run.iterdir()})
            self.assertFalse((run / "FAILED.json").exists())
            self.assertFalse(ckpts.exists())

    def test_train_warnings_not_silently_accepted(self):
        logs = self.parent / "run"
        logs.mkdir()
        args = module.build_parser().parse_args(["train", "--height", "6", "--width", "8", "--samples-per-episode", "3"])
        with patch("src.wp_ws.verified_data.load_verified_data", return_value=self.data), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, "acknowledge-audit-warnings"):
                module.prepare(args, load_config(), logs)
        self.assertEqual(list(logs.iterdir()), [])


class MetricAndBatchTests(unittest.TestCase):
    def test_masked_metrics_weight_elements_not_batch_means(self):
        target = np.zeros((2, 3, 14), dtype=np.float32)
        pred = np.ones_like(target)
        pred[0] = 3.0
        pred[1, 1:] = 99999  # Padded predictions contribute nothing.
        pad = np.array([[False, False, False], [False, True, True]])
        totals = module.ErrorTotals(np.arange(1, 15), 3)
        totals.add(pred[:1], target[:1], pad[:1], [0])
        totals.add(pred[1:], target[1:], pad[1:], [1])
        result = totals.result()
        self.assertEqual(result["normalized_l1"], 2.5)
        self.assertEqual(result["valid_action_elements"], 56)
        self.assertEqual(result["per_horizon"][0]["normalized_l1"], 2.0)
        self.assertEqual(result["per_horizon"][1]["normalized_l1"], 3.0)
        self.assertEqual(result["per_dimension"][13]["raw_mae"], 35.0)
        self.assertEqual(result["per_dimension"][13]["raw_unit"], "normalized_gripper")
        self.assertAlmostEqual(result["joint_mae_rad"], np.arange(1, 15)[module.JOINTS].mean() * 2.5)
        empty = module.ErrorTotals(np.ones(14), 3)
        empty.add(pred, target, np.ones_like(pad), [0, 1])
        with self.assertRaisesRegex(ValueError, "No valid"):
            empty.result()
        pred[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            totals.add(pred, target, pad, [0, 1])

    def test_batches_deterministic_full_coverage_no_rng_side_effects(self):
        state = np.random.get_state()
        a = module.fixed_batches(32, 15, seed=42, shuffle=True)
        self.assertEqual(a, module.fixed_batches(32, 15, seed=42, shuffle=True))
        self.assertNotEqual(a, module.fixed_batches(32, 15, seed=42, epoch=1, shuffle=True))
        self.assertEqual(sorted(i for batch in a for i in batch), list(range(32)))
        self.assertEqual([len(batch) for batch in a], [15, 15, 2])
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_report_scope_and_checkpoint_not_validation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            metric = {"normalized_l1": .3, "joint_mae_rad": .1, "gripper_mae_normalized_units": .2, "step": 0}
            module.write_report(path, [metric], {"train_mean_constant": {"normalized_l1": .4}})
            self.assertIn("不是验证集", (path / "report.md").read_text())
            self.assertFalse((path / "report.md.partial").exists())
            policy = SimpleNamespace(serialize=lambda: {"test": torch.ones(1)})
            manifest = {"format": "wp_ws.act_overfit.v1", "policy_config": {}}
            module.save_checkpoint(path / "best.ckpt", policy, {}, manifest, 0, metric, torch)
            saved = torch.load(path / "best.ckpt", map_location="cpu", weights_only=True)
            self.assertIn("overfit_evaluation", saved)
            self.assertNotIn("validation", saved)
            self.assertFalse(saved["live_deployment_ready"])
            self.assertFalse(saved["resume_entrypoint_implemented"])
            self.assertFalse((path / "best.ckpt.partial").exists())

    def test_parser_rejects_invalid_small_counts(self):
        with redirect_stderr(StringIO()):
            for args in [["train", "--batch-size", "0"], ["train", "--steps", "-1"],
                         ["train", "--samples-per-episode", "0"], ["train", "--episodes", "-1"]]:
                with self.assertRaises(SystemExit):
                    module.build_parser().parse_args(args)


class LauncherTests(unittest.TestCase):
    script = Path(__file__).resolve().parents[3] / "scripts/wp_ws/overfit_flip_the_pillow_act.sh"

    def run_script(self, *args, **env):
        return subprocess.run(["bash", str(self.script), *args], cwd="/tmp", capture_output=True, text=True,
                              env={**os.environ, "PYTHON_BIN": "/bin/false", "GPU_ID": "0", **env})

    def test_syntax_help_and_dry_run_without_executing_python(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(self.script)], capture_output=True).returncode, 0)
        self.assertEqual(self.run_script("--help").returncode, 0)
        for mode in ("train", "preflight"):
            result = self.run_script(mode, "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            line = next(s for s in result.stdout.splitlines() if s.startswith("Command: "))
            args = shlex.split(line.removeprefix("Command: "))
            parsed = module.build_parser().parse_args(args[5:])
            self.assertEqual((parsed.command, parsed.batch_size, parsed.samples_per_episode), (mode, 16, 16))
            self.assertIn("NOT validation", result.stdout)
            self.assertIn("DRY RUN", result.stdout)

    def test_server_gpu_and_overrides(self):
        result = self.run_script("train", "--dry-run", "--run-name", "server_test", "--steps", "250", GPU_ID="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(s for s in result.stdout.splitlines() if s.startswith("Command: "))
        args = shlex.split(line.removeprefix("Command: "))
        self.assertEqual(args[0], "CUDA_VISIBLE_DEVICES=1")
        parsed = module.build_parser().parse_args(args[5:])
        self.assertEqual((parsed.run_name, parsed.steps), ("server_test", 250))
        self.assertEqual(self.run_script("--dry-run", GPU_ID="1,2").returncode, 2)


if __name__ == "__main__":
    unittest.main()
