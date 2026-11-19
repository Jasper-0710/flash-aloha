"""Synthetic CPU-only evaluator tests; no recorded data, CUDA, or robot use."""
from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr, ExitStack
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import torch

from src.wp_ws import evaluate_act as module, evaluation_checkpoint as checkpoint
from src.wp_ws.act_dataset import ActDataset, fit_training_statistics, validation_batches
from src.wp_ws.audit_core import AuditConfig, audit_dataset, hash_file
from src.wp_ws.audit_dataset import write_artifacts
from src.wp_ws.baseline_data_config import load_config
from src.wp_ws.evaluation_metrics import DetailedErrors, example_rows, reproduction_check, write_report
from src.wp_ws.evaluation_plots import plot_episode, plot_summary
from src.wp_ws.tests.test_audit_dataset import fixture, edit_column
from src.wp_ws.train_act import policy_configuration, build_parser as training_parser
from src.wp_ws.verified_data import _verify_at
from src.wp_ws.view_loss import _plot_cache


class FakePolicy(torch.nn.Module):
    def __init__(self, chunk=5, bias=.25):
        super().__init__()
        self.chunk = chunk
        self.bias = torch.nn.Parameter(torch.tensor(bias))
        self.calls = 0

    def forward(self, qpos, images):
        assert not self.training and torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert images.shape[1:] == (3, 3, 6, 8)
        self.calls += 1
        return self.bias.expand(len(qpos), self.chunk, 14)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.parent = Path(temp.name)
        self.config = load_config()
        self.root = self.parent / "data" / self.config.dataset
        self.log_root, self.ckpt_root = self.parent / "logs", self.parent / "ckpts"
        self.audit = self.log_root / "wp_ws" / self.config.task_name / self.config.audit_run
        fixture(self.root, count=5, length=7)
        for eid in range(5):
            def values(rows, eid=eid):
                for i, row in enumerate(rows):
                    row[0] = i * .1 + eid * .03
            edit_column(self.root, "action", values, eid=eid)
            # A real saved-time gap produces two audited ranges without invented timestamps.
            edit_column(self.root, "recording.elapsed_time", lambda r: r.__setitem__(slice(3, None), [x + .5 for x in r[3:]]), eid=eid)
        write_artifacts(self.audit, audit_dataset(self.root, AuditConfig(val_episodes=2)))
        self.data = _verify_at(self.root, self.audit, self.config)
        self.stats = fit_training_statistics(self.data, progress=None)
        args = training_parser().parse_args(["preflight", "--height", "6", "--width", "8", "--chunk-size", "5", "--batch-size", "3"])
        self.policy_config = policy_configuration(args)
        self.ds = ActDataset(self.data, self.stats, self.policy_config, partition="val")

    def snapshot_inputs(self):
        return {str(p): hash_file(p) for root in (self.root, self.audit) for p in root.rglob("*") if p.is_file()}

    def make_checkpoint(self):
        from src.wp_ws.act_dataset import image_configuration
        self.source_logs = self.log_root / "wp_ws" / self.config.task_name / "trained"
        self.source_ckpts = self.ckpt_root / "wp_ws" / self.config.task_name / "trained"
        for p in (self.source_logs, self.source_ckpts):
            p.mkdir(parents=True)
        result = module.evaluate_stream(FakePolicy(), self.ds, 3, "cpu", progress=None)
        reference = {"step": 10, "normalized_l1": result["model"]["normalized_l1"],
                     "mean_constant_normalized_l1": result["train_mean_constant"]["normalized_l1"],
                     "samples": len(self.ds), "valid_action_elements": result["model"]["valid_action_elements"],
                     "mode": "full_fixed_validation"}
        manifest = {"format": "wp_ws.act_baseline.v1", "data_config": asdict(self.config),
                    "policy_config": self.policy_config, "image_preprocessing": image_configuration(self.policy_config, self.data),
                    "provenance": self.data.provenance(), "validation_selection": validation_batches(self.ds, 3, 0)[1],
                    "implementation_sha256": {k: hash_file(v) for k, v in checkpoint.inference_files().items()},
                    "environment": {}, "channels_last": True}
        self.payload = {"format": manifest["format"], "run_config": manifest,
                        "policy_config": self.policy_config, "normalization_stats": self.stats,
                        "validation": reference, "step": 10, "model_state_dict": FakePolicy().state_dict()}
        for directory in (self.source_logs, self.source_ckpts):
            module.write_json(directory / "run_config.json", manifest)
            module.write_json(directory / "normalization_stats.json", self.stats)
        for name in ("split.json", "continuity.json", "fingerprints.json"):
            module.write_json(self.source_logs / name, self.data.documents[name])
        module.write_json(self.source_logs / "COMPLETED.json", {"status": "TRAINING_STEPS_COMPLETED", "steps": 10,
            "last_validation": reference, "best_validation_step": 10, "best_normalized_val_l1": reference["normalized_l1"]})
        for name in ("best", "last"):
            torch.save(self.payload, self.source_ckpts / (name + ".ckpt"))


class MetricsTests(unittest.TestCase):
    def test_mask_weighting_raw_units_and_unobserved_horizons(self):
        pred = np.zeros((2, 4, 14), dtype=np.float32)
        target = np.ones_like(pred)
        target[1, 0] = 3
        mask = np.array([[False, False, True, True], [False, True, True, True]])
        pred[mask] = 999  # Never participates in averages.
        std = np.arange(1, 15, dtype=np.float64) * .01
        metrics = DetailedErrors(std, 4)
        metrics.add(pred, target, mask, [8, 19])
        result = metrics.result()
        self.assertAlmostEqual(result["normalized_l1"], 5 / 3)
        self.assertEqual(result["valid_action_elements"], 42)
        self.assertEqual(result["per_horizon"][0]["valid_chunks"], 2)
        self.assertIsNone(result["per_horizon"][2]["normalized_l1"])
        self.assertIsNone(result["per_horizon"][2]["joint_mae_deg"])
        self.assertAlmostEqual(result["per_dimension"][6]["raw_mae"], 5 / 3 * .07)
        self.assertAlmostEqual(result["joint_mae_deg"], np.degrees(result["joint_mae_rad"]))
        metrics2 = DetailedErrors(std, 4)
        for i in range(2):
            metrics2.add(pred[i:i+1], target[i:i+1], mask[i:i+1], [[8], [19]][i])
        self.assertEqual(result, metrics2.result())
        pred[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            metrics2.add(pred, target, mask, [8, 19])

    def test_reproduction_is_not_a_performance_threshold(self):
        actual = dict(samples=2, valid_action_elements=42, normalized_l1=.4)
        constant = dict(actual, normalized_l1=.8)
        recorded = dict(actual, normalized_l1=.4001, mean_constant_normalized_l1=.8)
        self.assertTrue(reproduction_check(actual, constant, recorded)["within_tolerance"])
        self.assertFalse(reproduction_check(actual, constant, dict(recorded, normalized_l1=.5))["within_tolerance"])
        for key, value in (("samples", 3), ("valid_action_elements", 43), ("mean_constant_normalized_l1", .9)):
            with self.assertRaises(ValueError):
                reproduction_check(actual, constant, dict(recorded, **{key: value}))

    def test_examples_deterministic_no_duplicates_no_error_selection(self):
        self.assertEqual(example_rows(100, 2), [25, 75])
        self.assertEqual(example_rows(1, 2), [0])
        self.assertEqual(example_rows(5, 100), list(range(5)))
        with self.assertRaises(ValueError):
            example_rows(10, 0)


class StreamingTests(FixtureTests):
    def test_full_validation_masked_metrics_archives_no_input_changes(self):
        before = self.snapshot_inputs()
        episodes = {}
        policy = FakePolicy()
        initial = policy.bias.detach().clone()
        result = module.evaluate_stream(policy, self.ds, 3, "cpu", on_episode=lambda eid, a: episodes.update({eid: a}), progress=None)
        self.assertTrue(policy.training)
        self.assertIsNone(policy.bias.grad)
        torch.testing.assert_close(initial, policy.bias)
        self.assertEqual(result["model"]["samples"], len(self.ds))
        self.assertEqual(set(episodes), set(self.data.split["val_episode_indices"]))
        self.assertFalse(set(episodes) & set(self.data.split["train_episode_indices"]))
        self.assertEqual(before, self.snapshot_inputs())
        output = self.parent / "derived"
        (output / "predictions").mkdir(parents=True)
        total = count = 0
        for eid, arrays in episodes.items():
            self.assertEqual(arrays["contiguous_ranges"].tolist(), [[0, 3], [3, 7]])
            self.assertEqual(arrays["frame_index"].tolist(), list(range(7)))
            self.assertEqual((~arrays["is_pad"][2]).sum(), 1)  # gap blocks the chunk
            raw = arrays["target_normalized"] * arrays["action_std"] + arrays["action_mean"]
            self.assertTrue((raw[~arrays["is_pad"]][:, 6] > 1).all())
            self.assertTrue((raw[~arrays["is_pad"]][:, 13] < 0).all())
            valid = ~arrays["is_pad"]
            total += np.abs(arrays["target_normalized"][valid]).astype(np.float64).sum()
            count += int(valid.sum()) * 14
            module.save_episode(output, eid, arrays)
            with np.load(output / "predictions" / f"episode_{eid:06d}.npz", allow_pickle=False) as saved:
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], saved[key])
                    self.assertNotEqual(saved[key].dtype, np.dtype(object))
            with self.assertRaises(FileExistsError):
                module.save_episode(output, eid, arrays)
        self.assertAlmostEqual(result["train_mean_constant"]["normalized_l1"], total / count)
        self.assertEqual(result["model"]["valid_action_elements"], count)
        # Batch size does not change weighting or row coverage.
        other = module.evaluate_stream(FakePolicy(), self.ds, 6, "cpu", progress=None)
        self.assertAlmostEqual(other["model"]["normalized_l1"], result["model"]["normalized_l1"], places=12)
        weighted = sum(r["model"]["normalized_l1"] * r["model"]["valid_action_elements"] for r in result["per_episode"].values())
        self.assertAlmostEqual(weighted / count, result["model"]["normalized_l1"])
        self.assertIsNone(self.ds._cache_table)

    def test_prediction_errors_restore_mode_and_clear_cache(self):
        policy = FakePolicy()
        with patch.object(policy, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                module.evaluate_stream(policy, self.ds, 3, "cpu", progress=None)
        self.assertTrue(policy.training)
        self.assertIsNone(self.ds._cache_table)

    def test_training_partition_and_incomplete_or_duplicate_rows_rejected(self):
        train = ActDataset(self.data, self.stats, self.policy_config, partition="train")
        with self.assertRaisesRegex(ValueError, "validation partition"):
            module.evaluate_stream(FakePolicy(), train, 3, "cpu", progress=None)
        for batches, message in (([[0, 1, 2]], "Incomplete"), ([[0, 1], [0, 1]], "missing, duplicated")):
            with patch.object(module, "EpisodeBatchSampler", return_value=batches):
                with self.assertRaisesRegex(ValueError, message):
                    module.evaluate_stream(FakePolicy(), self.ds, 3, "cpu", progress=None)

    def test_changed_mask_rejected_and_shared_processor_used(self):
        from src.contracts import standard_image_processing as shared
        original = ActDataset.__getitem__
        def broken(ds, index):
            row = original(ds, index)
            row["is_pad"] = ~row["is_pad"]
            return row
        with patch.object(ActDataset, "__getitem__", broken):
            with self.assertRaisesRegex(ValueError, "Padding mask"):
                module.evaluate_stream(FakePolicy(), self.ds, 3, "cpu", progress=None)
        with patch.object(shared, "image_to_chw", wraps=shared.image_to_chw) as process:
            module.evaluate_stream(FakePolicy(), self.ds, 3, "cpu", progress=None)
            self.assertEqual(process.call_count, len(self.ds) * 3)

    def test_plots_and_report_use_masks_and_no_overwrites(self):
        episodes = {}
        result = module.evaluate_stream(FakePolicy(), self.ds, 3, "cpu", on_episode=lambda eid, a: episodes.update({eid: a}), progress=None)
        result["status"] = "SYNTHETIC_TEST_ONLY"
        result["reproduction"] = reproduction_check(result["model"], result["train_mean_constant"], result["model"])
        plots = self.parent / "plots"
        plots.mkdir()
        eid, arrays = next(iter(episodes.items()))
        manifest = {"source": {"run_name": "synthetic", "checkpoint_name": "best.ckpt", "step": 10}}
        with _plot_cache(self.parent):
            plot_episode(plots, eid, arrays, [2])  # exactly one valid horizon before the gap
            plot_summary(plots, result)
            with self.assertRaises(FileExistsError):
                plot_summary(plots, result)
        write_report(self.parent, result, manifest, {str(eid): [2]})
        self.assertEqual(len(list(plots.glob("*.png"))), 4)
        for image in plots.glob("*.png"):
            with Image.open(image) as png:
                png.verify()
                self.assertGreater(png.width, 600)
        report = (self.parent / "report.md").read_text()
        self.assertIn("不是模型自回归或闭环", report)
        self.assertIn("不是全新的独立测试集", report)
        self.assertIn("共 1 个 chunk", report)


class CheckpointTests(FixtureTests):
    def setUp(self):
        super().setUp()
        self.make_checkpoint()

    def read(self, name="best"):
        return checkpoint.load_checkpoint(self.source_logs, self.source_ckpts, name)

    def test_safe_load_contract_and_unchanged_guards(self):
        with patch("torch.load", wraps=torch.load) as load:
            checked, docs = self.read()
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        checkpoint.verify_contract(checked, docs, self.config, self.data)
        checked.assert_unchanged()
        # Files with the same bytes but a new write/signature also invalidate the evidence.
        path = self.source_logs / "run_config.json"
        path.write_text(path.read_text())
        with self.assertRaisesRegex(ValueError, "changed"):
            checked.assert_unchanged()

    def test_rejects_unsafe_or_stale_checkpoint_and_missing_completion(self):
        original = deepcopy(self.payload)
        for key, value, message in (("format", "wp_ws.act_overfit.v1", "baseline checkpoint"),
                                    ("step", 11, "stale"),
                                    ("normalization_stats", {}, "statistics mismatch")):
            torch.save(dict(original, **{key: value}), self.source_ckpts / "best.ckpt")
            with self.assertRaisesRegex(ValueError, message):
                self.read()
        torch.save(original, self.source_ckpts / "best.ckpt")
        (self.source_logs / "COMPLETED.json").write_text('{"status":"RUNNING"}')
        with self.assertRaisesRegex(ValueError, "completed training"):
            self.read()

    def test_contract_mismatches_fail_closed(self):
        checked, docs = self.read()
        with self.assertRaisesRegex(ValueError, "Data configuration"):
            checkpoint.verify_contract(checked, docs, replace(self.config, task_name="other_task"), self.data)
        cases = [("validation_selection", {"mode": "fixed_evenly_spaced_smoke_subset"}, "batch size|full fixed"),
                 ("image_preprocessing", {}, "image preprocessing"),
                 ("provenance", {}, "provenance"), ("implementation_sha256", {}, "implementation changed")]
        for key, value, message in cases:
            modified = deepcopy(checked)
            modified.payload["run_config"][key] = value
            with self.assertRaisesRegex(ValueError, message):
                checkpoint.verify_contract(modified, docs, self.config, self.data)
        modified = deepcopy(checked)
        modified.payload["validation"]["valid_action_elements"] += 14
        with self.assertRaisesRegex(ValueError, "mask/count"):
            checkpoint.verify_contract(modified, docs, self.config, self.data)
        wrong = dict(docs, **{"audit_split.json": {}})
        with self.assertRaisesRegex(ValueError, "split.json"):
            checkpoint.verify_contract(checked, wrong, self.config, self.data)
        stats = deepcopy(self.stats)
        stats["train_episode_indices"] = list(self.ds.episode_ids)
        with self.assertRaisesRegex(ValueError, "Normalization provenance"):
            ActDataset(self.data, stats, self.policy_config, partition="val")

    def test_last_drops_optimizer_and_checks_exact_final_reference(self):
        payload = dict(self.payload, optimizer_state_dict={"not_used": torch.zeros(1)}, rng_state={"not_used": 3})
        torch.save(payload, self.source_ckpts / "last.ckpt")
        checked, _ = self.read("last")
        self.assertNotIn("optimizer_state_dict", checked.payload)
        self.assertNotIn("rng_state", checked.payload)
        completed = json.loads((self.source_logs / "COMPLETED.json").read_text())
        completed["steps"] += 1
        (self.source_logs / "COMPLETED.json").write_text(json.dumps(completed))
        with self.assertRaisesRegex(ValueError, "Last checkpoint"):
            self.read("last")

    def test_symlink_input_and_restricted_loader_failure_no_unsafe_retry(self):
        target = self.source_ckpts / "best.ckpt"
        target.rename(self.source_ckpts / "original.ckpt")
        target.symlink_to("original.ckpt")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.read()
        with patch("torch.load", side_effect=ValueError("safe loader refuses")) as load:
            with self.assertRaisesRegex(ValueError, "safe loader refuses"):
                self.read("last")
            self.assertEqual(load.call_count, 1)
            self.assertTrue(load.call_args.kwargs["weights_only"])

    def path_patches(self):
        stack = ExitStack()
        for owner, key, value in ((checkpoint, "LOG_DIR", self.log_root), (checkpoint, "CKPT_DIR", self.ckpt_root),
                                  (checkpoint, "DATA_DIR", self.parent / "data")):
            stack.enter_context(patch.object(owner, key, value))
        stack.enter_context(patch("src.wp_ws.audit_dataset.DATA_DIR", self.parent / "data"))
        stack.enter_context(patch("src.wp_ws.audit_dataset.LOG_DIR", self.log_root))
        return stack

    def test_paths_no_overwrite_traversal_symlinks_or_ckpt_collisions(self):
        with self.path_patches():
            logs, ckpts, out = checkpoint.locations(self.config, "trained", "eval1")
            self.assertEqual((logs, ckpts), (self.source_logs, self.source_ckpts))
            for name in ("trained", "audit_v1", "../x", "/tmp/eval"):
                with self.assertRaises(ValueError):
                    checkpoint.locations(self.config, "trained", name)
            out.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                checkpoint.locations(self.config, "trained", "eval1")
            (out.parent / "escape").symlink_to(self.parent / "escape", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                checkpoint.locations(self.config, "trained", "escape")
            with patch.object(checkpoint, "DATA_DIR", self.log_root):
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    checkpoint.locations(self.config, "trained", "eval2")

    def test_complete_cli_with_simulated_gpu_calls_and_cpu_fake_policy(self):
        # The CLI orchestration is exercised, but ALL CUDA APIs/inference are
        # replaced. This never constructs native ACT or touches a real GPU.
        class CpuShim(FakePolicy):
            def __init__(self, config):
                super().__init__(chunk=config["num_queries"])
                self.model = SimpleNamespace(use_channels_last=lambda: None)
            def deserialize(self, state):
                return self.load_state_dict(state, strict=True)
            def to(self, device):
                assert device == "cuda"
                return self
            def configure_optimizers(self):
                raise AssertionError("must not optimize")
            def load_pretrained_backbone(self):
                raise AssertionError("must not download")
        original_stream = module.evaluate_stream
        def cpu_stream(policy, dataset, batch_size, device, **kwargs):
            self.assertEqual(device, "cuda")
            return original_stream(policy, dataset, batch_size, "cpu", **kwargs)
        before = self.snapshot_inputs()
        with self.path_patches(), ExitStack() as stack, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            for method, value in (("is_available", True), ("is_bf16_supported", True),
                                  ("set_device", None), ("reset_peak_memory_stats", None)):
                stack.enter_context(patch.object(torch.cuda, method, return_value=value))
            # Restore ONLY this injected module. patch.dict(sys.modules) would
            # also remove Matplotlib C extensions first imported while plotting,
            # which cannot then be safely imported a second time in this process.
            previous = sys.modules.get("policy.act")
            sys.modules["policy.act"] = SimpleNamespace(ACTPolicy=CpuShim)
            def restore_import():
                if previous is None:
                    sys.modules.pop("policy.act", None)
                else:
                    sys.modules["policy.act"] = previous
            stack.callback(restore_import)
            stack.enter_context(patch.object(module, "evaluate_stream", side_effect=cpu_stream))
            stack.enter_context(patch.object(module, "_cuda_memory", return_value={"device": "SYNTHETIC_CPU_ONLY"}))
            self.assertEqual(module.main(["evaluate", "--source-run", "trained", "--run-name", "sim_eval", "--examples-per-episode", "1"]), 0)
        out = self.source_logs.parent / "sim_eval"
        completed = json.loads((out / "COMPLETED.json").read_text())
        metrics = json.loads((out / "metrics.json").read_text())
        self.assertEqual(completed["status"], "OFFLINE_EVALUATION_COMPLETED")
        self.assertFalse(completed["task_success_rate_measured"])
        self.assertFalse(completed["live_deployment_ready"])
        self.assertTrue(metrics["reproduction"]["within_tolerance"])
        self.assertEqual(metrics["model"]["samples"], len(self.ds))
        self.assertTrue((out / "report.md").is_file())
        self.assertFalse((out / "FAILED.json").exists())
        self.assertEqual(len(list((out / "predictions").glob("*.npz"))), 2)
        self.assertEqual(before, self.snapshot_inputs())

    def test_preflight_cli_cpu_only_completion_failure_and_no_reuse(self):
        before = self.snapshot_inputs()
        with self.path_patches(), patch("torch.cuda.is_available", side_effect=AssertionError("must not access CUDA")), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = module.main(["preflight", "--source-run", "trained", "--run-name", "preflight1"])
            self.assertEqual(code, 0)
            out = self.source_logs.parent / "preflight1"
            completed = json.loads((out / "COMPLETED.json").read_text())
            self.assertEqual(completed["status"], "EVALUATION_PREFLIGHT_PASSED")
            self.assertFalse(completed["gpu_evaluation_verified"])
            snapshot = {p.name: hash_file(p) for p in out.iterdir() if p.is_file()}
            code = module.main(["preflight", "--source-run", "trained", "--run-name", "preflight1"])
            self.assertEqual(code, 1)
            self.assertEqual(snapshot, {p.name: hash_file(p) for p in out.iterdir() if p.is_file()})
            with patch.object(module, "prepare", side_effect=RuntimeError("synthetic failure")):
                code = module.main(["preflight", "--source-run", "trained", "--run-name", "failed1"])
            self.assertEqual(code, 1)
            failed = self.source_logs.parent / "failed1"
            self.assertTrue((failed / "FAILED.json").is_file())
            self.assertFalse((failed / "COMPLETED.json").exists())
        self.assertEqual(before, self.snapshot_inputs())


if __name__ == "__main__":
    unittest.main()
