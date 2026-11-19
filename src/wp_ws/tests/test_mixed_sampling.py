"""Synthetic CPU/offline checks. No real data edits, GPU training or robot access."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import torch

from src.wp_ws import train_act
from src.wp_ws.act_dataset import ActDataset, EpisodeBatchSampler, fit_training_statistics
from src.wp_ws.audit_core import AuditConfig, audit_dataset
from src.wp_ws.audit_dataset import write_artifacts
from src.wp_ws.baseline_data_config import load_config
from src.wp_ws.mixed_sampling import WindowBatchSampler, WindowCachedActDataset
from src.wp_ws.tests.test_audit_dataset import fixture, edit_column
from src.wp_ws.training_probe import evaluate_probe, evaluate_validation, prepare_probe, probe_indices, write_report
from src.wp_ws.verified_data import _verify_at


class SamplingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root, self.audit = self.base / "data", self.base / "audit"
        fixture(self.root, count=7, length=6)
        # Nontrivial labels + a continuity boundary; unchanged image/action pairing must survive caching.
        for eid in range(7):
            def varied(values, offset=eid):
                for row, value in enumerate(values):
                    value[0] = offset * 3 + row
            edit_column(self.root, "action", varied, eid=eid)
            edit_column(self.root, "recording.elapsed_time",
                        lambda v: v.__setitem__(slice(3, 6), [.25, .27, .29]), eid=eid)
        write_artifacts(self.audit, audit_dataset(self.root, AuditConfig(val_episodes=2)))
        self.data = _verify_at(self.root, self.audit, load_config())
        self.stats = fit_training_statistics(self.data, progress=None)
        self.policy = {"height": 6, "width": 8, "num_queries": 5}
        self.plain = ActDataset(self.data, self.stats, self.policy, partition="train")
        self.mixed = WindowCachedActDataset(self.data, self.stats, self.policy, window_episodes=2, cache_mib=1)

    def test_exact_row_coverage_epoch_determinism_and_mixing(self):
        sampler = WindowBatchSampler(self.mixed, 4, seed=42)
        before = np.random.get_state()
        first = list(sampler)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(len(first), len(sampler))
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(len(self.mixed))))
        self.assertTrue(any(len(batch) < 4 for batch in first))
        self.assertTrue(any(len({self.mixed.locate(i)[0] for i in batch}) > 1 for batch in first))
        self.assertTrue(all(self.mixed.locate(i)[0] in self.data.split["train_episode_indices"]
                            for batch in first for i in batch))
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        self.assertEqual(len(list(sampler)), len(sampler))

    def test_every_array_and_mask_matches_original_adapter(self):
        has_short_chunk = has_raw_gripper_outside = False
        for batch in WindowBatchSampler(self.mixed, 4):
            for index in batch:
                actual, expected = self.mixed[index], self.plain[index]
                self.assertEqual(actual.keys(), expected.keys())
                for key in actual:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
                has_short_chunk |= bool(actual["is_pad"].any())
                raw = actual["actions"][~actual["is_pad"]] * self.mixed.std["action"] + self.mixed.mean["action"]
                has_raw_gripper_outside |= bool(((raw[:, [6, 13]] < 0) | (raw[:, [6, 13]] > 1)).any())
        self.assertTrue(has_short_chunk and has_raw_gripper_outside)
        self.data.assert_unchanged()

    def test_cache_reads_once_per_window_and_releases_old_buffers(self):
        ids = list(self.mixed.episode_ids[:2])
        self.mixed.begin_window(ids)
        with patch("src.wp_ws.act_dataset.pq.ParquetFile", wraps=pq.ParquetFile) as reader:
            for eid in ids * 3:
                self.mixed[self.mixed.sample_index(eid, 0)]
            self.assertEqual(reader.call_count, 2)
        self.assertEqual(self.mixed.table_reads, 2)
        self.assertIsNone(self.mixed._cache_table)
        self.assertIsNone(self.mixed._cache_eid)
        self.assertEqual(self.mixed.cache_bytes, sum(t.get_total_buffer_size() for t in self.mixed._tables.values()))
        self.assertLessEqual(self.mixed.cache_bytes, self.mixed.cache_limit_bytes)
        self.mixed.begin_window([self.mixed.episode_ids[2]])
        self.assertEqual((self.mixed._tables, self.mixed.cache_bytes), ({}, 0))
        with self.assertRaisesRegex(ValueError, "outside active"):
            self.mixed[self.mixed.sample_index(ids[0], 0)]
        self.mixed.clear_cache()
        self.assertEqual(self.mixed._window, ())

    def test_oversize_window_fails_instead_of_eviction_thrashing(self):
        ids = list(self.mixed.episode_ids[:2])
        self.mixed.begin_window(ids)
        first_size = self.mixed._table(ids[0]).get_total_buffer_size()
        self.mixed.cache_limit_bytes = first_size
        with self.assertRaisesRegex(ValueError, "exceeds --episode-cache-mib"):
            self.mixed._table(ids[1])
        self.assertEqual(self.mixed.cache_bytes, first_size)
        self.assertEqual(set(self.mixed._tables), {ids[0]})
        self.assertIsNone(self.mixed._cache_table)

    def test_pickle_drops_caches_and_cached_file_changes_fail(self):
        eid = self.mixed.episode_ids[0]
        self.mixed.begin_window([eid])
        self.mixed._table(eid)
        restored = pickle.loads(pickle.dumps(self.mixed))
        self.assertEqual((restored._window, restored._tables, restored.cache_bytes), ((), {}, 0))
        self.assertIsNone(restored._cache_table)
        edit_column(self.root, "action", lambda v: v[0].__setitem__(0, 999.), eid=eid)
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            self.mixed._table(eid)

    def test_invalid_sampler_usage_rejected(self):
        with self.assertRaisesRegex(ValueError, "training episodes"):
            self.mixed.begin_window(self.data.split["val_episode_indices"])
        with self.assertRaisesRegex(ValueError, "Invalid cache window"):
            self.mixed.begin_window([self.mixed.episode_ids[0]] * 2)
        with self.assertRaises(ValueError):
            WindowBatchSampler(self.plain, 4)
        args = train_act.build_parser().parse_args(["train", "--sampler", "episode_window", "--workers", "1"])
        with patch("src.wp_ws.verified_data.load_verified_data") as load, self.assertRaisesRegex(ValueError, "workers 0"):
            train_act.prepare(args, load_config(), self.base)
        load.assert_not_called()

    def test_default_sampler_is_unchanged(self):
        args = train_act.build_parser().parse_args(["train"])
        self.assertEqual(args.sampler, "episode_block")
        self.assertEqual(args.train_probe_per_episode, 0)
        self.assertFalse(args.eval_at_start)
        sampler = EpisodeBatchSampler(self.plain, 4, shuffle=True)
        self.assertTrue(all(len({self.plain.locate(i)[0] for i in batch}) == 1 for batch in sampler))

    def test_cpu_dataloader_crosses_windows_epochs_and_independent_validation(self):
        from torch.utils.data import DataLoader
        sampler = WindowBatchSampler(self.mixed, 4)
        loader = DataLoader(self.mixed, batch_sampler=sampler, num_workers=0)
        probe, _, _ = prepare_probe(self.plain, 2, 1)
        for epoch in range(2):
            sampler.set_epoch(epoch)
            seen = []
            for batch in loader:
                seen.extend(zip(batch["episode_index"].tolist(), batch["frame_index"].tolist()))
                # Probe eval clears only its own adapter cache, not the active training window.
                evaluate_probe(ZeroPolicy(5), probe, self.stats["action"]["std"], 4, "cpu", 1.)
                self.assertLessEqual(len(self.mixed._tables), 2)
            self.assertEqual(sorted(seen), sorted(self.plain.locate(i) for i in range(len(self.plain))))
        self.assertEqual(self.mixed.table_reads, 2 * len(self.mixed.episode_ids))

    def test_probe_selection_deterministic_train_only_and_shared_arrays(self):
        indices = probe_indices(self.plain, 2)
        self.assertEqual(indices, probe_indices(self.plain, 2))
        self.assertEqual(len(indices), len(self.plain.episode_ids) * 2)
        self.assertEqual({self.plain.locate(i)[1] for i in indices}, {1, 4})
        cache, selection, baselines = prepare_probe(self.plain, 2, 1)
        self.assertEqual(selection["algorithm"], "sorted-train-episodes-midpoint-strata-v1")
        self.assertIn("NOT_held_out", selection["evaluation_scope"])
        self.assertEqual(selection["episode_indices"], sorted(self.plain.episode_ids))
        for position, index in enumerate(indices):
            sample = self.plain[index]
            for key, array in cache.arrays.items():
                np.testing.assert_array_equal(array[position], sample[key])
                self.assertFalse(array.flags.writeable)
        self.assertGreaterEqual(baselines["train_mean_constant"]["normalized_l1"], 0)
        with self.assertRaisesRegex(ValueError, "training partition"):
            probe_indices(ActDataset(self.data, self.stats, self.policy, partition="val"), 2)
        with self.assertRaisesRegex(ValueError, "too short"):
            probe_indices(self.plain, 7)
        with self.assertRaisesRegex(ValueError, "cache-limit"):
            prepare_probe(self.plain, 2, 0)

    def test_probe_evaluation_matches_constant_and_restores_mode(self):
        cache, _, baselines = prepare_probe(self.plain, 2, 1)
        policy = ZeroPolicy(self.policy["num_queries"])
        baseline = baselines["train_mean_constant"]["normalized_l1"]
        result = evaluate_probe(policy, cache, self.stats["action"]["std"], 4, "cpu", baseline)
        self.assertAlmostEqual(result["normalized_l1"], baseline, places=7)
        self.assertTrue(policy.training)
        self.assertEqual(result["samples"], len(cache))
        self.assertIn("NOT_held_out", result["evaluation_scope"])

    def test_preflight_manifest_control_same_model_stats_split_and_no_gpu(self):
        outputs = []
        for mode in ("episode_block", "episode_window"):
            logs = self.base / mode
            logs.mkdir()
            args = train_act.build_parser().parse_args(["preflight", "--sampler", mode, "--mix-episodes", "2",
                "--height", "6", "--width", "8", "--chunk-size", "5", "--batch-size", "4",
                "--val-max-batches", "0", "--train-probe-per-episode", "2", "--eval-at-start"])
            with patch("src.wp_ws.verified_data.load_verified_data", return_value=self.data), \
                 patch.object(torch.cuda, "is_available", side_effect=AssertionError("No GPU checks")), \
                 redirect_stdout(io.StringIO()):
                prepared = train_act.prepare(args, load_config(), logs)
            self.assertEqual(len(prepared[-1]), 10)
            self.assertEqual(prepared[2]["validation_selection"]["sample_count"], 12)
            self.assertEqual(json.loads((logs / "preflight.json").read_text())["train_probe_samples"], 10)
            outputs.append(prepared)
        for key in ("policy_config", "provenance", "validation_selection", "train_probe_selection", "train_probe_baselines"):
            self.assertEqual(outputs[0][2][key], outputs[1][2][key], key)
        self.assertEqual(outputs[0][1], outputs[1][1])
        self.data.assert_unchanged()


class ZeroPolicy(torch.nn.Module):
    def __init__(self, chunk):
        super().__init__()
        self.chunk = chunk

    def forward(self, qpos, images, actions=None, is_pad=None):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Evaluation must use eval + inference_mode")
        pred = torch.zeros((len(qpos), self.chunk, 14), device=qpos.device)
        if actions is None:
            return pred
        valid = (~is_pad).unsqueeze(-1).expand_as(actions)
        return {"loss": (pred - actions).abs()[valid].mean()}


class ValidationTests(unittest.TestCase):
    def batches(self):
        return [{"qpos": torch.zeros(2, 14), "images": torch.zeros(2, 3, 3, 6, 8),
                 "actions": torch.stack([torch.ones(3, 14), torch.full((3, 14), 3.)]),
                 "is_pad": torch.tensor([[False, False, False], [False, True, True]])},
                {"qpos": torch.zeros(1, 14), "images": torch.zeros(1, 3, 3, 6, 8),
                 "actions": torch.full((1, 3, 14), 5.), "is_pad": torch.tensor([[False, True, True]])}]

    def test_weighted_native_loss_and_reference_same_masks_not_batch_means(self):
        policy = ZeroPolicy(3)
        result = evaluate_validation(policy, self.batches(), device="cpu", step=0, mode="full_fixed_validation")
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["valid_action_elements"], 5 * 14)
        self.assertAlmostEqual(result["normalized_l1"], 11 / 5)
        self.assertAlmostEqual(result["mean_constant_normalized_l1"], 11 / 5)
        self.assertTrue(policy.training)
        policy.eval()
        evaluate_validation(policy, self.batches(), device="cpu", step=0, mode="test")
        self.assertFalse(policy.training)

    def test_validation_failure_restores_mode(self):
        policy = ZeroPolicy(3)
        with self.assertRaisesRegex(ValueError, "Empty validation"):
            evaluate_validation(policy, [], device="cpu", step=0, mode="test")
        self.assertTrue(policy.training)
        batches = self.batches()
        batches[0]["actions"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            evaluate_validation(policy, batches, device="cpu", step=0, mode="test")
        self.assertTrue(policy.training)

    def test_report_chooses_validation_not_lower_train_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.md"
            history = [{"probe": {"normalized_l1": p, "mean_constant_normalized_l1": 1.},
                        "validation": {"step": s, "normalized_l1": v, "mean_constant_normalized_l1": .84}}
                       for s, p, v in ((0, 1., .9), (1000, .3, .6), (2000, .1, .7))]
            write_report(path, history, {"sampler_configuration": {"mode": "episode_window"},
                                        "validation_selection": {"mode": "full_fixed_validation"}})
            text = path.read_text()
            self.assertIn("step 1000", text)
            self.assertIn("不是验证集", text)
            self.assertFalse(path.with_suffix(".md.partial").exists())


if __name__ == "__main__":
    unittest.main()
