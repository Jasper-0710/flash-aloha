"""Synthetic offline tests; never read recordings or start GPU/robot training.

User-run: python -m unittest src.wp_ws.tests.test_act_baseline -v
"""
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from src.contracts import standard_image_processing as shared_images
from src.wp_ws.act_dataset import (
    ActDataset, EpisodeBatchSampler, Moments, fit_training_statistics, validation_batches,
)
from src.wp_ws.audit_core import AuditConfig, EXPECTED_CAMERAS, audit_dataset, hash_file
from src.wp_ws.audit_dataset import write_artifacts
from src.wp_ws.baseline_data_config import load_config
from src.wp_ws.tests.test_audit_dataset import edit_column, fixture
from src.wp_ws import train_act
from src.wp_ws.splits import fixed_episode_split
from src.wp_ws.verified_data import _verify_at


def hashes(root):
    return {str(p.relative_to(root)): hash_file(p) for p in root.rglob("*") if p.is_file()}


class ActBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        parent = Path(self.temporary.name)
        self.root, self.audit = parent / "data", parent / "audit"
        self.config = load_config()
        self.policy_config = {"height": 6, "width": 8, "num_queries": 5}
        fixture(self.root, count=5, length=6)
        for eid in range(5):
            def varied(values, offset=eid):
                for row, value in enumerate(values):
                    value[0] = offset * 10.0 + row
                    value[1] = row * .5
            edit_column(self.root, "action", varied, eid=eid)
            edit_column(self.root, "observation.state", varied, eid=eid)

    def publish(self):
        result = audit_dataset(self.root, AuditConfig(val_episodes=2))
        self.assertEqual(result["report"]["status"], "PASS_WITH_WARNINGS")
        self.assertTrue(result["continuity"]["usable"])
        write_artifacts(self.audit, result)
        return result

    def load(self):
        return _verify_at(self.root, self.audit, self.config)

    def edit_audit(self, name, edit):
        path = self.audit / name
        document = json.loads(path.read_text())
        edit(document)
        path.write_text(json.dumps(document), encoding="utf-8")

    def ready(self):
        self.publish()
        data = self.load()
        stats = fit_training_statistics(data, progress=None)
        return data, stats

    def dataset_for(self, data, stats, eid):
        partition = "train" if eid in data.split["train_episode_indices"] else "val"
        return ActDataset(data, stats, self.policy_config, partition=partition)

    def add_boundaries(self):
        edit_column(self.root, "recording.segment_index", lambda v: v.__setitem__(slice(3, 6), [1, 1, 1]))
        edit_column(self.root, "recording.elapsed_time", lambda v: v.__setitem__(5, .201))
        path = self.root / "meta/episodes.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        entries[0]["segments"].append({"segment_index": 1, "start_frame": 3})
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    def test_verify_and_dataset_never_modify_source(self):
        before = hashes(self.root)
        data, stats = self.ready()
        for partition in ("train", "val"):
            ds = ActDataset(data, stats, self.policy_config, partition=partition)
            ds[0]
            ds[len(ds) - 1]
        data.assert_unchanged()
        self.assertEqual(hashes(self.root), before)
        self.assertEqual(set(data.split["train_episode_indices"]) | set(data.split["val_episode_indices"]), set(range(5)))

    def test_source_tampering_rejected(self):
        self.publish()
        edit_column(self.root, "action", lambda values: values[0].__setitem__(0, 999.0))
        with self.assertRaisesRegex(ValueError, "does not match audit"):
            self.load()

    def test_runtime_source_change_rejected_even_with_cached_episode(self):
        data, stats = self.ready()
        ds = ActDataset(data, stats, self.policy_config, partition="train")
        sample = ds[0]
        edit_column(self.root, "action", lambda values: values[0].__setitem__(0, 999.0), eid=sample["episode_index"])
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            ds[0]

    def test_extra_uncommitted_parquet_rejected(self):
        self.publish()
        (self.root / "data/chunk-000/extra.parquet").write_bytes(
            (self.root / "data/chunk-000/episode_000000.parquet").read_bytes())
        with self.assertRaisesRegex(ValueError, "membership"):
            self.load()

    def test_audit_completion_and_fingerprint_disagreement_rejected(self):
        self.publish()
        original = (self.audit / "COMPLETED.json").read_bytes()
        self.edit_audit("COMPLETED.json", lambda d: d.update(split_ready=False))
        with self.assertRaisesRegex(ValueError, "COMPLETED"):
            self.load()
        (self.audit / "COMPLETED.json").write_bytes(original)
        self.edit_audit("split.json", lambda d: d["dataset_fingerprint"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.load()

    def test_edited_split_is_rejected_not_resampled(self):
        self.publish()
        self.edit_audit("split.json", lambda d: d["val_episode_indices"].__setitem__(0, d["train_episode_indices"][0]))
        with self.assertRaisesRegex(ValueError, "Fixed split"):
            self.load()

    def test_incomplete_continuity_rejected(self):
        self.publish()
        self.edit_audit("continuity.json", lambda d: d["episodes"][0]["contiguous_ranges"][0].update(stop_frame_exclusive=5))
        with self.assertRaisesRegex(ValueError, "continuity ranges"):
            self.load()

    def test_erasing_boundaries_in_both_json_files_still_rejected(self):
        self.add_boundaries()
        self.publish()
        merged = [{"start_frame": 0, "stop_frame_exclusive": 6, "segment_index": 0,
                   "start_reasons": ["episode_start"]}]
        self.edit_audit("continuity.json", lambda d: d["episodes"][0].update(contiguous_ranges=merged))
        self.edit_audit("report.json", lambda d: d["episodes"][0]["timing"].update(contiguous_ranges=merged))
        with self.assertRaisesRegex(ValueError, "saved rows"):
            self.load()

    def test_unexpected_task_label_rejected(self):
        self.publish()
        with self.assertRaisesRegex(ValueError, "source task label"):
            _verify_at(self.root, self.audit, replace(self.config, source_task_name="other"))

    def test_statistics_use_only_training_rows(self):
        # Holdout-only outliers would visibly corrupt a leaking estimator.
        holdouts = fixed_episode_split(range(5), val_episodes=2)["val_episode_indices"]
        for eid in holdouts:
            for key in ("action", "observation.state"):
                edit_column(self.root, key, lambda values: [v.__setitem__(0, 1000000.) for v in values], eid=eid)
        artifacts = self.publish()
        data = self.load()
        stats = fit_training_statistics(data, progress=None)
        rows = []
        for eid in artifacts["split"]["train_episode_indices"]:
            rows.extend(pq.ParquetFile(self.root / data.episodes[eid].data_path).read(columns=["action"])["action"].to_pylist())
        reference = np.asarray(rows, dtype=np.float64)
        np.testing.assert_allclose(stats["action"]["mean"], reference.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(stats["action"]["std"], np.maximum(reference.std(axis=0), .01), atol=1e-12)
        self.assertEqual(stats["rows"], reference.shape[0])
        self.assertEqual(stats["train_episode_indices"], data.split["train_episode_indices"])
        self.assertLess(stats["action"]["mean"][0], 100.)
        self.assertEqual(stats["rows"], 18)
        self.assertAlmostEqual(stats["action"]["std"][6], .01)
        self.assertNotEqual(stats["state"]["mean"][6], stats["action"]["mean"][6])

    def test_normalization_provenance_mismatch_refused(self):
        data, stats = self.ready()
        stats["train_episode_indices"] = data.split["val_episode_indices"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            ActDataset(data, stats, self.policy_config, partition="train")

    def test_raw_grippers_and_exact_same_row_actions_preserved(self):
        data, stats = self.ready()
        ds = self.dataset_for(data, stats, 0)
        sample = ds[ds.sample_index(0, 2)]
        table = pq.ParquetFile(self.root / data.episodes[0].data_path).read(columns=["action", "observation.state"])
        raw = sample["actions"] * ds.std["action"] + ds.mean["action"]
        np.testing.assert_allclose(raw[~sample["is_pad"]], table["action"].slice(2, 4).to_pylist(), atol=2e-6)
        np.testing.assert_allclose(sample["qpos"] * ds.std["state"] + ds.mean["state"],
                                   table["observation.state"][2].as_py(), atol=2e-6)
        self.assertGreater(raw[0, 6], 1.0)
        self.assertLess(raw[0, 13], 0.0)
        self.assertEqual((sample["frame_index"], sample["task_index"]), (2, 0))

    def test_chunk_padding_never_crosses_segment_gap_or_episode(self):
        self.add_boundaries()
        data, stats = self.ready()
        ds = self.dataset_for(data, stats, 0)
        self.assertEqual(data.episodes[0].ranges, ((0, 3), (3, 5), (5, 6)))
        for row, expected_valid in ((0, 3), (1, 2), (3, 2), (4, 1), (5, 1)):
            sample = ds[ds.sample_index(0, row)]
            self.assertEqual(int((~sample["is_pad"]).sum()), expected_valid)
            self.assertEqual(sample["is_pad"].dtype, np.bool_)
            np.testing.assert_array_equal(sample["actions"][expected_valid:], 0.)
        # The last saved row is retained even if only one target is valid.
        tail = ds[len(ds) - 1]
        self.assertEqual((~tail["is_pad"]).sum(), 1)

    def test_images_call_shared_rgb_processor_and_match_existing_profile(self):
        data, stats = self.ready()
        config = {**self.policy_config, "height": 3, "width": 4}
        ds = ActDataset(data, stats, config, partition="train")
        original = shared_images.image_to_chw
        with patch.object(shared_images, "image_to_chw", wraps=original) as processor:
            sample = ds[0]
        self.assertEqual(processor.call_count, 3)
        self.assertTrue(all(call.kwargs == {"source_color": "RGB"} for call in processor.call_args_list))
        eid, row = ds.locate(0)
        table = pq.ParquetFile(self.root / data.episodes[eid].data_path).read(columns=list(EXPECTED_CAMERAS))
        for i, key in enumerate(EXPECTED_CAMERAS):
            with Image.open(BytesIO(table[key][row].as_py()["bytes"])) as image:
                expected = original(np.asarray(image), shared_images.policy_image_spec({"policy_config": config}), source_color="RGB")
            np.testing.assert_array_equal(sample["images"][i], expected)
        self.assertEqual(sample["images"].shape, (3, 3, 3, 4))
        np.testing.assert_allclose(sample["images"][0, :, 0, 0], np.asarray([30, 60, 90]) / 255.)

    def test_episode_sampler_covers_every_row_without_dropping(self):
        data, stats = self.ready()
        ds = ActDataset(data, stats, self.policy_config, partition="train")
        sampler = EpisodeBatchSampler(ds, 4, shuffle=True, seed=42)
        first = list(sampler)
        self.assertEqual(len(first), len(sampler))
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(len(ds))))
        self.assertEqual(first, list(EpisodeBatchSampler(ds, 4, shuffle=True, seed=42)))
        self.assertTrue(all(len({ds.locate(i)[0] for i in batch}) == 1 for batch in first))
        self.assertTrue(any(len(batch) < 4 for batch in first))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_validation_subset_fixed_and_full_mode_covers_all(self):
        data, stats = self.ready()
        ds = ActDataset(data, stats, self.policy_config, partition="val")
        batches, selection = validation_batches(ds, 2, 3)
        self.assertEqual((batches, selection), validation_batches(ds, 2, 3))
        self.assertEqual(selection["sample_count"], 6)
        indices = [i for batch in batches for i in batch]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual((indices[0], indices[-1]), (0, len(ds) - 1))
        full, full_selection = validation_batches(ds, 4, 0)
        self.assertEqual(full_selection["mode"], "full_fixed_validation")
        self.assertEqual([i for batch in full for i in batch], list(range(len(ds))))


class TrainingCliTests(unittest.TestCase):
    def test_default_is_small_native_bf16_smoke(self):
        args = train_act.build_parser().parse_args(["train"])
        config = train_act.policy_configuration(args)
        self.assertEqual((args.steps, args.batch_size), (100, 4))
        self.assertEqual((config["height"], config["width"], config["num_queries"]), (240, 320, 50))
        self.assertEqual((config["precision"], config["state_dim"], config["action_dim"]), ("bf16", 14, 14))
        self.assertFalse(config["training_time_rtc"])
        self.assertIsNone(config["action_idx"])
        self.assertEqual(config["pretrained_backbone_weights"], "IMAGENET1K_V1")

    def test_output_locations_use_central_roots_and_reject_audit_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            data, logs, ckpts = (parent / name for name in ("data", "logs", "ckpts"))
            with patch("src.wp_ws.audit_dataset.DATA_DIR", data), patch("src.wp_ws.audit_dataset.LOG_DIR", logs), \
                    patch.object(train_act, "LOG_DIR", logs), patch.object(train_act, "CKPT_DIR", ckpts):
                paths = train_act.output_locations(load_config(), "smoke_v1", training=True)
                self.assertEqual(paths, [logs / "wp_ws/flip_the_pillow/smoke_v1", ckpts / "wp_ws/flip_the_pillow/smoke_v1"])
                with self.assertRaisesRegex(ValueError, "source/audit"):
                    train_act.output_locations(load_config(), "audit_v1", training=True)
                paths[0].mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    train_act.output_locations(load_config(), "smoke_v1", training=True)

    def test_existing_run_never_receives_failed_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            existing = parent / "logs/wp_ws/flip_the_pillow/existing"
            existing.mkdir(parents=True)
            (existing / "sentinel").write_text("do not change")
            with patch.object(train_act, "LOG_DIR", parent / "logs"), patch.object(train_act, "CKPT_DIR", parent / "ckpts"):
                self.assertEqual(train_act.main(["preflight", "--run-name", "existing"]), 1)
            self.assertEqual(list(existing.iterdir()), [existing / "sentinel"])

    def test_merged_moments_match_population_and_reject_nonfinite(self):
        values = np.arange(42, dtype=np.float64).reshape(3, 14)
        moments = Moments(14)
        moments.add(values[:1])
        moments.add(values[1:])
        result = moments.export(.01)
        np.testing.assert_allclose(result["mean"], values.mean(axis=0))
        np.testing.assert_allclose(result["std"], values.std(axis=0))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            moments.add(np.full((1, 14), np.nan))


if __name__ == "__main__":
    unittest.main()
