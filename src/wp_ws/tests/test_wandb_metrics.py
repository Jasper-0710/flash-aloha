"""Synthetic, stdlib-only tests; no W&B network, recorded data, training or GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.wp_ws import wandb_metrics as metrics


class FakeRun:
    url = None

    def __init__(self):
        self.summary = {}
        self.points = []
        self.definitions = []
        self.finished = []

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log(self, values):
        # Deliberately no step= argument: train/val can share a global_step.
        self.points.append(values)

    def finish(self, *, exit_code):
        self.finished.append(exit_code)


class FakeBackend:
    def __init__(self):
        self.run = FakeRun()

    def Settings(self, **kwargs):
        self.settings = kwargs
        return kwargs

    def init(self, **kwargs):
        self.arguments = kwargs
        return self.run


class WandbMetricsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, folder in (("LOG_DIR", "logs"), ("DATA_DIR", "data"), ("CKPT_DIR", "ckpts")):
            mocked = patch.object(metrics, name, self.root / folder)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.source = metrics.LOG_DIR / "wp_ws" / "flip_the_pillow" / "synthetic"
        self.source.mkdir(parents=True)
        manifest = {
            "format": "wp_ws.act_baseline.v1", "model_variant": "decoder_only_ACT",
            "data_config": {"task_name": "flip_the_pillow"},
            "validation_selection": {"mode": "full_fixed_validation"},
            "batch_size": 4, "seed": 42, "steps_requested": 100,
            "policy_config": {"height": 240, "width": 320, "action_dim": 14,
                              "api_key": "do-not-upload"},
            "cli_arguments": {"token": "do-not-upload"},
            "provenance": {"path": "/private/data"}, "environment": {"secret": "do-not-upload"},
        }
        (self.source / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.args = metrics.build_parser().parse_args(["--run-name", "synthetic"])

    def write_records(self, *records):
        path = self.source / "metrics.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        return path

    def source_hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}

    def test_default_offline_and_component_validation(self):
        self.assertEqual(self.args.mode, "offline")
        for bad in ("..", "../run", "x/y", "x\\y", " x", "x\n", "_wandb_uploads"):
            with self.assertRaises(argparse.ArgumentTypeError):
                metrics.source_directory("flip_the_pillow", bad)

    def test_maps_scalars_and_drops_arbitrary_payloads(self):
        row = {"event": "train", "step": 10, "normalized_l1": .4,
               "gradient_norm_before_clip": 2., "images": ["do-not-upload"],
               "action": [1.2], "cuda_memory": {"peak_allocated_gib": 1.1, "device": "name"}}
        self.assertEqual(metrics.metric_values(row), {
            "global_step": 10, "train/normalized_l1": .4,
            "train/gradient_norm_before_clip": 2., "cuda/peak_allocated_gib": 1.1})
        self.assertIsNone(metrics.metric_values({"event": "unrecognized", "secret": 1}))

    def test_nonfinite_loss_and_invalid_steps_rejected(self):
        for value in (float("nan"), float("inf"), "0.5", True):
            with self.assertRaises(ValueError):
                metrics.metric_values({"event": "train", "step": 1, "normalized_l1": value})
        for step in (-1, 1.5, True, None):
            with self.assertRaises(ValueError):
                metrics.metric_values({"event": "train", "step": step, "normalized_l1": .5})

    def test_configuration_is_allowlisted(self):
        result = metrics.scalar_configuration(self.source, "flip_the_pillow", "synthetic")
        serialized = json.dumps(result)
        for forbidden in ("do-not-upload", "provenance", "private", "cli_arguments", "environment"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(result["validation_mode"], "full_fixed_validation")
        self.assertEqual(result["policy"]["height"], 240)

    def test_same_step_train_validation_both_uploaded_and_source_unchanged(self):
        self.write_records(
            {"event": "model_ready", "trainable_parameters": 56},
            {"event": "train", "step": 50, "normalized_l1": .5},
            {"event": "validation", "step": 50, "normalized_l1": .8, "samples": 40},
            {"event": "completed", "steps": 50},
        )
        before = self.source_hashes()
        backend = FakeBackend()
        self.assertEqual(metrics.upload(self.args, backend=backend), 0)
        self.assertEqual(before, self.source_hashes())
        self.assertEqual([point["global_step"] for point in backend.run.points], [0, 50, 50, 50])
        self.assertEqual(backend.run.summary["source_status"], "completed")
        self.assertEqual(backend.run.finished, [0])
        self.assertIn(("val/normalized_l1", {"step_metric": "global_step", "summary": "min"}),
                      backend.run.definitions)
        output = Path(backend.arguments["dir"])
        self.assertTrue(output.is_relative_to(metrics.LOG_DIR / "wp_ws" / "_wandb_uploads"))
        self.assertFalse(output.is_relative_to(self.source))
        self.assertEqual(backend.settings["console"], "off")
        for setting in ("disable_code", "disable_git", "x_disable_stats", "x_disable_meta"):
            self.assertTrue(backend.settings[setting])

    def test_follow_waits_for_complete_line_without_dropping_it(self):
        path = self.source / "metrics.jsonl"
        path.write_bytes(b'{"event":"train","step":1,"normalized_l1":')
        sleeps = []

        def append(_):
            sleeps.append(1)
            with path.open("ab") as stream:
                stream.write(b'0.5}\n{"event":"completed","steps":1}\n')

        records = list(metrics.metric_records(self.source, follow=True, sleep=append))
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(records[0]["normalized_l1"], .5)
        self.assertEqual(len(records), 2)

    def test_follow_reads_new_records_at_eof(self):
        path = self.write_records({"event": "train", "step": 1, "normalized_l1": .5})

        def append(_):
            with path.open("ab") as stream:
                stream.write(b'{"event":"validation","step":1,"normalized_l1":0.8}\n')
                stream.write(b'{"event":"completed","steps":1}\n')

        records = list(metrics.metric_records(self.source, follow=True, sleep=append))
        self.assertEqual([row["event"] for row in records], ["train", "validation", "completed"])

    def test_partial_snapshot_and_malformed_records_rejected(self):
        path = self.source / "metrics.jsonl"
        for raw in (b'{"event":', b'[]\n', b'{invalid}\n'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                list(metrics.metric_records(self.source))

    def test_failed_training_without_metrics_stops_follower(self):
        (self.source / "FAILED.json").write_text('{"status":"FAILED"}')
        self.args.follow = True
        backend = FakeBackend()
        self.assertEqual(metrics.upload(self.args, backend=backend), 1)
        self.assertEqual(backend.run.summary["source_status"], "failed")
        self.assertEqual(backend.run.finished, [1])

    def test_snapshot_does_not_claim_training_completed(self):
        self.write_records({"event": "train", "step": 1, "normalized_l1": .5})
        backend = FakeBackend()
        self.assertEqual(metrics.upload(self.args, backend=backend), 0)
        self.assertEqual(backend.run.summary["source_status"], "snapshot")

    def test_backwards_step_finishes_failed_uploader(self):
        self.write_records({"event": "train", "step": 10, "normalized_l1": .5},
                           {"event": "validation", "step": 9, "normalized_l1": .6})
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "backwards"):
            metrics.upload(self.args, backend=backend)
        self.assertEqual(backend.run.finished, [1])

    def test_ctrl_c_only_stops_uploader(self):
        self.write_records({"event": "train", "step": 1, "normalized_l1": .5})
        before = self.source_hashes()
        self.args.follow = True
        backend = FakeBackend()

        def interrupt(_):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            metrics.upload(self.args, backend=backend, sleep=interrupt)
        self.assertEqual(before, self.source_hashes())
        self.assertEqual(backend.run.summary["source_status"], "uploader_stopped_by_user")
        self.assertEqual(backend.run.finished, [0])

    def test_source_symlink_and_output_overlap_rejected(self):
        outside = self.root / "outside.jsonl"
        outside.write_text('{}\n')
        (self.source / "metrics.jsonl").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside"):
            list(metrics.metric_records(self.source))
        with patch.object(metrics, "DATA_DIR", metrics.LOG_DIR):
            with self.assertRaisesRegex(ValueError, "overlap"):
                metrics._upload_directory(self.source, "flip_the_pillow", "synthetic")

    def test_repeated_imports_have_separate_local_and_wandb_runs(self):
        self.write_records({"event": "completed", "steps": 100})
        first, second = FakeBackend(), FakeBackend()
        metrics.upload(self.args, backend=first)
        metrics.upload(self.args, backend=second)
        self.assertNotEqual(first.arguments["id"], second.arguments["id"])
        self.assertNotEqual(first.arguments["dir"], second.arguments["dir"])

    def test_sdk_cache_and_staging_follow_central_logs_and_restore_environment(self):
        directory = self.root / "sdk"
        with patch.dict(os.environ, {"WANDB_CACHE_DIR": "/original/cache"}):
            before = dict(os.environ)
            with metrics._sdk_storage(directory):
                self.assertEqual(os.environ["WANDB_CACHE_DIR"], str(directory / "cache"))
                self.assertEqual(os.environ["WANDB_DATA_DIR"], str(directory / "staging"))
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
