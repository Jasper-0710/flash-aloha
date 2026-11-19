"""Config-only tests. Run: python -m unittest src.wp_ws.tests.test_baseline_data_config -v.

No recorded dataset, GPU, training, or robot is accessed.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.wp_ws import baseline_data_config as config


class BaselineDataConfigTests(unittest.TestCase):
    def test_task_alias_is_separate_from_source_label(self):
        value = config.load_config()
        self.assertEqual(value.task_name, "flip_the_pillow")
        self.assertEqual(value.task_description, "翻转枕头")
        self.assertEqual(value.source_task_name, "pick_cup")
        self.assertNotEqual(value.task_name, value.source_task_name)
        self.assertEqual(value.audit_run, "audit_v1")

    def test_no_source_reads_or_writes_when_previewing(self):
        value = config.load_config()
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            data, logs, checkpoints = parent / "data", parent / "logs", parent / "ckpts"
            with patch("src.wp_ws.audit_dataset.DATA_DIR", data), \
                    patch("src.wp_ws.audit_dataset.LOG_DIR", logs), \
                    patch.object(config, "CKPT_DIR", checkpoints):
                result = value.preview()
            self.assertEqual(result["status"], "CONFIG_PREVIEW_ONLY")
            self.assertEqual(result["paths"]["dataset"], str(data / "flip_the_pillow"))
            self.assertEqual(result["paths"]["split"],
                             str(logs / "wp_ws/flip_the_pillow/audit_v1/split.json"))
            self.assertEqual(result["paths"]["checkpoint_namespace"],
                             str(checkpoints / "wp_ws/flip_the_pillow"))
            self.assertEqual(list(parent.iterdir()), [])
            self.assertFalse(result["source_metadata_changed"])

    def test_custom_training_name_does_not_change_dataset_or_audit(self):
        original = config.load_config()
        renamed = replace(original, task_name="pillow_act_baseline").validate()
        self.assertEqual(renamed.source_task_name, original.source_task_name)
        for name in ("dataset", "split", "continuity", "audit_fingerprints"):
            self.assertEqual(renamed.preview()["paths"][name], original.preview()["paths"][name])

    def test_unsafe_baseline_changes_are_rejected(self):
        value = config.load_config()
        invalid = {
            "use_base_action": True, "gripper_labels": "clip_0_1", "action_dim": 16,
            "state_dim": 16, "action_key": "observation.state", "schema_version": True,
            "statistics_scope": "all_episodes", "split_source": "random_train_ratio",
            "continuity_source": "ignore_gaps", "policy": "Flow",
        }
        for name, changed in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(value, **{name: changed}).validate()

    def test_invalid_names_are_rejected(self):
        value = config.load_config()
        for name, changed in (("task_name", "../outside"), ("dataset", "/tmp/data"),
                              ("audit_run", ".."), ("source_task_name", ""),
                              ("task_description", "bad\nlabel")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(value, **{name: changed}).validate()

    def test_missing_or_unknown_fields_are_rejected(self):
        document = json.loads(config.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        missing = dict(document)
        del missing["source_task_name"]
        for changed in (missing, {**document, "data_dir": "/tmp/override"}, []):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "config.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    config.load_config(path)


if __name__ == "__main__":
    unittest.main()
