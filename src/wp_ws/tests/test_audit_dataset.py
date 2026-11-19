"""Run explicitly: python -m unittest discover -s src/wp_ws/tests -v.

Fixtures are generated in TemporaryDirectory, never in the recorded dataset.
"""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.wp_ws.audit_core import AuditConfig, EXPECTED_CAMERAS, audit_dataset, hash_file
from src.wp_ws.audit_dataset import resolve_locations, write_artifacts
from src.wp_ws.splits import fixed_episode_split


def fixture(root, *, count=5, length=4):
    """Small real Parquet files with embedded lossless RGB bytes."""
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    buffer = BytesIO()
    Image.new("RGB", (8, 6), (30, 60, 90)).save(buffer, format="PNG")
    image = dict(bytes=buffer.getvalue(), path=None)
    features = {name: dict(dtype="float32", shape=[width])
                for name, width in (("observation.state", 14), ("action", 14), ("base_action", 2))}
    integers = ("frame_index", "episode_index", "index", "task_index", "recording.segment_index")
    features.update({name: dict(dtype="int64", shape=[1]) for name in integers})
    features.update({"timestamp": dict(dtype="float32", shape=[1]),
                     "recording.elapsed_time": dict(dtype="float64", shape=[1])})
    features.update({name: dict(dtype="image", shape=[6, 8, 3]) for name in EXPECTED_CAMERAS})
    info = dict(codebase_version="v2.1", fps=50, chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path=None, total_videos=0, total_episodes=count,
                total_frames=count * length, total_tasks=1, features=features,
                splits={"train": f"0:{count}"})
    episodes = [dict(episode_index=i, length=length, tasks=["pick_cup"],
                     segments=[dict(segment_index=0, start_frame=0)]) for i in range(count)]
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes) + "\n", encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0, task="pick_cup")) + "\n", encoding="utf-8")
    for eid in range(count):
        action = np.zeros((length, 14), dtype=np.float32)
        action[:, 6] = 1.12  # Valid original labels outside the reference [0,1].
        action[:, 13] = -.05
        state = np.zeros((length, 14), dtype=np.float32)
        state[:, [6, 13]] = .7
        values = {
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "base_action": pa.array([[0., 0.]] * length, type=pa.list_(pa.float32())),
            "timestamp": pa.array(np.arange(length) / 50, type=pa.float32()),
            "recording.elapsed_time": pa.array(np.arange(length) / 50 + .001, type=pa.float64()),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "episode_index": pa.array([eid] * length, type=pa.int64()),
            "index": pa.array(range(eid * length, (eid + 1) * length), type=pa.int64()),
            "task_index": pa.array([0] * length, type=pa.int64()),
            "recording.segment_index": pa.array([0] * length, type=pa.int64()),
        }
        for camera in EXPECTED_CAMERAS:
            values[camera] = pa.array([image] * length, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
        pq.write_table(pa.table(values), root / f"data/chunk-000/episode_{eid:06d}.parquet")


def edit_column(root, name, edit, eid=0):
    path = root / f"data/chunk-000/episode_{eid:06d}.parquet"
    table = pq.ParquetFile(path).read()
    values = table[name].to_pylist()
    edit(values)
    index = table.schema.get_field_index(name)
    table = table.set_column(index, table.schema.field(name), pa.array(values, type=table.schema.field(name).type))
    pq.write_table(table, path)


class SplitTests(unittest.TestCase):
    def test_deterministic_order_independent_and_disjoint(self):
        ids = list(range(101))
        a = fixed_episode_split(ids, seed=42, val_episodes=10)
        self.assertEqual(a, fixed_episode_split(reversed(ids), seed=42, val_episodes=10))
        self.assertEqual(len(a["train_episode_indices"]), 91)
        self.assertEqual(len(a["val_episode_indices"]), 10)
        self.assertFalse(set(a["train_episode_indices"]) & set(a["val_episode_indices"]))
        self.assertEqual(sorted(a["train_episode_indices"] + a["val_episode_indices"]), ids)
        self.assertNotEqual(a["val_episode_indices"], fixed_episode_split(ids, seed=43)["val_episode_indices"])

    def test_bad_splits_refused(self):
        for ids, count in (([0, 0, 1], 1), ([0, 1], 0), ([0, 1], 2), ([False, 1], 1)):
            with self.subTest(ids=ids, count=count), self.assertRaises(ValueError):
                fixed_episode_split(ids, val_episodes=count)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        fixture(self.root)
        self.config = AuditConfig(val_episodes=2, batch_rows=2)

    def audit(self):
        return audit_dataset(self.root, self.config)

    def codes(self, artifacts):
        return {i["code"] for i in artifacts["report"]["issues"]}

    def test_full_decode_out_of_range_labels_and_readonly(self):
        before = {p.relative_to(self.root).as_posix(): hash_file(p) for p in self.root.rglob("*") if p.is_file()}
        artifacts = self.audit()
        self.assertEqual(artifacts["report"]["status"], "PASS_WITH_WARNINGS")
        self.assertEqual(artifacts["report"]["counts"]["images_decoded"], 5 * 4 * 3)
        self.assertEqual(artifacts["split"]["train_rows"], 12)
        self.assertEqual(artifacts["split"]["val_rows"], 8)
        self.assertIn("GRIPPER_OUTSIDE_REFERENCE_RANGE", self.codes(artifacts))
        self.assertIn("TASK_LABEL_MISMATCH", self.codes(artifacts))
        self.assertAlmostEqual(artifacts["report"]["grippers"]["action"]["left"]["max"], 1.12, places=6)
        after = {p.relative_to(self.root).as_posix(): hash_file(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_nonfinite_labels_fail_without_split(self):
        edit_column(self.root, "action", lambda v: v[1].__setitem__(0, float("nan")))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("NONFINITE_VECTOR", self.codes(artifacts))
        json.dumps(artifacts, allow_nan=False)

    def test_bad_vector_shape_fails(self):
        edit_column(self.root, "action", lambda v: v.__setitem__(0, [0.] * 13))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("BAD_VECTOR_SHAPE", self.codes(artifacts))

    def test_corrupt_image_fails(self):
        edit_column(self.root, EXPECTED_CAMERAS[0], lambda v: v.__setitem__(2, dict(bytes=b"not an image", path=None)))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("INVALID_IMAGE", self.codes(artifacts))

    def test_wrong_image_shape_fails_without_resizing(self):
        meta_path = self.root / "meta/info.json"
        meta = json.loads(meta_path.read_text())
        meta["features"][EXPECTED_CAMERAS[0]]["shape"] = [12, 16, 3]
        meta_path.write_text(json.dumps(meta))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("INVALID_IMAGE", self.codes(artifacts))

    def test_bad_row_index_fails(self):
        edit_column(self.root, "frame_index", lambda v: v.__setitem__(2, 99))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("INDEX_MISMATCH", self.codes(artifacts))

    def test_nonincreasing_elapsed_fails(self):
        edit_column(self.root, "recording.elapsed_time", lambda v: v.__setitem__(2, v[1]))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("NONINCREASING_ELAPSED", self.codes(artifacts))

    def test_incorrect_nominal_timestamp_is_not_repaired(self):
        edit_column(self.root, "timestamp", lambda v: v.__setitem__(1, .03))
        before = hash_file(self.root / "data/chunk-000/episode_000000.parquet")
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("BAD_NOMINAL_TIMESTAMP", self.codes(artifacts))
        self.assertEqual(before, hash_file(self.root / "data/chunk-000/episode_000000.parquet"))

    def test_gap_records_ranges_without_dropping_or_realignment(self):
        edit_column(self.root, "recording.elapsed_time", lambda v: v.__setitem__(slice(2, None), [x + 1 for x in v[2:]]))
        artifacts = self.audit()
        self.assertIsNotNone(artifacts["split"])
        ranges = artifacts["continuity"]["episodes"][0]["contiguous_ranges"]
        self.assertEqual([(r["start_frame"], r["stop_frame_exclusive"]) for r in ranges], [(0, 2), (2, 4)])
        self.assertIn("ELAPSED_GAPS", self.codes(artifacts))
        self.assertEqual(artifacts["report"]["counts"]["rows_read"], 20)

    def test_segment_change_records_boundary(self):
        edit_column(self.root, "recording.segment_index", lambda v: v.__setitem__(slice(2, None), [1, 1]))
        meta_path = self.root / "meta/episodes.jsonl"
        episodes = [json.loads(line) for line in meta_path.read_text().splitlines()]
        episodes[0]["segments"].append(dict(segment_index=1, start_frame=2))
        meta_path.write_text("\n".join(json.dumps(e) for e in episodes) + "\n")
        artifacts = self.audit()
        self.assertIsNotNone(artifacts["split"])
        ranges = artifacts["continuity"]["episodes"][0]["contiguous_ranges"]
        self.assertEqual(ranges[1]["start_reasons"], ["segment_change"])

    def test_missing_file_fails(self):
        (self.root / "data/chunk-000/episode_000001.parquet").unlink()
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("EPISODE_READ_FAILED", self.codes(artifacts))

    def test_unreferenced_file_is_not_silently_ignored(self):
        source = self.root / "data/chunk-000/episode_000000.parquet"
        (source.parent / "episode_999999.parquet").write_bytes(source.read_bytes())
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("UNREFERENCED_PARQUET", self.codes(artifacts))

    def test_declared_frame_count_mismatch_fails(self):
        meta_path = self.root / "meta/info.json"
        meta = json.loads(meta_path.read_text())
        meta["total_frames"] += 1
        meta_path.write_text(json.dumps(meta))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("FRAME_COUNT_MISMATCH", self.codes(artifacts))

    def test_path_escape_is_refused(self):
        meta_path = self.root / "meta/info.json"
        meta = json.loads(meta_path.read_text())
        meta["data_path"] = "../episode_{episode_index}.parquet"
        meta_path.write_text(json.dumps(meta))
        artifacts = self.audit()
        self.assertIsNone(artifacts["split"])
        self.assertIn("AUDIT_SETUP_FAILED", self.codes(artifacts))

    def test_dataset_fingerprint_is_stable_and_content_sensitive(self):
        first = self.audit()
        second = self.audit()
        self.assertEqual(first["report"]["dataset_fingerprint"], second["report"]["dataset_fingerprint"])
        self.assertEqual(first["split"], second["split"])
        edit_column(self.root, "action", lambda v: v[0].__setitem__(0, .01))
        third = self.audit()
        self.assertNotEqual(first["report"]["dataset_fingerprint"], third["report"]["dataset_fingerprint"])
        self.assertEqual(first["split"]["val_episode_indices"], third["split"]["val_episode_indices"])

    def test_reports_are_new_only_and_failed_audit_has_no_split_file(self):
        artifacts = self.audit()
        output = Path(self.temp.name) / "report"
        write_artifacts(output, artifacts)
        self.assertTrue((output / "COMPLETED.json").is_file())
        self.assertTrue((output / "split.json").is_file())
        with self.assertRaises(FileExistsError):
            write_artifacts(output, artifacts)
        failed = deepcopy(artifacts)
        failed["split"] = None
        failed["report"]["status"] = "FAIL"
        failed["report"]["split_ready"] = False
        failed_dir = Path(self.temp.name) / "failed_report"
        write_artifacts(failed_dir, failed)
        self.assertFalse((failed_dir / "split.json").exists())
        self.assertTrue((failed_dir / "report.json").exists())

    def test_central_paths_and_no_output_inside_dataset(self):
        base = Path(self.temp.name)
        with patch("src.wp_ws.audit_dataset.DATA_DIR", base / "data"), patch("src.wp_ws.audit_dataset.LOG_DIR", base / "logs"):
            source, output = resolve_locations("flip_the_pillow", "audit_v1")
            self.assertEqual(source, base / "data/flip_the_pillow")
            self.assertEqual(output, base / "logs/wp_ws/flip_the_pillow/audit_v1")
        with patch("src.wp_ws.audit_dataset.DATA_DIR", base), patch("src.wp_ws.audit_dataset.LOG_DIR", base / "flip_the_pillow"):
            with self.assertRaises(ValueError):
                resolve_locations("flip_the_pillow", "audit_v1")


if __name__ == "__main__":
    unittest.main()
