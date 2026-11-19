"""Offline audit of ALOHA LeRobot v2.1 datasets with embedded RGB images.

No robot/torch/LeRobot imports, transformations, relabeling, or source writes.
Stored row relationships are inspected as-is; camera synchronization is NOT
reconstructed or certified. Numerical summaries are diagnostics, not training
normalization statistics.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, __version__ as pillow_version

from src.wp_ws.splits import fixed_episode_split

SCHEMA_VERSION = 1
VECTOR_WIDTHS = {
    "observation.state": 14,
    "action": 14,
    "observation.velocity": 14,
    "observation.effort": 14,
    "base_action": 2,
}
REQUIRED_VECTORS = {"observation.state", "action"}
INTEGER_COLUMNS = ("frame_index", "episode_index", "index", "task_index",
                   "recording.segment_index")
TIME_COLUMNS = ("timestamp", "recording.elapsed_time")
EXPECTED_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ACTION_CONTRACT = {
    "order": "left_6_arm_joints,left_gripper,right_6_arm_joints,right_gripper",
    "arm_actions": "absolute master joint angles, radians",
    "gripper_indices": [6, 13],
    "gripper_action": "(master_motor_radians + 1.65) / 0.85; no clipping",
    "gripper_state": "(follower_single_finger_metres - 0.01844) / (0.05800 - 0.01844)",
    "execution_mapping_reference_only": "follower_motor_target_radians = action_gripper * 1.491",
    "hardware_limits_verified": False,
    "training_labels": "original action; do not multiply by 1.491 or replace with state",
}


@dataclass(frozen=True)
class AuditConfig:
    dataset_name: str = "flip_the_pillow"
    expected_task: str = "flip_the_pillow"
    seed: int = 42
    val_episodes: int = 10
    gap_factor: float = 3.0
    batch_rows: int = 64

    def validate(self):
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.val_episodes) is not int or self.val_episodes < 1:
            raise ValueError("val_episodes must be positive")
        if type(self.batch_rows) is not int or self.batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        if not math.isfinite(self.gap_factor) or self.gap_factor <= 1:
            raise ValueError("gap_factor must be finite and greater than 1")


class Issues:
    """Aggregate repeated problems, keeping only a small number of row examples."""
    def __init__(self):
        self._items = {}

    def add(self, severity, code, message, *, episode=None, field=None, row=None, count=1):
        key = severity, code, episode, field
        if key not in self._items:
            self._items[key] = dict(severity=severity, code=code, message=message,
                                    episode_index=episode, field=field, count=0,
                                    example_rows=[])
        item = self._items[key]
        item["count"] += count
        if row is not None and len(item["example_rows"]) < 8:
            item["example_rows"].append(row)

    @property
    def items(self):
        return list(self._items.values())

    @property
    def errors(self):
        return sum(i["severity"] == "error" for i in self.items)

    def error(self, code, message, **kwargs):
        self.add("error", code, message, **kwargs)

    def warning(self, code, message, **kwargs):
        self.add("warning", code, message, **kwargs)


class Ranges:
    """Per-dimension extrema and finite counts; never silently clip values."""
    def __init__(self, width):
        self.width = width
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)
        self.finite = np.zeros(width, dtype=np.int64)
        self.below_zero = np.zeros(width, dtype=np.int64)
        self.above_one = np.zeros(width, dtype=np.int64)
        self.rows = 0

    def add(self, value):
        finite = np.isfinite(value)
        self.minimum[finite] = np.minimum(self.minimum[finite], value[finite])
        self.maximum[finite] = np.maximum(self.maximum[finite], value[finite])
        self.finite += finite
        self.below_zero += finite & (value < 0)
        self.above_one += finite & (value > 1)
        self.rows += 1

    def json(self):
        return {
            "rows_with_valid_shape": self.rows,
            "min": [float(x) if np.isfinite(x) else None for x in self.minimum],
            "max": [float(x) if np.isfinite(x) else None for x in self.maximum],
            "finite_count": self.finite.tolist(),
        }

    def grippers(self):
        result = {}
        for side, index in (("left", 6), ("right", 13)):
            result[side] = dict(
                index=index,
                min=float(self.minimum[index]) if self.finite[index] else None,
                max=float(self.maximum[index]) if self.finite[index] else None,
                finite_count=int(self.finite[index]),
                below_zero_count=int(self.below_zero[index]),
                above_one_count=int(self.above_one[index]),
            )
        return result


def _is_int(value):
    return type(value) is int


def _under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_file(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Dataset references must be relative paths")
    candidate = (root / relative).resolve()
    if not _under(candidate, root):
        raise ValueError("Dataset reference escapes the dataset directory")
    return candidate


def _file_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(raw, name):
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{name}:{number}: expected an object")
            rows.append(value)
    return rows


def _metadata(root, issues):
    records, raw = [], {}
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
        path = root / "meta" / name
        if name == "episodes_stats.jsonl" and not path.exists():
            continue  # Existing global statistics are not needed or reused.
        payload = path.read_bytes()
        raw[f"meta/{name}"] = payload
        records.append(dict(path=f"meta/{name}", size_bytes=len(payload),
                            sha256=sha256(payload).hexdigest()))
    info = json.loads(raw["meta/info.json"])
    episodes = _jsonl(raw["meta/episodes.jsonl"], "episodes.jsonl")
    tasks = _jsonl(raw["meta/tasks.jsonl"], "tasks.jsonl")
    if not isinstance(info, dict):
        raise ValueError("info.json must be an object")
    if info.get("codebase_version") != "v2.1":
        issues.error("UNSUPPORTED_FORMAT", "Only embedded-image LeRobot v2.1 is supported")
    if info.get("video_path") is not None or info.get("total_videos", 0) != 0:
        issues.error("UNSUPPORTED_VIDEO_LAYOUT", "This tool expects embedded images, not external video")
    if not isinstance(info.get("features"), dict):
        raise ValueError("features must be an object")
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    chunk_size = info.get("chunks_size")
    if not _is_int(chunk_size) or chunk_size < 1:
        raise ValueError("chunks_size must be a positive integer")
    if not isinstance(info.get("data_path"), str):
        raise ValueError("data_path must be a string")
    ids = [e.get("episode_index") for e in episodes]
    if not ids or any(not _is_int(i) or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Episode IDs must be nonempty, unique nonnegative integers")
    episodes.sort(key=lambda e: e["episode_index"])
    if sorted(ids) != list(range(len(ids))):
        issues.error("NONCONTIGUOUS_EPISODES", "Expected committed episode IDs 0..N-1")
    for episode in episodes:
        if not _is_int(episode.get("length")) or episode["length"] < 1:
            raise ValueError("Episode length must be a positive integer")
    if not _is_int(info.get("total_episodes")) or info["total_episodes"] != len(episodes):
        issues.error("EPISODE_COUNT_MISMATCH", "total_episodes disagrees with episodes.jsonl")
    if not _is_int(info.get("total_frames")) or info["total_frames"] != sum(e["length"] for e in episodes):
        issues.error("FRAME_COUNT_MISMATCH", "total_frames disagrees with episodes.jsonl")
    task_ids = [t.get("task_index") for t in tasks]
    if (not task_ids or any(not _is_int(t) or t < 0 for t in task_ids)
            or len(set(task_ids)) != len(task_ids)
            or any(not isinstance(t.get("task"), str) for t in tasks)):
        raise ValueError("Invalid or duplicate task records")
    if not _is_int(info.get("total_tasks")) or info["total_tasks"] != len(tasks):
        issues.error("TASK_COUNT_MISMATCH", "total_tasks disagrees with tasks.jsonl")
    return info, episodes, tasks, records


def _validate_features(info, issues):
    features = info["features"]
    required = REQUIRED_VECTORS | set(INTEGER_COLUMNS) | set(TIME_COLUMNS) | set(EXPECTED_CAMERAS)
    for name in sorted(required - features.keys()):
        issues.error("MISSING_FEATURE", "Required feature is not declared", field=name)
    vectors = [name for name in VECTOR_WIDTHS if name in features]
    cameras = [name for name, spec in features.items()
               if isinstance(spec, dict) and spec.get("dtype") == "image"]
    for name in vectors:
        spec = features[name]
        if not isinstance(spec, dict) or spec.get("shape") != [VECTOR_WIDTHS[name]] or spec.get("dtype") != "float32":
            issues.error("BAD_FEATURE_DECLARATION", "Expected float32 vector with the recorder's width", field=name)
    for name in INTEGER_COLUMNS + TIME_COLUMNS:
        if name not in features:
            continue
        spec = features[name]
        expected_dtype = "int64" if name in INTEGER_COLUMNS else ("float32" if name == "timestamp" else "float64")
        if not isinstance(spec, dict) or spec.get("shape") != [1] or spec.get("dtype") != expected_dtype:
            issues.error("BAD_FEATURE_DECLARATION", f"Expected scalar {expected_dtype}", field=name)
    for name in cameras:
        shape = features[name].get("shape")
        if (not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3
                or any(not _is_int(x) or x <= 0 for x in shape)):
            issues.error("BAD_IMAGE_DECLARATION", "Expected positive H,W,3 image shape", field=name)
    for name in EXPECTED_CAMERAS:
        if name in features and name not in cameras:
            issues.error("BAD_IMAGE_DECLARATION", "Expected an embedded image feature", field=name)
    return vectors, cameras


def _valid_schema(schema, columns, vectors, cameras, eid, issues):
    missing = set(columns) - set(schema.names)
    for name in sorted(missing):
        issues.error("MISSING_COLUMN", "Parquet column is missing", episode=eid, field=name)
    if missing:
        return False
    valid = True
    for name in columns:
        kind = schema.field(name).type
        if name in vectors:
            if not ((pa.types.is_list(kind) or pa.types.is_fixed_size_list(kind))
                    and pa.types.is_float32(kind.value_type)):
                issues.error("BAD_COLUMN_TYPE", "Expected a list of float32", episode=eid, field=name)
                valid = False
        elif name in INTEGER_COLUMNS and not pa.types.is_int64(kind):
            issues.error("BAD_COLUMN_TYPE", "Expected int64", episode=eid, field=name)
            valid = False
        elif name in TIME_COLUMNS:
            expected = pa.float32() if name == "timestamp" else pa.float64()
            if kind != expected:
                issues.error("BAD_COLUMN_TYPE", f"Expected {expected}", episode=eid, field=name)
                valid = False
        elif name in cameras and not pa.types.is_struct(kind):
            issues.error("BAD_COLUMN_TYPE", "Expected HF image struct with embedded bytes", episode=eid, field=name)
            valid = False
    return valid


def _image(value, shape):
    if not isinstance(value, dict) or not isinstance(value.get("bytes"), bytes) or not value["bytes"]:
        raise ValueError("Missing embedded image bytes; external-path fallback is not supported")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("Invalid image shape declaration")
    with Image.open(BytesIO(value["bytes"])) as image:
        if image.mode != "RGB" or [image.height, image.width, 3] != shape:
            raise ValueError(f"Expected RGB {shape}, got {image.mode} {image.size}")
        image.load()  # Fully decode EVERY saved image, not merely its header.


def _timing(timestamps, elapsed, segments, episode, fps, gap_factor, issues):
    eid, length = episode["episode_index"], episode["length"]
    if not (len(elapsed) == len(segments) == len(timestamps) == length):
        return {"contiguous_ranges": [], "checks_complete": False}
    if not all(v is not None for seq in (timestamps, elapsed, segments) for v in seq):
        return {"contiguous_ranges": [], "checks_complete": False}
    elapsed, timestamps, segments = np.array(elapsed), np.array(timestamps), np.array(segments)
    if elapsed[0] < 0:
        issues.error("NEGATIVE_ELAPSED", "Elapsed time starts below zero", episode=eid, row=0)
    dt, segdiff = np.diff(elapsed), np.diff(segments)
    bad_dt = np.flatnonzero(dt <= 0)
    if bad_dt.size:
        issues.error("NONINCREASING_ELAPSED", "Stored elapsed time must strictly increase",
                     episode=eid, row=int(bad_dt[0] + 1), count=int(bad_dt.size))
    bad_seg = np.flatnonzero(segdiff < 0)
    if np.any(segments < 0) or bad_seg.size:
        issues.error("BAD_SEGMENT_ORDER", "Segment IDs must be nonnegative and nondecreasing", episode=eid)
    # The recorder defines timestamp = accepted row / fps. This is a structural
    # consistency check, NOT an assertion that cameras exposed at these times.
    expected = np.arange(length, dtype=np.float64) / fps
    wrong_time = np.flatnonzero(~np.isclose(timestamps, expected, rtol=1e-6, atol=1e-5))
    if wrong_time.size:
        issues.error("BAD_NOMINAL_TIMESTAMP", "timestamp disagrees with frame_index/fps; no timing is repaired",
                     episode=eid, row=int(wrong_time[0]), count=int(wrong_time.size))
    segment_starts = [0] + (np.flatnonzero(segdiff != 0) + 1).tolist()
    observed_segments = [{"segment_index": int(segments[i]), "start_frame": i} for i in segment_starts]
    if episode.get("segments") != observed_segments:
        issues.error("SEGMENT_METADATA_MISMATCH", "Recorded segment changes disagree with episode metadata", episode=eid)
    gap_rows = (np.flatnonzero(dt > gap_factor / fps) + 1).tolist()
    if gap_rows:
        issues.warning("ELAPSED_GAPS", "Large elapsed gaps; do not form action chunks across these boundaries",
                       episode=eid, row=gap_rows[0], count=len(gap_rows))
    starts = sorted(set(segment_starts + gap_rows))
    ranges = []
    for start, stop in zip(starts, starts[1:] + [length]):
        reasons = ["episode_start"] if start == 0 else []
        if start and start in segment_starts:
            reasons.append("segment_change")
        if start in gap_rows:
            reasons.append("elapsed_gap")
        ranges.append(dict(start_frame=start, stop_frame_exclusive=stop,
                           segment_index=int(segments[start]), start_reasons=reasons))
    positive = dt[dt > 0]
    return dict(
        checks_complete=True,
        elapsed_span_seconds=float(elapsed[-1] - elapsed[0]),
        delta_seconds_min=float(dt.min()) if dt.size else None,
        delta_seconds_median=float(np.median(dt)) if dt.size else None,
        delta_seconds_p95=float(np.quantile(dt, .95)) if dt.size else None,
        delta_seconds_max=float(dt.max()) if dt.size else None,
        median_rate_hz=float(1 / np.median(positive)) if positive.size else None,
        gap_threshold_seconds=gap_factor / fps,
        gap_count=len(gap_rows),
        segment_count=len(segment_starts),
        contiguous_ranges=ranges,
    )


def _scan_episode(path, episode, *, offset, tasks, info, vectors, cameras,
                  aggregate, config, issues):
    eid = episode["episode_index"]
    output = dict(episode_index=eid, expected_rows=episode["length"], rows_read=0,
                  images_checked={name: 0 for name in cameras}, numeric={}, timing={})
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != episode["length"]:
        issues.error("PARQUET_ROW_COUNT_MISMATCH", "Parquet rows disagree with episode length", episode=eid)
    columns = vectors + list(INTEGER_COLUMNS) + list(TIME_COLUMNS) + cameras
    if not _valid_schema(pf.schema_arrow, columns, vectors, cameras, eid, issues):
        return output
    ranges = {name: Ranges(VECTOR_WIDTHS[name]) for name in vectors}
    time_values = {name: [] for name in TIME_COLUMNS}
    segment_values = []
    task_map = {t["task_index"]: t["task"] for t in tasks}
    seen_tasks = set()
    for batch in pf.iter_batches(batch_size=config.batch_rows, columns=columns, use_threads=False):
        arrays = {name: batch.column(batch.schema.get_field_index(name)) for name in columns}
        for j in range(batch.num_rows):
            row = output["rows_read"]
            values = {name: values[j].as_py() for name, values in arrays.items()}
            for name in vectors:
                try:
                    value = np.asarray(values[name], dtype=np.float64)
                except (TypeError, ValueError):
                    value = np.empty(0)
                if value.shape != (VECTOR_WIDTHS[name],):
                    issues.error("BAD_VECTOR_SHAPE", f"Expected {VECTOR_WIDTHS[name]} values", episode=eid, field=name, row=row)
                    continue
                if not np.isfinite(value).all():
                    issues.error("NONFINITE_VECTOR", "NaN/Inf or null numeric element", episode=eid, field=name, row=row)
                ranges[name].add(value)
                aggregate[name].add(value)
            for name in INTEGER_COLUMNS:
                if not _is_int(values[name]):
                    issues.error("INVALID_INTEGER", "Missing or non-integer scalar", episode=eid, field=name, row=row)
            for name, expected in (("frame_index", row), ("episode_index", eid), ("index", offset + row)):
                if values[name] != expected:
                    issues.error("INDEX_MISMATCH", f"Expected {name}={expected}", episode=eid, field=name, row=row)
            task = values["task_index"]
            if _is_int(task) and task in task_map:
                seen_tasks.add(task_map[task])
            else:
                issues.error("UNKNOWN_TASK_INDEX", "task_index has no matching task record", episode=eid, row=row)
            segment_values.append(values["recording.segment_index"] if _is_int(values["recording.segment_index"]) else None)
            for name in TIME_COLUMNS:
                value = values[name]
                valid = isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
                if not valid:
                    issues.error("INVALID_TIME", "Time value must be finite", episode=eid, field=name, row=row)
                time_values[name].append(float(value) if valid else None)
            for name in cameras:
                try:
                    _image(values[name], info["features"][name].get("shape"))
                    output["images_checked"][name] += 1
                except Exception as exc:
                    issues.error("INVALID_IMAGE", str(exc), episode=eid, field=name, row=row)
            output["rows_read"] += 1
    output["numeric"] = {name: stats.json() for name, stats in ranges.items()}
    output["grippers"] = {name: ranges[name].grippers() for name in ("observation.state", "action") if name in ranges}
    if set(episode.get("tasks", [])) != seen_tasks:
        issues.error("EPISODE_TASK_MISMATCH", "Episode task names disagree with row task_index values", episode=eid)
    output["timing"] = _timing(time_values["timestamp"], time_values["recording.elapsed_time"], segment_values,
                               episode, info["fps"], config.gap_factor, issues)
    return output


def _fingerprint(records):
    payload = json.dumps(sorted(records, key=lambda x: x["path"]), sort_keys=True,
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def audit_dataset(root, config=AuditConfig(), *, progress=None):
    """Read the dataset and return artifacts in memory; never write into root.

    The CLI resolves source/output roots through src.config.paths. A root
    parameter here also permits tiny synthetic fixtures in offline tests.
    """
    config.validate()
    root = Path(root).resolve()
    issues, records, episode_reports = Issues(), [], []
    report = dict(schema_version=SCHEMA_VERSION, dataset_name=config.dataset_name,
                  source_directory=str(root), image_check="full_decode_all_rows_all_cameras",
                  environment=dict(python=sys.version, numpy=np.__version__, pyarrow=pa.__version__,
                                   pillow=pillow_version),
                  action_contract=ACTION_CONTRACT, issues=[], episodes=[], split_ready=False)
    split = None
    try:
        info, episodes, tasks, records = _metadata(root, issues)
        selection = fixed_episode_split([e["episode_index"] for e in episodes], seed=config.seed,
                                        val_episodes=config.val_episodes)
        vectors, cameras = _validate_features(info, issues)
        report["metadata"] = dict(codebase_version=info.get("codebase_version"), fps=info["fps"],
                                  episode_count=len(episodes), declared_rows=info.get("total_frames"),
                                  source_task_names=sorted(t["task"] for t in tasks),
                                  expected_task_label=config.expected_task,
                                  original_splits=info.get("splits"), cameras=cameras,
                                  gap_factor=config.gap_factor, seed=config.seed, val_episodes=config.val_episodes)
        if any(t["task"] != config.expected_task for t in tasks):
            issues.warning("TASK_LABEL_MISMATCH", "Source task label differs from the intended task; source metadata is preserved")
        issues.warning("LEGACY_CAPTURE_PROVENANCE", "Stored row relationships are preserved. Camera frame IDs/exposure timing and the native Flash ALOHA capture contract are not certified")
        aggregate = {name: Ranges(VECTOR_WIDTHS[name]) for name in vectors}
        planned_paths = []
        for episode in episodes:
            relative = info["data_path"].format(episode_chunk=episode["episode_index"] // info["chunks_size"],
                                                episode_index=episode["episode_index"])
            planned_paths.append(_safe_file(root, relative))
        if len(set(planned_paths)) != len(planned_paths):
            issues.error("DUPLICATE_EPISODE_PATH", "Multiple episodes reference the same Parquet file")
        extra = {p.resolve() for p in (root / "data").rglob("*.parquet")} - set(planned_paths)
        if extra:
            issues.error("UNREFERENCED_PARQUET", "Uncommitted/unreferenced Parquet files exist; no files will be silently excluded", count=len(extra))
            report["unreferenced_parquet_files"] = sorted(str(p.relative_to(root)) if _under(p, root) else str(p) for p in extra)
        offset = 0
        signatures = {}
        for number, (episode, path) in enumerate(zip(episodes, planned_paths), 1):
            eid = episode["episode_index"]
            if progress:
                progress(f"[体检 {number}/{len(episodes)}] episode={eid:06d}: SHA-256 + 数值/索引/时间 + 全量图片解码")
            try:
                before = _file_signature(path)
                signatures[path] = before
                digest = hash_file(path)
                records.append(dict(path=path.relative_to(root).as_posix(), size_bytes=before[0], sha256=digest))
                result = _scan_episode(path, episode, offset=offset, tasks=tasks, info=info,
                                       vectors=vectors, cameras=cameras, aggregate=aggregate,
                                       config=config, issues=issues)
                result["data_path"] = path.relative_to(root).as_posix()
                episode_reports.append(result)
                if _file_signature(path) != before:
                    issues.error("SOURCE_CHANGED_DURING_AUDIT", "Parquet changed while auditing; stop collectors before auditing", episode=eid)
            except Exception as exc:
                issues.error("EPISODE_READ_FAILED", f"{type(exc).__name__}: {exc}", episode=eid)
            offset += episode["length"]
        for record in records:
            if record["path"].startswith("meta/") and hash_file(root / record["path"]) != record["sha256"]:
                issues.error("SOURCE_CHANGED_DURING_AUDIT", "Metadata changed while auditing")
        for path, before in signatures.items():
            if _file_signature(path) != before:
                issues.error("SOURCE_CHANGED_DURING_AUDIT", "A previously scanned Parquet changed before the audit finished")
        current_paths = {p.resolve() for p in (root / "data").rglob("*.parquet")}
        if current_paths != set(planned_paths) | extra:
            issues.error("SOURCE_CHANGED_DURING_AUDIT", "Parquet file membership changed while auditing")
        report["numeric_ranges_all_data_diagnostic_only"] = {name: stats.json() for name, stats in aggregate.items()}
        report["grippers"] = {name: aggregate[name].grippers() for name in ("observation.state", "action") if name in aggregate}
        for name, sides in report["grippers"].items():
            for side, stats in sides.items():
                count = stats["below_zero_count"] + stats["above_one_count"]
                if count:
                    issues.warning("GRIPPER_OUTSIDE_REFERENCE_RANGE", f"{side} normalized gripper values outside [0,1]; diagnostic only, labels are NOT clipped",
                                   field=f"{name}[{stats['index']}]", count=count)
        if "base_action" in aggregate:
            stats = aggregate["base_action"]
            extrema = np.maximum(np.abs(stats.minimum), np.abs(stats.maximum))
            if np.any(np.isfinite(extrema) & (extrema > 1e-6)):
                issues.warning("BASE_MOTION_PRESENT", "base_action is measured velocity, not a verified base control label; review nonzero base motion before an arms-only baseline", field="base_action")
        report["dataset_fingerprint"] = dict(algorithm="sha256-sorted-file-manifest-v1", sha256=_fingerprint(records))
        if issues.errors == 0:
            lengths = {e["episode_index"]: e["length"] for e in episodes}
            split = dict(schema_version=SCHEMA_VERSION, dataset_name=config.dataset_name,
                         dataset_fingerprint=report["dataset_fingerprint"], **selection,
                         train_rows=sum(lengths[i] for i in selection["train_episode_indices"]),
                         val_rows=sum(lengths[i] for i in selection["val_episode_indices"]),
                         partition_unit="whole_episode", excluded_episode_indices=[],
                         warnings_require_review=True,
                         training_statistics="Not computed. Fit state/action statistics on train_episode_indices only.")
            report["split_ready"] = True
    except Exception as exc:
        issues.error("AUDIT_SETUP_FAILED", f"{type(exc).__name__}: {exc}")
    report["episodes"] = episode_reports
    report["issues"] = issues.items
    report["counts"] = dict(error_groups=issues.errors,
                            warning_groups=sum(i["severity"] == "warning" for i in issues.items),
                            episodes_scanned=len(episode_reports),
                            rows_read=sum(e["rows_read"] for e in episode_reports),
                            images_decoded=sum(sum(e["images_checked"].values()) for e in episode_reports))
    report["status"] = "FAIL" if issues.errors else ("PASS_WITH_WARNINGS" if issues.items else "PASS")
    return dict(report=report, split=split, fingerprints=dict(schema_version=SCHEMA_VERSION, files=records),
                continuity=dict(schema_version=SCHEMA_VERSION, dataset_name=config.dataset_name,
                                dataset_fingerprint=report.get("dataset_fingerprint"),
                                usable=report["split_ready"],
                                rule="Do not cross episode, segment-change, or elapsed-gap boundaries; no rows were removed or realigned",
                                timing_provenance="recording.elapsed_time is not camera exposure time",
                                episodes=[dict(episode_index=e["episode_index"], data_path=e.get("data_path"),
                                               rows=e["rows_read"], **e["timing"])
                                          for e in episode_reports]))


def markdown_report(report, split):
    counts = report["counts"]
    meta = report.get("metadata", {})
    lines = ["# 翻转枕头数据体检报告", "", f"**结果：{report['status']}**", "",
             f"- 数据：`{report['source_directory']}`",
             f"- 扫描：{counts['episodes_scanned']} episodes / {counts['rows_read']} rows",
             f"- 完整解码图片：{counts['images_decoded']}",
             f"- 错误组：{counts['error_groups']}；警告组：{counts['warning_groups']}",
             f"- 声明频率：{meta.get('fps')} Hz；源任务：{meta.get('source_task_names')}",
             "- 未改写任何图像、动作、状态、时间戳或元数据。", "",
             "## 固定划分", ""]
    if split:
        lines += [f"- 训练：{len(split['train_episode_indices'])} episodes / {split['train_rows']} rows",
                  f"- 验证：{len(split['val_episode_indices'])} episodes / {split['val_rows']} rows",
                  f"- 算法：`{split['algorithm']}`；seed：{split['seed']}",
                  f"- 验证 episode：`{split['val_episode_indices']}`",
                  "- 后续训练必须显式加载 split.json；源 info.json 的 train 划分没有被修改。"]
    else:
        lines += ["**未生成 split.json。先解决错误，不自动删除失败 episode。**"]
    lines += ["", "## 夹爪原始范围（只诊断，不裁剪）", "",
              "| 字段 | 侧 | 最小值 | 最大值 | <0 数量 | >1 数量 |",
              "|---|---|---:|---:|---:|---:|"]
    for field, sides in report.get("grippers", {}).items():
        for side, values in sides.items():
            lines.append(f"| {field} | {side} | {values['min']} | {values['max']} | {values['below_zero_count']} | {values['above_one_count']} |")
    lines += ["", "## 连续性", "",
              "continuity.json 的区间为 `[start_frame, stop_frame_exclusive)`；不改变原始行配对。",
              "时间间隔检查基于保存的 elapsed_time，只提示训练切片边界，不证明相机曝光同步。", "",
              "## 问题列表", "",
              "下表最多列出前 80 组；完整问题、次数与行号示例见 report.json。", "",
              "| 等级 | 代码 | episode | 字段 | 次数 | 说明 |",
              "|---|---|---|---|---:|---|"]
    for item in report["issues"][:80]:
        message = item["message"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['severity']} | {item['code']} | {item['episode_index']} | {item['field']} | {item['count']} | {message} |")
    lines += ["", "## 使用边界", "",
              "- PASS_WITH_WARNINGS 表示存储结构检查通过，警告仍需人工审阅；不是动作成功率或硬件安全证明。",
              "- 数据摘要覆盖全部数据，只用于诊断，不能作为训练标准化统计量。",
              "- 尚未实现训练加载器、标准化、共享图像预处理接入或训练。",
              "- 缺失的原始相机帧 ID / 时间戳不会被伪造；不声明满足 Flash ALOHA 原生采集契约。", ""]
    return "\n".join(lines)
