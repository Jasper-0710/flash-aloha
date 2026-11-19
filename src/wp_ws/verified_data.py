"""Read-only verification of an audit and its unchanged embedded-image source.

Public callers resolve all locations through BaselineDataConfig/paths.py. No
source writes, row repair, random splitting, or hardware imports are permitted.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from src.wp_ws.audit_core import (
    ACTION_CONTRACT, EXPECTED_CAMERAS, Issues, _fingerprint, _metadata,
    _safe_file, _timing, _validate_features, hash_file,
)
from src.wp_ws.audit_dataset import resolve_locations
from src.wp_ws.splits import fixed_episode_split

AUDIT_FILES = ("report.json", "COMPLETED.json", "fingerprints.json", "split.json", "continuity.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _indexed(rows, name):
    require(isinstance(rows, list) and rows, f"Missing {name} episodes")
    result = {}
    for row in rows:
        require(isinstance(row, dict), f"Invalid {name} entry")
        eid = row.get("episode_index")
        require(type(eid) is int and eid >= 0 and eid not in result,
                f"Invalid/duplicate {name} episode index: {eid!r}")
        result[eid] = row
    return result


@dataclass(frozen=True)
class Episode:
    episode_index: int
    data_path: str
    rows: int
    # (start inclusive, stop exclusive); indices refer to saved rows, never time interpolation.
    ranges: tuple


@dataclass(frozen=True)
class VerifiedData:
    root: Path
    audit: Path
    config: object
    info: dict
    episodes: dict
    documents: dict
    audit_sha256: dict
    signatures: dict

    @property
    def split(self):
        return self.documents["split.json"]

    @property
    def fingerprint(self):
        return self.documents["report.json"]["dataset_fingerprint"]

    @property
    def warnings(self):
        return [i for i in self.documents["report.json"]["issues"] if i["severity"] == "warning"]

    def source_file(self, relative):
        path = _safe_file(self.root, relative)
        require(relative in self.signatures, f"Unaudited source reference: {relative}")
        require(signature(path) == self.signatures[relative], f"Source changed after verification: {relative}")
        return path

    def assert_unchanged(self):
        for relative in self.signatures:
            self.source_file(relative)
        planned = {_safe_file(self.root, episode.data_path) for episode in self.episodes.values()}
        actual = {p.resolve() for p in (self.root / "data").rglob("*.parquet")}
        require(actual == planned, "Source Parquet membership changed after verification")
        for name, expected in self.audit_sha256.items():
            require(hash_file(self.audit / name) == expected, f"Audit changed after verification: {name}")

    def provenance(self):
        return {
            "dataset_fingerprint": self.fingerprint,
            "audit_artifact_sha256": self.audit_sha256,
            "source_task_name": self.config.source_task_name,
            "training_task_name": self.config.task_name,
            "camera_keys_in_order": list(EXPECTED_CAMERAS),
            "declared_fps": self.info["fps"],
            "action_contract": ACTION_CONTRACT,
            "warnings": self.warnings,
            "legacy_capture_certified": False,
            "source_metadata_changed": False,
        }


def load_verified_data(config, *, progress=print):
    """Verify real content hashes each invocation; never trust a config preview."""
    config.validate()
    root, audit = resolve_locations(config.dataset, config.audit_run)
    return _verify_at(root, audit, config, progress=progress)


def _verify_at(root, audit, config, *, progress=None):
    """Explicit paths are private and exist only for isolated synthetic tests."""
    root, audit = Path(root).resolve(), Path(audit).resolve()
    config.validate()
    raw = {name: (audit / name).read_bytes() for name in AUDIT_FILES}
    documents = {name: json.loads(payload) for name, payload in raw.items()}
    require(all(isinstance(d, dict) for d in documents.values()), "Audit artifacts must be JSON objects")
    report, completed = documents["report.json"], documents["COMPLETED.json"]
    split, continuity = documents["split.json"], documents["continuity.json"]
    manifest = documents["fingerprints.json"]
    for name in ("report.json", "split.json", "continuity.json", "fingerprints.json"):
        require(documents[name].get("schema_version") == 1, f"Unsupported schema: {name}")
    require(report.get("status") in ("PASS", "PASS_WITH_WARNINGS") and report.get("split_ready") is True,
            "Audit did not pass / split is not ready")
    require(completed.get("status") == report["status"] and completed.get("split_ready") is True,
            "COMPLETED.json does not confirm the successful audit")
    require(continuity.get("usable") is True, "Audit continuity is not usable")
    require(report.get("counts", {}).get("error_groups") == 0,
            "Audit reports error groups")
    require(isinstance(report.get("issues"), list) and
            not any(i.get("severity") == "error" for i in report["issues"]), "Audit contains errors")
    require(report.get("action_contract") == ACTION_CONTRACT, "Audit action semantics differ from this baseline")
    for name, doc in (("report", report), ("split", split), ("continuity", continuity)):
        require(doc.get("dataset_name") == config.dataset, f"Dataset name mismatch in {name}")
    records = manifest.get("files")
    require(isinstance(records, list) and records, "Missing audited file manifest")
    seen = set()
    for record in records:
        require(isinstance(record, dict) and set(record) == {"path", "size_bytes", "sha256"},
                "Invalid manifest record")
        relative = record["path"]
        _safe_file(root, relative)
        require(relative not in seen, f"Duplicate manifest path: {relative}")
        require(type(record["size_bytes"]) is int and record["size_bytes"] >= 0 and
                isinstance(record["sha256"], str) and len(record["sha256"]) == 64, "Invalid source fingerprint")
        seen.add(relative)
    fingerprint = {"algorithm": "sha256-sorted-file-manifest-v1", "sha256": _fingerprint(records)}
    for doc in (report, split, continuity):
        require(doc.get("dataset_fingerprint") == fingerprint, "Audit/split/continuity fingerprint mismatch")

    signatures = {}
    for number, record in enumerate(records, 1):
        path = _safe_file(root, record["path"])
        if progress:
            progress(f"[源指纹 {number}/{len(records)}] {record['path']}")
        before = signature(path)
        require(before[0] == record["size_bytes"] and hash_file(path) == record["sha256"],
                f"Source does not match audit: {record['path']}; do not reuse stale audit artifacts")
        require(before == signature(path), f"Source changed while hashing: {record['path']}")
        signatures[record["path"]] = before

    issues = Issues()
    info, metadata_episodes, tasks, metadata_records = _metadata(root, issues)
    _, cameras = _validate_features(info, issues)
    require(issues.errors == 0, f"Source metadata invalid: {issues.items}")
    require(set(cameras) == set(EXPECTED_CAMERAS), "This baseline requires exactly the three audited cameras")
    require(sorted({t["task"] for t in tasks}) == [config.source_task_name], "Unexpected source task label")
    meta = report.get("metadata", {})
    require(meta.get("source_task_names") == [config.source_task_name] and
            set(meta.get("cameras", [])) == set(EXPECTED_CAMERAS) and meta.get("fps") == info["fps"],
            "Audit metadata no longer matches source/task configuration")
    source_episodes = _indexed(metadata_episodes, "source")
    report_episodes = _indexed(report.get("episodes"), "report")
    ranges_episodes = _indexed(continuity.get("episodes"), "continuity")
    require(source_episodes.keys() == report_episodes.keys() == ranges_episodes.keys(),
            "Audit/source/continuity episode membership mismatch")
    expected_split = fixed_episode_split(source_episodes, seed=split.get("seed"),
                                         val_episodes=split.get("val_episode_count"))
    require(all(split.get(key) == value for key, value in expected_split.items()),
            "Fixed split was edited, is not disjoint/complete, or differs from its declared algorithm")
    require(split["seed"] == meta.get("seed") and split["val_episode_count"] == meta.get("val_episodes") and
            split.get("partition_unit") == "whole_episode" and split.get("excluded_episode_indices") == [],
            "Split provenance mismatch; no episode may be silently excluded")
    for partition in ("train", "val"):
        total = sum(source_episodes[eid]["length"] for eid in split[f"{partition}_episode_indices"])
        require(type(split.get(f"{partition}_rows")) is int and split[f"{partition}_rows"] == total,
                f"{partition} row count differs from the fixed split")

    episodes = {}
    expected_paths = {r["path"] for r in metadata_records}
    gap_factor = meta.get("gap_factor")
    require(isinstance(gap_factor, (int, float)) and not isinstance(gap_factor, bool) and math.isfinite(gap_factor) and gap_factor > 1,
            "Invalid audited elapsed-gap factor")
    for eid, entry in source_episodes.items():
        relative = info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid)
        path = _safe_file(root, relative)
        require(relative not in expected_paths, "Multiple episodes reference one Parquet file")
        expected_paths.add(relative)
        require(relative in signatures, f"Episode missing from audited manifest: {relative}")
        declared = ranges_episodes[eid]
        scanned = report_episodes[eid]
        require(declared.get("data_path") == relative and scanned.get("data_path") == relative and
                declared.get("rows") == scanned.get("rows_read") == entry["length"],
                f"Episode {eid}: path or row count differs from source")
        pf = pq.ParquetFile(path)
        require(pf.metadata.num_rows == entry["length"], f"Episode {eid}: invalid Parquet row count")
        # Validate boundaries against actual saved scalars, not just an editable JSON list.
        columns = ("timestamp", "recording.elapsed_time", "recording.segment_index")
        table = pf.read(columns=list(columns), use_threads=False)
        timing_issues = Issues()
        timing = _timing(*(table[k].to_pylist() for k in columns), entry, info["fps"], gap_factor, timing_issues)
        require(timing_issues.errors == 0 and timing.get("checks_complete") is True,
                f"Episode {eid}: invalid saved continuity scalars")
        require(declared.get("checks_complete") is True and
                declared.get("contiguous_ranges") == timing["contiguous_ranges"] ==
                scanned.get("timing", {}).get("contiguous_ranges"),
                f"Episode {eid}: continuity ranges disagree with saved rows")
        cursor, ranges = 0, []
        for interval in declared["contiguous_ranges"]:
            start, stop = interval["start_frame"], interval["stop_frame_exclusive"]
            require(type(start) is int and type(stop) is int and start == cursor < stop <= entry["length"],
                    f"Episode {eid}: continuity must cover every row exactly once")
            ranges.append((start, stop))
            cursor = stop
        require(cursor == entry["length"], f"Episode {eid}: incomplete continuity")
        episodes[eid] = Episode(eid, relative, entry["length"], tuple(ranges))
    require(expected_paths == seen, "Audited manifest does not exactly cover source metadata/episodes")
    require(report["counts"].get("episodes_scanned") == len(episodes) and
            report["counts"].get("rows_read") == info["total_frames"], "Audit source totals disagree")
    verified = VerifiedData(root, audit, config, info, episodes, documents,
                            {name: sha256(payload).hexdigest() for name, payload in raw.items()}, signatures)
    verified.assert_unchanged()
    return verified
