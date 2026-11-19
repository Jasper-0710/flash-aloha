"""Offline ACT adapter for the *original* embedded-image Parquet rows.

Only training episodes fit normalization. Images use the repository's existing
Standard ACT/Flow processor. No timestamp/image pairing, clipping or action
conversion happens here. At most one compressed Arrow episode is cached per
worker; only the requested row's images are decoded.
"""
from bisect import bisect_right
from dataclasses import asdict
from io import BytesIO
import math

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from src.contracts import standard_image_processing as shared_images
from src.wp_ws.audit_core import EXPECTED_CAMERAS
from src.wp_ws.verified_data import require


class Moments:
    """Merge batch moments in float64; do not use audit/global/validation stats."""
    def __init__(self, width):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        require(values.ndim == 2 and values.shape[1] == self.mean.size and np.isfinite(values).all(),
                "Nonfinite or incorrectly shaped state/action in training statistics")
        if not len(values):
            return
        count = self.count + len(values)
        delta = values.mean(axis=0) - self.mean
        self.m2 += ((values - values.mean(axis=0)) ** 2).sum(axis=0) + \
                   delta ** 2 * self.count * len(values) / count
        self.mean += delta * len(values) / count
        self.count = count
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def export(self, std_floor):
        require(self.count > 0, "No training samples for statistics")
        population_std = np.sqrt(np.maximum(self.m2 / self.count, 0))
        return {
            "mean": self.mean.tolist(),
            "std": np.maximum(population_std, std_floor).tolist(),
            "population_std_before_floor": population_std.tolist(),
            "min": self.minimum.tolist(), "max": self.maximum.tolist(),
            "rows": self.count,
        }


def fit_training_statistics(data, *, std_floor=0.01, progress=print):
    require(math.isfinite(std_floor) and std_floor > 0, "std_floor must be positive and finite")
    keys = (data.config.state_key, data.config.action_key)
    moments = {key: Moments(14) for key in keys}
    ids = data.split["train_episode_indices"]
    for number, eid in enumerate(ids, 1):
        if progress:
            progress(f"[训练集统计 {number}/{len(ids)}] episode={eid:06d}（不读取验证集统计）")
        episode = data.episodes[eid]
        path = data.source_file(episode.data_path)
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=list(keys), use_threads=False):
            for key in keys:
                moments[key].add(batch.column(batch.schema.get_field_index(key)).to_pylist())
            count += batch.num_rows
        require(count == episode.rows, f"Episode {eid}: row count changed during statistics")
        data.source_file(episode.data_path)
    require(all(m.count == data.split["train_rows"] for m in moments.values()), "Training statistics row mismatch")
    return {
        "schema_version": 1, "scope": "train_episode_indices_only",
        "train_episode_indices": list(ids), "rows": data.split["train_rows"],
        "dataset_fingerprint": data.fingerprint,
        "audit_artifact_sha256": data.audit_sha256,
        "method": "float64 merged population moments; (raw - mean) / max(population_std, std_floor)",
        "std_floor": std_floor, "gripper_labels": "preserve_raw",
        "state": moments[data.config.state_key].export(std_floor),
        "action": moments[data.config.action_key].export(std_floor),
    }


def image_configuration(policy_config, data):
    config = {"policy_config": policy_config, "aloha": False, "grayscale": False}
    spec = shared_images.policy_image_spec(config)
    return {
        "profile": "Standard ACT/Flow", "implementation": "src/contracts/standard_image_processing.py",
        "configuration": config, "spec": asdict(spec), "source_color": "RGB",
        "camera_keys_in_order": list(EXPECTED_CAMERAS),
        "native_shapes": {key: data.info["features"][key]["shape"] for key in EXPECTED_CAMERAS},
        "layout": "cameras,CHW", "dtype": "float32", "float_scaling": "uint8 / 255",
        "crop": None, "letterbox": False, "augmentation": None,
        "backbone_normalization": "inside native ACTPolicy._inputs only",
    }


class ActDataset:
    """PyTorch map-style compatible without importing torch or robot packages."""
    def __init__(self, data, statistics, policy_config, *, partition):
        require(partition in ("train", "val"), "partition must be train or val")
        require(statistics.get("dataset_fingerprint") == data.fingerprint and
                statistics.get("audit_artifact_sha256") == data.audit_sha256 and
                statistics.get("scope") == "train_episode_indices_only" and
                statistics.get("train_episode_indices") == data.split["train_episode_indices"] and
                statistics.get("rows") == data.split["train_rows"], "Normalization provenance mismatch")
        self.data = data
        self.episode_ids = tuple(data.split[f"{partition}_episode_indices"])
        self.chunk_size = policy_config["num_queries"]
        require(type(self.chunk_size) is int and self.chunk_size > 0, "num_queries must be a positive integer")
        self.spec = shared_images.policy_image_spec({"policy_config": policy_config, "aloha": False, "grayscale": False})
        require(self.spec.width is not None and self.spec.height is not None, "Explicit policy image size required")
        self.mean, self.std = {}, {}
        for key in ("state", "action"):
            self.mean[key] = np.asarray(statistics[key]["mean"], dtype=np.float32)
            self.std[key] = np.asarray(statistics[key]["std"], dtype=np.float32)
            require(self.mean[key].shape == self.std[key].shape == (14,) and
                    np.isfinite(self.mean[key]).all() and np.isfinite(self.std[key]).all() and
                    (self.std[key] > 0).all(), "Invalid normalization vectors")
        self.offsets = [0]
        self.episode_offsets, self.range_starts = {}, {}
        for eid in self.episode_ids:
            self.episode_offsets[eid] = self.offsets[-1]
            self.offsets.append(self.offsets[-1] + data.episodes[eid].rows)
            self.range_starts[eid] = tuple(start for start, _ in data.episodes[eid].ranges)
        self._cache_eid = self._cache_table = None

    def __len__(self):
        return self.offsets[-1]

    def locate(self, index):
        require(isinstance(index, (int, np.integer)) and 0 <= index < len(self), "Sample index out of range")
        position = bisect_right(self.offsets, int(index)) - 1
        return self.episode_ids[position], int(index) - self.offsets[position]

    def sample_index(self, eid, frame):
        require(eid in self.episode_offsets and 0 <= frame < self.data.episodes[eid].rows, "Invalid episode/frame")
        return self.episode_offsets[eid] + frame

    def _table(self, eid):
        episode = self.data.episodes[eid]
        path = self.data.source_file(episode.data_path)
        if self._cache_eid != eid:
            # Release previous episode before reading the next; never cache the decoded corpus.
            self._cache_eid = self._cache_table = None
            columns = [self.data.config.state_key, self.data.config.action_key,
                       "task_index", "frame_index", "recording.segment_index", *EXPECTED_CAMERAS]
            table = pq.ParquetFile(path).read(columns=columns, use_threads=False)
            require(table.num_rows == episode.rows, f"Episode {eid}: Parquet rows changed")
            self.data.source_file(episode.data_path)
            self._cache_eid, self._cache_table = eid, table
        return self._cache_table

    def clear_cache(self):
        self._cache_eid = self._cache_table = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache_eid"] = state["_cache_table"] = None
        return state

    def __getitem__(self, index):
        eid, row = self.locate(index)
        table = self._table(eid)
        episode = self.data.episodes[eid]
        interval = bisect_right(self.range_starts[eid], row) - 1
        stop = min(row + self.chunk_size, episode.ranges[interval][1])
        raw_state = np.asarray(table[self.data.config.state_key][row].as_py(), dtype=np.float32)
        raw_actions = np.asarray(table[self.data.config.action_key].slice(row, stop - row).to_pylist(), dtype=np.float32)
        require(raw_state.shape == (14,) and raw_actions.shape == (stop - row, 14) and
                np.isfinite(raw_state).all() and np.isfinite(raw_actions).all(), "Invalid saved state/action")
        # Action[0] is the original action at this exact row: no shifts or follower-angle conversion.
        actions = np.zeros((self.chunk_size, 14), dtype=np.float32)
        actions[:stop - row] = (raw_actions - self.mean["action"]) / self.std["action"]
        is_pad = np.arange(self.chunk_size) >= stop - row
        images = []
        for key in EXPECTED_CAMERAS:
            cell = table[key][row].as_py()
            require(isinstance(cell, dict) and cell.get("bytes"), f"Missing embedded image: {eid}/{row}/{key}")
            with Image.open(BytesIO(cell["bytes"])) as image:
                require(image.mode == "RGB", f"Expected audited RGB image: {eid}/{row}/{key}")
                image.load()
                rgb = np.asarray(image)
            require(list(rgb.shape) == self.data.info["features"][key]["shape"], "Native image shape changed")
            images.append(shared_images.image_to_chw(rgb, self.spec, source_color="RGB"))
        return {
            "qpos": (raw_state - self.mean["state"]) / self.std["state"],
            "images": np.stack(images), "actions": actions, "is_pad": is_pad,
            "episode_index": eid, "frame_index": int(table["frame_index"][row].as_py()),
            "task_index": int(table["task_index"][row].as_py()),
            "segment_index": int(table["recording.segment_index"][row].as_py()),
        }


class EpisodeBatchSampler:
    """Shuffle episodes and rows per epoch, but batch within one episode for I/O.

    Every saved row is sampled once per epoch, including short final batches.
    No drop_last, no overlapping train/val frames, no global RNG side effects.
    """
    def __init__(self, dataset, batch_size, *, shuffle, seed=42):
        require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __len__(self):
        return sum(math.ceil(self.dataset.data.episodes[eid].rows / self.batch_size) for eid in self.dataset.episode_ids)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        ids = list(self.dataset.episode_ids)
        if self.shuffle:
            rng.shuffle(ids)
        for eid in ids:
            episode = self.dataset.data.episodes[eid]
            rows = np.arange(episode.rows, dtype=np.int64) + self.dataset.episode_offsets[eid]
            if self.shuffle:
                rng.shuffle(rows)
            for start in range(0, len(rows), self.batch_size):
                yield rows[start:start + self.batch_size].tolist()


def validation_batches(dataset, batch_size, max_batches):
    require(type(max_batches) is int and max_batches >= 0, "val_max_batches must be nonnegative")
    if max_batches == 0:
        sampler = EpisodeBatchSampler(dataset, batch_size, shuffle=False)
        selection = {"mode": "full_fixed_validation", "sample_count": len(dataset)}
    else:
        count = min(len(dataset), batch_size * max_batches)
        indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64).tolist()
        sampler = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        selection = {"mode": "fixed_evenly_spaced_smoke_subset", "sample_count": count, "flat_indices": indices}
    return sampler, {**selection, "episode_indices": list(dataset.episode_ids), "batch_size": batch_size}
