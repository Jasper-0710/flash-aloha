"""Offline window-mixed sampling; sample contents still come from ActDataset.

Only num_workers=0 is supported: the sampler announces each window immediately
before its batches are fetched. It never reads or changes an image/action itself.
The cache holds original Arrow tables (embedded encoded images), not a decoded
corpus. The byte cap is retained table buffers, NOT a process RSS/peak-read cap.
"""
import math

import numpy as np

from src.wp_ws.act_dataset import ActDataset
from src.wp_ws.verified_data import require


class WindowCachedActDataset(ActDataset):
    def __init__(self, data, statistics, policy_config, *, window_episodes=8, cache_mib=1024):
        super().__init__(data, statistics, policy_config, partition="train")
        require(type(window_episodes) is int and window_episodes >= 2, "window_episodes must be >= 2")
        require(type(cache_mib) is int and cache_mib > 0, "cache_mib must be positive")
        self.window_episodes = window_episodes
        self.cache_limit_bytes = cache_mib * 1024 ** 2
        self._tables, self._window, self.cache_bytes = {}, (), 0
        self.peak_cache_bytes = self.table_reads = 0

    def begin_window(self, episode_ids):
        ids = tuple(episode_ids)
        require(0 < len(ids) <= self.window_episodes and len(set(ids)) == len(ids), "Invalid cache window")
        require(all(eid in self.episode_ids for eid in ids), "Window must contain only training episodes")
        self.clear_cache()  # release old buffers before reading the next window
        self._window = ids

    def _table(self, eid):
        require(eid in self._window, "Episode outside active cache window; use the window sampler with workers=0")
        self.data.source_file(self.data.episodes[eid].data_path)  # check even on cache hits
        if eid not in self._tables:
            try:
                table = super()._table(eid)  # exact original verified column reader
            finally:
                # No hidden reference in the base class may keep an evicted table alive.
                ActDataset.clear_cache(self)
            size = table.get_total_buffer_size()
            require(self.cache_bytes + size <= self.cache_limit_bytes,
                    "Active window exceeds --episode-cache-mib; reduce --mix-episodes or increase the cap. "
                    "Refusing cache thrashing or silent sampler changes.")
            self._tables[eid] = table
            self.table_reads += 1
            self.cache_bytes += size
            self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        return self._tables[eid]

    def clear_cache(self):
        super().clear_cache()
        self._tables, self._window, self.cache_bytes = {}, (), 0

    def __getstate__(self):
        state = super().__getstate__()
        state.update(_tables={}, _window=(), cache_bytes=0)
        return state


class WindowBatchSampler:
    """Reshuffle episodes each epoch, mix starts within windows, no drop/duplicates.

    Not global uniform shuffling. A window's tail is kept as a smaller batch;
    batch count depends on the epoch's grouping. Actions never cross boundaries.
    """
    def __init__(self, dataset, batch_size, *, seed=42):
        require(isinstance(dataset, WindowCachedActDataset), "Window sampler requires its bounded cache")
        require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
        self.set_epoch(0)

    def set_epoch(self, epoch):
        require(type(epoch) is int and epoch >= 0 and type(self.seed) is int and self.seed >= 0,
                "Invalid sampling seed/epoch")
        self.epoch = epoch

    def _plan(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        ids = list(self.dataset.episode_ids)
        rng.shuffle(ids)
        width = self.dataset.window_episodes
        return rng, [ids[i:i + width] for i in range(0, len(ids), width)]

    def __len__(self):
        _, windows = self._plan()
        return sum(math.ceil(sum(self.dataset.data.episodes[eid].rows for eid in ids) / self.batch_size)
                   for ids in windows)

    def __iter__(self):
        rng, windows = self._plan()
        for ids in windows:
            self.dataset.begin_window(ids)
            indices = np.concatenate([np.arange(self.dataset.data.episodes[eid].rows, dtype=np.int64) +
                                      self.dataset.episode_offsets[eid] for eid in ids])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size].tolist()
