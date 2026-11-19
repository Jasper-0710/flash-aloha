"""Versioned, episode-level splits without dependence on a PRNG implementation."""
import hashlib

SPLIT_ALGORITHM = "sha256-episode-rank-v1"


def fixed_episode_split(episode_indices, *, seed=42, val_episodes=10):
    """Choose validation IDs by SHA-256 rank; return ascending disjoint ID lists.

    All source episodes must have passed the audit before this is published.
    No episode is silently excluded or subdivided into train/validation frames.
    """
    ids = list(episode_indices)
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Episode indices must be nonnegative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode indices")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(val_episodes) is not int or not 0 < val_episodes < len(ids):
        raise ValueError("val_episodes must leave at least one episode in each split")

    def rank(index):
        payload = f"{SPLIT_ALGORITHM}:{seed}:{index}".encode("ascii")
        return hashlib.sha256(payload).digest(), index

    validation = set(sorted(ids, key=rank)[:val_episodes])
    return {
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "val_episode_count": val_episodes,
        "train_episode_indices": sorted(set(ids) - validation),
        "val_episode_indices": sorted(validation),
    }
