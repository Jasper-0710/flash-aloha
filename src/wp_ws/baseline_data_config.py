"""Data configuration for the offline ACT baseline; never rename source labels.

This module only parses/previews configuration. It does not start training,
verify dataset/audit fingerprints, or implement a training dataset adapter.
All persistent data/checkpoint/log locations use src.config.paths.
"""
import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import sys

from src.config.paths import CKPT_DIR
from src.wp_ws.audit_dataset import resolve_locations

DEFAULT_CONFIG = Path(__file__).with_name("flip_the_pillow_baseline.json")


def _text(value, name):
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"{name} must be a nonempty string without edge whitespace/control characters")


def _component(value, name):
    _text(value, name)
    if value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be one name, not a filesystem path")


@dataclass(frozen=True)
class BaselineDataConfig:
    schema_version: int
    policy: str
    task_name: str
    task_description: str
    dataset: str
    source_task_name: str
    audit_run: str
    state_key: str
    state_dim: int
    action_key: str
    action_dim: int
    use_base_action: bool
    gripper_labels: str
    statistics_scope: str
    split_source: str
    continuity_source: str

    def validate(self):
        for name in ("task_name", "dataset", "audit_run"):
            _component(getattr(self, name), name)
        for name in ("task_description", "source_task_name"):
            _text(getattr(self, name), name)
        # These are invariants of this initial baseline, not data transforms.
        required = {
            "schema_version": 1, "policy": "ACT",
            "state_key": "observation.state", "state_dim": 14,
            "action_key": "action", "action_dim": 14,
            "use_base_action": False, "gripper_labels": "preserve_raw",
            "statistics_scope": "train_episode_indices_only",
            "split_source": "audit_split_json", "continuity_source": "audit_continuity_json",
        }
        for name, expected in required.items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"This baseline requires {name}={expected!r}; got {value!r}")
        return self

    def preview(self):
        self.validate()
        root, audit = resolve_locations(self.dataset, self.audit_run)
        return {
            "status": "CONFIG_PREVIEW_ONLY",
            "scope": "No training, dataset writes, or source/audit fingerprint verification",
            "config": asdict(self),
            "paths": {
                "dataset": str(root),
                "source_tasks": str(root / "meta/tasks.jsonl"),
                "audit_report": str(audit / "report.json"),
                "audit_completed": str(audit / "COMPLETED.json"),
                "audit_fingerprints": str(audit / "fingerprints.json"),
                "split": str(audit / "split.json"),
                "continuity": str(audit / "continuity.json"),
                "checkpoint_namespace": str(CKPT_DIR / "wp_ws" / self.task_name),
            },
            "source_metadata_changed": False,
            "task_name_usage": "Training experiment/checkpoint identity only; keep source task_index and labels",
        }


def load_config(path=DEFAULT_CONFIG):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Baseline configuration must be a JSON object")
    expected = {field.name for field in fields(BaselineDataConfig)}
    if set(document) != expected:
        raise ValueError(f"Config fields: missing={sorted(expected - set(document))}, "
                         f"unknown={sorted(set(document) - expected)}")
    return BaselineDataConfig(**document).validate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Baseline data JSON (not a native Flash ALOHA training recipe)")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(load_config(args.config).preview(), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
        print(f"Baseline configuration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
