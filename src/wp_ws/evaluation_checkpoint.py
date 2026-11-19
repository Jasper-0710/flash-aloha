"""Read-only provenance checks for completed native ACT baseline checkpoints."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import json

from src.config.paths import CKPT_DIR, DATA_DIR, LOG_DIR
from src.wp_ws.act_dataset import image_configuration
from src.wp_ws.audit_core import hash_file
from src.wp_ws.audit_dataset import resolve_locations
from src.wp_ws.baseline_data_config import _component
from src.wp_ws.verified_data import require, signature


# These implementations affect observations, targets, or predictions. A changed
# trainer/optimizer is not used for inference; record, but do not require, those hashes.
def inference_files():
    here = Path(__file__).resolve().parent
    return {"native_act": here.parent / "policy/act.py",
            "shared_images": here.parent / "contracts/standard_image_processing.py",
            "adapter": here / "act_dataset.py", "verifier": here / "verified_data.py"}


def _child(parent, name):
    _component(name, "path component")
    path = parent / name
    require(not path.is_symlink() and path.resolve().parent == parent.resolve(),
            f"Symlink/escaped path is not allowed: {path}")
    return path.resolve()


def _overlap(a, b):
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def locations(config, source_run, run_name):
    """Only central paths, completed input runs, and a never-used output directory."""
    logs = _child((LOG_DIR / "wp_ws").resolve(), config.task_name)
    ckpts = _child((CKPT_DIR / "wp_ws").resolve(), config.task_name)
    source_logs, source_ckpts = _child(logs, source_run), _child(ckpts, source_run)
    output = _child(logs, run_name)
    root, audit = resolve_locations(config.dataset, config.audit_run)
    for protected in (DATA_DIR, CKPT_DIR, root, audit, source_logs, source_ckpts):
        require(not _overlap(output, protected), f"Evaluation output overlaps protected input: {protected}")
    require(not output.exists(), f"Run already exists; choose a new --run-name: {output}")
    require(source_logs.is_dir() and source_ckpts.is_dir(), "Source log/checkpoint run does not exist")
    return source_logs, source_ckpts, output


@dataclass
class CheckedCheckpoint:
    payload: dict
    evidence: dict
    signatures: dict

    def assert_unchanged(self):
        for key, (path, before) in self.signatures.items():
            require(not path.is_symlink() and signature(path) == before,
                    f"Checkpoint/source artifact changed: {key}")
            require(hash_file(path) == self.evidence[key]["sha256"], f"Checkpoint/source content changed: {key}")


def load_checkpoint(source_logs, source_ckpts, name):
    """Restricted CPU deserialization only; never unpickle arbitrary Python classes."""
    import torch
    require(name in ("best", "last"), "Checkpoint must be best or last")
    paths = {"checkpoint": _child(source_ckpts, name + ".ckpt")}
    for key, directory, filename in (
        ("manifest", source_logs, "run_config.json"),
        ("checkpoint_manifest", source_ckpts, "run_config.json"),
        ("statistics", source_logs, "normalization_stats.json"),
        ("checkpoint_statistics", source_ckpts, "normalization_stats.json"),
        ("completed", source_logs, "COMPLETED.json"),
        *((f"audit_{n}", source_logs, n) for n in ("split.json", "continuity.json", "fingerprints.json")),
    ):
        paths[key] = _child(directory, filename)
    evidence, signatures, documents = {}, {}, {}
    for key, path in paths.items():
        require(path.is_file(), f"Missing source artifact: {path}")
        before = signature(path)
        evidence[key] = {"path": str(path), "sha256": hash_file(path), "size_bytes": before[0]}
        if key != "checkpoint":
            documents[key] = json.loads(path.read_text(encoding="utf-8"))
        require(signature(path) == before, f"Source artifact changed while reading: {key}")
        signatures[key] = (path, before)
    require(documents["completed"].get("status") == "TRAINING_STEPS_COMPLETED",
            "Only a completed training run is supported; do not evaluate a moving checkpoint")
    # Explicit safe loading. If the runtime cannot load safely, stop: no unsafe fallback.
    payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and payload.get("format") == "wp_ws.act_baseline.v1",
            "Expected a native wp_ws ACT baseline checkpoint (not an overfit/foreign checkpoint)")
    require(payload.get("run_config") == documents["manifest"] == documents["checkpoint_manifest"],
            "Checkpoint/run manifest mismatch")
    require(payload.get("normalization_stats") == documents["statistics"] == documents["checkpoint_statistics"],
            "Checkpoint normalization statistics mismatch")
    require(isinstance(payload.get("model_state_dict"), dict) and payload["model_state_dict"],
            "Checkpoint has no model state")
    require(all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload["model_state_dict"].items()),
            "Invalid model state; expected named tensors")
    reference, step = payload.get("validation"), payload.get("step")
    require(type(step) is int and step >= 0 and isinstance(reference, dict) and reference.get("step") == step,
            "Checkpoint validation is stale or missing")
    completed = documents["completed"]
    if name == "best":
        require(completed.get("best_validation_step") == step and
                completed.get("best_normalized_val_l1") == reference.get("normalized_l1"),
                "Best checkpoint disagrees with completed training summary")
    else:
        require(completed.get("steps") == step and completed.get("last_validation") == reference,
                "Last checkpoint disagrees with completed training summary")
    for key in ("optimizer_state_dict", "rng_state"):
        payload.pop(key, None)  # Never resume or restore optimizer/RNG from a training checkpoint.
    checked = CheckedCheckpoint(payload, evidence, signatures)
    checked.assert_unchanged()
    return checked, documents


def verify_contract(checked, documents, config, data):
    payload = checked.payload
    manifest, policy = payload["run_config"], payload.get("policy_config")
    require(manifest.get("format") == payload["format"] and manifest.get("data_config") == asdict(config),
            "Data configuration differs from the training checkpoint")
    require(isinstance(policy, dict) and policy == manifest.get("policy_config"), "Policy configuration mismatch")
    require(policy.get("policy_class") == "ACT" and policy.get("precision") == "bf16" and
            policy.get("state_dim") == policy.get("action_dim") == 14 and
            policy.get("training_time_rtc") is False and
            policy.get("state_idx") is None and policy.get("action_idx") is None,
            "Evaluator supports the recorded 14D non-RTC bf16 ACT baseline only")
    require(manifest.get("provenance") == data.provenance(), "Source/audit provenance differs from training")
    for name in ("split.json", "continuity.json", "fingerprints.json"):
        require(documents[f"audit_{name}"] == data.documents[name], f"Training {name} differs from verified audit")
    selection = manifest.get("validation_selection", {})
    batch_size = selection.get("batch_size")
    require(type(batch_size) is int and batch_size > 0, "Invalid original validation batch size")
    require(selection == {"mode": "full_fixed_validation", "sample_count": data.split["val_rows"],
                          "episode_indices": data.split["val_episode_indices"], "batch_size": batch_size},
            "A full fixed validation run is required; smoke subsets are not comparable")
    require(manifest.get("image_preprocessing") == image_configuration(policy, data),
            "Shared image preprocessing declaration differs from checkpoint")
    reference = payload["validation"]
    require(reference.get("mode") == "full_fixed_validation" and reference.get("samples") == data.split["val_rows"],
            "Checkpoint validation scope/count mismatch")
    require(isinstance(reference.get("normalized_l1"), (int, float)) and
            math.isfinite(reference["normalized_l1"]) and reference["normalized_l1"] >= 0,
            "Invalid checkpoint validation loss")
    chunk_size = policy["num_queries"]
    require(type(chunk_size) is int and chunk_size > 0, "Invalid action chunk size")
    expected_elements = 14 * sum(min(chunk_size, stop - row)
        for eid in data.split["val_episode_indices"]
        for start, stop in data.episodes[eid].ranges for row in range(start, stop))
    require(reference.get("valid_action_elements") == expected_elements,
            "Checkpoint validation mask/count differs from continuity")
    current = {}
    for key, path in inference_files().items():
        before = signature(path)
        digest = hash_file(path)
        require(manifest.get("implementation_sha256", {}).get(key) == digest,
                f"Inference implementation changed since training: {key}; review instead of silently migrating")
        checked.signatures[f"code_{key}"] = (path, before)
        checked.evidence[f"code_{key}"] = {"path": str(path), "sha256": digest, "size_bytes": before[0]}
        current[key] = digest
    return current
