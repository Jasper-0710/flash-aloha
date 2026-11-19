"""Read-only W&B sidecar for train_act metrics; never starts or changes training.

Import existing JSONL records, optionally following appended records. Only explicitly
selected scalars/configuration reach W&B; no watch(), artifacts or file uploads.
All uploader files are separate from the original run, under central LOG_DIR.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

from src.config.paths import CKPT_DIR, DATA_DIR, LOG_DIR


def _component(value):
    if (not value or value in (".", "..", "_wandb_uploads") or value != value.strip()
            or any(c in value for c in ("/", "\\")) or any(ord(c) < 32 for c in value)):
        raise argparse.ArgumentTypeError("must be one nonempty name, not a path")
    return value


def _interval(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=_component, default="flip_the_pillow")
    parser.add_argument("--run-name", type=_component, required=True,
                        help="Existing train_act run under LOG_DIR/wp_ws/<task>")
    parser.add_argument("--project", default="flash-aloha")
    parser.add_argument("--entity", help="Optional W&B account/team; never an API key")
    parser.add_argument("--name", help="Optional W&B display name; default: source run name")
    parser.add_argument("--mode", choices=("online", "offline"), default="offline",
                        help="Default offline; explicitly select online to upload to W&B")
    parser.add_argument("--follow", action="store_true", help="Follow until completion/failure or Ctrl+C")
    parser.add_argument("--poll-seconds", type=_interval, default=1.0)
    return parser


def source_directory(task, run_name):
    _component(task)
    _component(run_name)
    namespace = (LOG_DIR / "wp_ws" / task).resolve()
    source = (namespace / run_name).resolve()
    if source.parent != namespace or not source.is_dir():
        raise ValueError(f"Existing training log directory not found/escapes namespace: {source}")
    return source


def _source_file(source, name):
    path = source / name
    if path.resolve().parent != source:
        raise ValueError(f"Source log must not point outside the training run: {name}")
    return path


def _document(source, name):
    with _source_file(source, name).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {name}")
    return value


def scalar_configuration(source, task, run_name):
    manifest = _document(source, "run_config.json")
    if (manifest.get("format") != "wp_ws.act_baseline.v1"
            or manifest.get("data_config", {}).get("task_name") != task):
        raise ValueError("Expected the matching wp_ws ACT training manifest")
    result = {"source_run_name": run_name, "task_name": task, "tracking": "metrics_sidecar",
              "model_variant": manifest["model_variant"],
              "validation_mode": manifest["validation_selection"]["mode"],
              "live_deployment_ready": False}
    for key in ("batch_size", "seed", "steps_requested", "gradient_clip_max_norm"):
        if key in manifest:
            result[key] = manifest[key]
    sampling = manifest.get("sampler_configuration", {})
    result["sampling"] = {key: sampling[key] for key in ("mode", "window_episodes", "epoch0_batches") if key in sampling}
    probe = manifest.get("train_probe_selection") or {}
    result["train_probe"] = {key: probe[key] for key in ("sample_count", "samples_per_episode", "evaluation_scope")
                             if key in probe}
    # Do not upload cli_arguments, provenance/paths, statistics, environment or arbitrary keys.
    policy = manifest.get("policy_config", {})
    result["policy"] = {key: policy[key] for key in (
        "policy_class", "precision", "backbone", "pretrained_backbone_weights", "camera_names",
        "state_dim", "action_dim", "num_queries", "height", "width", "hidden_dim", "nheads",
        "dec_layers", "dropout", "lr", "lr_backbone", "weight_decay",
    ) if key in policy}
    return result


def _terminal_state(source):
    if _source_file(source, "FAILED.json").exists():
        return "failed"
    if _source_file(source, "COMPLETED.json").exists():
        return "completed"
    return None


def metric_records(source, *, follow=False, poll_seconds=1.0, sleep=time.sleep):
    """Follow append-only JSONL without dropping a concurrently written partial line."""
    path = _source_file(source, "metrics.jsonl")
    while not path.exists():
        if _terminal_state(source):
            return
        if not follow:
            raise FileNotFoundError(f"No metrics yet: {path}; use --follow while training starts")
        sleep(poll_seconds)
    with path.open("rb") as stream:
        identity = (path.stat().st_dev, path.stat().st_ino)
        line_number = 0
        while True:
            _source_file(source, "metrics.jsonl")
            stat = path.stat()
            if (stat.st_dev, stat.st_ino) != identity or stat.st_size < stream.tell():
                raise ValueError("Source metrics were replaced/truncated; refusing to duplicate points")
            position = stream.tell()
            line = stream.readline()
            if line and line.endswith(b"\n"):
                line_number += 1
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"metrics.jsonl line {line_number}: expected an object")
                yield record
                if record.get("event") == "completed":
                    return
                continue
            if line:
                stream.seek(position)
                if not follow or _terminal_state(source):
                    raise ValueError("Incomplete final JSONL line; wait for its writer or use --follow")
            elif not follow or _terminal_state(source):
                return
            sleep(poll_seconds)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite scalar: {name}")
    return value


def metric_values(record):
    event = record.get("event")
    if event not in ("model_ready", "train", "train_probe", "validation", "completed"):
        return None
    step = 0 if event == "model_ready" else record.get("steps" if event == "completed" else "step")
    if type(step) is not int or step < 0:
        raise ValueError(f"Invalid training step for {event}: {step!r}")
    values = {"global_step": step}
    if event in ("train", "train_probe", "validation"):
        prefix = "val" if event == "validation" else event
        values[f"{prefix}/normalized_l1"] = _number(record.get("normalized_l1"), "normalized_l1")
    mappings = {
        "train": {"gradient_norm_before_clip": "train/gradient_norm_before_clip",
                  "batch_size_actual": "train/batch_size_actual", "epoch_zero_based": "train/epoch_zero_based",
                  "window_normalized_l1": "train/window_normalized_l1",
                  "window_valid_action_elements": "train/window_valid_action_elements",
                  "batch_episode_count": "train/batch_episode_count", "samples_seen": "train/samples_seen"},
        "train_probe": {"samples": "train_probe/samples", "valid_action_elements": "train_probe/valid_action_elements",
                        "mean_constant_normalized_l1": "train_probe/mean_constant_normalized_l1",
                        "joint_mae_rad": "train_probe/joint_mae_rad",
                        "gripper_mae_normalized_units": "train_probe/gripper_mae_normalized_units"},
        "validation": {"samples": "val/samples", "valid_action_elements": "val/valid_action_elements",
                       "mean_constant_normalized_l1": "val/mean_constant_normalized_l1"},
        "model_ready": {"trainable_parameters": "model/trainable_parameters"},
    }
    for source_key, target in mappings.get(event, {}).items():
        if source_key in record:
            values[target] = _number(record[source_key], source_key)
    if "elapsed_seconds" in record:
        values["timing/elapsed_seconds"] = _number(record["elapsed_seconds"], "elapsed_seconds")
    for key in ("peak_allocated_gib", "peak_reserved_gib", "free_gib_now", "total_gib"):
        if key in record.get("cuda_memory", {}):
            values[f"cuda/{key}"] = _number(record["cuda_memory"][key], key)
    return values


def _upload_directory(source, task, run_name):
    base = (LOG_DIR / "wp_ws" / "_wandb_uploads" / task / run_name).resolve()
    for protected in (source, DATA_DIR.resolve(), CKPT_DIR.resolve()):
        if base == protected or protected in base.parents or base in protected.parents:
            raise ValueError("W&B uploader output must not overlap source logs, datasets or checkpoints")
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="upload-", dir=base))


@contextmanager
def _sdk_storage(directory):
    # The core service otherwise puts its own debug log in ~/.cache/wandb,
    # even when init(dir=...) is explicit. Keep it and SDK staging under LOG_DIR.
    values = {"WANDB_CACHE_DIR": str(directory / "cache"),
              "WANDB_DATA_DIR": str(directory / "staging")}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def upload(args, *, backend=None, sleep=time.sleep):
    source = source_directory(args.task, args.run_name)
    config = scalar_configuration(source, args.task, args.run_name)
    directory = _upload_directory(source, args.task, args.run_name)
    with _sdk_storage(directory):
        if backend is None:
            try:
                import wandb as backend
            except ImportError as exc:
                raise ImportError("Install optional dependency: python -m pip install -r src/wp_ws/requirements-wandb.txt") from exc
        return _upload(args, source, config, directory, backend, sleep)


def _upload(args, source, config, directory, backend, sleep):
    print(f"只读指标来源：{source}\nW&B 本地文件：{directory}\n模式：{args.mode}", flush=True)
    settings = backend.Settings(console="off", disable_code=True, disable_git=True,
                                x_disable_stats=True, x_disable_meta=True,
                                x_save_requirements=False, symlink=False)
    # Fresh W&B identity on every invocation: no accidental merge with another upload/run.
    run = backend.init(project=args.project, entity=args.entity, name=args.name or args.run_name,
                       id=uuid.uuid4().hex[:12], group=args.task, job_type="metrics-sidecar",
                       mode=args.mode, dir=str(directory), config=config,
                       tags=["act", "metrics-only", config["validation_mode"]], settings=settings)
    if run is None:
        raise RuntimeError("W&B did not create a run")
    exit_code = 1
    count = 0
    completed = False
    last_step = -1
    try:
        run.define_metric("global_step")
        for prefix in ("train", "train_probe", "val", "cuda", "timing", "model"):
            run.define_metric(f"{prefix}/*", step_metric="global_step")
        run.define_metric("val/normalized_l1", step_metric="global_step", summary="min")
        run.summary["source_status"] = "following" if args.follow else "snapshot_import"
        if args.mode == "online" and run.url:
            print(f"W&B 页面：{run.url}", flush=True)
        for record in metric_records(source, follow=args.follow, poll_seconds=args.poll_seconds, sleep=sleep):
            values = metric_values(record)
            if values is None:
                continue
            if values["global_step"] < last_step:
                raise ValueError("Training step moved backwards; refusing to mix/replay duplicate histories")
            # Do NOT pass step=global_step: train + validation can share a training step.
            run.log(values)
            last_step = values["global_step"]
            count += 1
            if record["event"] == "validation":
                print(f"[W&B] step={last_step} val/normalized_l1={record['normalized_l1']:.6f}", flush=True)
            completed = record["event"] == "completed"
        state = _terminal_state(source) or ("completed" if completed else "snapshot")
        run.summary.update({"source_status": state, "uploaded_records": count,
                            "last_training_step": last_step, "task_success_rate_measured": False})
        exit_code = 1 if state == "failed" else 0
        print(f"W&B 指标处理完成：{count} 条；源训练状态：{state}；未改写源文件。", flush=True)
        return exit_code
    except KeyboardInterrupt:
        run.summary["source_status"] = "uploader_stopped_by_user"
        exit_code = 0
        raise
    except Exception:
        run.summary["source_status"] = "uploader_error_not_training_status"
        raise
    finally:
        run.finish(exit_code=exit_code)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return upload(args)
    except KeyboardInterrupt:
        print("已停止 W&B 指标跟随；不会停止或影响原训练。", flush=True)
        return 130
    except Exception as exc:
        print(f"W&B 指标上传失败（原训练不受影响）：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
