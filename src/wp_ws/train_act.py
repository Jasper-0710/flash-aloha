"""Offline, fixed-split native ACT baseline. Never opens robot/camera devices.

preflight: verify audit/source hashes, fit train-only statistics, decode CPU batches.
train: repeat preflight, then run explicitly bounded CUDA/bf16 optimization steps.
All persistent outputs use src.config.paths; existing runs are never reused.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

from src.config.paths import CKPT_DIR, LOG_DIR
from src.wp_ws.baseline_data_config import DEFAULT_CONFIG, _component, load_config


def _name(value):
    try:
        _component(value, "run_name")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "train"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--run-name", type=_name, help="New name under central LOG_DIR/CKPT_DIR; never overwrite a run")
        sub.add_argument("--batch-size", type=_positive, default=4)
        sub.add_argument("--chunk-size", type=_positive, default=50, help="50 saved actions = 1 nominal second at 50 Hz")
        sub.add_argument("--height", type=_positive, default=240)
        sub.add_argument("--width", type=_positive, default=320)
        sub.add_argument("--seed", type=_nonnegative, default=42)
        sub.add_argument("--val-max-batches", type=_nonnegative, default=10,
                         help="0: full fixed validation; >0: deterministic evenly spaced SMOKE subset")
        if command == "train":
            sub.add_argument("--steps", type=_positive, default=100)
            sub.add_argument("--validate-every", type=_positive, default=50)
            sub.add_argument("--save-every", type=_positive, default=1000)
            sub.add_argument("--log-every", type=_positive, default=10)
            sub.add_argument("--workers", type=_nonnegative, default=0,
                             help="Start with 0; each worker can cache one compressed Arrow episode")
            sub.add_argument("--lr", type=float, default=0.00012)
            sub.add_argument("--lr-backbone", type=float, default=0.00001)
            sub.add_argument("--backbone-init", choices=("imagenet", "scratch"), default="imagenet",
                             help="imagenet may download torchvision ResNet18 weights; never silently falls back")
            sub.add_argument("--acknowledge-audit-warnings", action="store_true",
                             help="Acknowledge preserved warnings; not a hardware-safety certification")
    return parser


def policy_configuration(args):
    return {
        "policy_class": "ACT", "precision": "bf16", "backbone": "resnet18",
        "pretrained_backbone_weights": "IMAGENET1K_V1" if getattr(args, "backbone_init", "imagenet") == "imagenet" else None,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "base_camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "state_dim": 14, "action_dim": 14, "num_queries": args.chunk_size,
        "height": args.height, "width": args.width,
        "hidden_dim": 512, "nheads": 8, "dec_layers": 7, "dropout": 0.1,
        "same_backbones": False, "training_time_rtc": False,
        "state_idx": None, "action_idx": None,
        "lr": getattr(args, "lr", 0.00012), "lr_backbone": getattr(args, "lr_backbone", 0.00001),
        "weight_decay": 0.0001,
    }


def output_locations(config, run_name, *, training):
    """No arbitrary output-root options, source/audit collisions, or symlink escapes."""
    _component(run_name, "run_name")
    from src.wp_ws.audit_dataset import resolve_locations
    root, audit = resolve_locations(config.dataset, config.audit_run)
    namespaces = [(LOG_DIR / "wp_ws" / config.task_name, "logs")]
    if training:
        namespaces.append((CKPT_DIR / "wp_ws" / config.task_name, "checkpoints"))
    paths = []
    for namespace, kind in namespaces:
        namespace = namespace.resolve()
        path = (namespace / run_name).resolve()
        if path.parent != namespace:
            raise ValueError(f"{kind} run escapes its configured namespace")
        for protected in (root, audit):
            if path == protected or protected in path.parents or path in protected.parents:
                raise ValueError(f"{kind} output overlaps source/audit: {path}")
        if path.exists():
            raise ValueError(f"Run already exists; choose a new --run-name: {path}")
        paths.append(path)
    if len(paths) == 2 and (paths[0] == paths[1] or paths[0] in paths[1].parents or paths[1] in paths[0].parents):
        raise ValueError("Central checkpoint/log locations overlap; configure them through src.config.paths")
    return paths


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _versions():
    result = {"python": sys.version}
    for name in ("numpy", "pyarrow", "Pillow", "opencv-python", "opencv-python-headless", "torch", "torchvision"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _batch_summary(dataset, indices):
    import numpy as np
    samples = [dataset[i] for i in indices]
    arrays = {key: np.stack([s[key] for s in samples]) for key in ("qpos", "images", "actions", "is_pad")}
    for key in ("qpos", "images", "actions"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite preflight batch: {key}")
    if not ((arrays["images"] >= 0).all() and (arrays["images"] <= 1).all()):
        raise ValueError("Shared image processor returned pixels outside [0,1]")
    valid = ~arrays["is_pad"]
    raw_actions = arrays["actions"] * dataset.std["action"] + dataset.mean["action"]
    return {
        "shapes": {key: list(array.shape) for key, array in arrays.items()},
        "dtypes": {key: str(array.dtype) for key, array in arrays.items()},
        "saved_rows": [{key: s[key] for key in ("episode_index", "frame_index", "task_index", "segment_index")}
                       for s in samples],
        "valid_action_timesteps": int(valid.sum()), "padded_action_timesteps": int((~valid).sum()),
        "decoded_raw_grippers_valid_only": {
            str(index): {"min": float(raw_actions[:, :, index][valid].min()),
                         "max": float(raw_actions[:, :, index][valid].max())} for index in (6, 13)
        },
        "labels_clipped": False,
    }


def prepare(args, config, logs):
    from src.wp_ws.act_dataset import (
        ActDataset, EpisodeBatchSampler, fit_training_statistics, image_configuration, validation_batches,
    )
    from src.wp_ws.verified_data import load_verified_data
    print("[1/3] 核验 audit_v1（以配置为准）与当前源文件 SHA-256；不改源数据", flush=True)
    data = load_verified_data(config, progress=lambda text: print(text, flush=True))
    if args.command == "train" and data.warnings and not args.acknowledge_audit_warnings:
        codes = sorted({w["code"] for w in data.warnings})
        raise ValueError(f"Review audit warnings {codes}, then explicitly pass --acknowledge-audit-warnings")
    print("[2/3] 只拟合训练 episode 的 state/action 标准化；验证集不参与", flush=True)
    statistics = fit_training_statistics(data, progress=lambda text: print(text, flush=True))
    policy_config = policy_configuration(args)
    train = ActDataset(data, statistics, policy_config, partition="train")
    val = ActDataset(data, statistics, policy_config, partition="val")
    train_sampler = EpisodeBatchSampler(train, args.batch_size, shuffle=True, seed=args.seed)
    val_sampler, val_selection = validation_batches(val, args.batch_size, args.val_max_batches)
    print("[3/3] 解码训练/验证 CPU batch，调用共享 Standard ACT/Flow 图像处理", flush=True)
    batches = {"train": _batch_summary(train, next(iter(train_sampler))),
               "val": _batch_summary(val, next(iter(val_sampler)))}
    source_files = {
        "native_act": Path(__file__).resolve().parents[1] / "policy/act.py",
        "native_optimizer": Path(__file__).resolve().parents[1] / "policy/mixed_precision.py",
        "shared_images": Path(__file__).resolve().parents[1] / "contracts/standard_image_processing.py",
        "adapter": Path(__file__).with_name("act_dataset.py"),
        "verifier": Path(__file__).with_name("verified_data.py"), "trainer": Path(__file__),
    }
    manifest = {
        "schema_version": 1, "format": "wp_ws.act_baseline.v1",
        "model_variant": "repository_native_decoder_only_ACT_not_VAE_ACT",
        "data_config": asdict(config), "policy_config": policy_config,
        "image_preprocessing": image_configuration(policy_config, data),
        "provenance": data.provenance(), "validation_selection": val_selection,
        "normalization_file": "normalization_stats.json",
        "sampling": "episode-block shuffle, then row shuffle; every row once per epoch; no drop_last",
        "steps_requested": getattr(args, "steps", 0), "batch_size": args.batch_size, "seed": args.seed,
        "cli_arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gradient_clip_max_norm": 1.0, "compile": False, "channels_last": True,
        "environment": _versions(),
        "implementation_sha256": {key: sha256(path.read_bytes()).hexdigest() for key, path in source_files.items()},
        "live_deployment_ready": False,
        "deployment_note": "Offline training artifact only; do not pass directly to a robot runner. No deployment adapter is implemented.",
    }
    data.assert_unchanged()
    for name in ("split.json", "continuity.json", "fingerprints.json"):
        write_json(logs / name, data.documents[name])
    write_json(logs / "normalization_stats.json", statistics)
    write_json(logs / "run_config.json", manifest)
    preflight = {"status": "PREFLIGHT_PASSED", "source_content_hashes_verified": True,
                 "gpu_training_verified": False, "train_rows": len(train), "val_rows": len(val),
                 "train_episodes": len(train.episode_ids), "val_episodes": len(val.episode_ids),
                 "batches": batches, "warnings": data.warnings,
                 "validation_mode": val_selection["mode"], "source_metadata_changed": False}
    write_json(logs / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    # CPU preflight may have sampled several validation episodes. Do not retain that cache.
    train.clear_cache()
    val.clear_cache()
    return data, statistics, manifest, train, val, train_sampler, val_sampler


def _worker_init(worker_id):
    # Offline data-loading subprocesses only; nothing live is imported or started.
    import cv2
    import numpy as np
    import torch
    cv2.setNumThreads(1)
    np.random.seed(torch.initial_seed() % (2 ** 32))
    random.seed(torch.initial_seed())


def _cuda_memory(torch):
    gib = 1024 ** 3
    free, total = torch.cuda.mem_get_info()
    return {"device": torch.cuda.get_device_name(), "total_gib": total / gib, "free_gib_now": free / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib}


def _cpu_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch) for item in value)
    return value


def _save_checkpoint(path, policy, optimizer, statistics, manifest, step, validation, torch, *, include_optimizer):
    payload = {"format": manifest["format"], "model_state_dict": policy.serialize(),
               "policy_config": manifest["policy_config"], "run_config": manifest,
               "normalization_stats": statistics, "step": step, "validation": validation,
               "live_deployment_ready": False, "resume_entrypoint_implemented": False}
    if include_optimizer:
        payload["optimizer_state_dict"] = _cpu_tree(optimizer.state_dict(), torch)
        payload["rng_state"] = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                                "python": random.getstate()}
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(args, prepared, checkpoints, logs):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    data, statistics, manifest, train, val, train_sampler, val_sampler = prepared
    if not torch.cuda.is_available():
        raise ValueError("Native ACT training requires a CUDA GPU; preflight can run without one")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This repository's ACT requires bf16. This GPU/runtime does not support it; no silent fp16 fallback")
    torch.cuda.set_device(0)  # Select a physical GPU through CUDA_VISIBLE_DEVICES.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Fixed seeds/validation selection, not a claim of bitwise CUDA reproducibility.
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    source = str(Path(__file__).resolve().parents[1])
    if source not in sys.path:
        sys.path.insert(0, source)  # Native policy imports top-level contracts/policy packages.
    from policy.act import ACTPolicy
    print("[CUDA] 构建原生 ACT：ResNet18 × 3，decoder=7×512，bf16；不启用 compile/RTC", flush=True)
    policy = ACTPolicy(manifest["policy_config"])
    if args.backbone_init == "imagenet":
        # Pretrained weight storage also goes through central CKPT_DIR, never the source dataset.
        hub = CKPT_DIR / "wp_ws" / "_torchvision_cache"
        for protected in (data.root, data.audit):
            if protected == hub.resolve() or protected in hub.resolve().parents or hub.resolve() in protected.parents:
                raise ValueError("Pretrained cache must not overlap source/audit; configure CKPT_DIR")
        torch.hub.set_dir(str(hub))
        print(f"[预训练权重] IMAGENET1K_V1；缓存 {hub}；缺少缓存时 torchvision 将尝试下载", flush=True)
        policy.load_pretrained_backbone()
    else:
        print("[警告] 显式选择 scratch：随机初始化骨干，不能冒充 ImageNet-pretrained baseline", flush=True)
    policy.to("cuda").train()
    policy.model.use_channels_last()
    optimizer = policy.configure_optimizers()  # Native bf16 model + fp32 master weights/moments.
    generator = torch.Generator().manual_seed(args.seed)
    kwargs = {"num_workers": args.workers, "pin_memory": True, "generator": generator}
    if args.workers:
        kwargs.update(worker_init_fn=_worker_init, multiprocessing_context="spawn", prefetch_factor=1,
                      persistent_workers=False)
    train_loader = DataLoader(train, batch_sampler=train_sampler, **kwargs)
    # Validation remains in the main process, so it never doubles worker caches/prefetch.
    val_loader = DataLoader(val, batch_sampler=val_sampler, num_workers=0, pin_memory=True)
    epoch = 0
    iterator = iter(train_loader)
    log_path = logs / "metrics.jsonl"
    started = time.monotonic()
    best = math.inf
    last_validation = None

    def transfer(batch):
        return tuple(batch[key].to("cuda", non_blocking=True) for key in ("qpos", "images", "actions", "is_pad"))

    def validate(step):
        policy.eval()
        numerator = denominator = samples = 0
        try:
            with torch.inference_mode():
                for batch in val_loader:
                    qpos, images, actions, is_pad = transfer(batch)
                    loss = policy(qpos, images, actions, is_pad)["loss"]
                    if not torch.isfinite(loss).item():
                        raise ValueError("Validation loss is nonfinite")
                    valid_elements = int((~is_pad).sum().item()) * actions.shape[-1]
                    numerator += loss.item() * valid_elements
                    denominator += valid_elements
                    samples += len(qpos)
            if not denominator:
                raise ValueError("Empty validation target set")
            return {"step": step, "normalized_l1": numerator / denominator, "samples": samples,
                    "valid_action_elements": denominator, "mode": manifest["validation_selection"]["mode"]}
        finally:
            policy.train()
            val.clear_cache()

    with log_path.open("x", encoding="utf-8") as log:
        def emit(record):
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        emit({"event": "model_ready", "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
              "cuda_memory": _cuda_memory(torch), "optimizer": "native_MasterWeightAdamW",
              "validation_mode": manifest["validation_selection"]["mode"]})
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            qpos, images, actions, is_pad = transfer(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = policy(qpos, images, actions, is_pad)["loss"]
            if not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite training loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                torch.cuda.synchronize()
                emit({"event": "train", "step": step, "epoch_zero_based": epoch,
                      "normalized_l1": loss.item(), "gradient_norm_before_clip": float(grad_norm.item()),
                      "batch_size_actual": len(qpos), "elapsed_seconds": time.monotonic() - started,
                      "cuda_memory": _cuda_memory(torch)})
            if step % args.validate_every == 0 or step == args.steps:
                last_validation = validate(step)
                emit({"event": "validation", **last_validation, "cuda_memory": _cuda_memory(torch)})
                if last_validation["normalized_l1"] < best:
                    best = last_validation["normalized_l1"]
                    _save_checkpoint(checkpoints / "best.ckpt", policy, optimizer, statistics, manifest,
                                     step, last_validation, torch, include_optimizer=False)
            if step % args.save_every == 0 or step == args.steps:
                _save_checkpoint(checkpoints / "last.ckpt", policy, optimizer, statistics, manifest,
                                 step, last_validation, torch, include_optimizer=True)
        optimizer.zero_grad(set_to_none=True)
        data.assert_unchanged()
        torch.cuda.synchronize()
        result = {"status": "TRAINING_STEPS_COMPLETED", "steps": args.steps,
                  "last_validation": last_validation, "best_normalized_val_l1": best,
                  "cuda_memory": _cuda_memory(torch), "elapsed_seconds": time.monotonic() - started,
                  "task_success_rate_measured": False, "live_deployment_ready": False,
                  "logs": str(logs), "checkpoints": str(checkpoints)}
        emit({"event": "completed", **result})
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    logs = None
    try:
        config = load_config(args.config)
        if args.command == "train":
            for key in ("lr", "lr_backbone"):
                value = getattr(args, key)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be positive and finite")
        run_name = args.run_name or f"act_{args.command}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        locations = output_locations(config, run_name, training=args.command == "train")
        # Assign only after successful mkdir: an error must never write FAILED.json into an old run.
        locations[0].mkdir(parents=True, exist_ok=False)
        logs = locations[0]
        print(f"日志：{logs}\n只读离线流程；不连接机器人、不改写数据。", flush=True)
        prepared = prepare(args, config, logs)
        if args.command == "preflight":
            result = {"status": "PREFLIGHT_PASSED", "gpu_training_verified": False, "logs": str(logs)}
        else:
            checkpoints = locations[1]
            checkpoints.mkdir(parents=True, exist_ok=False)
            write_json(checkpoints / "run_config.json", prepared[2])
            write_json(checkpoints / "normalization_stats.json", prepared[1])
            result = run_training(args, prepared, checkpoints, logs)
        write_json(logs / "COMPLETED.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        error = {"status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                 "error_type": type(exc).__name__, "message": str(exc)}
        if logs is not None:
            write_json(logs / "FAILED.json", error)
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr, flush=True)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
