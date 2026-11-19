"""Offline tiny-training-subset ACT diagnostic; NOT held-out validation or deployment.

Reuses the audited row/chunk adapter, train-only statistics, shared image processor,
and native ACT/optimizer. Never opens robot/camera devices or edits source data.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from src.config.paths import CKPT_DIR
from src.wp_ws import train_act
from src.wp_ws.baseline_data_config import DEFAULT_CONFIG, load_config
from src.wp_ws.verified_data import require

ARRAY_KEYS = ("qpos", "images", "actions", "is_pad")
DIMENSIONS = ([f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"] +
              [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"])
JOINTS = [i for i in range(14) if i not in (6, 13)]
EVAL_SCOPE = "same_fixed_training_samples_NOT_held_out_validation"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "train"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--run-name", type=train_act._name)
        sub.add_argument("--episodes", type=train_act._nonnegative, nargs="+",
                         help="Training episode IDs only; default: first two sorted train IDs")
        sub.add_argument("--samples-per-episode", type=train_act._positive, default=16)
        sub.add_argument("--batch-size", type=train_act._positive, default=16)
        sub.add_argument("--chunk-size", type=train_act._positive, default=50)
        sub.add_argument("--height", type=train_act._positive, default=240)
        sub.add_argument("--width", type=train_act._positive, default=320)
        sub.add_argument("--seed", type=train_act._nonnegative, default=42)
        sub.add_argument("--cache-limit-mib", type=train_act._positive, default=512)
        if command == "train":
            sub.add_argument("--steps", type=train_act._positive, default=1500)
            sub.add_argument("--eval-every", type=train_act._positive, default=100)
            sub.add_argument("--log-every", type=train_act._positive, default=50)
            sub.add_argument("--lr", type=float, default=0.00012)
            sub.add_argument("--lr-backbone", type=float, default=0.00001)
            sub.add_argument("--backbone-init", choices=("imagenet", "scratch"), default="imagenet")
            sub.add_argument("--acknowledge-audit-warnings", action="store_true")
    return parser


def select_samples(dataset, episodes=None, samples_per_episode=16):
    """Fixed endpoints-inclusive evenly spaced starts, not consecutive/duplicated rows."""
    require(tuple(dataset.episode_ids) == tuple(dataset.data.split["train_episode_indices"]),
            "Diagnostic fitting requires the training partition")
    require(type(samples_per_episode) is int and samples_per_episode > 0, "Invalid sample count")
    ids = sorted(dataset.episode_ids)[:2] if episodes is None else sorted(episodes)
    require(bool(ids) and len(ids) == len(set(ids)), "Episode selection is empty or duplicated")
    require(all(type(eid) is int and eid in dataset.episode_ids for eid in ids),
            "Only training episode IDs may be fitted; validation/unknown IDs are rejected")
    indices = []
    for eid in ids:
        rows = dataset.data.episodes[eid].rows
        require(samples_per_episode <= rows, f"Episode {eid} has only {rows} rows; reduce samples-per-episode")
        starts = np.linspace(0, rows - 1, num=samples_per_episode, dtype=np.int64)
        indices.extend(dataset.sample_index(eid, int(row)) for row in starts)
    require(len(indices) == len(set(indices)), "Duplicate sample indices")
    return indices


class FixedSamples:
    """Decode once through ActDataset; no new image transform or target construction."""
    def __init__(self, dataset, indices, *, limit_mib=512):
        require(bool(indices), "Empty fixed sample selection")
        # Include all large float arrays; the estimate excludes the temporary compressed Arrow episode.
        image_values = 3 * 3 * dataset.spec.height * dataset.spec.width
        estimated = len(indices) * (4 * (14 + image_values + dataset.chunk_size * 14) + dataset.chunk_size)
        require(estimated <= limit_mib * 1024 ** 2,
                f"Fixed decoded cache estimate {estimated / 1024 ** 2:.1f} MiB exceeds cache-limit-mib")
        self.metadata = []
        self.arrays = None
        try:
            for position, index in enumerate(indices):
                sample = dataset[index]
                if self.arrays is None:
                    self.arrays = {key: np.empty((len(indices), *sample[key].shape), dtype=sample[key].dtype)
                                   for key in ARRAY_KEYS}
                for key in ARRAY_KEYS:
                    self.arrays[key][position] = sample[key]
                self.metadata.append({"flat_index": int(index),
                                      **{key: int(sample[key]) for key in
                                         ("episode_index", "frame_index", "task_index", "segment_index")},
                                      "valid_action_timesteps": int((~sample["is_pad"]).sum())})
        finally:
            dataset.clear_cache()
        for key in ("qpos", "images", "actions"):
            require(np.isfinite(self.arrays[key]).all(), f"Nonfinite cached {key}")
        require(((self.arrays["images"] >= 0) & (self.arrays["images"] <= 1)).all(), "Invalid image scaling")
        require((~self.arrays["is_pad"]).any(axis=1).all(), "Sample without valid actions")
        # Advanced indexing in batch() returns independent writable copies for torch, never this cache.
        for array in self.arrays.values():
            array.setflags(write=False)

    def __len__(self):
        return len(self.metadata)

    def batch(self, indices):
        return {key: array[indices] for key, array in self.arrays.items()}

    def manifest(self):
        return {"algorithm": "sorted-train-episodes-evenly-spaced-starts-v1", "partition": "train",
                "evaluation_scope": EVAL_SCOPE, "samples": self.metadata,
                "sample_count": len(self), "episode_indices": sorted({s["episode_index"] for s in self.metadata}),
                "decoded_cache_mib": sum(a.nbytes for a in self.arrays.values()) / 1024 ** 2,
                "array_sha256": {key: sha256(array.tobytes()).hexdigest() for key, array in self.arrays.items()},
                "array_shapes": {key: list(array.shape) for key, array in self.arrays.items()},
                "array_dtypes": {key: str(array.dtype) for key, array in self.arrays.items()}}


def fixed_batches(count, batch_size, *, seed=42, epoch=0, shuffle=False):
    require(count > 0 and batch_size > 0 and epoch >= 0 and seed >= 0, "Invalid batch parameters")
    order = np.arange(count)
    if shuffle:
        np.random.default_rng(np.random.SeedSequence([seed, epoch])).shuffle(order)
    return [order[start:start + batch_size].tolist() for start in range(0, count, batch_size)]


class ErrorTotals:
    """Element-weighted masked L1, shared by model and constants; no mean-of-batch-means."""
    def __init__(self, action_std, chunk_size):
        self.std = np.asarray(action_std, dtype=np.float64)
        require(self.std.shape == (14,) and np.isfinite(self.std).all() and (self.std > 0).all(),
                "Invalid action standard deviations")
        self.dim = np.zeros(14, dtype=np.float64)
        self.horizon = np.zeros(chunk_size, dtype=np.float64)
        self.counts = np.zeros(chunk_size, dtype=np.int64)
        self.episodes = {}
        self.samples = 0

    def add(self, predictions, targets, is_pad, episode_ids):
        pred, target = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        mask = np.asarray(is_pad)
        require(pred.shape == target.shape == (len(episode_ids), len(self.horizon), 14), "Action shape mismatch")
        require(mask.shape == pred.shape[:2] and mask.dtype == np.bool_, "Invalid padding mask")
        require(np.isfinite(pred).all() and np.isfinite(target).all(), "Nonfinite actions/predictions")
        valid = ~mask
        err = np.where(valid[:, :, None], np.abs(pred - target), 0.0)
        self.dim += err.sum(axis=(0, 1))
        self.horizon += err.sum(axis=(0, 2))
        self.counts += valid.sum(axis=0) * 14
        self.samples += len(pred)
        for i, eid in enumerate(episode_ids):
            entry = self.episodes.setdefault(int(eid), [0.0, 0, 0])
            entry[0] += float(err[i].sum())
            entry[1] += int(valid[i].sum()) * 14
            entry[2] += 1

    def result(self):
        count = int(self.counts.sum())
        require(count > 0, "No valid target elements")
        per_dim = self.dim / (count // 14)
        raw_dim = per_dim * self.std
        return {"normalized_l1": float(self.dim.sum() / count), "samples": self.samples,
                "valid_action_elements": count,
                "joint_mae_rad": float(raw_dim[JOINTS].mean()),
                "gripper_mae_normalized_units": float(raw_dim[[6, 13]].mean()),
                "per_dimension": [{"index": i, "name": name, "normalized_l1": float(per_dim[i]),
                                   "raw_mae": float(raw_dim[i]), "raw_unit": "normalized_gripper" if i in (6, 13) else "rad"}
                                  for i, name in enumerate(DIMENSIONS)],
                "per_horizon": [{"offset": i, "valid_action_elements": int(n),
                                 "normalized_l1": float(self.horizon[i] / n) if n else None}
                                for i, n in enumerate(self.counts)],
                "per_episode": {str(eid): {"normalized_l1": total / n, "valid_action_elements": n, "samples": samples}
                                for eid, (total, n, samples) in sorted(self.episodes.items())}}


def constant_baselines(cache, action_std):
    actions, mask = cache.arrays["actions"], cache.arrays["is_pad"]
    constants = {"train_mean_constant": np.zeros(14, dtype=np.float64),
                 "subset_median_constant": np.median(actions[~mask], axis=0)}
    result = {}
    for name, vector in constants.items():
        totals = ErrorTotals(action_std, actions.shape[1])
        totals.add(np.broadcast_to(vector, actions.shape), actions, mask,
                   [s["episode_index"] for s in cache.metadata])
        result[name] = {**totals.result(), "normalized_constant_vector": vector.tolist()}
    return result


def evaluate_policy(policy, cache, action_std, batch_size, device):
    import torch
    totals = ErrorTotals(action_std, cache.arrays["actions"].shape[1])
    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            for indices in fixed_batches(len(cache), batch_size):
                batch = cache.batch(indices)
                qpos, images = (torch.from_numpy(batch[key]).to(device) for key in ("qpos", "images"))
                predictions = policy(qpos, images).float().cpu().numpy()
                totals.add(predictions, batch["actions"], batch["is_pad"],
                           [cache.metadata[i]["episode_index"] for i in indices])
        return {"evaluation_scope": EVAL_SCOPE, **totals.result()}
    finally:
        policy.train(was_training)


def prepare(args, config, logs):
    from src.wp_ws.act_dataset import ActDataset, fit_training_statistics, image_configuration
    from src.wp_ws.verified_data import load_verified_data
    print("[1/3] 核验源数据与 audit 指纹；只读，不重新采集/配对/裁剪。", flush=True)
    data = load_verified_data(config, progress=lambda text: print(text, flush=True))
    if args.command == "train" and data.warnings and not args.acknowledge_audit_warnings:
        raise ValueError("Review audit warnings, then pass --acknowledge-audit-warnings")
    print("[2/3] 沿用全训练集统计口径（验证集不参与）；选取固定小样本。", flush=True)
    statistics = fit_training_statistics(data, progress=lambda text: print(text, flush=True))
    policy_config = train_act.policy_configuration(args)
    dataset = ActDataset(data, statistics, policy_config, partition="train")
    indices = select_samples(dataset, args.episodes, args.samples_per_episode)
    print("[3/3] 经原 ActDataset 解码并缓存固定样本；计算同样本常数基线。", flush=True)
    cache = FixedSamples(dataset, indices, limit_mib=args.cache_limit_mib)
    selection = cache.manifest()
    baselines = constant_baselines(cache, statistics["action"]["std"])
    source = Path(__file__).resolve().parents[1]
    files = [Path(__file__), Path(train_act.__file__), source / "wp_ws/act_dataset.py",
             source / "wp_ws/verified_data.py", source / "policy/act.py", source / "policy/mixed_precision.py",
             source / "contracts/standard_image_processing.py"]
    manifest = {"schema_version": 1, "format": "wp_ws.act_overfit.v1",
                "model_variant": "repository_native_decoder_only_ACT_not_VAE_ACT",
                "data_config": asdict(config), "policy_config": policy_config,
                "image_preprocessing": image_configuration(policy_config, data), "provenance": data.provenance(),
                "selection": selection, "evaluation_scope": EVAL_SCOPE,
                "sampling": "shuffle all fixed starts across selected training episodes; no drop_last",
                "normalization_file": "normalization_stats.json", "gradient_clip_max_norm": 1.0,
                "batch_size": args.batch_size, "seed": args.seed, "steps_requested": getattr(args, "steps", 0),
                "cli_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "initialization": "fresh native model, NOT resumed from the stopped run",
                "environment": train_act._versions(),
                "implementation_sha256": {str(p.relative_to(source)): sha256(p.read_bytes()).hexdigest() for p in files},
                "live_deployment_ready": False, "held_out_validation_performed": False,
                "resume_entrypoint_implemented": False, "source_metadata_changed": False}
    data.assert_unchanged()
    for name, doc in [("run_config.json", manifest), ("selection.json", selection),
                      ("normalization_stats.json", statistics), ("baselines.json", baselines)]:
        train_act.write_json(logs / name, doc)
    for name in ("split.json", "continuity.json", "fingerprints.json"):
        train_act.write_json(logs / name, data.documents[name])
    preflight = {"status": "OVERFIT_PREFLIGHT_PASSED", "gpu_training_verified": False,
                 "evaluation_scope": EVAL_SCOPE, "samples": len(cache), "episodes": selection["episode_indices"],
                 "decoded_cache_mib": selection["decoded_cache_mib"], "warnings": data.warnings,
                 "baseline_l1": {k: v["normalized_l1"] for k, v in baselines.items()}}
    train_act.write_json(logs / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False), flush=True)
    return data, statistics, manifest, cache, baselines


def write_report(logs, evaluations, baselines):
    lines = ["# ACT 固定训练小样本拟合诊断", "",
             "**只评估参与优化的相同样本，不是验证集，不代表任务成功率或部署安全。**", "",
             "从新模型初始化；标准化仍只来自原固定训练集。原始标签/图像/划分未修改。", "",
             "| 常数预测器（相同样本、相同 padding 权重） | normalized L1 |", "|---|---:|"]
    lines += [f"| {key} | {value['normalized_l1']:.6f} |" for key, value in baselines.items()]
    lines += ["", "subset_median_constant 只用所选训练标签拟合每维单一常数，是该样本集的最优单常数向量 L1；",
              "不是独立验证指标，也不是按 horizon 单独拟合的常数。", "",
              "| step | 同样本 eval L1 | 12 关节 MAE (rad) | 2 夹爪 MAE (原归一化单位) |",
              "|---:|---:|---:|---:|"]
    lines += [f"| {row['step']} | {row['normalized_l1']:.6f} | {row['joint_mae_rad']:.6f} | "
              f"{row['gripper_mae_normalized_units']:.6f} |" for row in evaluations]
    lines += ["", "判断：检查同样本 eval 相比 step 0 是否持续下降、是否明显优于两个常数基线；",
              "再看逐 episode/维度/horizon 指标（metrics.jsonl）。不设置任务成功的虚假绝对阈值。",
              "拟合成功：下一轮用原固定验证 episode 做独立评估并受控比较全数据采样策略。",
              "拟合失败：先进一步缩小到 1 个样本，排查优化/模型/输入链路，不盲目继续 100K 或增大 batch。",
              "", "W&B 现有 wandb_metrics 上传器不解析 overfit_* 事件；不能将这些指标冒充 val/loss。", ""]
    temporary = logs / "report.md.partial"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(logs / "report.md")


def save_checkpoint(path, policy, statistics, manifest, step, evaluation, torch):
    # Model only: bounded diagnostic artifacts, no claim of training-state resume support.
    payload = {"format": manifest["format"], "model_state_dict": policy.serialize(),
               "policy_config": manifest["policy_config"], "run_config": manifest,
               "normalization_stats": statistics, "step": step, "overfit_evaluation": evaluation,
               "live_deployment_ready": False, "resume_entrypoint_implemented": False}
    temporary = path.with_suffix(".ckpt.partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(args, prepared, checkpoints, logs):
    import torch
    data, statistics, manifest, cache, baselines = prepared
    require(torch.cuda.is_available(), "Native ACT requires CUDA; preflight is CPU-only")
    torch.cuda.set_device(0)  # Physical GPU selected by CUDA_VISIBLE_DEVICES, not DDP.
    require(torch.cuda.is_bf16_supported(), "Native ACT requires bf16; no precision fallback")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    source = str(Path(__file__).resolve().parents[1])
    if source not in sys.path:
        sys.path.insert(0, source)
    from policy.act import ACTPolicy
    policy = ACTPolicy(manifest["policy_config"])
    if args.backbone_init == "imagenet":
        hub = (CKPT_DIR / "wp_ws" / "_torchvision_cache").resolve()
        for protected in (data.root, data.audit):
            require(hub != protected and protected not in hub.parents and hub not in protected.parents,
                    "Pretrained cache overlaps source/audit; configure CKPT_DIR")
        torch.hub.set_dir(str(hub))
        print(f"[CUDA] 全新原生 ACT + ImageNet 骨干；权重缓存 {hub}（缺失时下载）。", flush=True)
        policy.load_pretrained_backbone()
    else:
        print("[警告] 显式选择 scratch；不是 ImageNet 对照实验。", flush=True)
    policy.to("cuda").train()
    policy.model.use_channels_last()
    optimizer = policy.configure_optimizers()
    epoch, best = 0, math.inf
    iterator = iter(fixed_batches(len(cache), args.batch_size, seed=args.seed, shuffle=True))
    started, evaluations = time.monotonic(), []
    with (logs / "metrics.jsonl").open("x", encoding="utf-8") as log:
        def emit(record):
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()

        def evaluate(step):
            nonlocal best
            data.assert_unchanged()
            result = {"event": "overfit_eval", "step": step,
                      **evaluate_policy(policy, cache, statistics["action"]["std"], args.batch_size, "cuda")}
            evaluations.append(result)
            emit(result)
            print(f"[同训练样本 eval] step={step} L1={result['normalized_l1']:.6f} "
                  f"mean基线={baselines['train_mean_constant']['normalized_l1']:.6f} "
                  f"median基线={baselines['subset_median_constant']['normalized_l1']:.6f}", flush=True)
            if result["normalized_l1"] < best:
                best = result["normalized_l1"]
                save_checkpoint(checkpoints / "best.ckpt", policy, statistics, manifest, step, result, torch)
            save_checkpoint(checkpoints / "last.ckpt", policy, statistics, manifest, step, result, torch)
            write_report(logs, evaluations, baselines)
            return result

        emit({"event": "constant_baselines", "evaluation_scope": EVAL_SCOPE, "baselines": baselines})
        initial = evaluate(0)
        for step in range(1, args.steps + 1):
            try:
                indices = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(fixed_batches(len(cache), args.batch_size, seed=args.seed, epoch=epoch, shuffle=True))
                indices = next(iterator)
            batch = cache.batch(indices)
            qpos, images, actions, is_pad = (torch.from_numpy(batch[k]).to("cuda") for k in ARRAY_KEYS)
            optimizer.zero_grad(set_to_none=True)
            loss = policy(qpos, images, actions, is_pad)["loss"]
            require(torch.isfinite(loss).item(), f"Nonfinite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = {"event": "overfit_train", "step": step, "epoch_zero_based": epoch,
                          "normalized_l1": float(loss.item()), "gradient_norm_before_clip": float(norm.item()),
                          "batch_size_actual": len(indices), "elapsed_seconds": time.monotonic() - started,
                          "cuda_memory": train_act._cuda_memory(torch)}
                emit(record)
                print(f"[train] step={step}/{args.steps} batch_L1={record['normalized_l1']:.6f}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                optimizer.zero_grad(set_to_none=True)
                evaluate(step)
        data.assert_unchanged()
        final = evaluations[-1]
        result = {"status": "OVERFIT_STEPS_COMPLETED", "steps": args.steps, "evaluation_scope": EVAL_SCOPE,
                  "initial_eval_l1": initial["normalized_l1"], "final_eval_l1": final["normalized_l1"],
                  "best_eval_l1": best, "best_step": min(evaluations, key=lambda r: r["normalized_l1"])["step"],
                  "relative_l1_reduction_from_initial": (1 - final["normalized_l1"] / initial["normalized_l1"])
                  if initial["normalized_l1"] > 0 else None,
                  "baseline_l1": {k: v["normalized_l1"] for k, v in baselines.items()},
                  "task_success_rate_measured": False, "held_out_validation_performed": False,
                  "live_deployment_ready": False, "elapsed_seconds": time.monotonic() - started,
                  "cuda_memory": train_act._cuda_memory(torch), "logs": str(logs), "checkpoints": str(checkpoints)}
        emit({"event": "overfit_completed", **result})
        return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    logs = None
    try:
        for key in ("lr", "lr_backbone"):
            value = getattr(args, key, 1.0)
            require(math.isfinite(value) and value > 0, f"{key} must be positive and finite")
        config = load_config(args.config)
        name = args.run_name or f"act_overfit_{args.command}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        locations = train_act.output_locations(config, name, training=args.command == "train")
        locations[0].mkdir(parents=True, exist_ok=False)
        logs = locations[0]
        print(f"日志：{logs}\n固定训练小样本诊断；不连接机器人，不改写旧训练/数据。", flush=True)
        prepared = prepare(args, config, logs)
        if args.command == "preflight":
            result = {"status": "OVERFIT_PREFLIGHT_PASSED", "gpu_training_verified": False, "logs": str(logs)}
        else:
            checkpoints = locations[1]
            checkpoints.mkdir(parents=True, exist_ok=False)
            train_act.write_json(checkpoints / "run_config.json", prepared[2])
            train_act.write_json(checkpoints / "normalization_stats.json", prepared[1])
            result = run_training(args, prepared, checkpoints, logs)
        train_act.write_json(logs / "COMPLETED.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        error = {"status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                 "error_type": type(exc).__name__, "message": str(exc)}
        if logs is not None:
            train_act.write_json(logs / "FAILED.json", error)
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr, flush=True)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
