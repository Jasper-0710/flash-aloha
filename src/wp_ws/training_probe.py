"""Read-only fixed training probes and validation diagnostics for sampling trials.

The probe is NOT held out. Neither it nor the mean-action reference selects the
best checkpoint: only the original validation split does. No task success claim.
"""
import numpy as np

from src.wp_ws.overfit_act import FixedSamples, constant_baselines, evaluate_policy
from src.wp_ws.verified_data import require

SCOPE = "fixed_training_probe_NOT_held_out_validation"


def probe_indices(dataset, per_episode):
    require(tuple(dataset.episode_ids) == tuple(dataset.data.split["train_episode_indices"]),
            "Training probe requires the training partition")
    require(type(per_episode) is int and per_episode > 0, "Invalid probe size")
    indices = []
    for eid in sorted(dataset.episode_ids):
        rows = dataset.data.episodes[eid].rows
        require(per_episode <= rows, f"Episode {eid} too short for requested probe size")
        # Midpoints of equal time-index strata; with 2 samples: 25% and 75%, not padded endpoints.
        starts = ((2 * np.arange(per_episode, dtype=np.int64) + 1) * rows) // (2 * per_episode)
        indices.extend(dataset.sample_index(eid, int(row)) for row in starts)
    require(len(indices) == len(set(indices)), "Duplicate training probe indices")
    return indices


def prepare_probe(dataset, per_episode, cache_mib):
    cache = FixedSamples(dataset, probe_indices(dataset, per_episode), limit_mib=cache_mib)
    selection = {**cache.manifest(), "algorithm": "sorted-train-episodes-midpoint-strata-v1",
                 "evaluation_scope": SCOPE, "samples_per_episode": per_episode,
                 "weighting": "valid action elements across selected chunks; not full training population"}
    baselines = constant_baselines(cache, dataset.std["action"])
    return cache, selection, baselines


def evaluate_probe(policy, cache, action_std, batch_size, device, baseline):
    return {**evaluate_policy(policy, cache, action_std, batch_size, device),
            "evaluation_scope": SCOPE, "mode": "fixed_training_probe",
            "mean_constant_normalized_l1": baseline}


def evaluate_validation(policy, loader, *, device, step, mode):
    """Native masked loss, weighted by valid elements (same as the old trainer).

    Compute a zero-normalized-action reference on those exact labels/masks, not
    on a different set of rows; zero is the train-only mean, never a val fit.
    """
    import torch
    was_training = policy.training
    policy.eval()
    numerator = constant_numerator = denominator = samples = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                qpos, images, actions, is_pad = (batch[key].to(device, non_blocking=True)
                                                for key in ("qpos", "images", "actions", "is_pad"))
                loss = policy(qpos, images, actions, is_pad)["loss"]
                require(torch.isfinite(loss).item(), "Validation loss is nonfinite")
                valid = ~is_pad
                count = int(valid.sum().item()) * actions.shape[-1]
                numerator += loss.item() * count
                # float64 accumulation for the reference, using the actual float32 normalized labels.
                constant_numerator += actions[valid].double().abs().sum().item()
                denominator += count
                samples += len(qpos)
        require(denominator > 0, "Empty validation target set")
        return {"step": step, "normalized_l1": numerator / denominator, "samples": samples,
                "valid_action_elements": denominator, "mode": mode,
                "mean_constant_normalized_l1": constant_numerator / denominator}
    finally:
        policy.train(was_training)


def write_report(path, history, manifest):
    """Only a new training run's own derived report, updated after each evaluation."""
    rows = ["# ACT 全数据采样对照", "", f"采样：`{manifest['sampler_configuration']['mode']}`；"
            f"验证：`{manifest['validation_selection']['mode']}`。", "",
            "固定训练探针来自训练集，不是验证集；原始标签/图像/划分未修改。",
            "均值常数预测器使用原训练统计；各列基线在各自样本、相同 padding 权重上计算。", "",
            "| step | 固定 train probe L1 | probe 均值基线 | 验证 L1 | 验证均值基线 |",
            "|---:|---:|---:|---:|---:|"]
    for entry in history:
        probe, val = entry.get("probe"), entry["validation"]
        p = f"{probe['normalized_l1']:.6f} | {probe['mean_constant_normalized_l1']:.6f}" if probe else "— | —"
        rows.append(f"| {val['step']} | {p} | {val['normalized_l1']:.6f} | {val['mean_constant_normalized_l1']:.6f} |")
    if history:
        best = min((e["validation"] for e in history), key=lambda v: v["normalized_l1"])
        rows += ["", f"当前最佳验证：step {best['step']}，L1 **{best['normalized_l1']:.6f}**。"]
    rows += ["", "判断：probe 与验证均下降且验证持续优于均值常数，是继续训练的依据；",
             "probe 下降、验证停滞或恶化提示泛化差距；两者不降则继续排查优化/输入，不直接拉长至 100K。",
             "这不是硬阈值或任务成功证明。窗口混采不等于全局随机；对照需用相同配置/步数/seed，",
             "仍有每窗尾 batch、累计样本数和 CUDA 非逐位确定性差异，应查看日志，不只比较单个点。",
             "若中断，本表只覆盖最后完成的评估点。best.ckpt 只按验证选择，可能是 step 0。",
             "部署/视觉依赖/真机成功率尚未验证；不要直接上线机器人。", ""]
    temporary = path.with_suffix(".md.partial")
    temporary.write_text("\n".join(rows), encoding="utf-8")
    temporary.replace(path)
