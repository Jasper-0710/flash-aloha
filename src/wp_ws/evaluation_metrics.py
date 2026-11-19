"""Masked offline ACT diagnostics; no model, dataset, or hardware modifications."""
import math

import numpy as np

from src.wp_ws.overfit_act import JOINTS, ErrorTotals
from src.wp_ws.verified_data import require


class DetailedErrors(ErrorTotals):
    """Reuse diagnostic L1 accumulation and add raw-unit horizon breakdowns."""
    def __init__(self, action_std, chunk_size):
        require(type(chunk_size) is int and chunk_size > 0, "Invalid chunk size")
        super().__init__(action_std, chunk_size)
        self.horizon_dim = np.zeros((chunk_size, 14), dtype=np.float64)

    def add(self, predictions, targets, is_pad, episode_ids):
        super().add(predictions, targets, is_pad, episode_ids)
        err = np.abs(np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64))
        self.horizon_dim += np.where(np.asarray(is_pad)[..., None], 0, err).sum(axis=0)

    def result(self):
        result = super().result()
        result['joint_mae_deg'] = math.degrees(result['joint_mae_rad'])
        for row in result['per_dimension']:
            if row['index'] in JOINTS:
                row['raw_mae_deg'] = math.degrees(row['raw_mae'])
        for row, errors in zip(result['per_horizon'], self.horizon_dim):
            count = row['valid_action_elements'] // 14
            row['valid_chunks'] = count
            raw = errors * self.std / count if count else None
            row['joint_mae_rad'] = float(raw[JOINTS].mean()) if count else None
            row['joint_mae_deg'] = math.degrees(row['joint_mae_rad']) if count else None
            row['gripper_mae_normalized_units'] = float(raw[[6, 13]].mean()) if count else None
        return result


def reproduction_check(model, reference, recorded):
    """Same fixed targets/counts; small cross-device numerical drift is a warning, not success."""
    for key in ('samples', 'valid_action_elements'):
        require(model[key] == reference[key] == recorded[key], f'Validation {key} differs from checkpoint')
    expected = float(recorded['normalized_l1'])
    require(math.isfinite(expected) and expected >= 0, 'Invalid recorded validation L1')
    actual = model['normalized_l1']
    # Explicit diagnostic tolerance, NOT a task-performance threshold.
    atol, rtol = 5e-4, 5e-3
    delta = actual - expected
    result = {'recorded_normalized_l1': expected, 'evaluated_normalized_l1': actual,
              'difference': delta, 'absolute_tolerance': atol, 'relative_tolerance': rtol,
              'within_tolerance': abs(delta) <= atol + rtol * abs(expected)}
    if 'mean_constant_normalized_l1' in recorded:
        constant = float(recorded['mean_constant_normalized_l1'])
        require(math.isfinite(constant) and math.isclose(reference['normalized_l1'], constant,
                                                       abs_tol=1e-7, rel_tol=1e-6),
                'Mean-action reference differs from checkpoint validation')
    return result


def example_rows(rows, requested):
    """Deterministic midpoint strata, independent of observed model errors."""
    require(type(rows) is int and rows > 0 and type(requested) is int and requested > 0,
            'Invalid example selection size')
    count = min(rows, requested)
    return [int(((2 * i + 1) * rows) // (2 * count)) for i in range(count)]


def write_report(directory, result, manifest, examples):
    model, baseline = result['model'], result['train_mean_constant']
    source = manifest['source']
    reference = baseline['normalized_l1']
    improvement = 100 * (1 - model['normalized_l1'] / reference) if reference else None
    check = result['reproduction']
    rows = ['# ACT 固定验证集离线评估', '',
            f"来源：`{source['run_name']}/{source['checkpoint_name']}`；step **{source['step']}**。",
            f"状态：**{result['status']}**；这不是任务成功率或部署许可。", '',
            '## 总体指标', '', '| 指标 | 数值 |', '|---|---:|',
            f"| 完整验证输入数 | {model['samples']} |",
            f"| 有效动作元素数 | {model['valid_action_elements']} |",
            f"| 模型 normalized L1 | {model['normalized_l1']:.6f} |",
            f"| 训练集均值常数 L1 | {reference:.6f} |",
            f"| 相比均值常数的误差降幅（不是成功率） | {f'{improvement:.2f}%' if improvement is not None else 'N/A'} |",
            f"| 12 个关节平均 MAE / rad | {model['joint_mae_rad']:.6f} |",
            f"| 12 个关节平均 MAE / degree | {model['joint_mae_deg']:.4f} |",
            f"| 两侧夹爪平均 MAE / 原归一化单位 | {model['gripper_mae_normalized_units']:.6f} |", '',
            '## 原验证 loss 复现', '',
            f"原记录 **{check['recorded_normalized_l1']:.6f}**；本次 **{check['evaluated_normalized_l1']:.6f}**；"
            f"差值 {check['difference']:+.8f}；容差内：**{check['within_tolerance']}**。",
            '容差为绝对 0.0005 + 相对 0.005 × 原 L1，仅用于数值复现诊断，不是部署门槛。',
            '超出容差时先核对运行环境/批大小；不要直接把差异解释成性能变化。', '',
            '## 逐验证 episode', '',
            '| episode | 输入数 | 模型 L1 | 均值常数 L1 | 关节 MAE/度 | 夹爪 MAE |',
            '|---:|---:|---:|---:|---:|---:|']
    for eid, pair in result['per_episode'].items():
        m, b = pair['model'], pair['train_mean_constant']
        rows.append(f"| {eid} | {m['samples']} | {m['normalized_l1']:.6f} | {b['normalized_l1']:.6f} | "
                    f"{m['joint_mae_deg']:.4f} | {m['gripper_mae_normalized_units']:.6f} |")
    rows += ['', '## 逐动作维度', '', '| 维度 | 模型 L1 | 均值常数 L1 | MAE / 原单位 | MAE / 度 |',
             '|---|---:|---:|---|---:|']
    for m, b in zip(model['per_dimension'], baseline['per_dimension']):
        degree = f"{m['raw_mae_deg']:.4f}" if 'raw_mae_deg' in m else '—'
        rows.append(f"| {m['name']} | {m['normalized_l1']:.6f} | {b['normalized_l1']:.6f} | "
                    f"{m['raw_mae']:.6f} {m['raw_unit']} | {degree} |")
    rows += ['', '## 预测距离与对比图', '',
             '- `metrics.json` 的 `per_horizon` 覆盖每一个预测 offset，并记录有效 chunk 数；padding 不参与指标。',
             '- `plots/horizon_errors.png`：每个预测距离上的模型/均值常数误差。',
             '- `plots/episode_errors.png`：每条验证 episode 的模型/均值常数误差。',
             '- `plots/t0_episode_*.png`：每个录制观测分别预测的第一个动作，**不是模型自回归或闭环 rollout**。',
             '- `plots/chunk_episode_*_frame_*.png`：固定起点的完整预测 chunk 与对应录制动作；只显示有效段。',
             f'- 共 {sum(map(len, examples.values()))} 个 chunk 示例，由预先声明的等分中点选定；起点见 `examples.json`，不按误差挑片段。',
             '- 横轴为保存的 frame index/预测 offset，不用 FPS 伪造时间戳；跨连续性边界不连线。',
             '- `predictions/episode_*.npz` 保存所有标准化预测、目标、padding、frame index 和反标准化参数；'
             '可用 `numpy.load(..., allow_pickle=False)` 读取。',
             '- 绘图使用 checkpoint 统计量反标准化；可能有浮点舍入误差；夹爪不裁剪，也不转换为电机角度。', '',
             '## 使用边界', '',
             '- 这是模型选择时用过的固定验证集，不是全新的独立测试集；均值常数只是简单参照。',
             '- 所有预测均以录制的图像和状态为输入；未验证视觉依赖、闭环稳定性、时延或真机任务成功率。',
             '- 同一个真实动作可能出现在多个重叠 chunk；汇总按有效动作元素加权，不是去重后的单帧误差。',
             '- 每个 horizon 的有效样本数可能不同，不能只凭远端误差变化就判定因果。',
             '- 原有采集 provenance、夹爪越界等警告仍然保留；没有重新配对、裁剪或修复数据。',
             '- 不写源数据、audit、原训练日志或 checkpoint；不训练、不连接机器人。',
             '- 某个关节/夹爪的平均误差小，不等于关键接触或开合时刻正确；请结合完整动作对比图审阅。', '']
    with (directory / 'report.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(rows))
