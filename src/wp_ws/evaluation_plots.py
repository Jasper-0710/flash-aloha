"""Static offline prediction plots; no GUI, image resizing, or source writes."""
import numpy as np

from src.wp_ws.overfit_act import DIMENSIONS
from src.wp_ws.verified_data import require


def _figure(*, size):
    # Import only inside the caller's output-local MPLCONFIGDIR context.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    return fig


def _save(fig, path):
    try:
        # Never replace an existing plot (including a symlink).
        with path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=120)
    finally:
        fig.clear()


def _action_figure(title):
    from matplotlib.ticker import MaxNLocator
    fig = _figure(size=(15, 15))
    axes = fig.subplots(7, 2, sharex=True)
    for side in range(2):
        for joint in range(7):
            d = side * 7 + joint
            ax = axes[joint, side]
            ax.set_title(DIMENSIONS[d], fontsize=10, loc="left")
            ax.set_ylabel("gripper units (unclipped)" if joint == 6 else "rad", fontsize=8)
            ax.grid(alpha=.25)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=.94, bottom=.06, hspace=.55, wspace=.23)
    return fig, axes


def _draw(ax, x, target, pred, mean, *, legend):
    ax.plot(x, target, color="#111827", lw=1.2, label="Recorded action", marker="." if len(x) == 1 else None)
    ax.plot(x, pred, color="#2563eb", lw=1, alpha=.9, label="Model prediction", marker="." if len(x) == 1 else None)
    ax.plot(x, np.full(len(x), mean), color="#d97706", ls="--", lw=.8,
            label="Train-mean constant", marker="." if len(x) == 1 else None)
    if legend:
        ax.legend(fontsize=7, loc="best")


def plot_episode(directory, eid, arrays, examples):
    mean, std = arrays["action_mean"], arrays["action_std"]
    frames, mask = arrays["frame_index"], arrays["is_pad"]
    # These inversions are diagnostic only: targets were normalized in float32.
    pred = arrays["prediction_normalized"] * std + mean
    target = arrays["target_normalized"] * std + mean
    fig, axes = _action_figure(f"Episode {eid:06d} | chunk offset 0 at EACH recorded observation\n"
                               "Offline predictions, NOT a closed-loop rollout | labels and grippers unchanged")
    for i, (start, stop) in enumerate(arrays["contiguous_ranges"]):
        for d in range(14):
            _draw(axes[d % 7, d // 7], frames[start:stop], target[start:stop, 0, d],
                  pred[start:stop, 0, d], mean[d], legend=(i == 0 and d == 0))
    for ax in axes[-1]:
        ax.set_xlabel("Saved frame index (lines break at continuity boundaries)")
    _save(fig, directory / f"t0_episode_{eid:06d}.png")
    for start in examples:
        require(0 <= start < len(frames), "Plot example is outside episode")
        offsets = np.flatnonzero(~mask[start])
        require(len(offsets) > 0 and np.array_equal(offsets, np.arange(len(offsets))), "Invalid chunk plot mask")
        fig, axes = _action_figure(f"Episode {eid:06d} | ONE observation at saved frame {int(frames[start]):06d}\n"
                                   "Predicted action chunk vs recorded targets | padded actions omitted; not a rollout")
        for d in range(14):
            _draw(axes[d % 7, d // 7], offsets, target[start, offsets, d], pred[start, offsets, d],
                  mean[d], legend=(d == 0))
        for ax in axes[-1]:
            ax.set_xlabel("Prediction offset in saved rows (0 = same recorded row)")
        _save(fig, directory / f"chunk_episode_{eid:06d}_frame_{int(frames[start]):06d}.png")


def plot_summary(directory, result):
    from matplotlib.ticker import MaxNLocator
    fig = _figure(size=(12, 8))
    axes = fig.subplots(2, 2)
    for ax, key, title in zip(axes.flat, ("normalized_l1", "joint_mae_deg", "gripper_mae_normalized_units", "valid_chunks"),
                              ("Normalized L1", "Joint MAE (degrees; 12 joints)",
                               "Gripper MAE (original normalized units)", "Valid chunks at each offset")):
        for name, label, color in (("model", "Model", "#2563eb"), ("train_mean_constant", "Train-mean constant", "#d97706")):
            rows = result[name]["per_horizon"]
            if key == "valid_chunks":
                if name != "model":
                    continue
                label = "Valid chunks (model and constant)"
            ax.plot([r["offset"] for r in rows], [np.nan if r[key] is None else r[key] for r in rows],
                    label=label, color=color, ls="--" if name != "model" else "-",
                    marker="." if len(rows) == 1 else None)
        ax.set_xlim((0, len(rows) - 1) if len(rows) > 1 else (-.5, .5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Prediction offset (saved rows)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed validation | all valid chunk elements; padding excluded", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    _save(fig, directory / "horizon_errors.png")
    fig = _figure(size=(12, 5))
    ax = fig.subplots()
    episodes = list(result["per_episode"])
    x = np.arange(len(episodes))
    for offset, key, label, color in ((-.2, "model", "Model", "#2563eb"),
                                      (.2, "train_mean_constant", "Train-mean constant", "#d97706")):
        ax.bar(x + offset, [result["per_episode"][e][key]["normalized_l1"] for e in episodes],
               width=.4, label=label, color=color)
    ax.set_xticks(x, labels=episodes)
    ax.set_xlabel("Fixed validation episode index")
    ax.set_ylabel("Normalized L1 (valid-element weighted within episode)")
    ax.set_title("Per-episode offline errors | NOT task success rate")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    _save(fig, directory / "episode_errors.png")
