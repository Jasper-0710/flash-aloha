"""Read-only local ACT loss viewer; no W&B, torch, GPU or robot access.

Only an explicit --plot writes anything, under central LOG_DIR/wp_ws/_loss_views.
A snapshot ignores an explicitly reported, still-incomplete final JSONL record;
following reuses the W&B sidecar's append-only reader to wait for that record.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile

from src.config.paths import CKPT_DIR, DATA_DIR, LOG_DIR
from src.wp_ws import wandb_metrics as metrics


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=metrics._component, default="flip_the_pillow")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run-name", type=metrics._component)
    choice.add_argument("--list", action="store_true", help="List existing ACT training runs")
    parser.add_argument("--last", type=_positive, default=20, help="Recent train/val records to show initially")
    parser.add_argument("--follow", action="store_true", help="Print new losses until completion/failure or Ctrl+C")
    parser.add_argument("--poll-seconds", type=metrics._interval, default=1.0)
    parser.add_argument("--plot", action="store_true", help="Save PNG snapshot; with --follow, update again on exit")
    return parser


def snapshot(source):
    """Read only bytes present at open time, never consume a partial trailing record."""
    path = metrics._source_file(source, "metrics.jsonl")
    if not path.exists():
        return [], False
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        content = stream.read(opened.st_size)
    current = path.stat()
    if ((opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            or len(content) != opened.st_size or current.st_size < opened.st_size):
        raise ValueError("Source metrics were replaced/truncated during snapshot")
    pending = bool(content and not content.endswith(b"\n"))
    if pending:
        end = content.rfind(b"\n") + 1
        content = content[:end]
    records = []
    for index, line in enumerate(content.splitlines(), 1):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"metrics.jsonl line {index}: expected an object")
        metrics.metric_values(record)  # Reuse scalar/finite-value validation, no SDK import.
        records.append(record)
    return records, pending


def loss_rows(records):
    return [row for row in records if row.get("event") in ("train", "validation")]


def state(source, records):
    terminal = metrics._terminal_state(source)
    if terminal:
        return terminal
    if any(row.get("event") == "completed" for row in records):
        return "completed"
    return "no terminal marker (not proof the training process is alive)"


def print_row(row):
    if row.get("event") in ("train", "validation"):
        label = "train" if row["event"] == "train" else "val"
        extra = f" samples={row['samples']} mode={row.get('mode', 'unknown')}" if label == "val" else ""
        print(f"{label:>5}  step={row['step']:>8}  normalized_l1={row['normalized_l1']:.6f}{extra}", flush=True)


def print_summary(source, config, records):
    rows = loss_rows(records)
    train = [row for row in rows if row["event"] == "train"]
    val = [row for row in rows if row["event"] == "validation"]
    print(f"Source: {source}\nState: {state(source, records)}", flush=True)
    print(f"Requested steps: {config.get('steps_requested', '?')}; batch: {config.get('batch_size', '?')}; "
          f"validation: {config['validation_mode']}", flush=True)
    if train:
        print(f"Latest train: step={train[-1]['step']} normalized_l1={train[-1]['normalized_l1']:.6f}", flush=True)
    if val:
        best = min(val, key=lambda row: row["normalized_l1"])
        print(f"Latest val: step={val[-1]['step']} normalized_l1={val[-1]['normalized_l1']:.6f}; "
              f"best val: step={best['step']} normalized_l1={best['normalized_l1']:.6f}", flush=True)
    else:
        print("No validation record yet; the viewer cannot add validation to an existing training process.", flush=True)
    if not rows:
        print("No loss records yet. Training may still be preparing data/model; check the original terminal.", flush=True)
    print("Loss is train-stat-normalized action L1, not radians or task success rate.", flush=True)


def list_runs(task):
    namespace = LOG_DIR / "wp_ws" / task
    print(f"Training logs: {namespace}")
    found = 0
    if namespace.is_dir():
        for directory in sorted(namespace.iterdir()):
            if not directory.is_dir() or not (directory / "run_config.json").is_file():
                continue
            try:
                source = metrics.source_directory(task, directory.name)
                config = metrics.scalar_configuration(source, task, directory.name)
                if not isinstance(config.get("steps_requested"), int) or config["steps_requested"] < 1:
                    continue  # CPU preflight manifests are not training runs.
                records, pending = snapshot(source)
            except (ValueError, OSError) as exc:
                print(f"{directory.name}: cannot inspect ({exc})")
                continue
            train = [row for row in loss_rows(records) if row["event"] == "train"]
            latest = str(train[-1]["step"]) if train else "no loss yet"
            print(f"{directory.name}  requested={config.get('steps_requested', '?')}  "
                  f"latest_train_step={latest}  state={state(source, records)}"
                  + ("  [partial trailing record pending]" if pending else ""))
            found += 1
    if not found:
        print("No ACT training runs found. Check this machine's central LOG_DIR.")
    return 0


@contextmanager
def _plot_cache(directory):
    # Matplotlib cache must not create files in home, source logs, dataset or ckpts.
    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="mpl-cache-", dir=directory) as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def save_plot(source, task, run_name, config, records):
    rows = loss_rows(records)
    if not rows:
        raise ValueError("No train/validation loss records to plot")
    namespace = (LOG_DIR / "wp_ws" / "_loss_views" / task).resolve()
    directory = (namespace / run_name).resolve()
    if directory.parent != namespace:
        raise ValueError("Plot output escapes its central namespace")
    for protected in (source.resolve(), DATA_DIR.resolve(), CKPT_DIR.resolve()):
        if directory == protected or protected in directory.parents or directory in protected.parents:
            raise ValueError("Plot output must not overlap source logs, datasets or checkpoints")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "loss.png"
    if target.resolve().parent != directory:
        raise ValueError("Plot target must not escape its output directory")
    with _plot_cache(directory):
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise ImportError("--plot needs optional matplotlib; text viewing needs only the Python standard library") from exc
        figure = Figure(figsize=(10, 5), dpi=140)
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        for event, label in (("train", "Train (logged batches)"), ("validation", "Validation")):
            selected = [row for row in rows if row["event"] == event]
            if selected:
                axes.plot([row["step"] for row in selected], [row["normalized_l1"] for row in selected],
                          label=label, marker="o" if event == "validation" else None,
                          markersize=3, linewidth=1.2, alpha=0.85)
        axes.set(xlabel="Training step", ylabel="Normalized action L1 (lower is better)",
                 title=f"{run_name}\nValidation: {config['validation_mode']}")
        axes.grid(alpha=0.25)
        axes.legend()
        figure.tight_layout()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".png", delete=False) as stream:
                temporary = Path(stream.name)
            figure.savefig(temporary, format="png")
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
            figure.clear()
    print(f"PNG snapshot (not a live web page): {target}", flush=True)
    return target


def view(args):
    if args.list:
        if args.follow or args.plot:
            raise ValueError("--list cannot be combined with --follow or --plot")
        return list_runs(args.task)
    source = metrics.source_directory(args.task, args.run_name)
    config = metrics.scalar_configuration(source, args.task, args.run_name)
    path = metrics._source_file(source, "metrics.jsonl")
    opened = path.stat() if path.exists() else None
    identity = (opened.st_dev, opened.st_ino) if opened is not None else None
    snapshot_size = opened.st_size if opened is not None else 0
    records, pending = snapshot(source)
    print_summary(source, config, records)
    if pending:
        print("The incomplete final JSONL record is pending; it was not parsed or changed.", flush=True)
    print(f"Recent losses (up to {args.last} records):", flush=True)
    for row in loss_rows(records)[-args.last:]:
        print_row(row)
    if args.plot and loss_rows(records):
        save_plot(source, args.task, args.run_name, config, records)
    if args.follow:
        print("Following new losses. Ctrl+C stops this viewer only, not training.", flush=True)
        initial_count = len(records)
        try:
            # Replay through the existing append-only reader, skipping the shown snapshot.
            for index, record in enumerate(metrics.metric_records(
                    source, follow=True, poll_seconds=args.poll_seconds)):
                current = metrics._source_file(source, "metrics.jsonl").stat()
                current_identity = (current.st_dev, current.st_ino)
                if ((identity is not None and identity != current_identity)
                        or current.st_size < snapshot_size):
                    raise ValueError("Source metrics were replaced/truncated after snapshot")
                identity = current_identity
                if index < initial_count:
                    continue
                metrics.metric_values(record)
                records.append(record)
                print_row(record)
        except KeyboardInterrupt:
            print("\nViewer stopped; the original training process was not touched.", flush=True)
        print_summary(source, config, records)
        if args.plot and loss_rows(records):
            save_plot(source, args.task, args.run_name, config, records)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return view(args)
    except KeyboardInterrupt:
        print("\nViewer stopped; the original training process was not touched.", flush=True)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"Loss viewer failed (training unaffected): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
