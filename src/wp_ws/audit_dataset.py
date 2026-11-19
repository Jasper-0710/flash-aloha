"""User-run CLI: python -m src.wp_ws.audit_dataset (no robot or training)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Support both the documented module invocation and a direct file invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.paths import DATA_DIR, LOG_DIR


def _component(value):
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("Use a single name, not a filesystem path")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read-only ALOHA LeRobot v2.1 audit, full image decoding, and fixed episode split.")
    parser.add_argument("--dataset", type=_component, default="flip_the_pillow",
                        help="Dataset name under src.config.paths.DATA_DIR")
    parser.add_argument("--run-name", type=_component, default="audit_v1",
                        help="New report directory under LOG_DIR/wp_ws/DATASET; existing runs are never overwritten")
    parser.add_argument("--expected-task", default=None,
                        help="Intended task name for diagnostic comparison only; defaults to dataset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--gap-factor", type=float, default=3.0,
                        help="Mark chunk boundaries where saved elapsed delta exceeds this factor / fps (default 3)")
    parser.add_argument("--batch-rows", type=int, default=64,
                        help="Parquet streaming batch size, not a training batch size")
    return parser


def resolve_locations(dataset, run_name):
    """All persistent data/log roots come from the repository's central paths."""
    _component(dataset)
    _component(run_name)
    root = (DATA_DIR / dataset).resolve()
    output = (LOG_DIR / "wp_ws" / dataset / run_name).resolve()
    if output == root or root in output.parents:
        raise ValueError("Report output must not be inside the source dataset; check src.config.paths")
    return root, output


def write_artifacts(output, artifacts):
    """Create a new report directory only; split.json is absent on any error.

    COMPLETED.json is written last. Its absence means the process did not
    finish publishing artifacts, even if a partial report directory exists.
    """
    from src.wp_ws.audit_core import markdown_report

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def write_json(name, data):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

    write_json("report.json", artifacts["report"])
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown_report(artifacts["report"], artifacts["split"]))
    write_json("fingerprints.json", artifacts["fingerprints"])
    write_json("continuity.json", artifacts["continuity"])
    if artifacts["split"] is not None:
        write_json("split.json", artifacts["split"])
    write_json("COMPLETED.json", dict(
        status=artifacts["report"]["status"],
        split_ready=artifacts["report"]["split_ready"],
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    ))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # Delayed import gives an actionable error before any audit starts.
        from src.wp_ws.audit_core import AuditConfig, audit_dataset
    except ImportError as exc:
        print(f"依赖导入失败：{exc}\n请在你选定的环境中安装 src/wp_ws/requirements.txt。", file=sys.stderr)
        return 1
    try:
        config = AuditConfig(dataset_name=args.dataset,
                             expected_task=args.expected_task or args.dataset,
                             seed=args.seed, val_episodes=args.val_episodes,
                             gap_factor=args.gap_factor, batch_rows=args.batch_rows)
        config.validate()
        root, output = resolve_locations(args.dataset, args.run_name)
        if output.exists():
            raise ValueError(f"报告目录已存在，不覆盖：{output}\n请换一个 --run-name。")
        if not root.is_dir():
            raise ValueError(f"数据目录不存在：{root}")
        print(f"[1/3] 数据：{root}\n      报告：{output}\n"
              "仅离线读取；不连接机器人、不训练、不修改原始数据。", flush=True)
        artifacts = audit_dataset(root, config, progress=lambda text: print(text, flush=True))
        print("[2/3] 写入体检报告与固定划分（有错误则不生成 split.json）", flush=True)
        write_artifacts(output, artifacts)
    except (ValueError, OSError) as exc:
        print(f"无法完成：{exc}", file=sys.stderr)
        return 1
    report, split = artifacts["report"], artifacts["split"]
    print(f"[3/3] {report['status']}：{report['counts']}\n报告：{output / 'report.md'}", flush=True)
    if split:
        print(f"训练 {len(split['train_episode_indices'])} episodes / {split['train_rows']} rows；"
              f"验证 {len(split['val_episode_indices'])} episodes / {split['val_rows']} rows。\n"
              "请审阅警告；后续训练需显式使用 split.json，本工具没有修改源数据的 train 划分。")
        return 0
    print("体检未通过：没有生成 split.json。请提供 report.json 中的错误，不要先裁剪或删除数据。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
