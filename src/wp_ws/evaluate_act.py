"""Read-only, full fixed-validation evaluation of a completed native ACT run.

No training, weight downloads, hardware connections, dataset conversion or label
repair. All image processing and chunk boundaries come from the original adapter.
"""
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time

import numpy as np

from src.wp_ws.act_dataset import ActDataset, EpisodeBatchSampler
from src.wp_ws.audit_core import hash_file
from src.wp_ws.baseline_data_config import DEFAULT_CONFIG, load_config
from src.wp_ws.evaluation_checkpoint import locations, load_checkpoint, verify_contract
from src.wp_ws.evaluation_metrics import DetailedErrors, example_rows, reproduction_check, write_report
from src.wp_ws.train_act import _batch_summary, _cuda_memory, _name, _positive, _versions, write_json
from src.wp_ws.verified_data import load_verified_data, require


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "evaluate"):
        sub = subs.add_parser(name)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--source-run", type=_name, required=True)
        sub.add_argument("--checkpoint", choices=("best", "last"), default="best")
        sub.add_argument("--run-name", type=_name, help="New output name under central LOG_DIR; never overwritten")
        sub.add_argument("--batch-size", type=_positive, help="Default: original validation batch size; override is recorded")
        sub.add_argument("--examples-per-episode", type=_positive, default=2)
    return parser


def evaluate_stream(policy, dataset, batch_size, device, *, on_episode=None, progress=print):
    """Stream each recorded observation once; callback gets one episode of numeric chunks.

    Injectable CPU policies are for synthetic tests only; the CLI uses native CUDA
    ACT. Targets are never passed into prediction. No optimization is performed.
    """
    import torch
    from torch.utils.data import DataLoader
    require(tuple(dataset.episode_ids) == tuple(dataset.data.split["val_episode_indices"]),
            "Offline evaluator requires the fixed validation partition")
    sampler = EpisodeBatchSampler(dataset, batch_size, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=str(device).startswith("cuda"))
    totals = [DetailedErrors(dataset.std["action"], dataset.chunk_size) for _ in range(2)]
    per_episode, arrays = {}, []
    current = None
    episode_totals = None
    cursor = 0
    was_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                ids, frames = batch["episode_index"].numpy(), batch["frame_index"].numpy()
                require(len(ids) > 0 and cursor < len(dataset), "Unexpected empty/extra validation batch")
                eid, start = dataset.locate(cursor)
                require(np.all(ids == eid) and np.array_equal(frames, np.arange(start, start + len(ids))),
                        "Validation rows are missing, duplicated, shuffled, or cross episodes")
                episode = dataset.data.episodes[eid]
                require(start + len(ids) <= episode.rows, "Validation batch crosses episode boundary")
                if current != eid:
                    require(not arrays and str(eid) not in per_episode, "Repeated/incomplete validation episode")
                    current = eid
                    episode_totals = [DetailedErrors(dataset.std["action"], dataset.chunk_size) for _ in range(2)]
                target = batch["actions"].numpy()
                mask = batch["is_pad"].numpy()
                lengths = [min(dataset.chunk_size, episode.ranges[
                    bisect_right(dataset.range_starts[eid], int(row)) - 1][1] - row) for row in frames]
                expected_mask = np.arange(dataset.chunk_size)[None, :] >= np.asarray(lengths)[:, None]
                require(np.array_equal(mask, expected_mask), "Padding mask differs from audited continuity")
                qpos, images = (batch[key].to(device, non_blocking=True) for key in ("qpos", "images"))
                pred = policy(qpos, images).detach().float().cpu().numpy()
                for accumulator in (totals, episode_totals):
                    # The reference is standardized zero = original TRAIN action mean.
                    accumulator[0].add(pred, target, mask, ids)
                    accumulator[1].add(np.zeros_like(pred), target, mask, ids)
                arrays.append((pred, target, mask, frames))
                cursor += len(ids)
                if start + len(ids) == episode.rows:
                    per_episode[str(eid)] = {"model": episode_totals[0].result(),
                                             "train_mean_constant": episode_totals[1].result()}
                    if on_episode:
                        packed = {key: np.concatenate([item[i] for item in arrays], axis=0)
                                  for i, key in enumerate(("prediction_normalized", "target_normalized", "is_pad", "frame_index"))}
                        packed.update(episode_index=np.asarray(eid, dtype=np.int64),
                                      action_mean=dataset.mean["action"], action_std=dataset.std["action"],
                                      contiguous_ranges=np.asarray(episode.ranges, dtype=np.int64))
                        on_episode(eid, packed)
                        del packed
                    arrays.clear()
                    if progress:
                        progress(f"[离线评估 {len(per_episode)}/{len(dataset.episode_ids)}] episode={eid:06d} "
                                 f"rows={episode.rows} L1={per_episode[str(eid)]['model']['normalized_l1']:.6f}")
            require(cursor == len(dataset) and not arrays and
                    set(map(int, per_episode)) == set(dataset.episode_ids), "Incomplete validation coverage")
        return {"model": totals[0].result(), "train_mean_constant": totals[1].result(),
                "per_episode": per_episode}
    finally:
        policy.train(was_training)
        dataset.clear_cache()


def save_episode(directory, eid, arrays):
    # No Python objects/pickle in archives. Raw-unit plots invert only stored train statistics.
    with (directory / "predictions" / f"episode_{eid:06d}.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def prepare(args, config, source_logs, source_ckpts, output):
    print("[1/3] 只读核对完成记录、checkpoint、训练配置和统计量（CPU 安全加载）", flush=True)
    checked, documents = load_checkpoint(source_logs, source_ckpts, args.checkpoint)
    print("[2/3] 核验数据/audit 内容指纹、原固定划分和推理实现；不重算标准化", flush=True)
    data = load_verified_data(config)
    implementation = verify_contract(checked, documents, config, data)
    payload = checked.payload
    dataset = ActDataset(data, payload["normalization_stats"], payload["policy_config"], partition="val")
    original_batch_size = payload["run_config"]["validation_selection"]["batch_size"]
    batch_size = args.batch_size or original_batch_size
    print("[3/3] 用原 ActDataset/共享图像处理解码一个 CPU 验证 batch", flush=True)
    sampler = EpisodeBatchSampler(dataset, batch_size, shuffle=False)
    batch = _batch_summary(dataset, next(iter(sampler)))
    dataset.clear_cache()
    examples = {str(eid): example_rows(data.episodes[eid].rows, args.examples_per_episode)
                for eid in dataset.episode_ids}
    manifest = {
        "schema_version": 1, "format": "wp_ws.act_offline_evaluation.v1",
        "source": {"run_name": args.source_run, "checkpoint_name": args.checkpoint + ".ckpt", "step": payload["step"]},
        "source_evidence": checked.evidence,
        "policy_config": payload["policy_config"], "data_config": payload["run_config"]["data_config"],
        "image_preprocessing": payload["run_config"]["image_preprocessing"],
        "provenance": data.provenance(),
        "validation_episode_indices": list(dataset.episode_ids), "validation_rows": len(dataset),
        "batch_size": batch_size, "original_validation_batch_size": original_batch_size,
        "batch_size_changed": batch_size != original_batch_size,
        "implementation_sha256": implementation,
        "evaluator_sha256": {name: hash_file(Path(__file__).with_name(name)) for name in
                             ("evaluate_act.py", "evaluation_checkpoint.py", "evaluation_metrics.py", "evaluation_plots.py", "overfit_act.py")},
        "environment": _versions(), "training_environment": payload["run_config"].get("environment"),
        "example_selection": "per-episode midpoint strata v1; chosen before prediction, independent of errors",
        "examples_per_episode_requested": args.examples_per_episode,
        "weighting": "all valid action elements in all overlapping chunks; no deduplication",
        "prediction_scope": "recorded observation -> predicted chunk; not a closed-loop rollout",
        "source_metadata_changed": False, "task_success_rate_measured": False, "live_deployment_ready": False,
    }
    data.assert_unchanged()
    checked.assert_unchanged()
    write_json(output / "run_config.json", manifest)
    write_json(output / "examples.json", examples)
    preflight = {"status": "EVALUATION_PREFLIGHT_PASSED", "validation_rows": len(dataset),
                 "validation_episode_indices": list(dataset.episode_ids), "batch_size": batch_size,
                 "batch_size_changed": batch_size != original_batch_size, "batch": batch,
                 "source_content_hashes_verified": True, "normalization_refit": False,
                 "checkpoint_step": payload["step"], "recorded_validation": payload["validation"],
                 "native_model_weights_loaded": False, "gpu_evaluation_verified": False, "warnings": data.warnings}
    write_json(output / "preflight.json", preflight)
    return checked, dataset, manifest, examples, preflight


def run_evaluation(checked, dataset, manifest, examples, output):
    import torch
    from src.wp_ws.evaluation_plots import plot_episode, plot_summary
    from src.wp_ws.view_loss import _plot_cache

    require(torch.cuda.is_available(), "Native ACT evaluation requires CUDA; preflight is CPU-only")
    torch.cuda.set_device(0)  # Physical card chosen via CUDA_VISIBLE_DEVICES; not distributed.
    require(torch.cuda.is_bf16_supported(), "Native ACT requires bf16; no fp16 fallback")
    torch.cuda.reset_peak_memory_stats()
    source = str(Path(__file__).resolve().parents[1])
    if source not in sys.path:
        sys.path.insert(0, source)
    from policy.act import ACTPolicy
    policy = ACTPolicy(checked.payload["policy_config"])
    policy.deserialize(checked.payload["model_state_dict"])  # Native strict state_dict check.
    policy.to("cuda").eval()
    if checked.payload["run_config"].get("channels_last"):
        policy.model.use_channels_last()
    # Deliberately no load_pretrained_backbone(), optimizer, backward(), or step().
    (output / "predictions").mkdir()
    (output / "plots").mkdir()
    started = time.monotonic()
    with _plot_cache(output):
        def consume(eid, packed):
            save_episode(output, eid, packed)
            plot_episode(output / "plots", eid, packed, examples[str(eid)])
        result = evaluate_stream(policy, dataset, manifest["batch_size"], "cuda", on_episode=consume)
        result["reproduction"] = reproduction_check(result["model"], result["train_mean_constant"], checked.payload["validation"])
        result.update(status=("OFFLINE_EVALUATION_COMPLETED" if result["reproduction"]["within_tolerance"] else
                              "OFFLINE_EVALUATION_COMPLETED_REPRODUCTION_WARNING"),
                      elapsed_seconds=time.monotonic() - started, cuda_memory=_cuda_memory(torch),
                      evaluation_scope="full_fixed_validation_NOT_independent_test", source_metadata_changed=False,
                      task_success_rate_measured=False, live_deployment_ready=False)
        plot_summary(output / "plots", result)
    dataset.data.assert_unchanged()
    checked.assert_unchanged()
    write_json(output / "metrics.json", result)
    write_report(output, result, manifest, examples)
    # Final completion marker is written by main only after all outputs/checks succeed.
    return {"status": result["status"], "normalized_l1": result["model"]["normalized_l1"],
            "train_mean_constant_normalized_l1": result["train_mean_constant"]["normalized_l1"],
            "reproduction": result["reproduction"], "samples": result["model"]["samples"],
            "source_metadata_changed": False, "task_success_rate_measured": False, "live_deployment_ready": False,
            "report": str(output / "report.md")}


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = None
    created = False
    try:
        config = load_config(args.config)
        run_name = args.run_name or f"act_eval_{args.command}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
        source_logs, source_ckpts, output = locations(config, args.source_run, run_name)
        output.mkdir(parents=True, exist_ok=False)
        created = True
        checked, dataset, manifest, examples, preflight = prepare(args, config, source_logs, source_ckpts, output)
        if args.command == "preflight":
            result = {"status": preflight["status"], "gpu_evaluation_verified": False,
                      "validation_rows": len(dataset), "recorded_validation": preflight["recorded_validation"], "logs": str(output)}
        else:
            result = run_evaluation(checked, dataset, manifest, examples, output)
        dataset.data.assert_unchanged()
        checked.assert_unchanged()
        write_json(output / "COMPLETED.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        result = {"status": "EVALUATION_FAILED", "error": f"{type(exc).__name__}: {exc}",
                  "scope": "Only new evaluation output may be partial; source training/data unchanged"}
        if created and not (output / "FAILED.json").exists():
            write_json(output / "FAILED.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
