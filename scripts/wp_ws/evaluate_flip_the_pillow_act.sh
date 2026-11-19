#!/usr/bin/env bash
# Offline read-only evaluation; user-run only. No training, robot, or W&B calls.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage:
  bash scripts/wp_ws/evaluate_flip_the_pillow_act.sh preflight [--dry-run] [options]
  bash scripts/wp_ws/evaluate_flip_the_pillow_act.sh evaluate [--dry-run] [options]

Default: evaluate completed act_mix8_b16_10k_v1/best.ckpt on the FULL original
validation split. Original normalization/images/chunk configuration; no refitting.
Batch size defaults to the original validation batch size. 2 chunk examples/episode.
Outputs: report, metrics, raw numeric predictions and PNGs, in a NEW log run.
Preflight: CPU restricted checkpoint load + hashes + one decoded validation batch;
no CUDA/model inference, no full evaluation, no weight downloads.

Activate Conda first. GPU_ID=0 (local) or 1 (server), ONE numeric GPU, not DDP.
PYTHON_BIN=python selects the interpreter. Extra options override defaults.
Use --source-run/--run-name to choose source/new output. Existing outputs are
never overwritten. --dry-run does not invoke Python, CUDA or create files.
This is NOT a robot rollout or a task-success-rate evaluation.
EOF
}
mode=evaluate
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  preflight|evaluate) mode="$1"; shift ;;
esac
extra=(); dry_run=0
for argument in "$@"; do
  case "$argument" in
    --dry-run) dry_run=1 ;;
    -h|--help) usage; exit 0 ;;
    *) extra+=("$argument") ;;
  esac
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd -- "$repo_root"
python_bin="${PYTHON_BIN:-python}"
gpu="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  printf 'Error: select ONE numeric GPU with GPU_ID; evaluation is not DDP: %s\n' "$gpu" >&2
  exit 2
fi
if ! command -v -- "$python_bin" >/dev/null 2>&1; then
  printf 'Error: Python not found: %s. Activate Conda first.\n' "$python_bin" >&2
  exit 2
fi
run_name=act_mix8_eval_v1
[[ "$mode" != preflight ]] || run_name=act_mix8_eval_preflight_v1
command=("$python_bin" -m src.wp_ws.evaluate_act "$mode" --source-run act_mix8_b16_10k_v1
  --checkpoint best --run-name "$run_name" --examples-per-episode 2 "${extra[@]}")
printf 'Repository: %s\nMode: %s\nConda: %s\n' "$repo_root" "$mode" "${CONDA_DEFAULT_ENV:-not detected}"
printf 'Command: CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "$gpu"
printf '%q ' "${command[@]}"
printf '\n'
if (( dry_run )); then
  printf 'DRY RUN: no evaluation, downloads, CUDA access or output writes.\n'
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
exec "${command[@]}"
