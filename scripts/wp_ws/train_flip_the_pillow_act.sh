#!/usr/bin/env bash
# Offline ACT only: no robot, camera, replay, installation or automatic W&B upload.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/wp_ws/train_flip_the_pillow_act.sh smoke [--dry-run] [trainer options]
  bash scripts/wp_ws/train_flip_the_pillow_act.sh train [--dry-run] [trainer options]

Activate the intended Conda environment first. With no mode, defaults to train.
  smoke: 100 steps, batch 16, 40-start smoke validation
  train: 100000 steps, batch 16, full fixed validation every 5000 steps

Environment:
  GPU_ID=0          One physical GPU index; otherwise inherit a single CUDA_VISIBLE_DEVICES.
  PYTHON_BIN=python Python interpreter in your activated environment.

Examples:
  bash scripts/wp_ws/train_flip_the_pillow_act.sh smoke --dry-run
  bash scripts/wp_ws/train_flip_the_pillow_act.sh smoke
  bash scripts/wp_ws/train_flip_the_pillow_act.sh train
  GPU_ID=1 bash scripts/wp_ws/train_flip_the_pillow_act.sh train \
    --run-name act_baseline_gpu1_30k_v2 --steps 30000 --batch-size 4

Additional trainer options override the defaults. Existing runs are never reused;
there is no resume entry point. --dry-run prints the command without running Python.
The acknowledgement flag preserves, and does not bypass, the reviewed audit warnings.
EOF
}

mode=train
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  smoke|train) mode="$1"; shift ;;
esac
extra=()
dry_run=0
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
  printf 'Error: select ONE numeric GPU with GPU_ID (native trainer is not DDP): %s\n' "$gpu" >&2
  exit 2
fi
if ! command -v -- "$python_bin" >/dev/null 2>&1; then
  printf 'Error: Python not found: %s. Activate your Conda environment first.\n' "$python_bin" >&2
  exit 2
fi

if [[ "$mode" == smoke ]]; then
  defaults=(--run-name act_smoke_local_b16_v1 --steps 100
    --validate-every 50 --val-max-batches 10 --save-every 100 --log-every 10)
else
  defaults=(--run-name act_baseline_local_b16_100k_v1 --steps 100000
    --validate-every 5000 --val-max-batches 0 --save-every 5000 --log-every 100)
fi
command=("$python_bin" -m src.wp_ws.train_act train
  "${defaults[@]}" --batch-size 16 --chunk-size 50 --height 240 --width 320
  --workers 0 --backbone-init imagenet --acknowledge-audit-warnings "${extra[@]}")

printf 'Repository: %s\nMode: %s\nConda: %s\n' "$repo_root" "$mode" "${CONDA_DEFAULT_ENV:-not detected; check your interpreter}"
printf 'Command: CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "$gpu"
printf '%q ' "${command[@]}"
printf '\n'
if (( dry_run )); then
  printf 'DRY RUN: no training, downloads, CUDA access or output writes.\n'
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
exec "${command[@]}"
