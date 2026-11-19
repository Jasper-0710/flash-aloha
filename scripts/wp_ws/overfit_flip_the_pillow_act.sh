#!/usr/bin/env bash
# Offline only. User starts training; never controls a robot or overwrites existing runs.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage:
  bash scripts/wp_ws/overfit_flip_the_pillow_act.sh preflight [--dry-run] [options]
  bash scripts/wp_ws/overfit_flip_the_pillow_act.sh train [--dry-run] [options]

Activate Conda first. Default: train, fresh ACT/ImageNet initialization, 1500 steps,
2 training episodes x 16 fixed starts, batch 16, chunk 50, 240x320, eval every 100.
Evaluation reuses the SAME TRAINING SAMPLES: NOT held-out validation or task success.
GPU_ID=0 selects ONE numeric GPU (not DDP); PYTHON_BIN=python selects the interpreter.
Extra options override defaults. --dry-run performs no Python/CUDA work or writes.
No resume; no automatic W&B upload. New run names are required for repeated runs.
EOF
}
mode=train
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  preflight|train) mode="$1"; shift ;;
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
cd -- "$script_dir/../.."
python_bin="${PYTHON_BIN:-python}"
gpu="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  printf 'Error: select ONE numeric GPU with GPU_ID (not DDP): %s\n' "$gpu" >&2
  exit 2
fi
if ! command -v -- "$python_bin" >/dev/null 2>&1; then
  printf 'Error: Python not found: %s. Activate Conda first.\n' "$python_bin" >&2
  exit 2
fi
if [[ "$mode" == train ]]; then
  defaults=(--run-name act_overfit_32_local_v1 --steps 1500 --eval-every 100
    --log-every 50 --backbone-init imagenet --acknowledge-audit-warnings)
else
  defaults=(--run-name act_overfit_preflight_v1)
fi
command=("$python_bin" -m src.wp_ws.overfit_act "$mode" "${defaults[@]}"
  --samples-per-episode 16 --batch-size 16 --chunk-size 50 --height 240 --width 320 "${extra[@]}")
printf 'Repository: %s\nMode: %s (fixed training samples, NOT validation)\n' "$PWD" "$mode"
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
