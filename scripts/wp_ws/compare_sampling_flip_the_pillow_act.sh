#!/usr/bin/env bash
# Offline sampling experiment only. User starts each run; no hardware or W&B side effects.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage:
  bash scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh mixed [--dry-run] [trainer options]
  bash scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh control [--dry-run] [trainer options]
  bash scripts/wp_ws/compare_sampling_flip_the_pillow_act.sh preflight [--dry-run] [trainer options]

mixed (default): mix starts from shuffled windows of 8 TRAIN episodes; workers=0.
control: original episode-block sampler; same model/seed/steps/probe/evaluation settings.
preflight: CPU checks for mixed mode only; no CUDA training.

10,000 updates, batch 16, chunk 50, 240x320, ImageNet; full validation + fixed
training probe at step 0 and every 1000. 2 probe starts per TRAIN episode.
Arrow cache: 1024 MiB retained buffers; decoded probe: <=512 MiB. These are NOT RSS limits.
Source data, raw gripper labels and the existing audit split are never changed.
Existing runs are never overwritten. Fresh initialization; no resume entry point.

Activate Conda first. GPU_ID=0 (local), GPU_ID=1 (server); ONE GPU, not DDP.
PYTHON_BIN=python selects the interpreter. Extra trainer options override defaults.
--dry-run never invokes Python. This script never launches both runs automatically.
EOF
}
mode=mixed
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  mixed|control|preflight) mode="$1"; shift ;;
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
  printf 'Error: select ONE numeric GPU with GPU_ID; native trainer is not DDP: %s\n' "$gpu" >&2
  exit 2
fi
if ! command -v -- "$python_bin" >/dev/null 2>&1; then
  printf 'Error: Python not found: %s. Activate Conda first.\n' "$python_bin" >&2
  exit 2
fi
sampler=episode_window
run_name=act_mix8_b16_10k_v1
subcommand=train
training=(--steps 10000 --workers 0 --backbone-init imagenet --lr 0.00012 --lr-backbone 0.00001
  --validate-every 1000 --save-every 1000 --log-every 100 --acknowledge-audit-warnings)
if [[ "$mode" == control ]]; then
  sampler=episode_block
  run_name=act_block_b16_10k_v1
elif [[ "$mode" == preflight ]]; then
  subcommand=preflight
  run_name=act_mix8_preflight_v1
  training=()
fi
command=("$python_bin" -m src.wp_ws.train_act "$subcommand"
  --run-name "$run_name" --sampler "$sampler" --mix-episodes 8 --episode-cache-mib 1024
  --train-probe-per-episode 2 --train-probe-cache-mib 512 --eval-at-start
  --batch-size 16 --chunk-size 50 --height 240 --width 320 --seed 42 --val-max-batches 0
  "${training[@]}" "${extra[@]}")
printf 'Repository: %s\nMode: %s\nConda: %s\n' "$repo_root" "$mode" "${CONDA_DEFAULT_ENV:-not detected}"
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
