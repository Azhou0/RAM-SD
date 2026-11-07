#!/usr/bin/env bash
set -euo pipefail

task_data_dir="${1:?Usage: run_main.sh DATA_DIR PROFILE MODE OUTPUT_ROOT [ORDER_SEED] [ENV_FILE]}"
task_profile="${2:-appendix}"
task_mode="${3:-replay}"
task_output_root="${4:?Specify a new or resumable output root}"
task_seed="${5:-0}"
task_env_file="${6:-}"
task_run_uv() {
  if [[ -n "$task_env_file" ]]; then
    uv "$@" --env-file "$task_env_file"
  else
    uv "$@"
  fi
}

for task_dataset in iacv1 iacv2 semeval mustard; do
  if [[ "$task_dataset" != "mustard" ]]; then
    task_run_uv run --frozen --no-editable ram-sd build-index \
      --config "configs/$task_profile.json" --dataset "$task_dataset" \
      --data-dir "$task_data_dir" --mode "$task_mode" --retrieval-order-seed "$task_seed"
  fi
  task_run_uv run --frozen --no-editable ram-sd run \
    --config "configs/$task_profile.json" --dataset "$task_dataset" \
    --data-dir "$task_data_dir" --mode "$task_mode" --retrieval-order-seed "$task_seed" \
    --out "$task_output_root/$task_dataset" --resume
done

uv run --frozen --no-editable ram-sd summarize \
  "$task_output_root/iacv1/metrics.json" "$task_output_root/iacv2/metrics.json" \
  "$task_output_root/semeval/metrics.json" "$task_output_root/mustard/metrics.json" \
  --output "$task_output_root/summary.json"
