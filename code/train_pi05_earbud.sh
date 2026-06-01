#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
DATASET_ROOT=${DATASET_ROOT:-/root/autodl-tmp/lerobot_datasets}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/earbud_insert}
BASE_POLICY=${BASE_POLICY:-/root/autodl-tmp/hf_models/pi05_libero_finetuned_v044}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/checkpoints/pi05_earbud}
JOB_NAME=${JOB_NAME:-pi05_earbud}
STEPS=${STEPS:-1000}
BATCH_SIZE=${BATCH_SIZE:-4}

lerobot-train \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATASET_ROOT} \
  --policy.type=pi05 \
  --policy.repo_id=local/pi05_earbud \
  --policy.pretrained_path=${BASE_POLICY} \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --steps=${STEPS} \
  --batch_size=${BATCH_SIZE} \
  --policy.dtype=bfloat16 \
  --policy.device=cuda
