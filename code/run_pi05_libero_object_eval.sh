#!/usr/bin/env bash
set -euo pipefail

# Evaluate PI0.5 v044 on LIBERO-Object only.
# Start policy_server_pi05.py in another terminal before running this script.

ROOT="${ROOT:-/root/autodl-tmp/openpi_earbud_proto}"
LIBERO_PY="${LIBERO_PY:-/root/autodl-tmp/openpi/examples/libero/.venv/bin/python}"
SERVER="${SERVER:-http://127.0.0.1:8007}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark_eval_official_pi05_v044}"

SUITE="libero_object"
TRIALS="${TRIALS:-50}"
START_TASK="${START_TASK:-0}"
MAX_TASKS="${MAX_TASKS:--1}"
MAX_STEPS="${MAX_STEPS:-520}"
EXEC_HORIZON="${EXEC_HORIZON:-10}"
CAMERA_SIZE="${CAMERA_SIZE:-224}"
DEBUG_STEPS="${DEBUG_STEPS:-100}"
RECORD_VIDEO="${RECORD_VIDEO:-0}"

mkdir -p "${OUT_ROOT}/${SUITE}"

VIDEO_ARGS=()
if [[ "${RECORD_VIDEO}" == "1" ]]; then
  VIDEO_ARGS=(
    --record_video
    --video_dir "${OUT_ROOT}/videos"
    --video_cameras agentview,robot0_eye_in_hand
    --record_limit_per_task 1
  )
fi

echo "[run] suite=${SUITE} trials=${TRIALS} start_task=${START_TASK} max_tasks=${MAX_TASKS}"
echo "[run] output=${OUT_ROOT}/${SUITE}"

"${LIBERO_PY}" "${ROOT}/eval_pi05_libero_official_suites.py" \
  --suite "${SUITE}" \
  --server "${SERVER}" \
  --camera_size "${CAMERA_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --num_trials_per_task "${TRIALS}" \
  --start_task "${START_TASK}" \
  --max_tasks "${MAX_TASKS}" \
  --exec_horizon "${EXEC_HORIZON}" \
  --warmup_steps 10 \
  --pos_action_clip 0.08 \
  --rot_xy_action_clip 0.10 \
  --rot_z_action_clip 0.08 \
  --gripper_action_clip 1.0 \
  --debug_action_trace \
  --debug_action_trace_steps "${DEBUG_STEPS}" \
  --continue_on_error \
  --output_json "${OUT_ROOT}/${SUITE}/results.json" \
  --summary_csv "${OUT_ROOT}/${SUITE}/summary.csv" \
  --action_trace_csv "${OUT_ROOT}/${SUITE}/action_trace.csv" \
  "${VIDEO_ARGS[@]}" \
  2>&1 | tee "${OUT_ROOT}/${SUITE}/terminal.log"
