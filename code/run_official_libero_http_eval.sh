#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/openpi_earbud_proto

unset DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/root/autodl-tmp/openpi_earbud_proto/third_party/libero:$PYTHONPATH

CLIENT=/root/autodl-tmp/openpi_earbud_proto/eval_official_libero_http_client.py
PY=/root/autodl-tmp/openpi/examples/libero/.venv/bin/python
OUT_ROOT=/root/autodl-tmp/openpi_earbud_proto/eval_logs/http_pi05_official_$(date +%Y%m%d_%H%M%S)

mkdir -p "${OUT_ROOT}"

run_suite () {
  local SUITE=$1
  local EPS=$2
  local MAX_STEPS=$3
  echo ""
  echo "========== Evaluating ${SUITE} =========="
  "${PY}" "${CLIENT}" \
    --suite "${SUITE}" \
    --server http://127.0.0.1:8000 \
    --episodes-per-task "${EPS}" \
    --max-steps "${MAX_STEPS}" \
    --camera-size 224 \
    --exec-horizon 10 \
    --output-json "${OUT_ROOT}/${SUITE}.json" \
    --output-csv "${OUT_ROOT}/${SUITE}.csv" \
    2>&1 | tee "${OUT_ROOT}/${SUITE}.log"
}

run_suite libero_object 10 600
run_suite libero_10 10 600
run_suite libero_90 10 600

echo ""
echo "All done. Results saved to: ${OUT_ROOT}"
