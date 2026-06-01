#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-/root/autodl-tmp/checkpoints/pi05_earbud/checkpoints/001000/pretrained_model}
TOKENIZER=${TOKENIZER:-/root/autodl-tmp/cache/huggingface/google/paligemma-3b-pt-224}

# Terminal 1
python policy_server_pi05.py \
  --policy_path "$CKPT" \
  --tokenizer_path "$TOKENIZER" \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda

# Terminal 2 (run separately):
# python eval_pi0_libero_client.py --server http://127.0.0.1:8000 --camera_size 512 --output_json earbud_pi05_eval_results_finetuned.json
