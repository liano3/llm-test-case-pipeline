#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Qwen3.5-27B"

CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" exec vllm serve "$MODEL_DIR" \
    --served-model-name "${MODEL_NAME:-Qwen3.5-27B}" \
    --tensor-parallel-size "${TP_SIZE:-1}" \
    "$@"
