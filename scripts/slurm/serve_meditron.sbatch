#!/bin/bash
#SBATCH --job-name=bcv-vllm-meditron
#SBATCH --output=logs/vllm-%j.out
#SBATCH --error=logs/vllm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00

# Long-lived vLLM server hosting Meditron-70B for query generation and the
# RAGAS judge. Mirrors the values in configs/judge.yaml:
#   tensor_parallel_size: 4
#   max_model_len: 4096
#   dtype: bfloat16
#
# Other benchmark jobs find the endpoint via the hostname printed below.
# Set VLLM_ENDPOINT="http://<host>:${PORT}" in their submission script.

set -euo pipefail

source "$HOME/mmore-colvision-benchmark/.venv/bin/activate"
cd "$HOME/mmore-colvision-benchmark"

MODEL="${MEDITRON_MODEL:-epfl-llm/meditron-70b}"
PORT="${VLLM_PORT:-8000}"

HOST=$(hostname -f)
echo "[serve_meditron] node=${HOST} port=${PORT} model=${MODEL}"
echo "[serve_meditron] endpoint URL: http://${HOST}:${PORT}"
echo "[serve_meditron] downstream jobs: export VLLM_ENDPOINT=http://${HOST}:${PORT}"

# vLLM 0.5+ ships a `vllm serve` CLI; older versions need the explicit module.
if command -v vllm >/dev/null 2>&1; then
  exec vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests
else
  exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests
fi
