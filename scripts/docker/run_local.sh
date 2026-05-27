#!/usr/bin/env bash
# Run the benchmark inside a local Docker container (no Run:AI, no k8s).
#
# Useful for:
#   - A workstation with an NVIDIA GPU + nvidia-container-toolkit
#   - A CI runner with --gpus all
#   - Single-cell smoke tests outside RCP
#
# Usage:
#   ./scripts/docker/run_local.sh bcv-run track-a --model-id colpali_v1_3 ...
#   ./scripts/docker/run_local.sh bash       # drop into a shell
set -euo pipefail
cd "$(dirname "$0")/../.."

IMAGE="${BCV_IMAGE:-mmore-colvision-benchmark:${USER}-latest}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[bcv-local] image ${IMAGE} not found — building..."
    IMAGE_NAME="mmore-colvision-benchmark" GENERIC_TAG="generic-latest" \
        bash docker/build-generic.sh
    IMAGE_NAME="mmore-colvision-benchmark" USR_TAG="${USER}-latest" \
        bash docker/build-user.sh
fi

# By default, mount the repo at /workspace and point PROJECT_ROOT_AT to it.
# The entrypoint will pip install -e it.
docker run --rm -it \
    --gpus all \
    --shm-size=8g \
    -v "$(pwd):/workspace:rw" \
    -e PROJECT_ROOT_AT=/workspace \
    -e PACKAGE_NAME=benchmark_colvision \
    -e HF_HOME="${HF_HOME:-/workspace/data/.hf-cache}" \
    -w /workspace \
    "${IMAGE}" \
    "$@"
