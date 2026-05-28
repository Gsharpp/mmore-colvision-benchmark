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

# The image is lean (no Python deps). The dependency venv lives in a host dir
# mounted at /venv — the local equivalent of the RCP scratch venv. Build it once
# inside the container (so interpreter paths match the /venv mount), then reuse.
VENV_HOST="${BCV_VENV_HOST:-$(pwd)/.venv-docker}"
mkdir -p "${VENV_HOST}"

MOUNTS=(
    -v "$(pwd):/workspace:rw"
    -v "${VENV_HOST}:/venv:rw"
)
ENVS=(
    -e PROJECT_ROOT_AT=/workspace
    -e PACKAGE_NAME=benchmark_colvision
    -e BCV_VENV=/venv
    -e HF_HOME="${HF_HOME:-/workspace/data/.hf-cache}"
)

if [ ! -x "${VENV_HOST}/bin/python" ]; then
    echo "[bcv-local] /venv empty — populating it once (uv sync + vllm)..."
    docker run --rm -i --gpus all --shm-size=8g "${MOUNTS[@]}" "${ENVS[@]}" \
        -w /workspace "${IMAGE}" \
        bash -lc 'UV_PROJECT_ENVIRONMENT=/venv UV_LINK_MODE=copy uv sync --frozen --extra gpu \
                  && UV_PROJECT_ENVIRONMENT=/venv uv pip install "vllm>=0.6" "langchain-openai>=0.2"'
fi

docker run --rm -it --gpus all --shm-size=8g "${MOUNTS[@]}" "${ENVS[@]}" \
    -w /workspace "${IMAGE}" "$@"
