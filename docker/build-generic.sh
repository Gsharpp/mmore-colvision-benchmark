#!/usr/bin/env bash
# Build the generic (user-agnostic) benchmark image.
#
# Reads (from env, all optional):
#   IMAGE_NAME    image repository (default: mmore-colvision-benchmark)
#   GENERIC_TAG   tag for the generic image (default: generic-latest)
#   BASE_IMAGE    CUDA base image (default: nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04)
#   UV_VERSION    uv release pinned in the image (default: 0.5.13)
#   PYTHON_VERSION python version uv installs inside the image (default: 3.11)
#
# Build context is the repository root (one level up from this script).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE_NAME="${IMAGE_NAME:-mmore-colvision-benchmark}"
GENERIC_TAG="${GENERIC_TAG:-generic-latest}"

BUILD_ARGS=()
[ -n "${BASE_IMAGE:-}" ]      && BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
[ -n "${UV_VERSION:-}" ]      && BUILD_ARGS+=(--build-arg "UV_VERSION=${UV_VERSION}")
[ -n "${PYTHON_VERSION:-}" ]  && BUILD_ARGS+=(--build-arg "PYTHON_VERSION=${PYTHON_VERSION}")

echo "[bcv-build-generic] building ${IMAGE_NAME}:${GENERIC_TAG}"
DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile \
    -t "${IMAGE_NAME}:${GENERIC_TAG}" \
    --target runtime-generic \
    "${BUILD_ARGS[@]}" \
    .
