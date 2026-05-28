#!/usr/bin/env bash
# Build the user layer of the benchmark image (bakes uid/gid for NFS access).
#
# Reads (from env, all optional — sensible defaults from `id`):
#   IMAGE_NAME    image repository (default: mmore-colvision-benchmark)
#   GENERIC_TAG   tag of the upstream generic image (default: generic-latest)
#   USR_TAG       tag for the user image (default: ${USER}-latest)
#   USR / USRID / GRP / GRPID / PASSWD  host user identity
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE_NAME="${IMAGE_NAME:-mmore-colvision-benchmark}"
GENERIC_TAG="${GENERIC_TAG:-generic-latest}"

USR="${USR:-$(id -un)}"
USRID="${USRID:-$(id -u)}"
GRP="${GRP:-$(id -gn)}"
GRPID="${GRPID:-$(id -g)}"
PASSWD="${PASSWD:-${USR}}"
USR_TAG="${USR_TAG:-${USR}-latest}"

echo "[bcv-build-user] building ${IMAGE_NAME}:${USR_TAG}"
echo "[bcv-build-user]   USR=${USR} USRID=${USRID} GRP=${GRP} GRPID=${GRPID}"

# BuildKit resolves the base image's manifest against its registry even with
# --pull=false. On EPFL VPN the Harbor OIDC round-trip can take 30+ min and
# hang the build. Retag to a plain local name (no registry prefix) so BuildKit
# resolves the image from the local store without any network contact.
LOCAL_BASE="bcv-buildctx"
docker tag "${IMAGE_NAME}:${GENERIC_TAG}" "${LOCAL_BASE}:${GENERIC_TAG}"

DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile.user \
    -t "${IMAGE_NAME}:${USR_TAG}" \
    --pull=false \
    --build-arg "GENERIC_IMAGE=${LOCAL_BASE}" \
    --build-arg "GENERIC_TAG=${GENERIC_TAG}" \
    --build-arg "USR=${USR}" \
    --build-arg "USRID=${USRID}" \
    --build-arg "GRP=${GRP}" \
    --build-arg "GRPID=${GRPID}" \
    --build-arg "PASSWD=${PASSWD}" \
    docker/

docker rmi "${LOCAL_BASE}:${GENERIC_TAG}" >/dev/null 2>&1 || true
