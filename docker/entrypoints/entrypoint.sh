#!/bin/bash
# Main entrypoint — installs the benchmark package in editable mode from
# PROJECT_ROOT_AT, then execs the user's command.
#
# When PROJECT_ROOT_AT is not set, falls back to running with whatever code
# (if any) is baked in the image. This makes the image usable both in:
#   - PVC / bind-mount workflows (RCP, k8s, local dev) — code lives outside
#   - frozen-image workflows (CI / ad-hoc docker run) — code is built in
#
# Adapted from EPFLiGHT/LiGHT-cluster-template.
set -eo pipefail

PACKAGE_NAME="${PACKAGE_NAME:-benchmark_colvision}"

if [ -z "${PROJECT_ROOT_AT:-}" ]; then
    echo "[bcv-entrypoint] PROJECT_ROOT_AT not set — running with image as-is."
else
    echo "[bcv-entrypoint] PROJECT_ROOT_AT=${PROJECT_ROOT_AT}"
    if [ ! -f "${PROJECT_ROOT_AT}/pyproject.toml" ]; then
        echo "[bcv-entrypoint][ERROR] ${PROJECT_ROOT_AT}/pyproject.toml not found." >&2
        echo "[bcv-entrypoint][ERROR] Did you mount the repository at PROJECT_ROOT_AT?" >&2
        exit 2
    fi
    cd "${PROJECT_ROOT_AT}"
    # --no-deps because all transitive deps are already in /opt/bcv/.venv from
    # the image build. We only need to register the project's own source tree
    # in the venv so `bcv-run`, `bcv-corpus`, etc. resolve to the live code.
    echo "[bcv-entrypoint] installing project (editable, --no-deps)"
    uv pip install --no-deps --quiet -e "${PROJECT_ROOT_AT}"
    python -c "import ${PACKAGE_NAME}; print('[bcv-entrypoint] import OK:', ${PACKAGE_NAME}.__file__)"
fi

echo "[bcv-entrypoint] exec: $*"
exec "$@"
