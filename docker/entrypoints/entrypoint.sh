#!/bin/bash
# Main entrypoint — activates the shared scratch venv (BCV_VENV), verifies the
# benchmark package is importable, then execs the user's command.
#
# The heavy Python dependencies (and an editable install of the project) live
# in a venv on the scratch PVC, created once by scripts/rcp/bootstrap-venv.sh.
# Every job from the same image references that one venv, so jobs do NOT install
# anything here — that avoids many concurrent jobs writing to the same NFS venv.
#
# Adapted from EPFLiGHT/LiGHT-cluster-template.
set -eo pipefail

PACKAGE_NAME="${PACKAGE_NAME:-benchmark_colvision}"

# Activate the shared scratch venv built by bootstrap-venv.sh.
if [ -n "${BCV_VENV:-}" ]; then
    if [ ! -x "${BCV_VENV}/bin/python" ]; then
        echo "[bcv-entrypoint][ERROR] BCV_VENV=${BCV_VENV} has no python interpreter." >&2
        echo "[bcv-entrypoint][ERROR] Create it once with: ./scripts/rcp/bootstrap-venv.sh" >&2
        exit 3
    fi
    echo "[bcv-entrypoint] activating scratch venv ${BCV_VENV}"
    export VIRTUAL_ENV="${BCV_VENV}"
    export PATH="${BCV_VENV}/bin:${PATH}"
fi

if [ -z "${PROJECT_ROOT_AT:-}" ]; then
    echo "[bcv-entrypoint] PROJECT_ROOT_AT not set — running with environment as-is."
else
    echo "[bcv-entrypoint] PROJECT_ROOT_AT=${PROJECT_ROOT_AT}"
    if [ ! -f "${PROJECT_ROOT_AT}/pyproject.toml" ]; then
        echo "[bcv-entrypoint][ERROR] ${PROJECT_ROOT_AT}/pyproject.toml not found." >&2
        echo "[bcv-entrypoint][ERROR] Did you mount the repository at PROJECT_ROOT_AT?" >&2
        exit 2
    fi
    cd "${PROJECT_ROOT_AT}"
    # The project was installed editable into BCV_VENV during bootstrap, so the
    # common path is a no-op. We only (re)install if the package is missing
    # (venv not bootstrapped, or a stale checkout) — guarding against the
    # concurrent-write case where dozens of jobs share one venv.
    if ! python -c "import ${PACKAGE_NAME}" 2>/dev/null; then
        echo "[bcv-entrypoint] project not importable — installing editable (--no-deps)"
        uv pip install --no-deps --quiet -e "${PROJECT_ROOT_AT}"
    fi
    python -c "import ${PACKAGE_NAME}; print('[bcv-entrypoint] import OK:', ${PACKAGE_NAME}.__file__)"
fi

echo "[bcv-entrypoint] exec: $*"
exec "$@"
