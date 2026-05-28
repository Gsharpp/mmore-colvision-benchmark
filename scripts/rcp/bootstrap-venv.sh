#!/usr/bin/env bash
# One-time creation of the shared Python venv on the scratch PVC.
#
# Instead of baking the multi-GB dependency set (torch, vllm, mmore...) into the
# pushed Docker image, we install it ONCE here, on the cluster, into a venv that
# lives on the scratch PVC. Every benchmark job from the same image then just
# activates this venv (see docker/entrypoints/entrypoint.sh) — no per-job
# install, and the heavy download happens on the cluster's fast network rather
# than over your VPN link.
#
# This is normally run for you by `scripts/rcp/setup.sh`. Run it by hand only to
# rebuild the venv after dependencies change (pyproject.toml / uv.lock):
#   ./scripts/rcp/bootstrap-venv.sh            # submit + watch instructions
#   ./scripts/rcp/bootstrap-venv.sh --wait     # submit and block until ready
#   FORCE_BOOTSTRAP=1 ./scripts/rcp/bootstrap-venv.sh --wait   # force rebuild
#
# Prerequisites:
#   - ./scripts/rcp/setup.sh has produced .rcp-env (image built + pushed)
#   - the repo is cloned on the scratch PVC at PROJECT_ROOT_AT (one-time)
set -euo pipefail
cd "$(dirname "$0")/../.."

WAIT=0
[ "${1:-}" = "--wait" ] && WAIT=1

[ -f .rcp-env ] || {
    echo "[bcv-bootstrap][ERROR] .rcp-env not found — run ./scripts/rcp/setup.sh first." >&2
    exit 1
}
# shellcheck disable=SC1091
source .rcp-env

JOB_NAME="bcv-bootstrap"
PROJECT_ROOT_AT="/mloscratch/${USR}/bcv-dev"
BCV_VENV="/mloscratch/${USR}/bcv-venv"
UV_CACHE_DIR="/mloscratch/${USR}/uv-cache"
FB="${FORCE_BOOTSTRAP:-0}"

# What the pod does. ASCII sentinels (BCV_BOOTSTRAP_DONE / _FAIL) let the local
# waiter classify the outcome without parsing runai's status vocabulary.
#   - fail fast & clearly if the repo isn't on the scratch PVC,
#   - fast-path out in ~30s if the venv is already complete (idempotent re-runs),
#   - otherwise resolve deps + editable project, then add the judge libs.
read -r -d '' REMOTE_CMD <<EOF || true
set -e
if [ ! -f "${PROJECT_ROOT_AT}/pyproject.toml" ]; then
    echo "BCV_BOOTSTRAP_FAIL: repo not found at ${PROJECT_ROOT_AT} — clone it on the scratch PVC first"
    exit 1
fi
if [ "${FB}" != "1" ] && "${BCV_VENV}/bin/python" -c "import benchmark_colvision, torch, vllm" >/dev/null 2>&1; then
    echo "BCV_BOOTSTRAP_DONE: venv already complete at ${BCV_VENV} (FORCE_BOOTSTRAP=1 to rebuild)"
    exit 0
fi
cd "${PROJECT_ROOT_AT}"
export UV_PROJECT_ENVIRONMENT="${BCV_VENV}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export UV_PYTHON=3.11
export UV_LINK_MODE=copy
echo "[bootstrap] uv sync --frozen --extra gpu  (deps + editable project)"
uv sync --frozen --extra gpu
echo "[bootstrap] uv pip install vllm langchain-openai  (judge service)"
uv pip install --python "${BCV_VENV}/bin/python" "vllm>=0.6" "langchain-openai>=0.2"
"${BCV_VENV}/bin/python" -c "import benchmark_colvision; print('[bootstrap] import OK:', benchmark_colvision.__file__)"
echo "BCV_BOOTSTRAP_DONE: venv ready at ${BCV_VENV}"
EOF

# Poll the pod logs for our sentinel. Robust across runai versions (no status
# string parsing). Returns 0=ready, 1=failed, 2=timeout.
wait_for_bootstrap() {
    local timeout="${BCV_BOOTSTRAP_TIMEOUT:-2400}" elapsed=0 interval=15 logs
    echo "[bcv-bootstrap] waiting for ${JOB_NAME} (up to $((timeout / 60)) min, downloads several GB)…"
    while [ "${elapsed}" -lt "${timeout}" ]; do
        logs="$(runai logs "${JOB_NAME}" --suppress-deprecation-message 2>/dev/null || true)"
        if printf '%s' "${logs}" | grep -q "BCV_BOOTSTRAP_DONE"; then
            echo
            echo "[bcv-bootstrap] venv ready at ${BCV_VENV}."
            return 0
        fi
        if printf '%s' "${logs}" | grep -q "BCV_BOOTSTRAP_FAIL"; then
            echo
            echo "[bcv-bootstrap][ERROR] bootstrap job failed:" >&2
            printf '%s\n' "${logs}" | tail -n 20 >&2
            return 1
        fi
        sleep "${interval}"
        elapsed=$((elapsed + interval))
        printf '.'
    done
    echo
    echo "[bcv-bootstrap][WARN] still not ready after ${timeout}s." >&2
    echo "[bcv-bootstrap][WARN] keep watching with: runai logs ${JOB_NAME} -f" >&2
    return 2
}

# Clear any previous attempt so the name is free.
runai delete job "${JOB_NAME}" --suppress-deprecation-message >/dev/null 2>&1 || true

args=(
    --name "${JOB_NAME}"
    --image "${IMAGE}"
    --gpu 0
    --working-dir "${PROJECT_ROOT_AT}"
    --suppress-deprecation-message
)
if [ -n "${PVC_SCRATCH:-}" ]; then
    args+=(--existing-pvc "claimname=${PVC_SCRATCH},path=/mloscratch")
fi
if [ -n "${PVC_HOME:-}" ]; then
    args+=(--existing-pvc "claimname=${PVC_HOME},path=/home/${USR}")
fi

echo "[bcv-bootstrap] submitting ${JOB_NAME} (CPU-only, builds venv at ${BCV_VENV})"
runai submit "${args[@]}" --command -- bash -lc "${REMOTE_CMD}"

if [ "${WAIT}" -eq 1 ]; then
    wait_for_bootstrap
    exit $?
fi

cat <<EOM

────────────────────────────────────────────────────────────────────────
[bcv-bootstrap] submitted. Watch it (the uv sync downloads several GB):

    runai logs ${JOB_NAME} -f

Wait for:  BCV_BOOTSTRAP_DONE: venv ready

Then launch the benchmark:

    ./scripts/rcp/submit.sh all
────────────────────────────────────────────────────────────────────────
EOM
