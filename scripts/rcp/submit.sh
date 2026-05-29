#!/usr/bin/env bash
# Submit benchmark jobs to Run:AI on EPFL RCP / LiGHT.
#
# Usage:
#   ./scripts/rcp/submit.sh smoke
#   ./scripts/rcp/submit.sh serve-meditron
#   ./scripts/rcp/submit.sh track-a [model] [seed]
#   ./scripts/rcp/submit.sh track-b [model] [lang] [seed]
#   ./scripts/rcp/submit.sh all
#
# Without explicit model/seed/lang, `track-a` and `track-b` fan out into all
# (model × palier-via-runner × seed) cells of the benchmark.
#
# Reads `.rcp-env` produced by `scripts/rcp/setup.sh`.
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -f .rcp-env ] || {
    echo "[bcv-submit][ERROR] .rcp-env not found — run ./scripts/rcp/setup.sh first." >&2
    exit 1
}
# shellcheck disable=SC1091
source .rcp-env

ACTION="${1:-help}"
shift || true

# Personal scratch directory. LiGHT convention is /mloscratch/users/<gaspar>;
# override by setting SCRATCH_DIR in .rcp-env if your lab uses another layout.
SCRATCH_DIR="${SCRATCH_DIR:-/mloscratch/users/${USR}}"
PROJECT_ROOT_AT="${SCRATCH_DIR}/bcv-dev"
HF_HOME="${SCRATCH_DIR}/hf-cache"
# Shared venv on scratch, created once by scripts/rcp/bootstrap-venv.sh.
BCV_VENV="${SCRATCH_DIR}/bcv-venv"

# Inline command resolving the mmore commit recorded in pyproject.toml. The
# benchmark embeds it in every BenchmarkRecord for reproducibility.
MMORE_CMD='$(python -c "import tomllib; print(tomllib.loads(open(\"pyproject.toml\",\"rb\").read().decode())[\"tool\"][\"uv\"][\"sources\"][\"mmore\"][\"rev\"])")'

########################################################################
# submit_one — wraps `runai submit` with the conventions of this benchmark.

submit_one() {
    local name="$1"; shift
    local gpus="$1"; shift
    local cmd="$1";  shift

    # No --working-dir: runc would create it as root (NFS root_squash → nobody)
    # and the pod dies with StartError. The entrypoint cds to PROJECT_ROOT_AT
    # itself, running as the real user.
    local args=(
        --name "${name}"
        --image "${IMAGE}"
        --gpu "${gpus}"
        -e PROJECT_ROOT_AT="${PROJECT_ROOT_AT}"
        -e PACKAGE_NAME=benchmark_colvision
        -e BCV_VENV="${BCV_VENV}"
        -e HF_HOME="${HF_HOME}"
        --suppress-deprecation-message
    )
    if [ -n "${PVC_SCRATCH:-}" ]; then
        args+=(--existing-pvc "claimname=${PVC_SCRATCH},path=/mloscratch")
    fi
    if [ -n "${PVC_HOME:-}" ]; then
        args+=(--existing-pvc "claimname=${PVC_HOME},path=/home/${USR}")
    fi
    # Forward a Hugging Face token for gated models (e.g. epfl-llm/meditron-70b,
    # the Llama-2-derived judge). Read from the caller's env so the secret never
    # lives in the repo: `export HF_TOKEN=hf_xxx` before submitting.
    if [ -n "${HF_TOKEN:-}" ]; then
        args+=(-e HF_TOKEN="${HF_TOKEN}" -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}")
    fi

    # No --command: that would override the image ENTRYPOINT and skip the venv
    # activation in docker/entrypoints/entrypoint.sh (so `vllm`, `bcv-run` etc.
    # wouldn't be on PATH). Passing args without --command sends them THROUGH the
    # entrypoint, which activates BCV_VENV then execs the command. Plain `bash -c`
    # (not `-lc`) so a login shell doesn't reset the PATH the entrypoint set.
    echo "[bcv-submit] runai submit ${name} (gpus=${gpus})"
    runai submit "${args[@]}" -- bash -c "${cmd}"
}

########################################################################
# Actions.

cmd_smoke() {
    submit_one bcv-smoke 1 \
        "bcv-run track-a --model-id colpali_v1_3 --seed 0 --palier tiny \
            --config configs/track_a.yaml --models configs/models.yaml \
            --mmore-commit ${MMORE_CMD}"
}

cmd_serve_meditron() {
    submit_one bcv-vllm 4 \
        "vllm serve epfl-llm/meditron-70b \
            --host 0.0.0.0 --port 8000 \
            --tensor-parallel-size 4 \
            --max-model-len 4096 \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.90 \
            --disable-log-requests"
}

cmd_track_a() {
    local model="${1:-}" seed="${2:-}"
    shift 2 2>/dev/null || true
    # Optional: --palier <id>  (repeatable, passed through to bcv-run)
    local palier_flags=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --palier) palier_flags="${palier_flags} --palier $2"; shift 2 ;;
            *) warn "unknown track-a arg: $1"; shift ;;
        esac
    done
    if [ -n "${model}" ] && [ -n "${seed}" ]; then
        # Job names must be lowercase RFC 1123 — collapse underscores.
        local safe_model="${model//_/-}"
        submit_one "bcv-ta-${safe_model}-s${seed}" 1 \
            "bcv-run track-a --model-id ${model} --seed ${seed} \
                ${palier_flags} \
                --config configs/track_a.yaml --models configs/models.yaml \
                --mmore-commit ${MMORE_CMD}"
    else
        local models=(colpali_v1_3 colqwen2_v1_0 colqwen2_5_v0_2 colqwen3_v0_1 colgemma3_colnetra)
        for m in "${models[@]}"; do
            for s in 0 1 2; do
                cmd_track_a "${m}" "${s}"
            done
        done
    fi
}

cmd_track_b() {
    local model="${1:-}" lang="${2:-}" seed="${3:-}"
    if [ -n "${model}" ] && [ -n "${lang}" ] && [ -n "${seed}" ]; then
        local safe_model="${model//_/-}"
        submit_one "bcv-tb-${safe_model}-${lang}-s${seed}" 1 \
            "bcv-run track-b --model-id ${model} --language ${lang} --seed ${seed} \
                --config configs/track_b.yaml --models configs/models.yaml \
                --mmore-commit ${MMORE_CMD}"
    else
        local models=(colpali_v1_3 colqwen2_v1_0 colqwen2_5_v0_2 colqwen3_v0_1 colgemma3_colnetra)
        local langs=(en fr de es zh ar)
        for m in "${models[@]}"; do
            for l in "${langs[@]}"; do
                for s in 0 1 2; do
                    cmd_track_b "${m}" "${l}" "${s}"
                done
            done
        done
    fi
}

cmd_all() {
    cmd_serve_meditron
    cmd_smoke
    cmd_track_a
    cmd_track_b
}

cmd_help() {
    cat <<HELP
Usage: $0 <action> [args]

Actions:
  smoke                            run one Track A cell (colpali, tiny, seed 0)
  serve-meditron                   start the long-lived vLLM judge (4 GPUs)
  track-a [model] [seed]           one Track A cell, or all 15 if no args
  track-b [model] [lang] [seed]    one Track B cell, or all 90 if no args
  all                              serve-meditron + smoke + track-a + track-b

Examples:
  $0 smoke
  $0 track-a colpali_v1_3 0
  $0 track-b colqwen3_v0_1 ar 2
  $0 all

Reads .rcp-env (created by scripts/rcp/setup.sh).
HELP
}

case "${ACTION}" in
    smoke)           cmd_smoke           ;;
    serve-meditron)  cmd_serve_meditron  ;;
    track-a)         cmd_track_a "$@"    ;;
    track-b)         cmd_track_b "$@"    ;;
    all)             cmd_all             ;;
    help|--help|-h|*) cmd_help           ;;
esac
