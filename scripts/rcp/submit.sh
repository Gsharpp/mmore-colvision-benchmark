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

PROJECT_ROOT_AT="/mloscratch/${USR}/bcv-dev"
HF_HOME="/mloscratch/${USR}/hf-cache"

# Inline command resolving the mmore commit recorded in pyproject.toml. The
# benchmark embeds it in every BenchmarkRecord for reproducibility.
MMORE_CMD='$(python -c "import tomllib; print(tomllib.loads(open(\"pyproject.toml\",\"rb\").read().decode())[\"tool\"][\"uv\"][\"sources\"][\"mmore\"][\"rev\"])")'

########################################################################
# submit_one — wraps `runai submit` with the conventions of this benchmark.

submit_one() {
    local name="$1"; shift
    local gpus="$1"; shift
    local cmd="$1";  shift

    local args=(
        --name "${name}"
        --image "${IMAGE}"
        --gpu "${gpus}"
        --working-dir "${PROJECT_ROOT_AT}"
        -e PROJECT_ROOT_AT="${PROJECT_ROOT_AT}"
        -e PACKAGE_NAME=benchmark_colvision
        -e HF_HOME="${HF_HOME}"
        --suppress-deprecation-message
    )
    if [ -n "${PVC_SCRATCH:-}" ]; then
        args+=(--existing-pvc "claimname=${PVC_SCRATCH},path=/mloscratch")
    fi
    if [ -n "${PVC_HOME:-}" ]; then
        args+=(--existing-pvc "claimname=${PVC_HOME},path=/home/${USR}")
    fi

    echo "[bcv-submit] runai submit ${name} (gpus=${gpus})"
    runai submit "${args[@]}" --command -- bash -lc "${cmd}"
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
    if [ -n "${model}" ] && [ -n "${seed}" ]; then
        # Job names must be lowercase RFC 1123 — collapse underscores.
        local safe_model="${model//_/-}"
        submit_one "bcv-ta-${safe_model}-s${seed}" 1 \
            "bcv-run track-a --model-id ${model} --seed ${seed} \
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
