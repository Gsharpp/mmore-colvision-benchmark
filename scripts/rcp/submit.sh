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
# Dedicated CUDA-12 venv for the vLLM judge (see bootstrap-venv.sh for the why).
BCV_VENV_VLLM="${SCRATCH_DIR}/bcv-venv-vllm"
# Dedicated tf4 venv for the native mmore hybrid text baseline (mmore[rag]
# without the colvision extra — SPLADE's pymilvus model needs transformers 4.x).
BCV_VENV_MMORETEXT="${SCRATCH_DIR}/bcv-venv-mmoretext"
# Dedicated venv for mmore's own `process` pipeline (Marker+Surya OCR), used to
# extract page text from ViDoRe's image-only PDFs before running text baselines.
BCV_VENV_MMOREPROCESS="${SCRATCH_DIR}/bcv-venv-mmoreprocess"

# Resolve the mmore commit from pyproject.toml at submit time (local copy).
# Avoids quoting hell when the value is embedded in bash -c "...".
MMORE_REV=$(python3 -c 'import tomllib; d=tomllib.loads(open("pyproject.toml","rb").read().decode()); print(d["tool"]["uv"]["sources"]["mmore"]["rev"])' 2>/dev/null \
    || grep -oE 'rev = "[0-9a-f]{7,40}"' pyproject.toml | grep -oE '[0-9a-f]{7,40}' | head -1 \
    || echo "unknown")

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
        "bcv-run track-a --model-id colpali_v1_3 --seed 0 --palier full \
            --config configs/track_a.yaml --models configs/models.yaml \
            --mmore-commit ${MMORE_REV}"
}

cmd_serve_meditron() {
    # Serve from the dedicated CUDA-12 vLLM venv (NOT the PATH vllm, which is the
    # CUDA-13 build in bcv-venv that fails with libcudart.so.13). LD_LIBRARY_PATH
    # exposes libcuda.so.1 (NVIDIA driver lib) to vLLM's platform probe. No
    # --disable-log-requests: that flag does not exist in vllm<0.20.
    submit_one bcv-vllm 4 \
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-} && \
         ${BCV_VENV_VLLM}/bin/vllm serve epfl-llm/meditron-70b \
            --host 0.0.0.0 --port 8000 \
            --tensor-parallel-size 4 \
            --max-model-len 2048 \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.90"
}

cmd_gen_queries() {
    # Generate synthetic queries for all three modes (text / visual / mixed).
    # Usage: ./submit.sh gen-queries <manifest> <pdfs_dir> <out_prefix> <model>
    # Submits ONE job per mode; output files are <out_prefix>_{text,visual,mixed}.jsonl.
    # Default model is Qwen2.5-32B-Instruct (instruction-tuned, served via the
    # chat endpoint). Meditron base LLMs are abandoned for generation (0 yield).
    local manifest="${1:-${SCRATCH_DIR}/bcv-data/manifest.json}"
    local pdfs_dir="${2:-${SCRATCH_DIR}/bcv-data/pdfs}"
    local out_prefix="${3:-${PROJECT_ROOT_AT}/data/track_a/queries/tiny_qwen32b}"
    local model="${4:-Qwen/Qwen2.5-32B-Instruct}"
    # GPU/tp by model size; instruct models use the chat endpoint + 4k context
    # (the few-shot prompt + chat template overflow 2048 otherwise).
    local gpu_count=1 tp_size=1 health_iters=180 max_model_len=2048 chat_flag=""
    if [[ "${model}" == *"70b"* ]] || [[ "${model}" == *"70B"* ]]; then
        gpu_count=4; tp_size=4
        health_iters=1080   # 70B ≈ 140 GB from NFS → allow 90 min (1080 × 5s)
    elif [[ "${model}" == *"32b"* ]] || [[ "${model}" == *"32B"* ]]; then
        gpu_count=2; tp_size=2
        health_iters=600    # 32B ≈ 65 GB from NFS → up to 50 min
    fi
    if [[ "${model}" == *[Ii]nstruct* ]] || [[ "${model}" == *[Qq]wen* ]] || [[ "${model}" == *[Ll]lama-3* ]]; then
        chat_flag="--chat"; max_model_len=4096
    fi
    local TS; TS=$(date +%H%M)
    for mode in text visual mixed; do
        local out="${out_prefix}_${mode}.jsonl"
        submit_one "bcv-gen-${TS}-${mode}" "${gpu_count}" \
            "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-} && \
             ${BCV_VENV_VLLM}/bin/vllm serve ${model} \
                --host 127.0.0.1 --port 8000 --max-model-len ${max_model_len} \
                --tensor-parallel-size ${tp_size} \
                --dtype bfloat16 --gpu-memory-utilization 0.90 > /tmp/vllm.log 2>&1 & \
             VLLM_PID=\$! && echo [gen] vLLM starting pid=\${VLLM_PID} && \
             for i in \$(seq 1 ${health_iters}); do sleep 5; \
               if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then \
                 echo [gen] vLLM healthy after \$((i*5))s; break; fi; \
               if [ \$i -eq ${health_iters} ]; then echo [gen] vLLM TIMEOUT; tail -20 /tmp/vllm.log; exit 1; fi; done && \
             echo [gen] dumping vllm.log tail: && tail -20 /tmp/vllm.log && \
             ${BCV_VENV}/bin/bcv-queries generate \
                --corpus-manifest ${manifest} --corpus-root ${pdfs_dir} \
                --out ${out} --vllm-endpoint http://127.0.0.1:8000 \
                --vllm-model ${model} --n-per-page 2 \
                --temperature 0.7 --max-tokens 400 \
                --query-mode ${mode} --few-shot --guided-json ${chat_flag} && \
             echo DONE_GEN_${mode} && wc -l ${out}"
    done
}

cmd_gen_tb() {
    # Generate Track B queries for ONE language: serve Qwen once, generate mixed-mode
    # queries, write straight to the per-language <out_file>.jsonl the orchestrator reads.
    # Cross-lingual by design: questions are in ENGLISH regardless of the document
    # language (--query-language en), to test cross-script visual retrieval and avoid
    # depending on the generator's FR/ZH fluency.
    # Usage: ./submit.sh gen-tb <lang> <manifest> <pdfs_dir> <out_file> [model]
    local lang="${1:?lang required}" manifest="${2:?manifest required}"
    local pdfs_dir="${3:?pdfs_dir required}" out="${4:?out_file required}"
    local model="${5:-Qwen/Qwen2.5-32B-Instruct}"
    local gpu_count=2 tp_size=2 health_iters=600 max_model_len=4096
    local TS; TS=$(date +%H%M)
    submit_one "bcv-gentb-${lang}-${TS}" "${gpu_count}" \
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-} && \
         ${BCV_VENV_VLLM}/bin/vllm serve ${model} \
            --host 127.0.0.1 --port 8000 --max-model-len ${max_model_len} \
            --tensor-parallel-size ${tp_size} \
            --dtype bfloat16 --gpu-memory-utilization 0.90 > /tmp/vllm.log 2>&1 & \
         echo [gentb] vLLM starting && \
         for i in \$(seq 1 ${health_iters}); do sleep 5; \
           if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then \
             echo [gentb] vLLM healthy after \$((i*5))s; break; fi; \
           if [ \$i -eq ${health_iters} ]; then echo [gentb] vLLM TIMEOUT; tail -20 /tmp/vllm.log; exit 1; fi; done && \
         rm -f ${out} && mkdir -p \$(dirname ${out}) && \
         ${BCV_VENV}/bin/bcv-queries generate \
            --corpus-manifest ${manifest} --corpus-root ${pdfs_dir} \
            --out ${out} --vllm-endpoint http://127.0.0.1:8000 \
            --vllm-model ${model} --n-per-page 2 \
            --temperature 0.7 --max-tokens 400 \
            --query-mode mixed --query-language en --few-shot --guided-json --chat && \
         echo DONE_GENTB_${lang} && wc -l ${out}"
}

cmd_gen_tb_native() {
    # Same generation as gen-tb, but the questions are written in the DOCUMENT's
    # language instead of English, so a cell can be scored both cross-lingually
    # (English query) and monolingually (native query) over the same index.
    # Writes to data/track_b[_<lang>]/queries_native/<lang>.jsonl, which the
    # configs/track_b_<lang>_native.yaml cells read.
    # Usage: ./submit.sh gen-tb-native <lang> [model]
    local lang="${1:?lang required}"
    local model="${2:-Qwen/Qwen2.5-32B-Instruct}"
    local dir; [ "${lang}" = "en" ] && dir="track_b" || dir="track_b_${lang}"
    local base="${PROJECT_ROOT_AT}/data/${dir}"
    local manifest="${base}/corpus_manifest.json" pdfs_dir="${base}/pdfs"
    local out="${base}/queries_native/${lang}.jsonl"
    local gpu_count=2 tp_size=2 health_iters=600 max_model_len=4096
    local TS; TS=$(date +%H%M)
    submit_one "bcv-gentbnat-${lang}-${TS}" "${gpu_count}" \
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-} && \
         ${BCV_VENV_VLLM}/bin/vllm serve ${model} \
            --host 127.0.0.1 --port 8000 --max-model-len ${max_model_len} \
            --tensor-parallel-size ${tp_size} \
            --dtype bfloat16 --gpu-memory-utilization 0.90 > /tmp/vllm.log 2>&1 & \
         echo [gentbnat] vLLM starting && \
         for i in \$(seq 1 ${health_iters}); do sleep 5; \
           if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then \
             echo [gentbnat] vLLM healthy after \$((i*5))s; break; fi; \
           if [ \$i -eq ${health_iters} ]; then echo [gentbnat] vLLM TIMEOUT; tail -20 /tmp/vllm.log; exit 1; fi; done && \
         rm -f ${out} && mkdir -p \$(dirname ${out}) && \
         ${BCV_VENV}/bin/bcv-queries generate \
            --corpus-manifest ${manifest} --corpus-root ${pdfs_dir} \
            --out ${out} --vllm-endpoint http://127.0.0.1:8000 \
            --vllm-model ${model} --n-per-page 2 \
            --temperature 0.7 --max-tokens 400 \
            --query-mode mixed --query-language ${lang} --few-shot --guided-json --chat && \
         echo DONE_GENTBNAT_${lang} && wc -l ${out}"
}

cmd_corpus_fr() {
    # Build the French Track B corpus from HAL.
    # Usage: ./submit.sh corpus-fr [n_articles=100] [seed=0]
    local n="${1:-100}" seed="${2:-0}"
    local TS; TS=$(date +%H%M)
    submit_one "bcv-corpus-fr-${TS}" 1 \
        "bcv-corpus sample-hal --n ${n} --seed ${seed} --language fr --out ${SCRATCH_DIR}/bcv-data/hal_manifest.json && bcv-corpus download-urls ${SCRATCH_DIR}/bcv-data/hal_manifest.json --out-dir ${SCRATCH_DIR}/bcv-data/pdfs-fr && bcv-corpus build-manifest ${SCRATCH_DIR}/bcv-data/pdfs-fr --out ${SCRATCH_DIR}/bcv-data/manifest-fr.json --name track_b_fr --track B --source hal --language-override fr --density-threshold 0.0 && echo FR_CORPUS_DONE"
}

cmd_corpus_pmc_lang() {
    # Build a native-language Track B corpus from PMC OA via NCBI EUtils.
    # Native literature, NOT translations (filtered by <Language>[Language]).
    # Usage: ./submit.sh corpus-lang <lang> [n_articles=60] [seed=0]
    local lang="${1:?lang required (zh, de, …)}" n="${2:-60}" seed="${3:-0}"
    local TS; TS=$(date +%H%M)
    # CPU-only (download + manifest); request 0 GPU to avoid queueing behind GPU jobs.
    submit_one "bcv-corpus-${lang}-${TS}" 0 \
        "bcv-corpus sample-pmc-lang --language ${lang} --n ${n} --seed ${seed} --out ${SCRATCH_DIR}/bcv-data/${lang}_pmcids.json && bcv-corpus download-pmc --pmcids-json ${SCRATCH_DIR}/bcv-data/${lang}_pmcids.json --cache-dir ${SCRATCH_DIR}/bcv-data/pmc-cache --out-dir ${SCRATCH_DIR}/bcv-data/pdfs-${lang} && bcv-corpus build-manifest ${SCRATCH_DIR}/bcv-data/pdfs-${lang} --out ${SCRATCH_DIR}/bcv-data/manifest-${lang}.json --name track_b_${lang} --track B --source pmc-oa --language-override ${lang} --density-threshold 0.0 && echo ${lang}_CORPUS_DONE"
}

cmd_corpus_zh() {
    # Usage: ./submit.sh corpus-zh [n_articles=60] [seed=0]
    cmd_corpus_pmc_lang zh "$@"
}

cmd_corpus_de() {
    # Usage: ./submit.sh corpus-de [n_articles=60] [seed=0]
    cmd_corpus_pmc_lang de "$@"
}

cmd_corpus_vidore() {
    # Build the Track A corpus from a ViDoRe v2 dataset (biomedical lectures):
    # download page images, reconstruct one PDF per document, emit a graded qrels
    # sidecar, then wire the Track A symlinks the orchestrator reads. CPU-only.
    # Usage: ./submit.sh corpus-vidore [repo=vidore/biomedical_lectures_eng_v2]
    local repo="${1:-vidore/biomedical_lectures_eng_v2}"
    local tag; tag=$(basename "${repo}")
    local build="${SCRATCH_DIR}/bcv-data/vidore-${tag}"
    local ta="${PROJECT_ROOT_AT}/data/track_a"
    local TS; TS=$(date +%H%M)
    # Single line: RunAI strips backslash continuations. rm -rf before ln -s so a
    # real directory left from a previous corpus does not swallow the symlink.
    submit_one "bcv-vidore-${TS}" 0 \
        "bcv-corpus build-vidore --out-dir ${build} --repo ${repo} && mkdir -p ${ta}/queries && rm -rf ${ta}/pdfs ${ta}/corpus_manifest.json ${ta}/queries/full.jsonl ${ta}/queries/full.qrels.json ${ta}/process ${ta}/milvus ${ta}/retrieve && ln -s ${build}/pdfs ${ta}/pdfs && ln -s ${build}/corpus_manifest.json ${ta}/corpus_manifest.json && ln -s ${build}/queries.jsonl ${ta}/queries/full.jsonl && ln -s ${build}/qrels.json ${ta}/queries/full.qrels.json && echo VIDORE_CORPUS_DONE && ls -l ${ta} ${ta}/queries"
}

cmd_corpus_pmc500() {
    # Build the Track A scale corpus: 500 PDFs × 3 seeds.
    # Usage: ./submit.sh corpus-pmc500 [seed]  (no arg = seeds 0 1 2)
    local seeds=("${@:-0 1 2}")
    [ $# -gt 0 ] && seeds=("$@")
    local TS; TS=$(date +%H%M)
    for seed in "${seeds[@]}"; do
        submit_one "bcv-pmc500-s${seed}-${TS}" 0 \
            "bcv-corpus sample-pmc --n 500 --seed ${seed} --packages-out ${SCRATCH_DIR}/bcv-data/pkgs500_seed${seed}.json && \
             bcv-corpus download-packages --packages-json ${SCRATCH_DIR}/bcv-data/pkgs500_seed${seed}.json \
               --cache-dir ${SCRATCH_DIR}/bcv-data/pmc-cache \
               --out-dir ${SCRATCH_DIR}/bcv-data/pdfs-500-seed${seed} && \
             bcv-corpus build-manifest ${SCRATCH_DIR}/bcv-data/pdfs-500-seed${seed} \
               --out ${SCRATCH_DIR}/bcv-data/manifest-500-seed${seed}.json \
               --name track_a_500_seed${seed} --track A --source pmc-oa && \
             echo PMC500_SEED${seed}_DONE"
    done
}

cmd_baseline() {
    # Text-retrieval baseline (mmore non-ColVision, BAAI/bge-m3) for one cell.
    # Page-level indexing so doc_ids align with the ColVision relevance judgments.
    # Usage: ./submit.sh baseline <track A|B> <lang|palier>
    #   B en|fr|zh|de   → Track B per-language corpus
    #   A tiny          → Track A corpus
    local track="${1:?track required (A|B)}" tag="${2:?lang/palier required}"
    local D="${SCRATCH_DIR}/bcv-data" R="${PROJECT_ROOT_AT}"
    local manifest corpus queryset workdir record
    if [ "${track}" = "B" ]; then
        case "${tag}" in
            en) manifest="${D}/manifest-tb-en.json"; corpus="${D}/pdfs-tb-en"; queryset="${R}/data/track_b/queries/en.jsonl" ;;
            fr) manifest="${D}/manifest-tb-fr.json"; corpus="${D}/pdfs-tb-fr"; queryset="${R}/data/track_b_fr/queries/fr.jsonl" ;;
            zh) manifest="${D}/manifest-tb-zh.json"; corpus="${D}/pdfs-tb-zh"; queryset="${R}/data/track_b_zh/queries/zh.jsonl" ;;
            de) manifest="${D}/manifest-tb-de.json"; corpus="${D}/pdfs-tb-de"; queryset="${R}/data/track_b_de/queries/de.jsonl" ;;
            es) manifest="${D}/manifest-tb-es.json"; corpus="${D}/pdfs-tb-es"; queryset="${R}/data/track_b_es/queries/es.jsonl" ;;
            *) echo "[baseline] unknown Track B lang: ${tag}" >&2; return 1 ;;
        esac
        workdir="${R}/data/baseline_text/B/${tag}"; record="${R}/results/baseline_text/B/${tag}/seed_0.json"
    else
        manifest="${D}/manifest-track-a-tiny.json"; corpus="${D}/pdfs-trackA-tiny"
        queryset="${R}/data/track_a/queries/tiny.jsonl"
        workdir="${R}/data/baseline_text/A/${tag}"; record="${R}/results/baseline_text/A/${tag}/seed_0.json"
    fi
    # HF_HUB_DISABLE_XET=1: bge-m3 ships pytorch_model.bin via HF's xet backend,
    # which hangs in this cluster's network — force the plain LFS HTTP download.
    submit_one "bcv-baseline-${track,,}-${tag}" 1 \
        "export HF_HUB_DISABLE_XET=1 && python -m benchmark_colvision.runners.text_baseline \
            --track ${track} --language ${tag} \
            --manifest ${manifest} --corpus-root ${corpus} --queryset ${queryset} \
            --workdir ${workdir} --record-out ${record} \
            --dense-model BAAI/bge-m3 --mmore-commit ${MMORE_REV} && echo BASELINE_${track}_${tag}_DONE"
}

cmd_ocr_vidore() {
    # OCR the ViDoRe Track A corpus with mmore's own `process` pipeline
    # (Marker + Surya, use_fast_processors=false so every page is OCR'd even
    # though it carries no embedded text layer — full-page images by
    # construction, see corpus.vidore_v2). Output feeds the text baselines via
    # --ocr-results (corpus.ocr_pages). CPU crawl + GPU OCR, ~4s/page.
    local D="${SCRATCH_DIR}/bcv-data/vidore-biomedical_lectures_eng_v2"
    local out="${D}/ocr_process"
    submit_one "bcv-ocr-vidore" 1 \
        "mkdir -p ${out} && cat > ${D}/process.yaml << EOF
data_path: ${D}/pdfs
google_drive_ids: []
previous_results: null
dispatcher_config:
  output_path: ${out}
  use_fast_processors: false
  distributed: false
  extract_images: false
  processor_config: {}
EOF
${BCV_VENV_MMOREPROCESS}/bin/python -m mmore.run_process --config_file ${D}/process.yaml && echo OCR_VIDORE_DONE && wc -l ${out}/merged/merged_results.jsonl"
}

cmd_baseline_vidore() {
    # Text-retrieval baselines (dense bge-m3 + native mmore hybrid) on the
    # ViDoRe Track A corpus. Unlike cmd_baseline, pages have no embedded text
    # layer, so both read from mmore's own OCR output (cmd_ocr_vidore) via
    # --ocr-results instead of --manifest/--corpus-root, and score against the
    # graded ViDoRe qrels (multi-relevant nDCG) instead of single-page relevance.
    # Usage: ./submit.sh baseline-vidore <dense|mmore>
    local which="${1:?dense|mmore required}"
    local D="${SCRATCH_DIR}/bcv-data/vidore-biomedical_lectures_eng_v2"
    local R="${PROJECT_ROOT_AT}"
    local ocr="${D}/ocr_process/merged/merged_results.jsonl"
    local queryset="${D}/queries.jsonl"
    local qrels="${D}/qrels.json"
    if [ "${which}" = "dense" ]; then
        local workdir="${R}/data/baseline_text/A/vidore" record="${R}/results/baseline_text/A/vidore/seed_0.json"
        submit_one "bcv-baseline-vidore-dense" 1 \
            "export HF_HUB_DISABLE_XET=1 && python -m benchmark_colvision.runners.text_baseline \
                --track A --language en \
                --ocr-results ${ocr} --qrels ${qrels} --queryset ${queryset} \
                --workdir ${workdir} --record-out ${record} \
                --dense-model BAAI/bge-m3 --mmore-commit ${MMORE_REV} && echo BASELINE_VIDORE_DENSE_DONE"
    elif [ "${which}" = "mmore" ]; then
        local workdir="${R}/data/mmore_text/A/vidore" record="${R}/results/mmore_text/A/vidore/seed_0.json"
        submit_one "bcv-baseline-vidore-mmore" 1 \
            "export PYTHONPATH=${R}/src && ${BCV_VENV_MMORETEXT}/bin/python -m benchmark_colvision.runners.mmore_text_baseline \
                --track A --language en \
                --ocr-results ${ocr} --qrels ${qrels} --queryset ${queryset} \
                --workdir ${workdir} --record-out ${record} \
                --mmore-commit ${MMORE_REV} && echo BASELINE_VIDORE_MMORE_DONE"
    else
        echo "[baseline-vidore] unknown baseline: ${which} (expected dense|mmore)" >&2
        return 1
    fi
}

cmd_corpus_vidore_lang() {
    # Build one query-language slice (queries.jsonl + qrels.json) of the ViDoRe
    # v2 MULTILINGUAL release (vidore/biomedical_lectures_v2) for Track B. Same
    # corpus-id space as the eng_v2 Track A build — no PDFs/manifest written,
    # just the translated queries + qrels. CPU-only.
    # Usage: ./submit.sh corpus-vidore-lang <english|french|german|spanish>
    local lang="${1:?language required (english|french|german|spanish)}"
    local out="${PROJECT_ROOT_AT}/data/track_b_vidore/${lang}"
    submit_one "bcv-vidore-lang-${lang}" 0 \
        "bcv-corpus build-vidore-lang --out-dir ${out} --language ${lang} && echo VIDORE_LANG_${lang}_DONE"
}

cmd_track_b_vidore() {
    # Re-retrieve translated queries on Track A's FIXED Milvus index (no
    # re-embedding: same corpus across languages, only queries are translated).
    # Track A must already have been run for the model (its
    # configs/mmore/cells/<model>/retrieve.yaml + Milvus DB must exist).
    # Usage: ./submit.sh track-b-vidore <model> <english|french|german|spanish> [seed=0]
    local model="${1:-}" lang="${2:-}" seed="${3:-0}"
    if [ -n "${model}" ] && [ -n "${lang}" ]; then
        local safe_model="${model//_/-}"
        local ts; ts=$(date +%H%M%S)
        submit_one "bcv-tbv-${safe_model}-${lang}-s${seed}-${ts}" 1 \
            "bcv-run track-b-vidore --model-id ${model} --language ${lang} --seed ${seed} \
                --models configs/models.yaml --mmore-commit ${MMORE_REV} \
                --base-dir ${PROJECT_ROOT_AT}"
    else
        local models=(colpali_v1_3 colqwen2_v1_0 colqwen2_5_v0_2 colgemma3_colnetra colsmol_256m colsmol_500m)
        local langs=(french german spanish)  # english == Track A, already scored
        for m in "${models[@]}"; do
            for l in "${langs[@]}"; do
                cmd_track_b_vidore "${m}" "${l}" 0
            done
        done
    fi
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
        # Job names must be lowercase RFC 1123 — collapse underscores. RunAI keeps
        # names reserved even after deletion, so suffix with a timestamp to avoid
        # "already exists" on re-runs.
        local safe_model="${model//_/-}"
        local ts; ts=$(date +%H%M%S)
        submit_one "bcv-ta-${safe_model}-s${seed}-${ts}" 1 \
            "bcv-run track-a --model-id ${model} --seed ${seed} \
                ${palier_flags} \
                --config configs/track_a.yaml --models configs/models.yaml \
                --mmore-commit ${MMORE_REV}"
    else
        # ViDoRe retrieval is deterministic (fixed queries + qrels) → seed 0 only.
        local models=(colpali_v1_3 colqwen2_v1_0 colqwen2_5_v0_2 colgemma3_colnetra colsmol_256m colsmol_500m)
        for m in "${models[@]}"; do
            cmd_track_a "${m}" 0
        done
    fi
}

cmd_track_b_native() {
    # Same cells as track-b, but scored against the NATIVE-language queries
    # (configs/track_b_<lang>_native.yaml). The pages are already processed and
    # indexed by the cross-lingual run, so only `retrieve` actually re-runs.
    # Usage: ./submit.sh track-b-native [model] [lang] [seed=0]
    local model="${1:-}" lang="${2:-}" seed="${3:-0}"
    if [ -n "${model}" ] && [ -n "${lang}" ]; then
        local safe_model="${model//_/-}"
        submit_one "bcv-tbnat-${safe_model}-${lang}-s${seed}" 1 \
            "bcv-run track-b --model-id ${model} --language ${lang} --seed ${seed} \
                --config configs/track_b_${lang}_native.yaml --models configs/models.yaml \
                --mmore-commit ${MMORE_REV}"
    else
        local models=(colqwen2_5_v0_2 colgemma3_colnetra colqwen2_v1_0 colpali_v1_3 colsmol_500m colsmol_256m)
        for l in fr zh de es; do
            for m in "${models[@]}"; do cmd_track_b_native "${m}" "${l}" 0; done
        done
    fi
}

cmd_track_b_serial() {
    # Repair path for track-b-native. Co-scheduling several cells of the same
    # language makes them read one Milvus index concurrently; the loser returns
    # nothing and the record lands with ndcg_at_5=null, or the cell hangs
    # outright. Here one job per language walks its models *in sequence*, and
    # each cell's retrieve output is purged first so a half-written directory
    # cannot be mistaken for a finished one.
    # The same repair applies to track-b-phi4, so the suite is a parameter.
    # Usage: ./submit.sh track-b-serial <native|phi4> <lang> <model> [model...]
    local suite="$1"; shift
    local lang="$1"; shift
    local seed=0 tag out_dir
    case "${suite}" in
        native) tag="tbnatser"; out_dir="retrieve_native" ;;
        phi4)   tag="tbphiser"; out_dir="retrieve_phi4"   ;;
        *) echo "suite inconnue: ${suite} (attendu: native|phi4)" >&2; return 1 ;;
    esac
    # English lives in data/track_b, the other languages in data/track_b_<lang>.
    local data_dir="data/track_b_${lang}"
    [ "${lang}" = "en" ] && data_dir="data/track_b"
    local steps=""
    for m in "$@"; do
        # Single line per step: RunAI strips backslash continuations.
        steps="${steps} rm -rf ${data_dir}/${out_dir}/${m} results/track_b_${suite}/${m}/${lang} ;"
        steps="${steps} bcv-run track-b --model-id ${m} --language ${lang} --seed ${seed} --config configs/track_b_${lang}_${suite}.yaml --models configs/models.yaml --mmore-commit ${MMORE_REV} ;"
    done
    steps="${steps} echo SERIAL_DONE_${suite}_${lang}"
    submit_one "bcv-${tag}-${lang}-s${seed}" 1 "${steps}"
}

cmd_track_b_phi4() {
    # Query-generator variance: the same cells as track-b, but scored against
    # queries written by microsoft/phi-4 instead of Qwen2.5-32B-Instruct. Phi-4
    # shares no backbone with any encoder under test, so the gap between the two
    # runs measures how much the generator's family matters.
    # Usage: ./submit.sh track-b-phi4 [model] [lang] [seed=0]
    local model="${1:-}" lang="${2:-}" seed="${3:-0}"
    if [ -n "${model}" ] && [ -n "${lang}" ]; then
        local safe_model="${model//_/-}"
        submit_one "bcv-tbphi4-${safe_model}-${lang}-s${seed}" 1 \
            "bcv-run track-b --model-id ${model} --language ${lang} --seed ${seed} \
                --config configs/track_b_${lang}_phi4.yaml --models configs/models.yaml \
                --mmore-commit ${MMORE_REV}"
    else
        local models=(colqwen2_5_v0_2 colgemma3_colnetra colqwen2_v1_0 colpali_v1_3 colsmol_500m colsmol_256m)
        for l in en zh; do
            for m in "${models[@]}"; do cmd_track_b_phi4 "${m}" "${l}" 0; done
        done
    fi
}

cmd_track_b() {
    local model="${1:-}" lang="${2:-}" seed="${3:-}"
    if [ -n "${model}" ] && [ -n "${lang}" ] && [ -n "${seed}" ]; then
        local safe_model="${model//_/-}"
        # Select language-specific config (EN uses the validated existing layout).
        local tb_config="configs/track_b_${lang}.yaml"
        [ -f "${tb_config}" ] || tb_config="configs/track_b.yaml"
        submit_one "bcv-tb-${safe_model}-${lang}-s${seed}" 1 \
            "bcv-run track-b --model-id ${model} --language ${lang} --seed ${seed} \
                --config ${tb_config} --models configs/models.yaml \
                --mmore-commit ${MMORE_REV}"
    else
        local models=(colpali_v1_3 colqwen2_v1_0 colqwen2_5_v0_2 colgemma3_colnetra colsmol_256m colsmol_500m)
        local langs=(en fr zh de)
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
    # Query generation is NOT part of this: it needs a vLLM server and is run
    # once per corpus, not per benchmark sweep.
    cmd_smoke
    cmd_track_a
    cmd_track_b
}

cmd_help() {
    cat <<HELP
Usage: $0 <action> [args]

Actions:
  smoke                              run one Track A cell (colpali, tiny, seed 0)
  serve-meditron                     start the long-lived vLLM judge (4 GPUs)
  gen-queries [manifest] [pdfs] [out] [model]  generate queries (3 modes) with Qwen-32B
  gen-tb <lang> <manifest> <pdfs> <out> [model] one Track B language: EN queries (cross-lingual)
  corpus-fr [n=100] [seed=0]         build French Track B corpus from HAL
  corpus-zh [n=60] [seed=0]          build Chinese Track B corpus from PMC OA
  corpus-de [n=60] [seed=0]          build German Track B corpus from PMC OA
  corpus-lang <lang> [n=60] [seed=0] build native-language Track B corpus from PMC OA
  corpus-vidore [repo]               build Track A corpus from ViDoRe v2 (biomedical lectures)
  ocr-vidore                          OCR the ViDoRe corpus (mmore process, Marker+Surya)
  baseline <A|B> <lang|palier>       text-retrieval baseline (bge-m3, page-level)
  baseline-vidore <dense|mmore>      text baseline on ViDoRe (OCR'd text + graded qrels)
  corpus-vidore-lang <lang>          build translated queries+qrels (ViDoRe v2 multilingual)
  track-b-vidore [model] [lang] [seed]  re-retrieve on Track A's index, or all 18 if no args
  track-a [model] [seed]             one Track A cell, or all 6 if no args
  track-b [model] [lang] [seed]      one Track B cell, or all 90 if no args
  all                                smoke + every Track A and Track B cell

Examples:
  $0 smoke
  $0 track-a colpali_v1_3 0
  $0 track-b colqwen2_5_v0_2 zh 0
  $0 all

Reads .rcp-env (created by scripts/rcp/setup.sh).
HELP
}

case "${ACTION}" in
    smoke)           cmd_smoke           ;;
    serve-meditron)  cmd_serve_meditron  ;;
    gen-queries)     cmd_gen_queries "$@" ;;
    gen-tb)          cmd_gen_tb "$@"     ;;
    gen-tb-native)   cmd_gen_tb_native "$@" ;;
    corpus-pmc500)   cmd_corpus_pmc500 "$@" ;;
    corpus-fr)       cmd_corpus_fr "$@"  ;;
    corpus-zh)       cmd_corpus_zh "$@"  ;;
    corpus-de)       cmd_corpus_de "$@"  ;;
    corpus-lang)     cmd_corpus_pmc_lang "$@" ;;
    corpus-vidore)   cmd_corpus_vidore "$@" ;;
    ocr-vidore)      cmd_ocr_vidore       ;;
    baseline)        cmd_baseline "$@"   ;;
    baseline-vidore) cmd_baseline_vidore "$@" ;;
    corpus-vidore-lang) cmd_corpus_vidore_lang "$@" ;;
    track-b-vidore)  cmd_track_b_vidore "$@" ;;
    track-a)         cmd_track_a "$@"    ;;
    track-b)         cmd_track_b "$@"    ;;
    track-b-native)  cmd_track_b_native "$@" ;;
    track-b-serial)  cmd_track_b_serial "$@" ;;
    track-b-phi4)    cmd_track_b_phi4 "$@" ;;
    all)             cmd_all             ;;
    help|--help|-h|*) cmd_help           ;;
esac
