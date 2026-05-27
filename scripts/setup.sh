#!/usr/bin/env bash
# Bootstrap the development environment using uv.
#
# Usage:
#   scripts/setup.sh            # local install (CPU)
#   scripts/setup.sh --gpu      # add CUDA wheels (machine with NVIDIA driver)
#   scripts/setup.sh --rcp      # --gpu + vLLM in a local venv + dir layout
#                                 (intended for SLURM / interactive GPU node)
#
# > For EPFL RCP (Run:AI / Kubernetes), this script is NOT the recommended path.
# > Use `scripts/rcp/setup.sh` instead, which builds + pushes a Docker image
# > and does not require uv sync on the cluster.
# > See docs/RCP_QUICKSTART.md.

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-local}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install from https://docs.astral.sh/uv/." >&2
  exit 1
fi

case "${MODE}" in
  --gpu)
    EXTRAS=("--extra" "dev" "--extra" "gpu")
    ;;
  --rcp)
    EXTRAS=("--extra" "dev" "--extra" "gpu")
    ;;
  local|--local|"")
    EXTRAS=("--all-extras" "--extra" "dev")
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: scripts/setup.sh [--gpu|--rcp]" >&2
    exit 2
    ;;
esac

# Best-effort CUDA module load on RCP / SCITAS. Failure is non-fatal; the user
# may have CUDA available system-wide or via conda.
if [[ "${MODE}" == "--rcp" ]] && command -v module >/dev/null 2>&1; then
  echo "[setup] attempting: module load cuda/12.6 (best-effort)"
  module load cuda/12.6 2>/dev/null || echo "[setup] cuda/12.6 not available — continuing"
fi

echo "[setup] uv sync ${EXTRAS[*]}"
uv sync "${EXTRAS[@]}"

if [[ "${MODE}" == "--rcp" ]]; then
  # vLLM + LangChain-OpenAI are RCP-only (multi-GB CUDA wheels, only useful
  # with a GPU). Installing here rather than as a pyproject extra keeps the
  # local install light.
  echo "[setup] uv pip install vllm langchain-openai"
  uv pip install vllm langchain-openai

  echo "[setup] creating directory layout under data/, results/, logs/, configs/mmore/"
  mkdir -p \
    data/track_a/{queries,retrieve,pdfs,cache} \
    data/track_b/{queries,retrieve,pdfs,cache} \
    results/track_a \
    results/track_b \
    logs \
    report/figures \
    report/tables \
    configs/mmore
fi

if [[ -f .pre-commit-config.yaml ]]; then
  uv run pre-commit install
fi

echo
echo "[setup] Done. Activate with: source .venv/bin/activate"

if [[ "${MODE}" == "--rcp" ]]; then
  cat <<'NEXT'

────────────────────────────────────────────────────────────────────────
[setup --rcp] Next steps (you'll need to do these by hand):

  1. Write the mmore-side configs (process / index / retrieve YAML):
       configs/mmore/process.yaml
       configs/mmore/index.yaml
       configs/mmore/retrieve.yaml
     Templates: https://github.com/Gsharpp/mmore (PR #305 branch)

  2. Start the vLLM judge in a long-lived SLURM job:
       sbatch scripts/slurm/serve_meditron.sbatch
     Note its hostname:port from the log, then:
       export VLLM_ENDPOINT="http://<host>:8000"

  3. Build the corpus for each track:
       bcv-corpus download-pmc PMC12345 ... --cache-dir data/track_a/cache \
                                             --out-dir   data/track_a/pdfs
       bcv-corpus build-manifest data/track_a/pdfs \
           --out data/track_a/corpus_manifest.json \
           --name track_a --track A --source pmc-oa \
           --language-override en

  4. Generate synthetic queries:
       bcv-queries generate \
           --corpus-manifest data/track_a/corpus_manifest.json \
           --corpus-root     data/track_a/pdfs \
           --out             data/track_a/queries/raw.jsonl \
           --vllm-endpoint   "$VLLM_ENDPOINT" \
           --vllm-model      epfl-llm/meditron-70b
       bcv-queries filter \
           --in-jsonl  data/track_a/queries/raw.jsonl \
           --out-jsonl data/track_a/queries/tiny.jsonl \
           --corpus-manifest data/track_a/corpus_manifest.json \
           --corpus-root     data/track_a/pdfs \
           --vllm-endpoint   "$VLLM_ENDPOINT" \
           --vllm-model      epfl-llm/meditron-70b

  5. Run a smoke cell:
       bash scripts/smoke_rcp.sh

  6. If green, submit the full arrays:
       sbatch scripts/slurm/run_track_a.sbatch
       sbatch scripts/slurm/run_track_b.sbatch

> For EPFL RCP / LiGHT specifically, prefer the Docker + Run:AI path:
>   ./scripts/rcp/setup.sh && ./scripts/rcp/submit.sh all
> See docs/RCP_QUICKSTART.md.
────────────────────────────────────────────────────────────────────────
NEXT
fi
