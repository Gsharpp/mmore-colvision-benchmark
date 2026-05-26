#!/bin/bash
# Smoke test on RCP: runs one Track A cell (ColPali on the `tiny` palier, seed 0)
# end-to-end through the real mmore CLI. Use this once after a fresh `uv sync`
# on RCP to validate the full pipeline before submitting the SLURM arrays.
#
# Prerequisites (must already exist on RCP):
#   - data/track_a/corpus_manifest.json (built by bcv-corpus download-pmc + manifest)
#   - data/track_a/queries/tiny.jsonl   (built by bcv-queries generate + filter)
#   - configs/mmore/{process,index,retrieve}.yaml (mmore-side YAML, written by hand)
#
# This script does NOT submit to SLURM — run it in an interactive job (`srun`)
# or on a dev node with a GPU.

set -euo pipefail

cd "$(dirname "$0")/.."

# Mirrors what the SLURM scripts do for the BenchmarkRecord provenance.
MMORE_COMMIT=$(python -c "import tomllib; d=tomllib.loads(open('pyproject.toml','rb').read().decode()); print(d['tool']['uv']['sources']['mmore']['rev'])")

echo "=== smoke_rcp.sh ==="
echo "  mmore_commit=${MMORE_COMMIT}"
echo "  cell:        track=A model=colpali_v1_3 palier=tiny seed=0"
echo ""

uv run bcv-run track-a \
  --model-id colpali_v1_3 \
  --seed 0 \
  --config configs/track_a.yaml \
  --models configs/models.yaml \
  --mmore-commit "${MMORE_COMMIT}" \
  --palier tiny

echo ""
echo "Expected output: results/track_a/colpali_v1_3/tiny/seed_0.json"
ls -l results/track_a/colpali_v1_3/tiny/seed_0.json
