#!/usr/bin/env bash
# Bootstrap the development environment using uv.
#
# Usage:
#   scripts/setup.sh            # local install (CPU)
#   scripts/setup.sh --gpu      # add CUDA wheels (only on a machine with NVIDIA driver)

set -euo pipefail

cd "$(dirname "$0")/.."

EXTRAS=("--all-extras" "--extra" "dev")
if [[ "${1:-}" == "--gpu" ]]; then
  EXTRAS=("--extra" "dev" "--extra" "gpu")
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install from https://docs.astral.sh/uv/." >&2
  exit 1
fi

uv sync "${EXTRAS[@]}"

if [[ -f .pre-commit-config.yaml ]]; then
  uv run pre-commit install
fi

echo "Done. Activate with: source .venv/bin/activate"
