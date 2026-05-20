#!/bin/bash
#SBATCH --job-name=bcv-track-a
#SBATCH --output=logs/track-a-%A_%a.out
#SBATCH --error=logs/track-a-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-14    # 5 models x 3 seeds = 15 jobs; per-palier loop inside

set -euo pipefail

# Activate the uv-managed venv prepared on the shared filesystem.
source "$HOME/mmore-colvision-benchmark/.venv/bin/activate"
cd "$HOME/mmore-colvision-benchmark"

MODELS=(colpali_v1_3 colqwen2_v1_0 colqwen2_5_v0_2 colqwen3_v0_1 colgemma3_colnetra)
SEEDS=(0 1 2)

idx=${SLURM_ARRAY_TASK_ID}
model_idx=$((idx / 3))
seed_idx=$((idx % 3))
MODEL="${MODELS[$model_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "Track A — model=${MODEL} seed=${SEED}"

bcv-run track-a \
  --model-id "${MODEL}" \
  --seed "${SEED}" \
  --config configs/track_a.yaml \
  --models configs/models.yaml \
  --output "results/track_a/${MODEL}/seed_${SEED}/"
