#!/bin/bash
#SBATCH --job-name=bcv-track-b
#SBATCH --output=logs/track-b-%A_%a.out
#SBATCH --error=logs/track-b-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-89    # 5 models x 6 langs x 3 seeds = 90 jobs

set -euo pipefail

source "$HOME/mmore-colvision-benchmark/.venv/bin/activate"
cd "$HOME/mmore-colvision-benchmark"
mkdir -p logs

MODELS=(colpali_v1_3 colqwen2_v1_0 colqwen2_5_v0_2 colqwen3_v0_1 colgemma3_colnetra)
LANGS=(en fr de es zh ar)
SEEDS=(0 1 2)

idx=${SLURM_ARRAY_TASK_ID}
seed_idx=$((idx % 3))
lang_idx=$(( (idx / 3) % 6 ))
model_idx=$(( idx / 18 ))

MODEL="${MODELS[$model_idx]}"
LANG="${LANGS[$lang_idx]}"
SEED="${SEEDS[$seed_idx]}"

MMORE_COMMIT=$(python -c "import tomllib; d=tomllib.loads(open('pyproject.toml','rb').read().decode()); print(d['tool']['uv']['sources']['mmore']['rev'])")

echo "Track B — model=${MODEL} lang=${LANG} seed=${SEED} mmore=${MMORE_COMMIT}"

bcv-run track-b \
  --model-id "${MODEL}" \
  --language "${LANG}" \
  --seed "${SEED}" \
  --config configs/track_b.yaml \
  --models configs/models.yaml \
  --mmore-commit "${MMORE_COMMIT}"
