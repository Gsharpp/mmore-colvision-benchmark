#!/usr/bin/env bash
# Propager les fichiers modifiés sur le cluster via b64 (à lancer depuis la racine du repo,
# après "source .rcp-env"). Les chunks b64 sont générés DYNAMIQUEMENT depuis les fichiers
# courants — pas de blob codé en dur qui pourrait se périmer.
#
# Fichiers inclus :
#   pyproject.toml                  — pin mmore 4102f96 + extra colvision (tf 5.3.0)
#   uv.lock                         — lockfile aligné (colpali-engine 0.3.16, transformers 5.3.0)
#   configs/models.yaml             — 6 modèles (ColQwen3 retiré, ColSmol 256M/500M)
#   runners/orchestrate.py          — Track B per-(model,lang) config rendering + data_path
#   clients/vllm_client.py          — guided_json parameter
#   queries/inverse_query_gen.py    — few-shot + GUIDED_JSON_SCHEMA
#   queries/cli.py                  — --few-shot / --guided-json flags
#   corpus/cli.py                   — --pmcids-json on download-pmc
#   scripts/rcp/submit.sh           — corpus-fr / corpus-zh + --few-shot --guided-json
#   corpus/hal_downloader.py        — connecteur HAL (sort stable + seeded offset)
#   corpus/pmc_zh_sampler.py        — connecteur PMC ZH
#   corpus/ocr_pages.py             — page text from mmore `process` OCR (ViDoRe baselines)
#   queries/pdf_pages.py            — page_base param (1-based ids, ViDoRe qrels alignment)
#   corpus/vidore_v2.py             — + multilingual query/qrels builder (Track B ViDoRe)
#   runners/run_track_b_vidore.py   — Track B ViDoRe: re-retrieve on Track A's FIXED index
#   runners/cli.py                  — track-b-vidore command
# Après ce patch, re-bootstrapper le venv cluster :
#   FORCE_BOOTSTRAP=1 ./scripts/rcp/bootstrap-venv.sh --wait
set -euo pipefail

[ -f .rcp-env ] || { echo "Lancer depuis la racine du repo (où .rcp-env existe)." >&2; exit 1; }
source .rcp-env

FILES=(
    pyproject.toml
    uv.lock
    src/benchmark_colvision/runners/mmore_wrapper.py
    src/benchmark_colvision/runners/orchestrate.py
    src/benchmark_colvision/runners/run_track_a.py
    src/benchmark_colvision/runners/text_baseline.py
    src/benchmark_colvision/runners/mmore_text_baseline.py
    src/benchmark_colvision/clients/vllm_client.py
    src/benchmark_colvision/queries/inverse_query_gen.py
    src/benchmark_colvision/queries/cli.py
    src/benchmark_colvision/queries/pdf_pages.py
    src/benchmark_colvision/queries/schema.py
    src/benchmark_colvision/corpus/cli.py
    src/benchmark_colvision/corpus/corpus_manifest.py
    src/benchmark_colvision/corpus/vidore_v2.py
    src/benchmark_colvision/corpus/ocr_pages.py
    src/benchmark_colvision/runners/run_track_b_vidore.py
    src/benchmark_colvision/runners/cli.py
    scripts/rcp/submit.sh
    scripts/rcp/materialize_tb_lang.py
    src/benchmark_colvision/corpus/hal_downloader.py
    src/benchmark_colvision/corpus/pmc_zh_sampler.py
    configs/models.yaml
    configs/track_a.yaml
    configs/track_b_en.yaml
    configs/track_b_fr.yaml
    configs/track_b_zh.yaml
    configs/track_b_de.yaml
    configs/track_b_es.yaml
)

for f in "${FILES[@]}"; do
    [ -f "$f" ] || { echo "[patch][ERROR] fichier manquant : $f" >&2; exit 1; }
done

# Archive déterministe → base64 sur une seule ligne.
TMP_TGZ=$(mktemp /tmp/bcv-patch.XXXXXX.tar.gz)
trap 'rm -f "${TMP_TGZ}"' EXIT
tar -czf "${TMP_TGZ}" "${FILES[@]}"
B64=$(base64 -w0 "${TMP_TGZ}")
NBYTES=$(wc -c < "${TMP_TGZ}")
echo "[patch] archive ${NBYTES} bytes → ${#B64} chars b64"

# Découpe en chunks de 2700 chars (sous la limite d'une var d'env k8s).
CHUNK=2700
ENV_ARGS=()
CAT_CMD=""
i=0
off=0
while [ "${off}" -lt "${#B64}" ]; do
    i=$((i + 1))
    part="${B64:${off}:${CHUNK}}"
    ENV_ARGS+=(-e "P${i}=${part}")
    if [ "${i}" -eq 1 ]; then
        CAT_CMD="printf \"%s\" \"\${P${i}}\" > /tmp/p.b64"
    else
        CAT_CMD="${CAT_CMD} && printf \"%s\" \"\${P${i}}\" >> /tmp/p.b64"
    fi
    off=$((off + CHUNK))
done
echo "[patch] ${i} chunks"

REMOTE_CMD="${CAT_CMD} && base64 -d /tmp/p.b64 > /tmp/patch.tar.gz && tar -xzf /tmp/patch.tar.gz -C /mloscratch/users/mbonnet/bcv-dev && echo PATCH_OK"

NAME="bcv-patch-$(date +%H%M%S)"
echo "[patch] Soumission ${NAME} — ${#FILES[@]} fichiers, ${i} chunks b64"
runai submit "${NAME}" \
  --image "${IMAGE}" \
  --gpu 0 \
  --existing-pvc "claimname=${PVC_SCRATCH},path=/mloscratch" \
  -e PROJECT_ROOT_AT=/mloscratch/users/mbonnet/bcv-dev \
  -e PACKAGE_NAME=benchmark_colvision \
  -e BCV_VENV=/mloscratch/users/mbonnet/bcv-venv \
  -p light-mbonnet \
  --suppress-deprecation-message \
  "${ENV_ARGS[@]}" \
  -- bash -c "${REMOTE_CMD}"
echo "[patch] Soumis — vérifier avec : runai logs ${NAME} -p light-mbonnet --suppress-deprecation-message"
