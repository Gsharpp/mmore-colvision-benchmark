#!/usr/bin/env bash
# Tear down benchmark jobs from Run:AI.
#
# Usage:
#   ./scripts/rcp/teardown.sh              # delete every bcv-* job
#   ./scripts/rcp/teardown.sh smoke        # delete only bcv-smoke
#   ./scripts/rcp/teardown.sh track-a      # delete only Track A jobs
#   ./scripts/rcp/teardown.sh track-b      # delete only Track B jobs
#   ./scripts/rcp/teardown.sh serve        # delete only the vLLM judge
set -euo pipefail

SCOPE="${1:-all}"

case "${SCOPE}" in
    all)      PREFIX_RE='^bcv-'           ;;
    smoke)    PREFIX_RE='^bcv-smoke'      ;;
    track-a)  PREFIX_RE='^bcv-ta-'        ;;
    track-b)  PREFIX_RE='^bcv-tb-'        ;;
    serve)    PREFIX_RE='^bcv-vllm'       ;;
    *)
        echo "[bcv-teardown] unknown scope '${SCOPE}'." >&2
        echo "Valid scopes: all, smoke, track-a, track-b, serve." >&2
        exit 1
        ;;
esac

jobs="$(
    runai list jobs --suppress-deprecation-message 2>/dev/null \
        | awk 'NR>1 && $1 != "NAME" {print $1}' \
        | grep -E "${PREFIX_RE}" \
        || true
)"

if [ -z "${jobs}" ]; then
    echo "[bcv-teardown] no matching jobs found for scope '${SCOPE}'."
    exit 0
fi

echo "[bcv-teardown] will delete:"
echo "${jobs}" | sed 's/^/  /'

for j in ${jobs}; do
    echo "[bcv-teardown] runai delete job ${j}"
    runai delete job "${j}" --suppress-deprecation-message 2>/dev/null || true
done
