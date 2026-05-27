#!/bin/bash
# Pre-entrypoint — deduplicates per-task entrypoint executions under SLURM
# (Pyxis ntasks-per-node), then delegates to entrypoint.sh.
# Pattern lifted from EPFLiGHT/LiGHT-cluster-template.
set -eo pipefail

if [ -n "${SLURM_ONE_ENTRYPOINT_SCRIPT_PER_JOB:-}" ] && [ "${SLURM_PROCID:-0}" -gt 0 ]; then
  exec "$@"
fi
if [ -n "${SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE:-}" ] && [ "${SLURM_LOCALID:-0}" -gt 0 ]; then
  exec "$@"
fi

exec "${ENTRYPOINTS_ROOT:-/opt/bcv-entrypoints}/entrypoint.sh" "$@"
