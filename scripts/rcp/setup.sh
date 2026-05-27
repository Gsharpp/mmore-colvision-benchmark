#!/usr/bin/env bash
# Auto-configuring one-shot setup for EPFL RCP / LiGHT lab.
#
# Detects the caller's GASPAR identity and Run:AI project, builds the
# benchmark image (generic + user layers), pushes it to the RCP registry,
# and writes a `.rcp-env` consumed by `scripts/rcp/submit.sh`.
#
# Usage:
#   ./scripts/rcp/setup.sh
#
# Prerequisites (one-time, see docs/RCP_SETUP.md):
#   - Connected to EPFL VPN
#   - kubectl + runai CLIs installed
#   - `runai login` completed (token in ~/.kube/config)
#   - `docker login registry.rcp.epfl.ch -u <gaspar>` completed
set -euo pipefail
cd "$(dirname "$0")/../.."

log()  { echo "[bcv-setup] $*"; }
warn() { echo "[bcv-setup][WARN] $*" >&2; }
fail() { echo "[bcv-setup][ERROR] $*" >&2; exit 1; }

REGISTRY="registry.rcp.epfl.ch"

########################################################################
# 1. Sanity-check the toolchain.

log "checking tools..."
for tool in docker runai kubectl; do
    command -v "${tool}" >/dev/null || fail "${tool} not found in PATH. See docs/RCP_SETUP.md."
done
docker info >/dev/null 2>&1 || fail "docker daemon not reachable."

########################################################################
# 2. Detect the user identity to bake into the image.
#
# IMPORTANT: this must be the *RCP* uid/gid (so the container can write to
# light-scratch PVCs through NFS root_squash), NOT the local WSL/laptop one.
# Default to `id` but warn loudly if uid<1000 looks like a local-only account
# (RCP uids are 6-digit). Override via env: USR=, USRID=, GRP=, GRPID=.

USR="${USR:-$(id -un)}"
USRID="${USRID:-$(id -u)}"
GRPID="${GRPID:-$(id -g)}"
GRP="${GRP:-$(id -gn)}"

# Probe HaaS to get the right ids if local ones look local-only and ssh is set up.
if [ "${USRID}" -lt 100000 ]; then
    warn "host uid=${USRID} looks like a local account, not an EPFL GASPAR uid"
    warn "(RCP NFS PVCs are owned by your GASPAR uid in the 300000+ range)"
    warn "an image built with uid=${USRID} will fail to read /mloscratch on RCP"
    warn ""
    warn "options:"
    warn "  1. abort, then re-run as:"
    warn "       USR=<gaspar> USRID=<rcp-uid> GRP=<rcp-group> GRPID=<rcp-gid> $0"
    warn "  2. abort, then fetch your ids from HaaS:"
    warn "       ssh <gaspar>@haas001.rcp.epfl.ch id"
    warn "     and re-run with those values"
    warn ""
    read -r -p "[bcv-setup] continue anyway with local uid? [y/N] " ans
    case "${ans}" in
        y|Y|yes|YES) log "continuing with local uid (image won't work on RCP NFS)" ;;
        *) fail "aborted — re-run with the override env vars above" ;;
    esac
fi

log "user identity: USR=${USR} USRID=${USRID} GRP=${GRP} GRPID=${GRPID}"

########################################################################
# 3. Detect the Run:AI project + lab (department).
#
# `runai list projects` emits a header line then one row per project the user
# can submit to. We take the first project row.

log "detecting Run:AI project..."
RUNAI_OUT="$(mktemp -t bcv-runai.XXXXXX)"
trap 'rm -f "${RUNAI_OUT}"' EXIT
if ! runai list projects --suppress-deprecation-message > "${RUNAI_OUT}" 2>&1; then
    warn "runai list projects failed. Output:"
    sed 's/^/    /' "${RUNAI_OUT}" >&2
    fail "could not enumerate runai projects — try \`runai login\` and re-run."
fi

# Skip header + deprecation warning lines; first row matching PROJECT layout wins.
RUNAI_ROW="$(awk 'NR>1 && NF>0 && $1 != "PROJECT" && $1 != "CLI" && $1 !~ /^However|^Can|^Contact|^=/ {print; exit}' "${RUNAI_OUT}")"
if [ -z "${RUNAI_ROW}" ]; then
    warn "could not find a project row in runai output:"
    sed 's/^/    /' "${RUNAI_OUT}" >&2
    fail "no Run:AI project visible — does your user have one provisioned?"
fi

PROJECT="$(echo "${RUNAI_ROW}" | awk '{print $1}')"
LAB="${LAB:-$(echo "${RUNAI_ROW}" | awk '{print $2}')}"
[ -n "${PROJECT}" ] || fail "could not parse runai project (got '${RUNAI_ROW}')"

# Fall back to deriving the lab from the project name (LiGHT convention:
# <lab>-<user>) when the DEPARTMENT column reports "(default)" or anything
# that wouldn't be a valid Docker image path component.
if [ -z "${LAB}" ] || ! echo "${LAB}" | grep -qE '^[a-z0-9][a-z0-9._-]*$'; then
    derived="${PROJECT%%-*}"
    warn "DEPARTMENT column reported '${LAB}' which is not a valid Docker path"
    warn "deriving lab='${derived}' from project name '${PROJECT}'"
    warn "override with: LAB=<your-lab> $0"
    LAB="${derived}"
fi
log "Run:AI: project=${PROJECT} lab=${LAB}"

########################################################################
# 4. Construct the image identifier and verify registry access.

IMAGE_REPO="${REGISTRY}/${LAB}/${USR}/bcv"
GENERIC_TAG="generic-latest"
USR_TAG="${USR}-latest"
IMAGE_USR="${IMAGE_REPO}:${USR_TAG}"
IMAGE_GEN="${IMAGE_REPO}:${GENERIC_TAG}"
log "target image: ${IMAGE_USR}"

# Probe the registry credentials. A 401/403 means we are not logged in.
if ! docker pull "${IMAGE_GEN}" >/tmp/bcv-docker-probe.log 2>&1; then
    if grep -qiE "unauthorized|denied|authentication" /tmp/bcv-docker-probe.log; then
        warn "docker pull denied — you are probably not logged in:"
        warn "    docker login ${REGISTRY} -u ${USR}"
        fail "abort. Run the docker login above, then re-run setup.sh."
    fi
    # 404 / not found is expected on first build — keep going.
fi

########################################################################
# 5. Discover the PVCs in the user's namespace.

NS="runai-${PROJECT}"
log "checking PVCs in namespace ${NS}..."
PVC_LIST="$(kubectl get pvc -n "${NS}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)"
if [ -z "${PVC_LIST}" ]; then
    warn "no PVCs found in ${NS}. Submitted jobs will not have persistent storage."
    PVC_SCRATCH=""
    PVC_HOME=""
else
    # Conventional names on RCP-LiGHT; fall back to the first PVC we find.
    PVC_SCRATCH="$(echo "${PVC_LIST}" | awk '/scratch/{print; exit}')"
    PVC_HOME="$(echo    "${PVC_LIST}" | awk '/home/{print; exit}')"
    [ -n "${PVC_SCRATCH}" ] || PVC_SCRATCH="$(echo "${PVC_LIST}" | head -n1)"
    log "PVCs: scratch='${PVC_SCRATCH}' home='${PVC_HOME}'"
fi

########################################################################
# 6. Build generic + user images and push.

log "building generic image (this can take ~10 min on first run)..."
IMAGE_NAME="${IMAGE_REPO}" GENERIC_TAG="${GENERIC_TAG}" \
    bash docker/build-generic.sh

log "building user image..."
IMAGE_NAME="${IMAGE_REPO}" GENERIC_TAG="${GENERIC_TAG}" USR_TAG="${USR_TAG}" \
    USR="${USR}" USRID="${USRID}" GRP="${GRP}" GRPID="${GRPID}" \
    bash docker/build-user.sh

log "pushing ${IMAGE_USR}..."
docker push "${IMAGE_USR}" || fail "docker push failed. Logs above."

########################################################################
# 7. Persist detected values for submit.sh.

cat > .rcp-env <<EOF
# Auto-generated by scripts/rcp/setup.sh on $(date -Iseconds).
# Do NOT commit this file (it's in .gitignore).
USR=${USR}
USRID=${USRID}
GRP=${GRP}
GRPID=${GRPID}
PROJECT=${PROJECT}
LAB=${LAB}
NAMESPACE=${NS}
IMAGE=${IMAGE_USR}
PVC_SCRATCH=${PVC_SCRATCH}
PVC_HOME=${PVC_HOME}
EOF
log "wrote .rcp-env"

########################################################################
# 8. Help the user prep the runtime PROJECT_ROOT_AT on the scratch PVC.
#
# The benchmark code is mounted from the scratch PVC at runtime. The user
# must clone the repo there once. We can't do it for them (this script may
# run on a workstation that doesn't have the PVC mounted), but we print the
# exact commands.

cat <<EOM

────────────────────────────────────────────────────────────────────────
[bcv-setup] DONE. Next steps:

1. (one-time) clone the benchmark on your light-scratch PVC. From HaaS:

    ssh ${USR}@haas001.rcp.epfl.ch
    mkdir -p /mnt/light/scratch/${USR}
    cd /mnt/light/scratch/${USR}
    git clone <repo-url> bcv-dev

2. Run a smoke job to validate the full pipeline (~15 min):

    ./scripts/rcp/submit.sh smoke

3. Once green, submit the full benchmark:

    ./scripts/rcp/submit.sh all

Monitoring:

    runai list jobs
    runai logs bcv-smoke
    runai describe job bcv-smoke

Cleanup:

    ./scripts/rcp/teardown.sh
────────────────────────────────────────────────────────────────────────
EOM
