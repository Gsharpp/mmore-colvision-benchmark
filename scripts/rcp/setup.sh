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
# 3. Detect the Run:AI project (used as the Kubernetes namespace prefix).
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
[ -n "${PROJECT}" ] || fail "could not parse runai project (got '${RUNAI_ROW}')"
log "Run:AI: project=${PROJECT}"

########################################################################
# 4. Construct the image identifier and verify registry access.
#
# EPFL RCP Harbor convention: each GASPAR user is "Project Admin" on a
# personal Harbor project named after their GASPAR username. Override via
# HARBOR_PROJECT=<name> if your lab uses a shared project instead.

HARBOR_PROJECT="${HARBOR_PROJECT:-${USR}}"
IMAGE_REPO="${REGISTRY}/${HARBOR_PROJECT}/bcv"
GENERIC_TAG="generic-latest"
USR_TAG="${USR}-latest"
IMAGE_USR="${IMAGE_REPO}:${USR_TAG}"
IMAGE_GEN="${IMAGE_REPO}:${GENERIC_TAG}"
log "Harbor project: ${HARBOR_PROJECT} (override with HARBOR_PROJECT=<name> if your lab uses a shared registry project)"
log "target image: ${IMAGE_USR}"

# Probe the registry credentials with a HEAD-only manifest inspect (cheap, <1s).
# `docker pull` here would re-download the full generic on every run (~10 min on VPN).
# A 401/403 means we are not logged in; a "not found" is expected on first build.
if ! docker manifest inspect "${IMAGE_GEN}" >/tmp/bcv-docker-probe.log 2>&1; then
    if grep -qiE "unauthorized|denied|authentication" /tmp/bcv-docker-probe.log; then
        warn "registry access denied — you are probably not logged in:"
        warn "    docker login ${REGISTRY} -u ${USR}"
        fail "abort. Run the docker login above, then re-run setup.sh."
    fi
    # "manifest unknown" / 404 is expected on first build — keep going.
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

push_with_diag() {
    local image="$1"
    local attempt=0
    local max_attempts=3
    while [ "${attempt}" -lt "${max_attempts}" ]; do
        attempt=$((attempt + 1))
        local tmplog
        tmplog="$(mktemp -t bcv-push.XXXXXX)"
        local rc=0
        # `script -q` preserves Docker's compact in-place progress bar.
        # -e propagates docker's exit code; </dev/null avoids input hang.
        if command -v script >/dev/null 2>&1; then
            script -q -e -c "docker push '${image}'" "${tmplog}" </dev/null
            rc=$?
        else
            docker push "${image}" 2>&1 | tee "${tmplog}"
            rc="${PIPESTATUS[0]}"
        fi
        if [ "${rc}" -eq 0 ]; then rm -f "${tmplog}"; return 0; fi
        # Classify the failure: network drop vs auth error.
        if grep -qiE "closed network connection|connection reset|use of closed|dial tcp|i/o timeout|EOF" "${tmplog}"; then
            warn "push attempt ${attempt}/${max_attempts} failed (network drop). Retrying..."
            rm -f "${tmplog}"
            sleep 5
            continue
        fi
        if grep -qiE "unauthorized|denied|401|403" "${tmplog}"; then
            warn ""
            warn "push denied to ${image}. Your Harbor project is probably not"
            warn "'${HARBOR_PROJECT}'. Check https://${REGISTRY}/ for the project"
            warn "name where you have Developer/Maintainer/Project Admin role, then re-run with:"
            warn "    HARBOR_PROJECT=<your-project-name> $0"
            rm -f "${tmplog}"
            return 1
        fi
        warn "push failed (attempt ${attempt}/${max_attempts}):"
        sed 's/^/    /' "${tmplog}" >&2
        rm -f "${tmplog}"
        return 1
    done
    warn "push failed after ${max_attempts} attempts (persistent network error)."
    return 1
}

log "building generic image (this can take ~10 min on first run)..."
IMAGE_NAME="${IMAGE_REPO}" GENERIC_TAG="${GENERIC_TAG}" \
    bash docker/build-generic.sh

# Skip the generic push when it was already pushed and Docker holds its
# RepoDigest (set after a successful push, stable across all-cache rebuilds).
# Force a re-push with FORCE_GENERIC_PUSH=1.
_generic_already_pushed() {
    [ "${FORCE_GENERIC_PUSH:-0}" = "1" ] && return 1
    local digests
    digests=$(docker image inspect "${IMAGE_GEN}" \
        --format '{{range .RepoDigests}}{{.}}{{"\n"}}{{end}}' 2>/dev/null || true)
    echo "${digests}" | grep -q "^${REGISTRY}/" || return 1
    docker manifest inspect "${IMAGE_GEN}" >/dev/null 2>&1 || return 1
    return 0
}

if _generic_already_pushed; then
    log "generic image already in registry (RepoDigest present) — skipping push."
    log "  use FORCE_GENERIC_PUSH=1 to override."
else
    log "pushing ${IMAGE_GEN}..."
    push_with_diag "${IMAGE_GEN}" || fail "docker push generic failed."
fi

log "building user image..."
IMAGE_NAME="${IMAGE_REPO}" GENERIC_TAG="${GENERIC_TAG}" USR_TAG="${USR_TAG}" \
    USR="${USR}" USRID="${USRID}" GRP="${GRP}" GRPID="${GRPID}" \
    bash docker/build-user.sh

log "pushing ${IMAGE_USR}..."
push_with_diag "${IMAGE_USR}" || fail "docker push user failed."

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
HARBOR_PROJECT=${HARBOR_PROJECT}
NAMESPACE=${NS}
IMAGE=${IMAGE_USR}
PVC_SCRATCH=${PVC_SCRATCH}
PVC_HOME=${PVC_HOME}
EOF
log "wrote .rcp-env"

########################################################################
# 8. Build the shared scratch venv on the cluster (one-time, idempotent).
#
# The image is lean (no Python deps), so the heavy `uv sync` runs here, on the
# cluster, into a venv on the scratch PVC — it never has to survive your VPN
# link. This needs the repo cloned on the scratch PVC already; if it isn't, the
# bootstrap job fails fast and we print the exact clone command below.

log "bootstrapping the scratch venv on the cluster (waits until ready)…"
BOOTSTRAP_RC=0
bash scripts/rcp/bootstrap-venv.sh --wait || BOOTSTRAP_RC=$?

if [ "${BOOTSTRAP_RC}" -eq 0 ]; then
    cat <<EOM

────────────────────────────────────────────────────────────────────────
[bcv-setup] DONE — image pushed and scratch venv ready. Run the benchmark:

    ./scripts/rcp/submit.sh smoke      # one cell, ~15 min, validates the chain
    ./scripts/rcp/submit.sh all        # full benchmark

Monitoring:

    runai list jobs
    runai logs bcv-smoke
    runai describe job bcv-smoke

Cleanup:

    ./scripts/rcp/teardown.sh
────────────────────────────────────────────────────────────────────────
EOM
else
    cat <<EOM

────────────────────────────────────────────────────────────────────────
[bcv-setup] Image is built and pushed, but the scratch venv is NOT confirmed
ready (bootstrap rc=${BOOTSTRAP_RC}).

Most likely cause: the repo isn't cloned on the scratch PVC yet. From HaaS:

    ssh ${USR}@haas001.rcp.epfl.ch
    mkdir -p /mnt/light/scratch/${USR}
    cd /mnt/light/scratch/${USR}
    git clone <repo-url> bcv-dev
    exit

Then build the venv (re-runnable, idempotent):

    ./scripts/rcp/bootstrap-venv.sh --wait

…and launch:

    ./scripts/rcp/submit.sh all

(If it just timed out, watch progress with: runai logs bcv-bootstrap -f)
────────────────────────────────────────────────────────────────────────
EOM
fi
