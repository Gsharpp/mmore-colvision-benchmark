# syntax=docker/dockerfile:1.7
#
# Generic (user-agnostic) image for the mmore-colvision benchmark.
#
# Layered design adapted from EPFLiGHT/LiGHT-cluster-template:
#   - This Dockerfile builds a *lean* runtime toolbox (apt + uv + a managed
#     Python). It deliberately does NOT bake the multi-GB Python dependency set
#     (torch, vllm, mmore...) — that lives in a venv created once on the cluster
#     scratch PVC by `scripts/rcp/bootstrap-venv.sh`. Keeping the deps out of
#     the image makes the push to Harbor small and resilient on a flaky link
#     (the heavy `uv sync` download then runs on the cluster's fast network).
#   - The benchmark *code* is likewise NOT copied in — it is mounted at runtime
#     from a PVC / bind-mount under PROJECT_ROOT_AT.
#   - A second Dockerfile (`Dockerfile.user`) layers a host-matching user on top
#     so NFS root_squash on RCP does not deny access to the mounted PVCs.
#
# Build context: repository root.
#   docker build -f docker/Dockerfile -t mmore-colvision-benchmark:generic-latest .

ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
ARG UV_VERSION=0.5.13
ARG PYTHON_VERSION=3.11

########################################################################
# Stage 1 — system packages.

FROM ${BASE_IMAGE} AS apt-base
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    --mount=type=cache,target=/var/lib/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        openssh-client \
        openssl \
        passwd \
        sudo \
        tini \
        wget \
    && rm -rf /var/lib/apt/lists/*

########################################################################
# Stage 2 — install uv and a managed Python interpreter (no project deps).
#
# The interpreter is baked at a stable, world-readable path so the scratch venv
# (created by bootstrap-venv.sh and shared by every job from this same image)
# can reference it. It must NOT live under /root: the user image runs as a
# non-root user (Dockerfile.user) and /root is mode 700, which would make the
# venv's python symlink unresolvable. No project dependencies are installed here.

FROM apt-base AS uv-base
ARG UV_VERSION
ARG PYTHON_VERSION

# uv discovers/installs managed Pythons here; inherited by all later stages.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    mv /root/.local/bin/uv  /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

RUN uv python install ${PYTHON_VERSION} && \
    chmod -R a+rX /opt/uv

########################################################################
# Stage 3 — generic runtime (no user, no deps). Small and pushable.

FROM uv-base AS runtime-generic

ENV UV_PYTHON=3.11 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV ENTRYPOINTS_ROOT=/opt/bcv-entrypoints
COPY docker/entrypoints ${ENTRYPOINTS_ROOT}
RUN chmod +x ${ENTRYPOINTS_ROOT}/*.sh

# tini reaps zombies and forwards signals — important for long jobs.
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/bcv-entrypoints/pre-entrypoint.sh"]
CMD ["bash"]
