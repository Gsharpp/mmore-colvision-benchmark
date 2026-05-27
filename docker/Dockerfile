# syntax=docker/dockerfile:1.7
#
# Generic (user-agnostic) image for the mmore-colvision benchmark.
#
# Layered design adapted from EPFLiGHT/LiGHT-cluster-template:
#   - This Dockerfile builds the runtime *environment* (apt + uv + Python deps).
#   - The benchmark *code* is NOT copied in — it is mounted at runtime from a
#     PVC / bind-mount under PROJECT_ROOT_AT, and installed editable by the
#     entrypoint. This decouples image rebuilds from code iteration.
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
# Stage 2 — install uv and resolve project dependencies (no project install).

FROM apt-base AS uv-deps
ARG UV_VERSION
ARG PYTHON_VERSION

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    mv /root/.local/bin/uv  /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

RUN uv python install ${PYTHON_VERSION}

WORKDIR /opt/bcv

# uv needs pyproject.toml + uv.lock to resolve, and README.md is referenced
# from pyproject `readme = "README.md"`.
COPY pyproject.toml uv.lock README.md ./

# --no-install-project installs all dependencies but not the package itself.
# The package will be installed editable at runtime from PROJECT_ROOT_AT.
ENV UV_PROJECT_ENVIRONMENT=/opt/bcv/.venv \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra gpu

# vLLM + langchain-openai are installed unfrozen on top: they pin
# torchvision/triton tighter than mmore[cu126] and would otherwise force
# lockfile downgrades. We accept the looser resolution because they are only
# used to run the local LLM judge service.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/bcv/.venv/bin/python \
        "vllm>=0.6" "langchain-openai>=0.2"

########################################################################
# Stage 3 — generic runtime (no user). Pushable to a public registry.

FROM uv-deps AS runtime-generic

ENV VIRTUAL_ENV=/opt/bcv/.venv \
    PATH=/opt/bcv/.venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV ENTRYPOINTS_ROOT=/opt/bcv-entrypoints
COPY docker/entrypoints ${ENTRYPOINTS_ROOT}
RUN chmod +x ${ENTRYPOINTS_ROOT}/*.sh

# tini reaps zombies and forwards signals — important for long jobs.
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/bcv-entrypoints/pre-entrypoint.sh"]
CMD ["bash"]
