# Docker images for the benchmark

Two-layer build adapted from
[EPFLiGHT/LiGHT-cluster-template](https://github.com/EPFLiGHT/LiGHT-cluster-template):

- `Dockerfile` builds a **lean generic** image: apt packages, `uv`, and a
  uv-managed Python interpreter — but **no** Python dependencies. The multi-GB
  dependency set (torch, vllm, mmore...) is deliberately kept out so the push
  to Harbor stays small and resilient on a flaky link. Those deps are installed
  once on the cluster into a venv on the scratch PVC (`scripts/rcp/bootstrap-venv.sh`),
  and every job activates that shared venv via `BCV_VENV`. The benchmark *code*
  is also not in the image — it is mounted at runtime from `PROJECT_ROOT_AT`.
- `Dockerfile.user` adds a host-matching Unix user on top of the generic
  image. This is required on platforms with NFS `root_squash` (such as EPFL
  RCP): a root container would be mapped to `nobody` and lose access to the
  mounted PVCs. Each user rebuilds this layer locally with their own
  `uid`/`gid`.

## Local build (no RCP)

```bash
# Lean generic image (~ 3 GB, mostly the CUDA base). No Python deps.
docker/build-generic.sh

# User layer (~ small). Defaults to your local uid/gid.
docker/build-user.sh
```

Output images (local Docker daemon):

| Tag                                         | What it is                          |
| ------------------------------------------- | ----------------------------------- |
| `mmore-colvision-benchmark:generic-latest`  | toolbox only (apt + uv, no deps)    |
| `mmore-colvision-benchmark:${USER}-latest`  | toolbox + user matching `id`        |

## Push to a registry (RCP and friends)

`scripts/rcp/setup.sh` does the auto-configured RCP build + push. For any
other registry:

```bash
IMAGE_NAME="ghcr.io/<you>/bcv" docker/build-generic.sh
IMAGE_NAME="ghcr.io/<you>/bcv" USR_TAG="user-latest" docker/build-user.sh
docker push ghcr.io/<you>/bcv:user-latest
```

## Run

The image is lean, so the entrypoint expects **two** mounts: the project at
`PROJECT_ROOT_AT`, and a venv (with deps installed) at `BCV_VENV`. On RCP that
venv is built by `scripts/rcp/bootstrap-venv.sh`. Locally, build it once *inside*
the container so the interpreter paths match the mount point, then reuse it:

```bash
mkdir -p .venv-docker          # host dir backing the venv, mounted at /venv

# one-time: populate the venv (deps + editable project) at /venv
docker run --rm -it --gpus all \
    -v "$(pwd):/workspace" -v "$(pwd)/.venv-docker:/venv" \
    -e PROJECT_ROOT_AT=/workspace -e BCV_VENV=/venv \
    mmore-colvision-benchmark:${USER}-latest \
    bash -lc 'cd /workspace && UV_PROJECT_ENVIRONMENT=/venv uv sync --frozen --extra gpu \
              && UV_PROJECT_ENVIRONMENT=/venv uv pip install "vllm>=0.6" "langchain-openai>=0.2"'

# then run as many jobs as you like against that venv
docker run --rm -it --gpus all \
    -v "$(pwd):/workspace" -v "$(pwd)/.venv-docker:/venv" \
    -e PROJECT_ROOT_AT=/workspace -e BCV_VENV=/venv \
    mmore-colvision-benchmark:${USER}-latest \
    bcv-run track-a --model-id colpali_v1_3 --seed 0 --palier tiny ...
```

See `scripts/docker/run_local.sh` for a wrapper.
