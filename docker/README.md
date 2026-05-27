# Docker images for the benchmark

Two-layer build adapted from
[EPFLiGHT/LiGHT-cluster-template](https://github.com/EPFLiGHT/LiGHT-cluster-template):

- `Dockerfile` builds a **generic** image with the full Python environment
  (apt packages, `uv`-managed `.venv` containing all dependencies for the
  `gpu` and `vllm` extras). No user is baked in, so the image is portable
  across hosts. The benchmark *code* is not in this image — the entrypoint
  installs it at runtime from `PROJECT_ROOT_AT`.
- `Dockerfile.user` adds a host-matching Unix user on top of the generic
  image. This is required on platforms with NFS `root_squash` (such as EPFL
  RCP): a root container would be mapped to `nobody` and lose access to the
  mounted PVCs. Each user rebuilds this layer locally with their own
  `uid`/`gid`.

## Local build (no RCP)

```bash
# Generic image (~ 6–8 GB). Slow the first time, fast on rebuilds.
docker/build-generic.sh

# User layer (~ small). Defaults to your local uid/gid.
docker/build-user.sh
```

Output images (local Docker daemon):

| Tag                                         | What it is                        |
| ------------------------------------------- | --------------------------------- |
| `mmore-colvision-benchmark:generic-latest`  | environment only (no user)        |
| `mmore-colvision-benchmark:${USER}-latest`  | environment + user matching `id`  |

## Push to a registry (RCP and friends)

`scripts/rcp/setup.sh` does the auto-configured RCP build + push. For any
other registry:

```bash
IMAGE_NAME="ghcr.io/<you>/bcv" docker/build-generic.sh
IMAGE_NAME="ghcr.io/<you>/bcv" USR_TAG="user-latest" docker/build-user.sh
docker push ghcr.io/<you>/bcv:user-latest
```

## Run

The entrypoint expects the project to be mounted at `PROJECT_ROOT_AT`:

```bash
docker run --rm -it --gpus all \
    -v "$(pwd):/workspace" \
    -e PROJECT_ROOT_AT=/workspace \
    mmore-colvision-benchmark:${USER}-latest \
    bcv-run track-a --model-id colpali_v1_3 --seed 0 --palier tiny ...
```

See `scripts/docker/run_local.sh` for a wrapper.
