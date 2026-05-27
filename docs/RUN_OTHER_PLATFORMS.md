# Lancer le benchmark hors EPFL RCP

L'image Docker du benchmark est volontairement portable (cf.
[`docker/README.md`](../docker/README.md)) : la couche générique n'embarque
aucun identifiant spécifique à EPFL/LiGHT, seulement l'environnement Python.
Quatre cibles de déploiement sont documentées ci-dessous.

## 1. Local — workstation avec GPU NVIDIA

Prérequis :
- Docker + `nvidia-container-toolkit`
- ~30 Go d'espace disque libre pour l'image générique

```bash
./scripts/docker/run_local.sh \
    bcv-run track-a --model-id colpali_v1_3 --seed 0 --palier tiny \
        --config configs/track_a.yaml --models configs/models.yaml \
        --mmore-commit $(uv run python -c \
            'import tomllib; print(tomllib.loads(open("pyproject.toml","rb").read().decode())["tool"]["uv"]["sources"]["mmore"]["rev"])')
```

Le script :

- build l'image à la volée si elle n'existe pas localement,
- monte le repo à `/workspace`,
- alloue tous les GPUs visibles (`--gpus all`),
- exécute la commande passée en argument.

Pour un shell interactif :

```bash
./scripts/docker/run_local.sh bash
```

## 2. Un autre cluster Kubernetes / Run:AI

`scripts/rcp/setup.sh` détecte automatiquement le projet / lab / registry à
partir de `runai list projects`. Sur un autre déploiement Run:AI il devrait
fonctionner tel quel à deux conditions :

1. Le namespace expose des PVCs nommés `home` et `*scratch*` (le script
   tombe sur le premier match).
2. Le registry est accessible via `docker login`.

`scripts/rcp/submit.sh` est plus EPFL-spécifique (point de montage
`/mloscratch`, env var `HF_HOME`). Adapter ces deux constantes en haut du
script si nécessaire.

## 3. SLURM (sans Docker — via Singularity / Apptainer)

Sur les clusters HPC traditionnels (CSCS, JSC, etc.) Docker n'est en général
pas autorisé : on passe par Singularity / Apptainer.

```bash
# Sur ta workstation, build et convertis l'image au format SIF
docker/build-generic.sh
docker/build-user.sh
singularity build bcv.sif docker-daemon://mmore-colvision-benchmark:user-latest

# Pousse le .sif vers le cluster
scp bcv.sif <user>@cluster:/scratch/<user>/

# Adapter les scripts/slurm/*.sbatch en remplaçant
#   source ".venv/bin/activate"
# par
#   singularity exec --nv --bind $PWD:/workspace --env PROJECT_ROOT_AT=/workspace bcv.sif <command>
```

Les scripts `scripts/slurm/{run_track_a,run_track_b,serve_meditron}.sbatch`
restent fournis comme point de départ.

## 4. Cloud (AWS / GCP / Azure)

Pousser l'image générique sur un registry cloud :

```bash
IMAGE_NAME="<aws-acct>.dkr.ecr.<region>.amazonaws.com/bcv" \
    docker/build-generic.sh
docker push <aws-acct>.dkr.ecr.<region>.amazonaws.com/bcv:generic-latest
```

Puis utiliser l'orchestrateur de votre choix (Batch, Kubernetes EKS,
SageMaker, etc.). Les commandes `bcv-run track-a ...` et `bcv-run track-b ...`
sont identiques à celles utilisées sur RCP.

## Image générique publique

(*futur* — pas encore publié)

Une fois validée par les auteurs, l'image générique sera publiée sur GHCR
pour que les utilisateurs hors EPFL puissent `docker pull` directement sans
rebuild :

```bash
docker pull ghcr.io/<owner>/mmore-colvision-benchmark:generic-latest
```
