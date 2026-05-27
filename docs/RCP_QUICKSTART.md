# Quickstart — lancer le benchmark sur EPFL RCP

Trois commandes du clone à la fin du benchmark.

## Prérequis (une fois, par utilisateur)

- Compte GASPAR avec accès LiGHT (demander au responsable du labo)
- Sur le poste de travail local (Linux / WSL2 / macOS) :
  - VPN EPFL connecté
  - `kubectl` v1.29+ (Linux amd64)
  - `runai` CLI ([docs](https://docs.run.ai/v2.20/Researcher/cli-reference/Introduction/))
  - Docker (avec `nvidia-container-toolkit` si on veut tester en local)
  - Authentification :
    ```bash
    runai login                                                      # OIDC navigateur
    docker login registry.rcp.epfl.ch -u <gaspar>                    # mot de passe GASPAR
    ```

Si une étape coince, le détail complet de la mise en place est dans
[`RCP_SETUP.md`](RCP_SETUP.md).

## 1. Cloner le repo sur le scratch LiGHT

À faire une seule fois, depuis HaaS :

```bash
ssh <gaspar>@haas001.rcp.epfl.ch
mkdir -p /mnt/light/scratch/$USER
cd /mnt/light/scratch/$USER
git clone <repo-url> bcv-dev
exit
```

`bcv-dev` est le checkout vivant — toute modif `git pull` est immédiatement
visible des jobs suivants, sans rebuild d'image.

## 2. Setup local (build + push de l'image)

Depuis le clone local (où `kubectl`/`runai`/`docker` sont configurés) :

```bash
cd <repo>
./scripts/rcp/setup.sh
```

Ce que fait `setup.sh` automatiquement :

1. détecte ton uid/gid (`id`)
2. détecte ton projet Run:AI et ton lab (`runai list projects`)
3. découvre les PVCs `home` et `light-scratch` (`kubectl get pvc`)
4. build l'image générique (`docker/Dockerfile`)
5. build l'image user avec tes uid/gid (`docker/Dockerfile.user`)
6. push vers `registry.rcp.epfl.ch/<lab>/<user>/bcv:<user>-latest`
7. écrit `.rcp-env` (ignoré par git)

Compter ~10 min la première fois (download de la base CUDA + sync uv). Les
rebuilds sont rapides grâce au cache BuildKit.

## 3. Lancer

**Smoke test** (1 cell, ~15 min) pour valider la chaîne complète :

```bash
./scripts/rcp/submit.sh smoke
runai logs bcv-smoke -f         # streamer la sortie
```

**Benchmark complet** (~24 h selon disponibilité GPU) :

```bash
./scripts/rcp/submit.sh all
```

Ça soumet en cascade :

- 1 job `bcv-vllm` (4 GPUs, juge Meditron-70B en service long)
- 1 job `bcv-smoke`
- 15 jobs `bcv-ta-*` (Track A : 5 modèles × 3 seeds, paliers traités à l'intérieur)
- 90 jobs `bcv-tb-*` (Track B : 5 modèles × 6 langues × 3 seeds)

**Soumettre une cellule isolée** :

```bash
./scripts/rcp/submit.sh track-a colpali_v1_3 0
./scripts/rcp/submit.sh track-b colqwen3_v0_1 ar 2
```

## Suivi

```bash
runai list jobs
runai logs <job-name> [-f]
runai describe job <job-name>
runai exec -it <job-name> bash   # shell dans le pod
```

## Cleanup

```bash
./scripts/rcp/teardown.sh             # tous les jobs bcv-*
./scripts/rcp/teardown.sh smoke       # juste le smoke
./scripts/rcp/teardown.sh serve       # juste le vLLM judge
```

## Résultats

Les `BenchmarkRecord` JSON et logs sont écrits dans le scratch LiGHT, à
`/mnt/light/scratch/<user>/bcv-dev/results/`. Pour les rapatrier en local :

```bash
scp -r <gaspar>@haas001.rcp.epfl.ch:/mnt/light/scratch/<user>/bcv-dev/results ./results
```

## En cas d'échec d'une étape

- `setup.sh` échoue sur `docker login` → fais-le manuellement puis relance.
- `setup.sh` échoue sur `runai list projects` → relance `runai login`.
- Un job reste `Pending` longtemps → quotas Run:AI saturés (`runai list nodes`)
  ou un autre user occupe les GPUs du département.
- `runai logs <job>` montre `ImagePullBackOff` → l'image n'est pas pushée ou
  les credentials registry du namespace sont expirées.

Voir [`RCP_SETUP.md`](RCP_SETUP.md) pour les recettes de debug détaillées.

## Lancer hors RCP

Voir [`RUN_OTHER_PLATFORMS.md`](RUN_OTHER_PLATFORMS.md).
