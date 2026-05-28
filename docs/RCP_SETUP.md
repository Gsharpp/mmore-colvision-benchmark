# Connexion EPFL RCP (LiGHT lab) — installation des outils

Procédure pour mettre en place **les outils d'accès** au RCP depuis Linux/WSL2
(commandes adaptées de la procédure macOS arm64). Le RCP côté LiGHT est
**Kubernetes + Run:AI + Docker**, pas SLURM.

> ℹ️ **Pour lancer le benchmark une fois les outils en place**, voir
> [`RCP_QUICKSTART.md`](RCP_QUICKSTART.md) — 3 commandes du clone au résultat.
> Cette page-ci décrit uniquement la mise en place initiale (VPN, kubectl,
> runai, registry, build/push de l'image Docker).

## 1. Prérequis

- VPN EPFL installé et connecté (sinon `haas001.rcp.epfl.ch` est inaccessible)
- Compte GASPAR EPFL avec accès LiGHT (demande via le responsable du labo)
- Au moins 30 Go d'espace disque libre (build Docker)

## 2. SSH vers HaaS

```bash
ssh <gaspar>@haas001.rcp.epfl.ch
# mot de passe = mot de passe email EPFL (GASPAR)
```

Une fois sur HaaS, récupère les ids dont tu auras besoin pour le `.env` :

```bash
id
# uid=12345(<gaspar>) gid=67890(light-scratch) groups=...
```

Note le `uid` et le `gid` (light-scratch).

## 3. kubectl (en local, Linux amd64)

Si tu n'as pas sudo, installe dans `~/.local/bin/` (qui est dans `$PATH` sur
la plupart des distributions modernes) :

```bash
curl -LO "https://dl.k8s.io/release/v1.29.6/bin/linux/amd64/kubectl"
chmod +x ./kubectl
mkdir -p ~/.local/bin && mv ./kubectl ~/.local/bin/kubectl
hash -r

mkdir -p ~/.kube
curl -o ~/.kube/config https://raw.githubusercontent.com/epfml/getting-started/main/kubeconfig.yaml

# Vérifier
kubectl version --client
```

⚠️ Si tu as un `kubectl` cassé hérité d'une install précédente (par exemple un
binaire macOS / ARM dans `/usr/local/bin/kubectl`), `~/.local/bin/kubectl` le
masque car il vient avant dans `$PATH`. Sinon, `sudo mv ./kubectl /usr/local/bin/`
suffit pour le remplacer.

## 4. CLI Run:AI (en local, Linux)

```bash
wget --content-disposition https://rcp-caas-prod.rcp.epfl.ch/cli/linux
chmod +x ./runai
sudo mv ./runai /usr/local/bin/runai
sudo chown root: /usr/local/bin/runai

# Vérifier
runai version
runai list jobs
```

## 5. Authentification Docker au registry

```bash
docker login registry.rcp.epfl.ch -u <gaspar>
# mot de passe = mot de passe GASPAR / email EPFL
```

Le registry crée automatiquement le namespace `<lab>/<user>/...` lors du
premier push, pas besoin de créer un projet à la main.

## 6. Build et push de l'image du benchmark

Plutôt que d'utiliser le `template.sh` du template LiGHT (qui dépend de
`docker compose` et d'une certaine structure de repo), le benchmark expose ses
propres scripts :

```bash
cd <repo>
./scripts/rcp/setup.sh
```

Le script :

- détecte ton uid / gid / projet Run:AI automatiquement,
- build l'image générique (`docker/Dockerfile`),
- build l'image user avec tes uid / gid (`docker/Dockerfile.user`),
- push vers `registry.rcp.epfl.ch/<gaspar>/bcv:<gaspar>-latest`
  (par défaut Harbor project homonyme du GASPAR ; override via
  `HARBOR_PROJECT=<name> ./scripts/rcp/setup.sh` si ton lab utilise
  un project partagé),
- écrit `.rcp-env` (consommé par `scripts/rcp/submit.sh`).

Compter ~10 min la première fois (30 Go libres recommandés pour le cache
BuildKit). Les rebuilds incrémentaux sont rapides.

Détails de l'architecture Docker (deux couches, code monté à runtime) :
[`../docker/README.md`](../docker/README.md).

## 7. Soumission Run:AI

Voir [`RCP_QUICKSTART.md`](RCP_QUICKSTART.md) : `scripts/rcp/submit.sh smoke`,
`./scripts/rcp/submit.sh all`, etc. Les `scripts/slurm/*.sbatch` historiques
restent fournis pour les utilisateurs sur SLURM (voir
[`RUN_OTHER_PLATFORMS.md`](RUN_OTHER_PLATFORMS.md)).
