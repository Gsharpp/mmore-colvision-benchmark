# Connexion EPFL RCP (LiGHT lab)

Procédure pour accéder à la plateforme RCP du labo LiGHT depuis Linux/WSL2
(commandes adaptées de la procédure macOS arm64). Le RCP côté LiGHT est
**Kubernetes + Run:AI + Docker**, pas SLURM — voir `scripts/slurm/*.sbatch`
pour ce qui doit être refactoré (cf. tâche #25 dans le project tracker).

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

```bash
curl -LO "https://dl.k8s.io/release/v1.29.6/bin/linux/amd64/kubectl"
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
sudo chown root: /usr/local/bin/kubectl

mkdir -p ~/.kube
curl -o ~/.kube/config https://raw.githubusercontent.com/epfml/getting-started/main/kubeconfig.yaml

# Vérifier
kubectl get pods
```

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

## 5. Registry Docker

1. Aller sur https://registry.rcp.epfl.ch/ et se connecter (GASPAR)
2. Créer son projet personnel (ex. `mbonnet` ou un nom de ton choix — c'est ton
   `LAB_NAME` plus bas)

## 6. Template LiGHT

```bash
cd ~
git clone https://github.com/EPFLiGHT/LiGHT-cluster-template.git
cd LiGHT-cluster-template/installation/docker-amd64-cuda
```

Éditer le `.env` avec tes valeurs (toutes en minuscules pour `USR` /
`LAB_NAME`) :

```dotenv
USR=<gaspar lowercase>
USRID=<uid récupéré sur HaaS>
GRPID=<gid récupéré sur HaaS>
GRP=light-scratch
PASSWD=<choix libre, password du user à l'intérieur du conteneur>
LAB_NAME=<nom du projet créé sur registry.rcp.epfl.ch>
```

Build :

```bash
./template.sh build_generic --ignore-uncommitted
./template.sh build_user    --ignore-uncommitted
```

(30 Go libres minimum — l'image embarque CUDA, PyTorch, etc.)

## 7. Push de l'image

```bash
docker login registry.rcp.epfl.ch -u <gaspar>
./template.sh push RCP
```

## 8. (à venir) Soumission Run:AI

Une fois l'image poussée, soumettre des jobs avec `runai submit ...` ou via une
spec YAML. Cette étape reste **à scripter** : nos `scripts/slurm/*.sbatch` ne
sont pas compatibles avec Run:AI/Kubernetes (cf. tâche #25). Le pattern attendu
sera :

```bash
runai submit bcv-track-a-colpali-seed0 \
  --image registry.rcp.epfl.ch/<lab_name>/<repo>:<tag> \
  --gpu 1 \
  --pvc light-scratch:/scratch \
  --command -- bcv-run track-a --model-id colpali_v1_3 --seed 0 ...
```

À adapter quand le Dockerfile sera prêt et qu'on aura la convention de
montage des volumes scratch utilisée par LiGHT.
