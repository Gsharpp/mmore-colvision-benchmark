# SPRINT 24h — Rendu Cassiopée Télécom SudParis + LiGHT EPFL (2026-05-29)

## Contexte du sprint

Le plan complet du projet prévoit 2-3 semaines de compute (corpus 50k pages, queries 2k, Track A scaling 4 paliers × 3 seeds, Track B 6 langues). **Le rendu est demain (2026-05-29)** — c'est la note finale Cassiopée, possibilité d'ajouter après mais sans impact sur la note.

L'audit du code (2026-05-28) confirme que **tout est codé** côté pipeline : downloader PMC, générateur de queries via vLLM, runner Track A, aggregation, figures, tables LaTeX. Le tech report est à **~60%** : Intro, Related Work, Methodology subsections et Limitations sont rédigés ; Track A Results (4 subsections), Track B Results (2 subsections), Discussion vides ; Conclusion stub 3 lignes.

**Objectif** : produire un livrable défendable en 24h — tech report PDF + résultats partiels honnêtes — en utilisant le code déjà en place sur un mini-corpus, et reporter scaling 50k + Track B + validation Pearson à "Future Work" explicite.

## Stratégie

Réduire **agressivement** le scope expérimental sans toucher au code :

| Dimension | Plan original | Sprint 24h |
|---|---|---|
| Corpus EN | 50 000 pages | 300 pages |
| Queries auto | ~2 000 | 50-100 |
| Paliers scaling | 100 / 1k / 10k / 50k | 1 palier `tiny` |
| Seeds | 3 | 1 |
| Modèles Track A | 5 | **5 (couverture maintenue)** |
| Track B (6 langues) | Inclus | **Reporté Future Work** |
| Validation Pearson | Inclus (#18) | **Reportée Future Work** |
| RAGAS faithfulness | Inclus | Best-effort si compute reste |

5 cells Track A au lieu de 60 → faisable en ~1-2h sur les 3 GPUs deserved de mbonnet après teardown vLLM Meditron.

## Décisions confirmées par l'utilisateur (2026-05-28)

1. **Modèles** : 5 (ColPali v1.3, ColQwen2 v1.0, ColQwen2.5 v0.2, ColQwen3 v0.1, ColGemma3) — couverture plutôt que profondeur
2. **Source corpus** : listing public PMC OA CSV + filtre PyMuPDF densité visuelle 30% (reproductible)
3. **vLLM Meditron** : à vérifier via `runai list jobs` ; lance `bcv-vllm` si rien ne tourne, fallback mock queries si vraiment bloqué

## Plan d'exécution

### Étape 0 — Pré-requis (≤30 min) — débloquer la chaîne RCP

- `setup.sh` end-to-end terminé (DONE message), `.rcp-env` écrit. **STATUT 2026-05-28 16h+ : setup.sh en cours, push generic possiblement bloqué — à diagnostiquer**
- Clone `bcv-dev` sur HaaS :
  ```bash
  ssh mbonnet@haas001.rcp.epfl.ch
  mkdir -p /mnt/light/scratch/mbonnet && cd /mnt/light/scratch/mbonnet
  git clone <repo> bcv-dev
  ```
- Smoke run : `./scripts/rcp/submit.sh smoke && runai logs bcv-smoke -f` — valide pipeline RCP de bout en bout (#19)

### Étape 1 — Mini-corpus PMC (≤45 min) (#27)

- Récupérer index PMC OA + échantillonner ~500 PMCIDs EN du subset commercial-use, biais radiology/pathology/cardiology
- `bcv-corpus download-pmc <PMCIDs> --cache-dir /mloscratch/bcv/pmc-cache --out-dir /mloscratch/bcv/track_a/pdfs`
- `bcv-corpus build-manifest /mloscratch/bcv/track_a/pdfs --out /mloscratch/bcv/track_a/corpus_manifest.json --name sprint_mini_en --track A --source pmc-oa --density-threshold 0.30 --target-pages 300`
- Gate : `cat corpus_manifest.json | jq '.pages | length'` ≈ 300

### Étape 2 — Lancement vLLM Meditron-70B (parallèle, ≤30 min cold start)

- `runai list jobs` → check si un endpoint vLLM Meditron tourne déjà
- Sinon : `./scripts/rcp/submit.sh serve` (lance `bcv-vllm` 4 GPUs)
- `runai logs bcv-vllm -f` jusqu'à "model loaded" + endpoint URL accessible
- Note : vLLM consomme 4 GPUs > 3 GPUs deserved → Track A devra attendre teardown vLLM

### Étape 3 — Génération queries (1.5-2h compute) (#28)

```bash
bcv-queries generate \
  --corpus-manifest /mloscratch/bcv/track_a/corpus_manifest.json \
  --corpus-root /mloscratch/bcv/track_a/pdfs \
  --out /mloscratch/bcv/track_a/queries/sprint_mini.jsonl \
  --vllm-endpoint <endpoint> --vllm-model meditron-70b \
  --n-per-page 2 --max-pages 50
bcv-queries filter-ambiguity \
  --in /mloscratch/bcv/track_a/queries/sprint_mini.jsonl \
  --out /mloscratch/bcv/track_a/queries/sprint_mini_filtered.jsonl
```

- Cible : 50-100 queries `requires_visual=true` après filter
- **Fallback** si vLLM bloqué : créer `src/benchmark_colvision/queries/mock.py` (30 min de code) — queries templées par mesh tag de la page

### Étape 4 — Teardown vLLM + Track A réduit (2-3h compute) (#29)

- `./scripts/rcp/teardown.sh serve` (libère 3 GPUs pour Track A)
- Pour chaque `model_id` ∈ {colpali_v1_3, colqwen2_v1_0, colqwen2_5_v0_2, colqwen3_v0_1, colgemma3} :
  ```bash
  ./scripts/rcp/submit.sh track-a ${model_id} 0 --palier tiny
  ```
- 5 jobs Run:AI en file ; 3 GPUs deserved → 2 batches concurrents (3 + 2) → ~1-2h total
- Chaque cell produit `/mloscratch/bcv/results/track_a/${model_id}/tiny/seed_0.json`

### Étape 5 — Récupération résultats + figures + tables (≤30 min) (#30)

```bash
scp -r mbonnet@haas001.rcp.epfl.ch:/mloscratch/bcv/results ./results
bcv-report figures-a --results-dir results/track_a --out-dir report/figures --metric retrieval.ndcg_at_5
bcv-report tables --results-dir results/track_a --out-dir report/tables
```

- Output : throughput PNG + latency PNG + GPU mem PNG + summary table TEX. Scaling curve désactivé (1 palier seulement).

### Étape 6 — Finaliser le tech report (3-5h) (#31)

Modifs `report/main.tex` par section :

- **Track A — Results (général)** : insérer Table summary (model × {nDCG@5, MRR, throughput, GPU mem}), commenter quel modèle gagne sur nDCG@5/MRR
- **Track A — Generation** : RAGAS faithfulness + answer relevancy si on a réussi à tourner la génération RAG, sinon courte note "deferred to Future Work for compute reasons"
- **Track A — Performance** : insérer figures throughput, latency p50/p95/p99, GPU mem peak
- **Track A — Statistical analysis** : bootstrap CI sur nDCG@5 ; déclarer explicitement que **Wilcoxon signed-rank n'est pas applicable avec 1 seed** (limitation à mentionner)
- **Track B** : remplacer subsections vides par 1 paragraphe "Planned but not executed in this sprint — see Future Work"
- **Discussion** (à écrire, ~1 page) : tendances observées (modèle gagnant ColPali baseline vs Qwen series vs Gemma3), hypothèses pour scaling, comparaison aux résultats ViDoRe publiés des auteurs des modèles
- **Limitations** : compléter avec scope sprint réduit (300 pages au lieu de 50k, 1 seed, pas de Track B multilingue, validation Pearson différée)
- **Future Work** (nouvelle sous-section avant Conclusion) : roadmap pointant vers le plan complet (50k pages × 4 paliers × 3 seeds, 6 langues, Pearson validation, RAGAS complet)
- **Conclusion** : étoffer à ~10 lignes — recap résultats partiels + roadmap explicite
- Compiler : `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Critical files (sprint)

**Code (NE PAS modifier, juste utiliser via CLIs)** :
- `src/benchmark_colvision/corpus/cli.py` — CLI `bcv-corpus`
- `src/benchmark_colvision/queries/cli.py` — CLI `bcv-queries`
- `src/benchmark_colvision/runners/cli.py` — CLI `bcv-run`
- `src/benchmark_colvision/reporting/cli.py` — CLI `bcv-report`
- `scripts/rcp/submit.sh` — submission RCP (track-a, smoke, serve)
- `configs/track_a.yaml`, `configs/models.yaml` — déjà configurés

**À écrire / modifier (sprint uniquement)** :
- `report/main.tex` — rédaction Results (Track A), Discussion, Future Work, Conclusion ; réduire Track B à pointer vers Future Work
- `report/figures/` — placement PNG générés par `bcv-report figures-a`
- `report/tables/` — placement TEX généré par `bcv-report tables`

**Fallback potentiel à créer si bloqué** :
- `src/benchmark_colvision/queries/mock.py` — génération templée par mesh tag, ~30 min de code, à n'écrire QUE si vLLM Meditron est indisponible et que `bcv-queries generate` est bloqué

## Verification — gates par étape

| Étape | Gate de validation |
|---|---|
| 0 | `.rcp-env` présent + `runai list jobs` montre `bcv-smoke` terminé en succès |
| 1 | `jq '.pages \| length' corpus_manifest.json` ≈ 300 |
| 2 | `curl <endpoint>/v1/models` retourne `meditron-70b` |
| 3 | `wc -l queries/sprint_mini_filtered.jsonl` entre 50 et 100 |
| 4 | 5 fichiers `results/track_a/${model}/tiny/seed_0.json` valides |
| 5 | `report/figures/*.png` et `report/tables/track_a_summary.tex` existent |
| 6 | `report/main.pdf` recompilé ; aucun placeholder `% Figure: …` restant ; aucune section sans contenu |

## Risques résiduels et mitigations

| Risque | Probabilité | Mitigation |
|---|---|---|
| setup.sh bloqué sur push generic | **Actuel** | Diagnostic immédiat (lecture logs Docker + dernière ligne terminal) |
| vLLM Meditron indispo ou job pending long | Moyen | Fallback `queries/mock.py` (30 min code, queries templées) |
| 1 modèle plante (HF download, OOM) | Moyen | Continuer les 4 autres, mentionner "model X failed" dans Limitations |
| Queue Run:AI saturée le soir | Faible-moyen | Soumettre tôt (avant 19h), prioriser vLLM puis Track A, réduire à 3 modèles si vraiment bloqué |
| VPN coupe pendant la nuit | Moyen | Jobs Run:AI en `--interactive=false` (persistent côté RCP même si VPN tombe) |
| Tech report ne compile pas | Faible | PDF compilé OK le 26/05, bib stable ; éviter d'ajouter packages LaTeX |

## Out of scope (= Future Work dans le tech report)

- Phases 2-7 complètes du plan original (50k pages × 4 paliers × 3 seeds, 6 langues, validation Pearson)
- Multi-seed bootstrap CI rigoureux et Wilcoxon signed-rank
- RAGAS faithfulness avec juge Meditron-70B si compute tight (best-effort dans sprint)
- Publication HF Hub / arXiv / GitHub public / PR upstream mmore (déjà différé hors sprint)
