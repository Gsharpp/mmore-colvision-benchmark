# Feuille de route benchmark-colvision — rendu LUNDI MIDI

## Context

Le sprint a livré un run Track A `tiny` (90 PDFs / 709 pages, 4 modèles, 1 seed) avec
mesures d'empreinte/latence fiables et scores retrieval préliminaires (requêtes mock
templatées). **Contrainte dure : tout le benchmark (étapes 1-4) doit être rendu lundi midi.
Seul le formatage repro du repo (étape 6) est autorisé après lundi.** L'exploration du code
a montré que **beaucoup d'infrastructure existe déjà** :

- **Track B est entièrement codé** (`run_track_b.py`, `orchestrate.run_track_b_for_model`,
  CLI `bcv-run track-b`) — seul le *corpus multilingue* manque.
- **Le pipeline de génération LLM existe** (`queries/inverse_query_gen.py`,
  `ambiguity_filter.py`, `clients/vllm_client.py`, `bcv-queries generate/filter`,
  `configs/judge.yaml` déclare Meditron-70B/7B ; `submit.sh` a `serve-meditron`).
- **Le déploiement est mûr** (`scripts/rcp/setup.sh`, `bootstrap-venv.sh`, `submit.sh`).

**Décisions validées :**
- Meditron : valider sur **7B**, génération finale sur **70B**.
- ColQwen3 : **venv scratch isolé** (ne pas casser les 4 modèles sur colpali-engine 0.3.16).
- Échelle : **500 PDFs × 5 modèles × 3 seeds**.
- Track B : **littérature médicale réelle**, langues **EN + FR + ZH**.
- Repro/LiGHT : **expliquer d'abord** le delta, décider ensuite.

Contraintes : réseau local instable (downloads lourds sur le cluster, pas en local) ; le
cluster lit son code depuis `/mloscratch/users/mbonnet/bcv-dev` (propager via `git pull`).

### Échéance & honnêteté sur la faisabilité

- **Rendu lundi midi** : Étapes **1, 2, 3, 4**. **Après lundi (autorisé)** : Étape **6**
  (formatage repro). Étape 5 (RAGAS/convergence étendue) reste *bien après*, hors de ce rendu
  (à confirmer si jamais Mathieu la veut dans le lot).
- **Risque réel** : faire tenir 3 (connecteur ZH réel) + 4 (500 × 5 × 3 seeds) dans le
  week-end n'est **pas garanti** (GPU mur + file cluster + connecteur chinois difficile).
- **Dégradation gracieuse intégrée** (replis automatiques pour garantir un livrable cohérent
  lundi) :
  - **ZH** : si le connecteur littérature OA chinoise ne tient pas → repli **Wikipédia ZH**
    (endpoint REST PDF uniforme, fiable) pour garder le test cross-script ; EN/FR restent en
    littérature réelle.
  - **Échelle** : si 3 seeds ne finissent pas → repli **1 seed à 500 PDFs** ; seeds 2-3
    complétés ensuite si la file le permet.

---

## Étape 1 — ColQwen3 (venv isolé) + Track B `en` (code-path) — LUNDI

### 1a. ColQwen3 dans un venv scratch dédié
**Problème** : `colpali-engine 0.3.16` ne charge pas les poids ColQwen3 ; le `hf_name`
`vidore/colqwen3-v0.1` **n'existe pas** (bon nom : `goodman2001/colqwen3-v0.1`, dim 320).
1. Déterminer la version `colpali-engine`/`transformers` supportant ColQwen3 (job jetable).
2. Créer `bcv-venv-qwen3` (variante de `bootstrap-venv.sh`, override `BCV_VENV` +
   `uv pip install colpali-engine==<ver> transformers==<ver>`). `bcv-venv` (0.3.16) intact.
3. Corriger le `hf_name` (override via rendu de config en attendant un commit).
4. Vérifier mmore avec la nouvelle colpali-engine (process→index→retrieve tiny).
   `_render_cell_configs` injecte déjà `embed_dim=320` (fix local — vérifier présence cluster).
5. Soumettre ColQwen3 `tiny` avec `BCV_VENV=bcv-venv-qwen3`. **Un seul essai** ; sinon exclure.

### 1b. Track B `en` — validation du chemin de code
Valider Track B sur `en` (corpus PMC existant + requêtes étape 2) →
`results/track_b/<m>/en/seed_0.json`. Le vrai multilingue (FR/ZH) est l'Étape 3.

**Fichiers** : `scripts/rcp/bootstrap-venv.sh`, `configs/models.yaml`,
`orchestrate._render_cell_configs` ; symlinks `data/track_b/*`.

---

## Étape 2 — Requêtes Meditron (7B → 70B) — LUNDI

Remplace les requêtes mock (biais d'homogénéité) par des requêtes cliniques LLM ancrées sur
le contenu visuel.
1. **Valider sur 7B (1 GPU)** : `serve-meditron` variante 7B ; vérifier `HF_TOKEN` (gated) ;
   `bcv-queries generate` puis `filter` (seuil 0.8).
2. **Durcir le prompt JSON** (`inverse_query_gen.py` l.19-32) : few-shot si besoin, langue des
   requêtes alignée sur le corpus.
3. **Run final 70B (4 GPU)**.
4. Regénérer les requêtes Track A → re-scorer les 5 modèles → **scores défendables**.

**Fichiers** : `scripts/rcp/submit.sh` (variante 7B), `queries/inverse_query_gen.py`,
`configs/judge.yaml`. Aucun nouveau module.

---

## Étape 3 — Track B multilingue réel : EN / FR / ZH — LUNDI (haut risque)

Vrai moyen de comparer les perfs sur **documents multilingues** (contenu visuel = point clé).
Code Track B existant ; acquérir le corpus par langue.
1. **EN** : PMC OA (connecteur existant `corpus/pmc_downloader.py`).
2. **FR** : connecteur **HAL** (API REST `api.archives-ouvertes.fr`, JSON → URLs PDF), domaine
   médical, sur le modèle de `pmc_downloader.py` (sample→download→extract). Brancher dans
   `bcv-corpus`. *Plausible en quelques heures.*
3. **ZH** : connecteur source **OA médicale chinoise**. **Point dur** (CNKI payant). **Repli
   si non tenable lundi : Wikipédia ZH via endpoint REST PDF** (uniforme, fiable) pour garder
   le test CJK. Surveiller polices/encodage CJK côté mmore.
4. **Manifests par langue** : `bcv-corpus build-manifest --track B --source ...
   --language-override {en,fr,zh}` → `data/track_b/corpus_manifest.json` (vérifier le modèle
   un-manifest-multi-langue ; adapter `orchestrate` si besoin).
5. **Requêtes par langue** via Meditron (prompts FR/ZH), filtrées.
6. **Run** `bcv-run track-b` par (modèle, langue) → comparaison cross-langue/cross-script.
7. Caveats : sources hétérogènes par langue, tailles modestes, repli Wikipédia ZH le cas échéant.

**Fichiers** : nouveau `corpus/hal_downloader.py` (+ `corpus/<zh>_downloader.py` ou utilitaire
Wikipédia) ; `corpus/cli.py` ; `configs/track_b.yaml` ; éventuellement `orchestrate`.

---

## Étape 4 — Montée en échelle : 500 PDFs × 5 modèles × 3 seeds — LUNDI (haut risque)

Pipeline corpus complet existant. 500 PDFs ≈ 3500-4000 pages.
1. `bcv-corpus sample-pmc --n 500 --seed {0,1,2} --packages-out pkgs_seed{n}.json`.
2. `bcv-corpus download-packages ...` **sur le cluster** (retries/streaming déjà gérés).
3. `bcv-corpus build-manifest --name track_a_500 --track A --source pmc-oa` (hash/seed).
4. Palier `configs/track_a.yaml` (`id: p500, n_pages: ~3800`) + `queries.per_palier`.
5. Requêtes Meditron (étape 2) sur le nouveau corpus, par seed.
6. Matrice 5 × 3 (ColQwen3 via venv isolé). Surveiller VRAM (ColQwen2.5 pic 15 GB) et latence
   (∝ vecteurs ; ColPali ~1031 vec/page coûteux). **Repli : 1 seed d'abord**, seeds 2-3 si la
   file le permet.
7. `bcv-report` → tables moyenne ± σ.

**Fichiers** : `configs/track_a.yaml` + commandes corpus/run/report. Pas de nouveau code attendu.

---

## Étape 5 — Benchmark complet / RAGAS — BIEN APRÈS (hors rendu)

Extension paliers (`medium` 10k), langues Track B supplémentaires (DE/ES/AR), **métriques de
génération** (RAGAS + juge Meditron, `configs/judge.yaml`). À re-planifier ; hors du rendu lundi.

---

## Étape 6 — Reproductibilité « 2-3 commandes » alignée LiGHT — SEUL ITEM AUTORISÉ APRÈS LUNDI

**Mathieu veut d'abord l'explication du delta** avant tout refactor. Livrable initial = une note.

### Delta nous vs `EPFLiGHT/LiGHT-cluster-template`
| Aspect | Notre setup | Template LiGHT |
|---|---|---|
| Structure | bash maison standalone | scripts standardisés |
| Entrypoints | empruntés au template | source originale |
| Image | 2 étages lean, deps en venv scratch | même philosophie |
| Orchestration | wrappers `runai submit` custom | conventions du template |

**Déjà fonctionnel** : `setup.sh` puis `submit.sh smoke/all` (proche de 2 commandes).
**Points fragiles** : clone HaaS manuel préalable ; récup résultats par `scp` ; friction
uid/gid ; **fixes locaux non poussés** (`mmore_wrapper.py`, `run_track_a.py`, `process.yaml`,
`orchestrate._render_cell_configs`) → à committer (sous réserve d'accord push).
**Options** : (a) polir l'existant (automatiser clone + récup, committer fixes, doc 2-cmd) ;
(b) réaligner sur le template (refonte lourde).
**Plan** : note de comparaison → Mathieu tranche → refactor.

---

## Vérification (par étape)
- **1a** : `results/track_a/colqwen3_v0_1/tiny/seed_0.json` `ndcg@5 != None` (ou exclusion
  documentée) ; 4 autres modèles inchangés.
- **1b** : `results/track_b/<m>/en/seed_0.json`.
- **2** : requêtes filtrées (judge_score ≥ 0.8) ; re-score Track A.
- **3** : `results/track_b/<m>/{en,fr,zh}/...` ; comparaison cross-langue (repli ZH documenté
  si Wikipédia).
- **4** : manifests `track_a_500` (hash/seed), matrice 5 × N seeds, tables moyenne ± σ (N=3
  visé, 1 garanti).
- **6** : repro d'un smoke en ≤ 3 commandes documentées.

## Ordre, dépendances & critical path
1 → 2 séquentiels (requêtes LLM alimentent 3 et 4). **Critical path week-end** : (Sam) venv
ColQwen3 + Meditron 7B→requêtes ; (Sam soir) lancer process+index 500×5 en file ; (Dim)
connecteurs HAL/ZH + requêtes par langue + Track B run ; (Dim soir/Lun matin) agrégation +
rapport. Replis ZH-Wikipédia et 1-seed activés dès qu'une brique menace l'échéance. Étape 6
seule peut déborder après lundi.
