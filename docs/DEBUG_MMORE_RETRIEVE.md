# Debug : mmore colpali retrieve → context: [] (ndcg@5=0.0)

**Date** : 2026-05-29  
**Branch** : `rcp-lean-venv`  
**Commit mmore utilisé** : `498560047e19ddb57ecc0fdb5b01f9f4ccd0d651`

---

## Symptôme

Le pipeline Track A tourne sans erreur apparente (returncode 0 pour process, index, retrieve),
mais le fichier de sortie du retrieve contient `"context": []` pour chaque requête :

```json
[
  {"query": "Quelle page documente conjointement : ...", "context": []},
  ...
]
```

Résultat : toutes les métriques sont à `0.0` (`ndcg@5=0.0`, `recall@5=0.0`, etc.).

---

## Ce qui fonctionne

| Étape | État | Preuve |
|---|---|---|
| `mmore colpali process` | ✅ OK | `pdf_page_objects.parquet` = 140 MB |
| `mmore colpali index` | ✅ OK | `smoke.db` = 600 MB, collection `bcv_smoke_pages` = **730 979 vecteurs** |
| `mmore colpali retrieve` | ⚠️ returncode=0 mais vide | `context: []` × 32 requêtes |

L'index est correct. Les vecteurs ColPali-v1.3 (dim=128) sont bien insérés.

---

## Root cause

**Fichier** : `mmore/colpali/milvuscolpali.py`, méthode `rerank_page()` (dans `search_embeddings()`)

```python
# CODE BUGUÉ — syntaxe de filtre invalide pour pymilvus
docs = self.client.query(
    collection_name=self.collection_name,
    filter="pdf_path == $pdf_path and page_number == $page_number",  # ← INVALIDE
    output_fields=["embedding", "pdf_path"],
    limit=10000,
    params={"pdf_path": pdf_path, "page_number": page_number},  # ← ignoré silencieusement
)
```

pymilvus ne supporte **pas** la syntaxe `$variable` dans les filtres (ce n'est pas du MongoDB).
Le `params=` est un kwarg inconnu, silencieusement ignoré.
Le filtre est invalide → `client.query()` lève une exception →
catchée par le `try/except` de l'appelant → `score = None` →
page exclue des résultats → **tous les candidats sont ignorés** → `context: []`.

Le process ne crashe pas car l'exception est avalée ici dans `search_embeddings()` :

```python
for f in concurrent.futures.as_completed(futures):
    try:
        score, pdf_path, page_number = f.result()
        if score is not None:
            reranked.append(...)
    except Exception as e:
        logger.error(f"Rerank failed: {e}")  # ← avalé, pas re-raise
```

---

## ⚠️ Second bug, démasqué par le fix du filtre : `ndcg@5=None` (subprocess retrieve tué)

Une fois le filtre corrigé, le rerank fonctionne et renvoie de **vrais** scores — mais le
retrieve plante alors **pire qu'avant** (`ndcg@5=None` au lieu de `0.0`). Symptôme observé :
« process tué pendant le `json.dump` du premier résultat ».

**Ce n'est PAS un timeout RunAI.** C'est un `TypeError` de sérialisation JSON :

- `rerank_page()` calculait `score = np.dot(...).max(1).sum()` → un **`np.float32`**, pas un `float` Python.
- Ce score remonte dans `result["score"]` → `Document.metadata["similarity"]`.
- `run_retriever.save_results()` fait `json.dump(...)` ; or `np.float32` **n'est pas sérialisable JSON**.
- `json.dump` lève `TypeError` **dès le premier résultat** (d'où « tué pendant le json.dump du premier résultat »).
- Cet appel n'est PAS dans un `try/except` → tout le subprocess retrieve échoue (returncode≠0) →
  pas de fichier de sortie → côté benchmark, branche `else` de `run_track_a.py` → `RetrievalScores` tout à `None`.

Avant le fix du filtre, le rerank échouait toujours → `reranked=[]` → `context:[]` → `json.dump`
d'une liste vide réussissait → `0.0`. Le fix du filtre a donc **démasqué** ce bug latent.

> **Important pour un job retrieve-only** : l'index dans `smoke.db` est déjà valide, relancer
> uniquement le retrieve est la bonne approche pour itérer vite. **Mais** il faut que les **deux**
> correctifs ci-dessous soient présents dans le `milvuscolpali.py` du venv, sinon le retrieve-only
> replantera exactement au même endroit (`json.dump` du premier résultat).

---

## Fix appliqué au repo

**Corrigé dans `mmore-vlm`** — `src/mmore/colpali/milvuscolpali.py` (2 correctifs dans `rerank_page()`),
avec un test de régression dans `tests/test_colpali.py`
(`test_rerank_page_filter_uses_fstring_not_dollar_syntax`).

Le test vérifie que :
- le filtre passé à `client.query()` ne contient pas `$pdf_path` / `$page_number` (ancienne syntaxe invalide) ;
- les valeurs réelles sont interpolées dans le filtre ;
- `search_embeddings()` retourne une liste non vide (symptôme du 1ᵉʳ bug) ;
- le `score` est un `float` Python et `json.dumps(results)` ne lève pas (symptôme du 2ᵉ bug).

---

## Détail des fixes

### Fix 1 — syntaxe du filtre pymilvus

**Remplacer** dans `milvuscolpali.py` :

```python
# AVANT
docs = self.client.query(
    collection_name=self.collection_name,
    filter="pdf_path == $pdf_path and page_number == $page_number",
    output_fields=["embedding", "pdf_path"],
    limit=10000,
    params={"pdf_path": pdf_path, "page_number": page_number},
)
```

```python
# APRÈS
docs = self.client.query(
    collection_name=self.collection_name,
    filter=f'pdf_path == "{pdf_path}" and page_number == {int(page_number)}',
    output_fields=["embedding", "pdf_path"],
    limit=10000,
)
```

### Fix 2 — score JSON-sérialisable (`np.float32` → `float`)

**Remplacer** dans `rerank_page()` :

```python
# AVANT
doc_vecs = np.vstack([d["embedding"] for d in docs]).astype(np.float32)
score = np.dot(query_vecs, doc_vecs.T).max(1).sum()   # np.float32 → crash json.dump
return (score, pdf_path, page_number)
```

```python
# APRÈS
doc_vecs = np.vstack([d["embedding"] for d in docs]).astype(np.float32)
score = float(np.dot(query_vecs, doc_vecs.T).max(1).sum())
return (score, pdf_path, page_number)
```

### Application des fixes sur le cluster (workaround pré-merge)

Le fichier à patcher est dans le venv scratch (pas dans le repo git) :

```
/mloscratch/users/mbonnet/bcv-venv/lib/python3.11/site-packages/mmore/colpali/milvuscolpali.py
```

Option A — session interactive HaaS :
```bash
MMORE_MILVUS=/mloscratch/users/mbonnet/bcv-venv/lib/python3.11/site-packages/mmore/colpali/milvuscolpali.py

python3 - << 'EOF'
import pathlib
f = pathlib.Path("/mloscratch/users/mbonnet/bcv-venv/lib/python3.11/site-packages/mmore/colpali/milvuscolpali.py")
t = f.read_text()
old = '''            docs = self.client.query(
                collection_name=self.collection_name,
                filter="pdf_path == $pdf_path and page_number == $page_number",
                output_fields=["embedding", "pdf_path"],
                limit=10000,
                params={"pdf_path": pdf_path, "page_number": page_number},
            )'''
new = '''            docs = self.client.query(
                collection_name=self.collection_name,
                filter=f\'pdf_path == "{pdf_path}" and page_number == {int(page_number)}\',
                output_fields=["embedding", "pdf_path"],
                limit=10000,
            )'''
assert old in t, "pattern not found — check indentation"
f.write_text(t.replace(old, new, 1))
print("PATCH OK")
EOF
```

Option B — job RunAI (script base64 via env var DIAG_B64, pattern habituel de ce projet).

---

## État des données après le fix

Les parquets et le Milvus DB sont déjà générés et valides, **pas besoin de relancer process + index**.
Il suffit de relancer uniquement le retrieve :

```bash
cd /mloscratch/users/mbonnet/bcv-dev
source /mloscratch/users/mbonnet/bcv-venv/bin/activate
python -m mmore colpali retrieve \
  --config-file configs/mmore/retrieve.yaml \
  -f data/track_a/queries/tiny_mmore.jsonl \
  -o data/track_a/retrieve/colpali_v1_3/tiny/seed_0.json
```

Ou relancer le pipeline complet via `bcv-run track-a` (le `skip_already_processed: true` dans
`process.yaml` évitera de re-processer les PDFs déjà traités).

---

## Autres bugs identifiés et corrigés localement (non encore pushés)

| Fichier | Fix | Statut |
|---|---|---|
| `runners/mmore_wrapper.py` | `_ensure_milvus_dir()` crée le dossier Milvus avant index | Corrigé local |
| `runners/mmore_wrapper.py` | `_make_mmore_qf()` convertit le JSONL SyntheticQuery en strings mmore | Corrigé local |
| `runners/mmore_wrapper.py` | `output_file.parent.mkdir()` crée le dossier retrieve avant retrieve | Corrigé local |
| `runners/retrieval_output.py` | Ajoute `"context"` dans `_RESULT_KEYS` (mmore retourne `context`, pas `results`) | Corrigé local |
| `configs/mmore/process.yaml` | `skip_already_processed: true` | Corrigé local |

Ces fixes locaux ont été appliqués en runtime sur le cluster via des jobs de patch base64.
**Ne pas pusher sans accord explicite de Mathieu.**

---

## Queries mock (problème secondaire)

Les queries dans `data/track_a/queries/tiny.jsonl` sont en **français** et suivent un template
générique peu discriminant :
```
"Quelle page documente conjointement : transcription, transduction, Carbohydrate ?"
```

Une fois le fix mmore appliqué, les scores ne seront probablement pas excellents
(corpus anglais médical × requêtes françaises génériques),
mais ils seront non-nuls et permettront de valider le pipeline end-to-end.
