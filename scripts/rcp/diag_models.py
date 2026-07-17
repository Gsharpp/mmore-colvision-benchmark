"""
Diagnostic de chargement pour tous les modèles ColVision.

Pour chaque modèle :
  1. Import de la classe colpali-engine (capture warnings Python)
  2. from_pretrained (capture stderr Transformers : poids manquants, incompatibilités)
  3. Passe forward minimale (image synthétique 224×224 + texte)
  4. Rapport structuré : OK / WARN / ERROR

Usage (cluster) :
  python3 /tmp/diag_models.py
"""

import io
import os
import re
import sys
import time
import traceback
import warnings
from contextlib import contextmanager, redirect_stderr

import torch
from PIL import Image

HF_CACHE = os.environ.get("HF_HOME", "/mloscratch/users/mbonnet/hf-cache")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MODELS = [
    dict(id="colqwen2_5_v0_2",   hf="vidore/colqwen2.5-v0.2",           cls="ColQwen2_5",  proc="ColQwen2_5_Processor"),
    dict(id="colgemma3_colnetra",hf="Cognitive-Lab/ColNetraEmbed",       cls="ColGemma3",   proc="ColGemmaProcessor3"),
]

SEP = "=" * 72


@contextmanager
def _capture_stderr():
    buf = io.StringIO()
    with redirect_stderr(buf):
        yield buf


def _filter_warnings(text: str) -> list[str]:
    """Garde uniquement les lignes qui semblent indiquer un problème."""
    keep = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in (
            "missing", "unexpected", "not initialized", "warning",
            "error", "size mismatch", "incompatible", "some weights",
            "ignore", "not used", "loaded",
        )):
            keep.append(line.rstrip())
    return keep


def _dummy_image(h: int = 336, w: int = 336) -> Image.Image:
    import numpy as np
    arr = (np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
    return Image.fromarray(arr)


def run_diag(spec: dict) -> dict:
    result = dict(id=spec["id"], hf=spec["hf"], cls=spec["cls"],
                  status="OK", warnings=[], errors=[])

    # ── 1. Import classe ──────────────────────────────────────────────────
    try:
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            mod = __import__("colpali_engine.models", fromlist=[spec["cls"], spec["proc"]])
            ModelCls = getattr(mod, spec["cls"])
            ProcCls  = getattr(mod, spec["proc"])
        for w in w_list:
            msg = str(w.message)
            if not any(k in msg.lower() for k in ("beta", "torchvision")):
                result["warnings"].append(f"[import] {msg}")
    except Exception as exc:
        result["errors"].append(f"[import] {exc}")
        result["status"] = "ERROR"
        return result

    # ── 2. from_pretrained (model + processor) ────────────────────────────
    try:
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w_load, _capture_stderr() as stderr_buf:
            warnings.simplefilter("always")
            proc  = ProcCls.from_pretrained(spec["hf"], cache_dir=HF_CACHE)
            model = ModelCls.from_pretrained(
                spec["hf"],
                torch_dtype=DTYPE,
                device_map=DEVICE,
                cache_dir=HF_CACHE,
            )
        elapsed = time.time() - t0
        result["load_s"] = round(elapsed, 1)

        # Warnings Python levés pendant le chargement
        for w in w_load:
            msg = str(w.message)
            if not any(k in msg.lower() for k in ("beta", "torchvision", "userwarning")):
                result["warnings"].append(f"[load] {msg[:200]}")

        # Stderr Transformers (poids manquants, incompatibilités, etc.)
        stderr_text = stderr_buf.getvalue()
        for line in _filter_warnings(stderr_text):
            result["warnings"].append(f"[stderr] {line[:200]}")

    except Exception as exc:
        result["errors"].append(f"[load] {traceback.format_exc()[-600:]}")
        result["status"] = "ERROR"
        return result

    # ── 3. Forward pass minimal ───────────────────────────────────────────
    try:
        model.eval()
        img   = _dummy_image()
        query = "What is the main finding described in this document?"

        with warnings.catch_warnings(record=True) as w_fwd, _capture_stderr() as stderr_fwd:
            warnings.simplefilter("always")
            with torch.no_grad():
                # Image embedding
                img_inputs = proc.process_images([img]).to(DEVICE)
                img_emb    = model(**img_inputs)
                # Query embedding
                q_inputs   = proc.process_queries([query]).to(DEVICE)
                q_emb      = model(**q_inputs)

        for w in w_fwd:
            msg = str(w.message)
            if not any(k in msg.lower() for k in ("beta", "torchvision")):
                result["warnings"].append(f"[fwd] {msg[:200]}")
        for line in _filter_warnings(stderr_fwd.getvalue()):
            result["warnings"].append(f"[fwd-stderr] {line[:200]}")

        # Vérifier la dimension des embeddings
        img_dim = img_emb.shape[-1]
        q_dim   = q_emb.shape[-1]
        result["img_emb_shape"] = list(img_emb.shape)
        result["q_emb_shape"]   = list(q_emb.shape)

        expected_dims = {
            "colpali_v1_3": 128, "colqwen2_v1_0": 128,
            "colqwen2_5_v0_2": 128, "colqwen3_v0_1": 320,
            "colgemma3_colnetra": 128,
        }
        exp = expected_dims.get(spec["id"])
        if exp and img_dim != exp:
            result["warnings"].append(
                f"[dim] img embed_dim={img_dim}, expected {exp} — MISMATCH"
            )
            result["status"] = "WARN"

        # Score cosinus image-query (doit être positif)
        score = torch.einsum("bnd,csd->", img_emb.float(), q_emb.float()).item()
        result["cos_score"] = round(score, 4)

    except Exception as exc:
        result["errors"].append(f"[fwd] {traceback.format_exc()[-600:]}")
        result["status"] = "ERROR"
        return result

    # ── Statut final ──────────────────────────────────────────────────────
    if result["warnings"]:
        result["status"] = "WARN"
    return result


def main():
    print(f"\n{SEP}")
    print(f"  DIAGNOSTIC MODÈLES ColVision — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  device={DEVICE}  dtype={DTYPE}  hf_cache={HF_CACHE}")
    print(SEP)

    summary = []
    for spec in MODELS:
        print(f"\n>>> {spec['id']}  ({spec['hf']})")
        sys.stdout.flush()
        res = run_diag(spec)

        icon = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}.get(res["status"], "?")
        print(f"  {icon} status : {res['status']}")
        if "load_s" in res:
            print(f"  load    : {res['load_s']}s")
        if "img_emb_shape" in res:
            print(f"  img_emb : {res['img_emb_shape']}")
        if "q_emb_shape" in res:
            print(f"  q_emb   : {res['q_emb_shape']}")
        if "cos_score" in res:
            print(f"  score   : {res['cos_score']}")
        for w in res["warnings"]:
            print(f"  ⚠  {w}")
        for e in res["errors"]:
            print(f"  ✗  {e}")

        summary.append(res)
        sys.stdout.flush()

    print(f"\n{SEP}")
    print("  RÉSUMÉ")
    print(SEP)
    for r in summary:
        icon = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}.get(r["status"], "?")
        n_warn = len(r["warnings"])
        n_err  = len(r["errors"])
        print(f"  {icon} {r['id']:<30}  warnings={n_warn}  errors={n_err}")
    print(SEP)


if __name__ == "__main__":
    main()
