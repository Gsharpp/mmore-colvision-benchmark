"""
Vérité-terrain pour le remapping de clés ColPali / ColGemma3.

Pour chaque checkpoint « cassé » :
  - liste les fichiers du snapshot + présence d'un adapter_config.json (PEFT ?)
  - clés brutes du checkpoint (.safetensors)  → SOURCE
  - clés du modèle construit (state_dict)     → TARGET (layout transformers 5.x)
  - diff effondré sur les indices de couches → révèle la transformation de préfixe

But : écrire un key_mapping correct (non spéculatif) dans mmore model_utils.
"""

import glob
import io
import os
import re
import warnings
from contextlib import redirect_stderr
from importlib.metadata import version

import torch
from safetensors import safe_open

HF = os.environ.get("HF_HOME", "/mloscratch/users/mbonnet/hf-cache")


def _v(pkg):
    try:
        return version(pkg)
    except Exception:
        return "?"


print("versions:",
      "transformers", _v("transformers"),
      "| colpali-engine", _v("colpali-engine"),
      "| peft", _v("peft"),
      "| torch", torch.__version__)


def snapshot(hf: str):
    base = os.path.join(HF, "models--" + hf.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(base, "*")))
    return snaps[-1] if snaps else None


def ckpt_keys(d: str) -> set:
    keys = set()
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            keys |= set(sf.keys())
    return keys


def collapse(k: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", k)


def analyze(name: str, hf: str, cls_name: str):
    import colpali_engine.models as M

    d = snapshot(hf)
    print(f"\n{'='*70}\n{name}  ({hf})\n  snapshot: {d}")
    if not d:
        print("  !! snapshot introuvable dans le cache")
        return

    files = sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, "*")))
    print("  files:", files)
    print("  adapter_config.json:", os.path.exists(os.path.join(d, "adapter_config.json")))

    src = ckpt_keys(d)
    print(f"  clés checkpoint: {len(src)}")

    Cls = getattr(M, cls_name)
    print("  _checkpoint_conversion_mapping de la classe:")
    for k, v in (getattr(Cls, "_checkpoint_conversion_mapping", {}) or {}).items():
        print(f"      {k!r} -> {v!r}")

    with warnings.catch_warnings(), redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        m = Cls.from_pretrained(hf, torch_dtype=torch.float32,
                                device_map="cpu", cache_dir=HF)
    tgt = set(m.state_dict().keys())
    print(f"  clés modèle (state_dict): {len(tgt)}")

    src_only = sorted({collapse(k) for k in (src - tgt)})
    tgt_only = sorted({collapse(k) for k in (tgt - src)})

    print(f"  --- DANS le checkpoint, ABSENT du modèle ({len(src_only)} motifs) ---")
    for k in src_only[:18]:
        print("    S", k)
    print(f"  --- le modèle ATTEND, ABSENT du checkpoint ({len(tgt_only)} motifs) ---")
    for k in tgt_only[:18]:
        print("    T", k)


analyze("ColPali",   "vidore/colpali-v1.3",          "ColPali")
analyze("ColGemma3", "Cognitive-Lab/ColNetraEmbed",  "ColGemma3")
print(f"\n{'='*70}\nPROBE_DONE")
