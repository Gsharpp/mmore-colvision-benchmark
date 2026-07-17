"""Smoke-test + config-introspection for the mmore-text (tf4) venv.

Run with the bcv-venv-mmoretext interpreter on the cluster. Prints the dataclass
fields of the mmore text indexing/retrieval configs so we can wire a runner that
drives `mmore process` -> `mmore index` (hybrid dense+SPLADE) -> `mmore retrieve`
against the existing corpora, aligning doc ids to "<pdf>#page=<N>".
"""
import dataclasses
import importlib
import inspect


def dump_dataclass(label, obj):
    print(f"--- {label}: {obj} ---")
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            default = f.default if f.default is not dataclasses.MISSING else "<required>"
            print(f"    {f.name}: {f.type}  = {default}")
    else:
        print("    (not a dataclass)")


print("=== version check ===")
import transformers
print("transformers", transformers.__version__)
import mmore
print("mmore at", mmore.__file__)

print("=== core imports ===")
from mmore.index.indexer import Indexer, IndexerConfig  # noqa: E402
from mmore.rag.model.sparse.splade import SpladeSparseEmbedding  # noqa: E402
print("Indexer + SpladeSparseEmbedding import OK")

print("=== config dataclasses ===")
dump_dataclass("IndexerConfig", IndexerConfig)
# Try to locate the sparse/dense model config dataclasses.
for modname, names in [
    ("mmore.index.indexer", ["SparseModelConfig", "DenseModelConfig", "DBConfig"]),
    ("mmore.rag.model.dense.base", ["DenseModelConfig"]),
    ("mmore.rag.model.sparse.base", ["SparseModelConfig"]),
]:
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        print(f"(skip {modname}: {e})")
        continue
    for n in names:
        obj = getattr(mod, n, None)
        if obj is not None:
            dump_dataclass(f"{modname}.{n}", obj)

print("=== Indexer.from_documents / run signatures ===")
for attr in ["__init__", "from_config", "index_documents", "from_documents"]:
    fn = getattr(Indexer, attr, None)
    if fn is not None:
        try:
            print(f"Indexer.{attr}{inspect.signature(fn)}")
        except (TypeError, ValueError):
            print(f"Indexer.{attr}: <no signature>")

print("=== mmore CLI entrypoints ===")
for modname in ["mmore.run_index", "mmore.run_index_api", "mmore.run_retriever"]:
    try:
        mod = importlib.import_module(modname)
        funcs = [n for n, o in vars(mod).items() if inspect.isfunction(o) and not n.startswith("_")]
        print(f"{modname}: {funcs}")
    except Exception as e:
        print(f"(skip {modname}: {e})")

print("PROBE_DONE")
