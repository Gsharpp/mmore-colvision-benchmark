"""CLI entry point: bcv-queries."""

from __future__ import annotations

import json
from pathlib import Path

import click

from benchmark_colvision.clients.vllm_client import VLLMClient, VLLMSamplingDefaults
from benchmark_colvision.corpus.corpus_manifest import CorpusManifest
from benchmark_colvision.queries.ambiguity_filter import filter_queries
from benchmark_colvision.queries.inverse_query_gen import generate_for_pages
from benchmark_colvision.queries.methodology_validation import correlate
from benchmark_colvision.queries.mock import generate_mock_for_pages
from benchmark_colvision.queries.pdf_pages import iter_manifest_pages, page_text_index
from benchmark_colvision.queries.schema import QuerySet


@click.group()
def main() -> None:
    """Synthetic query generation, ambiguity filtering and methodology validation."""


@main.command("generate")
@click.option("--corpus-manifest", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--corpus-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--vllm-endpoint", required=True, help="Base URL of the vLLM server (e.g. http://node:8000)")
@click.option("--vllm-model", required=True, help="Model name as registered on the vLLM server")
@click.option("--n-per-page", type=int, default=2, show_default=True)
@click.option("--max-pages", type=int, default=None, help="Cap for smoke tests")
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--max-tokens", type=int, default=512, show_default=True)
@click.option("--name", default=None, help="QuerySet name (defaults to manifest name)")
@click.option(
    "--query-mode",
    type=click.Choice(["text", "visual", "mixed"]),
    default="mixed",
    show_default=True,
    help=(
        "text: text questions only (all pages). "
        "visual: visual questions only (figures/tables pages). "
        "mixed: 1 visual + 1 text on visual pages, 2 text on others."
    ),
)
@click.option("--few-shot/--no-few-shot", default=False, show_default=True,
              help="Prepend a one-shot example to the prompt (helps base LLMs like Meditron).")
@click.option("--guided-json/--no-guided-json", default=False, show_default=True,
              help="Enable vLLM guided JSON decoding (xgrammar backend).")
@click.option("--chat/--no-chat", default=False, show_default=True,
              help="Use the /v1/chat/completions endpoint (applies the model's chat "
                   "template). Required for instruction-tuned models (e.g. Qwen2.5-Instruct).")
@click.option("--query-language", default=None,
              help="ISO code (e.g. 'en') forcing the language the questions are written in, "
                   "regardless of the document language (cross-lingual retrieval). "
                   "Defaults to the document language.")
def generate(
    corpus_manifest: Path,
    corpus_root: Path,
    out: Path,
    vllm_endpoint: str,
    vllm_model: str,
    n_per_page: int,
    max_pages: int | None,
    temperature: float,
    max_tokens: int,
    name: str | None,
    query_mode: str,
    few_shot: bool,
    guided_json: bool,
    chat: bool,
    query_language: str | None,
) -> None:
    """Generate inverse queries by walking the corpus manifest and calling vLLM."""
    manifest = CorpusManifest.load(corpus_manifest)
    pages_iter = iter_manifest_pages(manifest, corpus_root)
    if max_pages is not None:
        pages_iter = (p for i, p in enumerate(pages_iter) if i < max_pages)

    defaults = VLLMSamplingDefaults(temperature=temperature, max_tokens=max_tokens)
    with VLLMClient(endpoint=vllm_endpoint, model=vllm_model, defaults=defaults, chat=chat) as llm:
        queries = generate_for_pages(pages_iter, llm, n_per_page=n_per_page,
                                     query_mode=query_mode,
                                     few_shot=few_shot, use_guided_json=guided_json,
                                     query_language=query_language)

    language = query_language or (manifest.languages[0] if len(manifest.languages) == 1 else "mixed")
    qs = QuerySet(name=name or manifest.name, language=language, queries=queries)
    qs.save_jsonl(out)
    click.echo(json.dumps({"out": str(out), "n_queries": len(queries), "sha256": qs.sha256()}))


@main.command("mock")
@click.option("--corpus-manifest", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--corpus-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--n-per-page", type=int, default=2, show_default=True)
@click.option("--max-pages", type=int, default=None, help="Cap the number of pages walked")
@click.option("--name", default=None, help="QuerySet name (defaults to manifest name)")
@click.option("--keep-visual-only/--all-pages", default=True, show_default=True,
              help="Keep only pages with figures/tables (the visual-retrieval focus)")
def mock(
    corpus_manifest: Path,
    corpus_root: Path,
    out: Path,
    n_per_page: int,
    max_pages: int | None,
    name: str | None,
    keep_visual_only: bool,
) -> None:
    """LLM-free fallback: build templated, page-grounded queries (PRELIMINARY).

    No vLLM/API needed. Each query is grounded on its source page for exact
    retrieval ground truth; phrasing is templated from the page's distinctive
    terms. Real LLM generation (`generate`) is the production path.
    """
    manifest = CorpusManifest.load(corpus_manifest)
    pages_iter = iter_manifest_pages(manifest, corpus_root)
    if max_pages is not None:
        pages_iter = (p for i, p in enumerate(pages_iter) if i < max_pages)

    queries = generate_mock_for_pages(
        pages_iter, n_per_page=n_per_page, keep_visual_only=keep_visual_only
    )
    language = manifest.languages[0] if len(manifest.languages) == 1 else "mixed"
    qs = QuerySet(name=name or manifest.name, language=language, queries=queries)
    qs.save_jsonl(out)
    click.echo(json.dumps({"out": str(out), "n_queries": len(queries), "sha256": qs.sha256()}))


@main.command("filter")
@click.option("--in-jsonl", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--corpus-manifest", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--corpus-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--vllm-endpoint", required=True)
@click.option("--vllm-model", required=True)
@click.option("--threshold", type=float, default=0.8, show_default=True)
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--max-tokens", type=int, default=16, show_default=True)
def filter_(
    in_jsonl: Path,
    out_jsonl: Path,
    corpus_manifest: Path,
    corpus_root: Path,
    vllm_endpoint: str,
    vllm_model: str,
    threshold: float,
    temperature: float,
    max_tokens: int,
) -> None:
    """Score each query with the judge and drop those below threshold."""
    qs = QuerySet.load_jsonl(in_jsonl, name=in_jsonl.stem, language="unknown")
    manifest = CorpusManifest.load(corpus_manifest)
    needed = {f"{q.source_pdf}#page={q.source_page}" for q in qs.queries}
    text_for = page_text_index(manifest, corpus_root, needed_ids=needed)

    defaults = VLLMSamplingDefaults(temperature=temperature, max_tokens=max_tokens)
    with VLLMClient(endpoint=vllm_endpoint, model=vllm_model, defaults=defaults) as llm:
        kept, report = filter_queries(qs.queries, text_for, llm, threshold=threshold)

    out_qs = QuerySet(name=qs.name, language=qs.language, queries=kept)
    out_qs.save_jsonl(out_jsonl)
    click.echo(
        json.dumps(
            {
                "out": str(out_jsonl),
                "n_in": report.n_in,
                "n_kept": report.n_kept,
                "n_dropped": report.n_dropped,
                "mean_score": round(report.mean_score, 4),
                "threshold": report.threshold,
                "sha256": out_qs.sha256(),
            }
        )
    )


@main.command("validate")
@click.option("--auto-scores", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--human-scores", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--subset", required=True)
@click.option("--threshold", type=float, default=0.85, show_default=True)
def validate(auto_scores: Path, human_scores: Path, subset: str, threshold: float) -> None:
    """Compare auto vs human per-query nDCG@k (JSON list of floats each)."""
    a = json.loads(auto_scores.read_text())
    h = json.loads(human_scores.read_text())
    rep = correlate(a, h, subset_name=subset, threshold=threshold)
    click.echo(json.dumps(rep.__dict__, indent=2))


@main.command("hash")
@click.argument("jsonl", type=click.Path(exists=True, path_type=Path))
@click.option("--language", default="unknown")
def hash_(jsonl: Path, language: str) -> None:
    """Print the SHA-256 of a query set JSONL (used for provenance in results)."""
    qs = QuerySet.load_jsonl(jsonl, name=jsonl.stem, language=language)
    click.echo(qs.sha256())
