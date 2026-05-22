"""CLI entry point: bcv-queries."""

from __future__ import annotations

import json
from pathlib import Path

import click

from benchmark_colvision.queries.methodology_validation import correlate
from benchmark_colvision.queries.schema import QuerySet


@click.group()
def main() -> None:
    """Synthetic query generation, ambiguity filtering and methodology validation."""


@main.command("generate")
@click.option("--corpus-manifest", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--n-per-page", type=int, default=2, show_default=True)
def generate(corpus_manifest: Path, out: Path, n_per_page: int) -> None:
    """Generate inverse queries via Meditron-70B (vLLM client wired up externally).

    The actual LLM connection is wired by `scripts/generate_queries.sh` which
    binds an `LLMClient` adapter — this command only orchestrates the corpus
    walk and writes the JSONL output.
    """
    raise click.ClickException(
        "Wire a vLLM client adapter before running this command "
        "(see src/benchmark_colvision/queries/inverse_query_gen.py:LLMClient)"
    )


@main.command("filter")
@click.option("--in-jsonl", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--threshold", type=float, default=0.8, show_default=True)
def filter_(in_jsonl: Path, out_jsonl: Path, threshold: float) -> None:
    """Ambiguity filter — requires a configured judge client (see scripts/)."""
    raise click.ClickException(
        "Wire a judge client adapter before running this command "
        "(see src/benchmark_colvision/queries/ambiguity_filter.py:filter_queries)"
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
