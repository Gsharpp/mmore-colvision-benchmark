"""CLI entry point: bcv-queries (synthetic query generation, currently TODO)."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """Synthetic query generation and methodology validation."""


@main.command()
def generate() -> None:
    """Generate inverse queries from a corpus via Meditron-70B (vLLM)."""
    raise NotImplementedError("Pending Meditron-70B prompt design; see plan Phase 3.")


@main.command()
def validate() -> None:
    """Validate the auto-generation methodology against an annotated subset."""
    raise NotImplementedError("Pending choice of annotated reference subset; see plan Phase 4.")
