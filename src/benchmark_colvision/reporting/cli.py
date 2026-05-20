"""CLI entry point: bcv-report."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """Figure and table generation from result JSONs."""


@main.command()
def figures() -> None:
    """Render all matplotlib figures from results/."""
    raise NotImplementedError("Pending final results JSON schema; see plan Phase 7.")


@main.command()
def tables() -> None:
    """Render LaTeX tables for the technical report."""
    raise NotImplementedError("Pending final results JSON schema; see plan Phase 7.")
