"""End-to-end smoke test: fake the mmore subprocess, exercise the full pipeline.

This test wires together the orchestrator, the retrieval output parser, the
metrics, the result schema, the aggregator (CI + pairwise Wilcoxon), and the
reporting layer (figures + LaTeX). The mmore CLI is monkeypatched to write a
synthetic retrieve.json instead of running the real binary, so the test runs
without GPU, network or torch.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import pytest

from benchmark_colvision.queries.schema import (
    QuerySet,
    SyntheticQuery,
    query_id_for,
)
from benchmark_colvision.reporting.figures_track_a import scaling_curve
from benchmark_colvision.reporting.figures_track_b import heatmap_metric
from benchmark_colvision.reporting.latex_tables import track_a_table, track_b_table
from benchmark_colvision.results.aggregate import (
    load_records,
    pairwise_wilcoxon,
    per_cell_ci,
    records_to_frame,
)
from benchmark_colvision.runners import mmore_wrapper
from benchmark_colvision.runners.mmore_wrapper import CommandResult
from benchmark_colvision.runners.run_track_a import TrackACell
from benchmark_colvision.runners.run_track_a import run_cell as run_cell_a
from benchmark_colvision.runners.run_track_b import TrackBCell
from benchmark_colvision.runners.run_track_b import run_cell as run_cell_b


def _deterministic_seed(*parts) -> int:
    """Hash-based seed that doesn't depend on PYTHONHASHSEED."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _make_queryset(scope_id: str, language: str, n_queries: int = 8) -> QuerySet:
    queries = []
    for i in range(n_queries):
        pdf = f"{scope_id}/doc_{i // 2}.pdf"
        page = i % 2
        queries.append(
            SyntheticQuery(
                query_id=query_id_for(pdf, page, i),
                question=f"What does page {page} of doc {i // 2} show?",
                expected_answer=f"Synthetic ground truth {i}",
                source_pdf=pdf,
                source_page=page,
                language=language,
            )
        )
    return QuerySet(name=scope_id, language=language, queries=queries)


def _make_fake_run(skill_by_model: dict[str, float]):
    """Monkeypatch replacement for `mmore_wrapper._run`.

    process / index are no-ops. retrieve loads the queries file, then writes a
    synthetic retrieve.json where the relevant page is returned at rank 0 with
    probability `skill`, otherwise pushed behind two distractors. Each cell
    gets its own deterministic rng (seeded from model + output path) so
    different (model, palier, seed) cells produce different scores — enough
    variance for pairwise Wilcoxon to compute a p-value.
    """
    state = {"current_model": None}

    def fake_run(cmd, cwd=None, env=None):
        # `python -m mmore colpali process -m <model_name>` — the first `-m`
        # is Python's module flag, the second is mmore's model selector. We
        # want the one that comes after the subcommand.
        m_indices = [i for i, tok in enumerate(cmd) if tok == "-m"]
        if len(m_indices) >= 2:
            state["current_model"] = cmd[m_indices[-1] + 1]
        is_process = "process" in cmd
        is_index = "index" in cmd
        is_retrieve = "retrieve" in cmd

        if is_retrieve:
            queries_path = Path(cmd[cmd.index("-f") + 1])
            output_path = Path(cmd[cmd.index("-o") + 1])
            # run_pipeline passes mmore's converted "<stem>_mmore.jsonl" (plain
            # question strings) to -f; recover the original SyntheticQuery JSONL.
            if queries_path.stem.endswith("_mmore"):
                queries_path = queries_path.parent / (
                    queries_path.stem[: -len("_mmore")] + ".jsonl"
                )
            qs = QuerySet.load_jsonl(queries_path, name="x", language="en")
            model = state.get("current_model") or "unknown"
            skill = skill_by_model.get(model, 0.5)
            rng = random.Random(_deterministic_seed(model, str(output_path)))
            data = []
            for q in qs.queries:
                relevant_doc = f"{q.source_pdf}#page={q.source_page}"
                if rng.random() < skill:
                    docs = [relevant_doc, "distractor1.pdf#page=0", "distractor2.pdf#page=0"]
                else:
                    docs = ["distractor1.pdf#page=0", "distractor2.pdf#page=0", relevant_doc]
                data.append(
                    {
                        "query_id": q.query_id,
                        "results": [
                            {"document_id": d, "score": 1.0 - 0.1 * k}
                            for k, d in enumerate(docs)
                        ],
                    }
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(data))

        now = time.time()
        duration = 1.0 if is_process else 0.5 if is_index else 0.2
        return CommandResult(
            command=cmd,
            returncode=0,
            stdout="",
            stderr="",
            duration_s=duration,
            started_at=now,
            finished_at=now + duration,
        )

    return fake_run


def _empty_yaml(tmp_path: Path, name: str) -> Path:
    p = tmp_path / f"{name}.yaml"
    p.write_text(f"name: {name}\n")
    return p


def test_end_to_end_track_a_smoke(tmp_path: Path, monkeypatch) -> None:
    """3 models × 2 paliers × 2 seeds, mocked, end-to-end through reporting."""
    models = [
        ("colpali_v1_3", "vidore/colpali-v1.3", 0.50),
        ("colqwen3_v0_1", "vidore/colqwen3-v0.1", 0.95),
        ("colgemma3_colnetra", "Cognitive-Lab/ColNetraEmbed", 0.75),
    ]
    paliers = [("tiny", 100), ("small", 1000)]
    seeds = [0, 1]

    skill = {hf: s for _, hf, s in models}
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run(skill))

    results_dir = tmp_path / "results"
    queries_dir = tmp_path / "queries"
    queries_dir.mkdir()
    proc_cfg = _empty_yaml(tmp_path, "process")
    idx_cfg = _empty_yaml(tmp_path, "index")
    ret_cfg = _empty_yaml(tmp_path, "retrieve")

    queryset_path_for: dict[str, Path] = {}
    for palier_id, _ in paliers:
        qs = _make_queryset(palier_id, language="en", n_queries=8)
        p = queries_dir / f"{palier_id}.jsonl"
        qs.save_jsonl(p)
        queryset_path_for[palier_id] = p

    n_expected = len(models) * len(paliers) * len(seeds)
    for model_id, hf_name, _ in models:
        for palier_id, n_pages in paliers:
            for seed in seeds:
                cell = TrackACell(
                    model_id=model_id,
                    model_hf_name=hf_name,
                    palier_id=palier_id,
                    seed=seed,
                    process_config=proc_cfg,
                    index_config=idx_cfg,
                    retrieve_config=ret_cfg,
                    queries_file=queryset_path_for[palier_id],
                    queryset_jsonl=queryset_path_for[palier_id],
                    output_file=tmp_path / "retrieve" / f"{model_id}__{palier_id}__seed{seed}.json",
                    record_out=results_dir / f"{model_id}__{palier_id}__seed{seed}.json",
                )
                rec = run_cell_a(
                    cell,
                    mmore_commit="abc123",
                    benchmark_version="0.1.0",
                    corpus_manifest_sha256="deadbeefcafe",
                    n_pages_in_palier=n_pages,
                )
                assert rec.retrieval.n_queries == 8
                assert rec.performance.process_duration_s == 1.0
                assert rec.performance.index_duration_s == 0.5
                assert rec.performance.retrieve_duration_total_s == 0.2
                assert rec.performance.throughput_pages_per_s == n_pages
                assert rec.mmore_commit == "abc123"

    # Records on disk
    records = load_records(results_dir)
    assert len(records) == n_expected

    df = records_to_frame(records)
    assert set(df["model_id"].unique()) == {m[0] for m in models}
    assert set(df["palier_id"].unique()) == {p[0] for p in paliers}

    # Bootstrap CIs are well-formed
    ci = per_cell_ci(df, metric="retrieval.ndcg_at_5", group_cols=["model_id", "palier_id"])
    assert len(ci) == len(models) * len(paliers)
    assert (ci["lower"] <= ci["point"]).all()
    assert (ci["point"] <= ci["upper"]).all()

    # The high-skill model should out-score the low-skill model on at least one palier
    by_model = ci.groupby("model_id")["point"].max()
    assert by_model["colqwen3_v0_1"] > by_model["colpali_v1_3"]

    # Pairwise Wilcoxon — structure check; tolerate the all-zero-diff edge case
    try:
        pw = pairwise_wilcoxon(df, metric="retrieval.ndcg_at_5")
    except ValueError:
        pytest.skip("rng produced all-tied scores for at least one pair")
    assert set(pw.columns) >= {"a", "b", "statistic", "pvalue", "n_pairs", "pvalue_holm"}
    assert len(pw) == 3  # 3 models → 3 unordered pairs
    assert ((pw["pvalue_holm"] >= 0) & (pw["pvalue_holm"] <= 1)).all()

    # Figures + LaTeX tables render
    fig_path = scaling_curve(
        df, metric="retrieval.ndcg_at_5", out_path=tmp_path / "scaling.png"
    )
    assert fig_path.exists() and fig_path.stat().st_size > 0

    tex = track_a_table(df, metric="retrieval.ndcg_at_5")
    assert "\\begin{tabular}" in tex
    for model_id, _, _ in models:
        assert model_id in tex
    for palier_id, _ in paliers:
        assert palier_id in tex


def test_end_to_end_track_b_smoke(tmp_path: Path, monkeypatch) -> None:
    """2 models × 3 languages × 1 seed = 6 cells, including a CJK and a RTL code."""
    models = [
        ("colpali_v1_3", "vidore/colpali-v1.3", 0.55),
        ("colgemma3_colnetra", "Cognitive-Lab/ColNetraEmbed", 0.80),
    ]
    languages = [("en", 1500), ("zh", 1500), ("ar", 1500)]
    seed = 0

    skill = {hf: s for _, hf, s in models}
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run(skill))

    results_dir = tmp_path / "results"
    queries_dir = tmp_path / "queries"
    queries_dir.mkdir()
    proc_cfg = _empty_yaml(tmp_path, "process")
    idx_cfg = _empty_yaml(tmp_path, "index")
    ret_cfg = _empty_yaml(tmp_path, "retrieve")

    queryset_path_for: dict[str, Path] = {}
    for lang, _ in languages:
        qs = _make_queryset(lang, language=lang, n_queries=8)
        p = queries_dir / f"{lang}.jsonl"
        qs.save_jsonl(p)
        queryset_path_for[lang] = p

    for model_id, hf_name, _ in models:
        for lang, n_pages in languages:
            cell = TrackBCell(
                model_id=model_id,
                model_hf_name=hf_name,
                language=lang,
                seed=seed,
                process_config=proc_cfg,
                index_config=idx_cfg,
                retrieve_config=ret_cfg,
                queries_file=queryset_path_for[lang],
                queryset_jsonl=queryset_path_for[lang],
                output_file=tmp_path / "retrieve" / f"{model_id}__{lang}__seed{seed}.json",
                record_out=results_dir / f"{model_id}__{lang}__seed{seed}.json",
            )
            rec = run_cell_b(
                cell,
                mmore_commit="abc123",
                benchmark_version="0.1.0",
                corpus_manifest_sha256="deadbeefcafe",
                n_pages_in_language=n_pages,
            )
            assert rec.cell.track == "B"
            assert rec.cell.language == lang
            assert rec.retrieval.n_queries == 8

    records = load_records(results_dir)
    assert len(records) == len(models) * len(languages)
    df = records_to_frame(records)
    assert set(df["language"].unique()) == {lang for lang, _ in languages}

    # The Track B figures + tables render with the language axis populated
    fig = heatmap_metric(
        df, metric="retrieval.ndcg_at_5", out_path=tmp_path / "heat.png"
    )
    assert fig.exists() and fig.stat().st_size > 0

    tex = track_b_table(df, metric="retrieval.ndcg_at_5")
    assert "\\begin{tabular}" in tex
    for lang, _ in languages:
        assert lang in tex
