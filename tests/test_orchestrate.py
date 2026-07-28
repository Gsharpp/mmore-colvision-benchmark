"""Tests for `runners.orchestrate` — the glue invoked by `bcv-run track-a/b`.

The mmore subprocess is monkeypatched to write a synthetic retrieve.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from benchmark_colvision.queries.schema import (
    QuerySet,
    SyntheticQuery,
    query_id_for,
)
from benchmark_colvision.runners import mmore_wrapper
from benchmark_colvision.runners.mmore_wrapper import CommandResult
from benchmark_colvision.runners.orchestrate import (
    run_track_a_for_model,
    run_track_b_for_model,
)


def _fake_mmore_item(question: str, pages: list[tuple[str, int]]) -> dict:
    """One entry in the real `mmore colvision retrieve` output format.

    Keyed on the query *text* with a `context` list whose `metadata` carries
    `pdf_name` + 1-based `page_number` — the shape the production parser reads.
    """
    return {
        "query": question,
        "context": [
            {
                "page_content": "...",
                "metadata": {
                    "pdf_name": pdf,
                    "pdf_path": f"data/pdfs/{pdf}",
                    "page_number": page,
                    "rank": rank,
                    "similarity": 1.0 - 0.1 * rank,
                },
            }
            for rank, (pdf, page) in enumerate(pages, start=1)
        ],
    }



def _make_queryset(scope: str, language: str, n: int = 4) -> QuerySet:
    queries = [
        SyntheticQuery(
            query_id=query_id_for(f"{scope}.pdf", i, i),
            question=f"Question {i} for {scope}",
            expected_answer=f"Answer {i}",
            source_pdf=f"{scope}.pdf",
            source_page=i,
            language=language,
        )
        for i in range(n)
    ]
    return QuerySet(name=scope, language=language, queries=queries)


def _empty_corpus_manifest(path: Path, track: str, languages: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "track": track,
                "languages": languages,
                "total_pages": 0,
                "pdfs": [],
            }
        )
    )


def _make_fake_run():
    """Replacement for mmore_wrapper._run that always succeeds and writes a
    synthetic retrieve.json with the relevant doc at rank 0."""

    def fake_run(cmd, cwd=None, env=None):
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
            data = []
            for q in qs.queries:
                # mmore's real output: page numbers are 1-based (page_num + 1).
                data.append(
                    _fake_mmore_item(
                        q.question,
                        [(q.source_pdf, q.source_page + 1), ("distractor.pdf", 99)],
                    )
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(data))
        now = time.time()
        duration = 0.1
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


def _write_configs_track_a(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal pair of (track_a.yaml, models.yaml) under tmp_path/configs/."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    queries_dir = tmp_path / "data" / "track_a" / "queries"
    queries_dir.mkdir(parents=True)
    mmore_cfg_dir = cfg_dir / "mmore"
    mmore_cfg_dir.mkdir()
    for name in ("process.yaml", "index.yaml", "retrieve.yaml"):
        (mmore_cfg_dir / name).write_text(f"name: {name}\n")

    manifest_path = tmp_path / "data" / "track_a" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _empty_corpus_manifest(manifest_path, track="A", languages=["en"])

    # Two paliers, two queries each
    for palier in ("tiny", "small"):
        _make_queryset(palier, "en", n=4).save_jsonl(queries_dir / f"{palier}.jsonl")

    track_cfg = {
        "scaling_paliers": [
            {"id": "tiny", "n_pages": 100},
            {"id": "small", "n_pages": 1000},
        ],
        "seeds": [0, 1, 2],
        "paths": {
            "corpus_manifest": "data/track_a/corpus_manifest.json",
            "queries_dir": "data/track_a/queries",
            "mmore_process_config": "configs/mmore/process.yaml",
            "mmore_index_config": "configs/mmore/index.yaml",
            "mmore_retrieve_config": "configs/mmore/retrieve.yaml",
            "mmore_output_dir": "data/track_a/retrieve",
            "records_dir": "results/track_a",
        },
    }
    models_cfg = {
        "models": [
            {"id": "colpali_v1_3", "hf_name": "vidore/colpali-v1.3"},
            {"id": "colqwen3_v0_1", "hf_name": "vidore/colqwen3-v0.1"},
        ]
    }
    (cfg_dir / "track_a.yaml").write_text(yaml.safe_dump(track_cfg))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump(models_cfg))
    return cfg_dir / "track_a.yaml", cfg_dir / "models.yaml"


def _write_configs_track_b(tmp_path: Path) -> tuple[Path, Path]:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    queries_dir = tmp_path / "data" / "track_b" / "queries"
    queries_dir.mkdir(parents=True)
    mmore_cfg_dir = cfg_dir / "mmore"
    mmore_cfg_dir.mkdir()
    for name in ("process.yaml", "index.yaml", "retrieve.yaml"):
        (mmore_cfg_dir / name).write_text(f"name: {name}\n")

    manifest_path = tmp_path / "data" / "track_b" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _empty_corpus_manifest(manifest_path, track="B", languages=["en", "fr"])
    (tmp_path / "data" / "track_b" / "pdfs").mkdir(parents=True, exist_ok=True)

    for lang in ("en", "fr"):
        _make_queryset(lang, lang, n=4).save_jsonl(queries_dir / f"{lang}.jsonl")

    track_cfg = {
        "languages": [
            {"code": "en", "name": "English", "source": "pmc-oa", "pages_target": 1500},
            {"code": "fr", "name": "French", "source": "hal", "pages_target": 1500},
        ],
        "seeds": [0, 1, 2],
        "queries": {"per_language": 200},
        "paths": {
            "corpus_manifest": "data/track_b/corpus_manifest.json",
            "corpus_pdfs": "data/track_b/pdfs",
            "queries_dir": "data/track_b/queries",
            "mmore_process_config": "configs/mmore/process.yaml",
            "mmore_index_config": "configs/mmore/index.yaml",
            "mmore_retrieve_config": "configs/mmore/retrieve.yaml",
            "mmore_output_dir": "data/track_b/retrieve",
            "records_dir": "results/track_b",
        },
    }
    models_cfg = {
        "models": [{"id": "colpali_v1_3", "hf_name": "vidore/colpali-v1.3"}]
    }
    (cfg_dir / "track_b.yaml").write_text(yaml.safe_dump(track_cfg))
    (cfg_dir / "models.yaml").write_text(yaml.safe_dump(models_cfg))
    return cfg_dir / "track_b.yaml", cfg_dir / "models.yaml"


def test_run_track_a_for_model_writes_records(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run())
    track_cfg, models_cfg = _write_configs_track_a(tmp_path)

    records = run_track_a_for_model(
        model_id="colpali_v1_3",
        seed=0,
        track_config=track_cfg,
        models_config=models_cfg,
        mmore_commit="abc123",
        benchmark_version="0.1.0",
        base_dir=tmp_path,
    )
    assert len(records) == 2
    palier_ids = {r.cell.palier_id for r in records}
    assert palier_ids == {"tiny", "small"}
    for r in records:
        assert r.cell.track == "A"
        assert r.cell.model_id == "colpali_v1_3"
        assert r.cell.seed == 0
        assert r.mmore_commit == "abc123"
        # ndcg@5 should be 1.0 since the fake puts the relevant doc at rank 0
        assert r.retrieval.ndcg_at_5 == 1.0

    # Record JSONs on disk
    for palier in ("tiny", "small"):
        out = tmp_path / "results" / "track_a" / "colpali_v1_3" / palier / "seed_0.json"
        assert out.exists()


def test_run_track_a_palier_filter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run())
    track_cfg, models_cfg = _write_configs_track_a(tmp_path)
    records = run_track_a_for_model(
        model_id="colpali_v1_3",
        seed=0,
        track_config=track_cfg,
        models_config=models_cfg,
        mmore_commit="abc",
        benchmark_version="0.1.0",
        base_dir=tmp_path,
        palier_filter=["tiny"],
    )
    assert len(records) == 1
    assert records[0].cell.palier_id == "tiny"


def test_run_track_a_unknown_model_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run())
    track_cfg, models_cfg = _write_configs_track_a(tmp_path)
    with pytest.raises(KeyError, match="not found"):
        run_track_a_for_model(
            model_id="nonexistent",
            seed=0,
            track_config=track_cfg,
            models_config=models_cfg,
            mmore_commit="abc",
            benchmark_version="0.1.0",
            base_dir=tmp_path,
        )


def test_run_track_a_missing_manifest_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run())
    track_cfg, models_cfg = _write_configs_track_a(tmp_path)
    # Delete the manifest
    (tmp_path / "data" / "track_a" / "corpus_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="corpus manifest"):
        run_track_a_for_model(
            model_id="colpali_v1_3",
            seed=0,
            track_config=track_cfg,
            models_config=models_cfg,
            mmore_commit="abc",
            benchmark_version="0.1.0",
            base_dir=tmp_path,
        )


def test_run_track_b_for_model_writes_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run())
    track_cfg, models_cfg = _write_configs_track_b(tmp_path)
    record = run_track_b_for_model(
        model_id="colpali_v1_3",
        language="fr",
        seed=1,
        track_config=track_cfg,
        models_config=models_cfg,
        mmore_commit="def456",
        benchmark_version="0.1.0",
        base_dir=tmp_path,
    )
    assert record.cell.track == "B"
    assert record.cell.language == "fr"
    assert record.cell.seed == 1
    assert record.mmore_commit == "def456"
    out = tmp_path / "results" / "track_b" / "colpali_v1_3" / "fr" / "seed_1.json"
    assert out.exists()

    # Per-(model, language) configs are rendered with isolated output paths and
    # the language-specific PDF directory injected as data_path.
    cell_dir = tmp_path / "configs" / "mmore" / "cells" / "colpali_v1_3_fr"
    proc_cfg = yaml.safe_load((cell_dir / "process.yaml").read_text())
    assert proc_cfg["data_path"] == str(tmp_path / "data" / "track_b" / "pdfs")
    assert proc_cfg["output_path"] == str(
        tmp_path / "data" / "track_b" / "process" / "colpali_v1_3_fr"
    )
    idx_cfg = yaml.safe_load((cell_dir / "index.yaml").read_text())
    assert idx_cfg["milvus"]["collection_name"] == "bcv_colpali_v1_3_fr_pages"


def test_run_track_b_missing_pdfs_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run())
    track_cfg, models_cfg = _write_configs_track_b(tmp_path)
    # Remove the language PDF directory the renderer needs.
    (tmp_path / "data" / "track_b" / "pdfs").rmdir()
    with pytest.raises(FileNotFoundError, match="corpus PDF directory"):
        run_track_b_for_model(
            model_id="colpali_v1_3",
            language="fr",
            seed=0,
            track_config=track_cfg,
            models_config=models_cfg,
            mmore_commit="x",
            benchmark_version="0.1.0",
            base_dir=tmp_path,
        )


def test_run_track_b_unknown_language_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mmore_wrapper, "_run", _make_fake_run())
    track_cfg, models_cfg = _write_configs_track_b(tmp_path)
    with pytest.raises(KeyError, match="not declared"):
        run_track_b_for_model(
            model_id="colpali_v1_3",
            language="ja",
            seed=0,
            track_config=track_cfg,
            models_config=models_cfg,
            mmore_commit="x",
            benchmark_version="0.1.0",
            base_dir=tmp_path,
        )
