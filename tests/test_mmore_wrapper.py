"""Unit tests for runners.mmore_wrapper (subprocess is mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from benchmark_colvision.runners import mmore_wrapper
from benchmark_colvision.runners.mmore_wrapper import (
    MmoreRun,
    format_command,
    run_pipeline,
    save_run,
)


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_fake_run(returncode=0):
    return lambda *args, **kwargs: FakeProc(returncode=returncode, stdout="ok", stderr="")


def test_run_pipeline_invokes_three_steps(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run = run_pipeline(
        model_name="vidore/colpali-v1.3",
        process_config=tmp_path / "p.yml",
        index_config=tmp_path / "i.yml",
        retrieve_config=tmp_path / "r.yml",
        queries_file=tmp_path / "q.jsonl",
        output_file=tmp_path / "out.json",
    )
    assert len(calls) == 3
    assert "process" in calls[0]
    assert "index" in calls[1]
    assert "retrieve" in calls[2]
    assert run.process and run.process.succeeded
    assert run.index and run.index.succeeded
    assert run.retrieve and run.retrieve.succeeded


def test_run_pipeline_skips_after_failure(monkeypatch, tmp_path: Path):
    state = {"i": 0}

    def fake_run(cmd, **kwargs):
        i = state["i"]
        state["i"] += 1
        return FakeProc(returncode=0 if i == 0 else 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run = run_pipeline(
        model_name="m",
        process_config=tmp_path / "p.yml",
        index_config=tmp_path / "i.yml",
        retrieve_config=tmp_path / "r.yml",
        queries_file=tmp_path / "q.jsonl",
        output_file=tmp_path / "out.json",
    )
    assert run.process and run.process.succeeded
    assert run.index and not run.index.succeeded
    assert run.retrieve is None


def test_save_run_writes_json(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc(0, "ok", ""))
    run = MmoreRun(model_name="m")
    out = tmp_path / "nested/run.json"
    save_run(run, out)
    assert out.exists()
    data = out.read_text()
    assert "model_name" in data


def test_format_command_quotes_spaces():
    assert format_command(["python", "-m", "mmore", "--config", "/tmp/a b.yml"]) == \
        "python -m mmore --config '/tmp/a b.yml'"


@pytest.mark.parametrize("rc, ok", [(0, True), (1, False), (137, False)])
def test_command_result_succeeded(monkeypatch, rc, ok, tmp_path: Path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc(returncode=rc))
    result = mmore_wrapper.run_index(tmp_path / "x.yml")
    assert result.succeeded is ok
