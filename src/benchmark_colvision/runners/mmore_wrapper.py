"""Subprocess wrappers around `python -m mmore colpali {process,index,retrieve}`.

We invoke the real mmore CLI rather than importing its internals so that the
benchmark measures the same code path a user would actually run.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    """Outcome of a single mmore CLI invocation."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    started_at: float
    finished_at: float

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


@dataclass
class MmoreRun:
    """Aggregate result of a process → index → retrieve sequence."""

    model_name: str
    process: CommandResult | None = None
    index: CommandResult | None = None
    retrieve: CommandResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "process": _result_to_dict(self.process),
            "index": _result_to_dict(self.index),
            "retrieve": _result_to_dict(self.retrieve),
            "extra": self.extra,
        }


def _result_to_dict(r: CommandResult | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return {
        "command": r.command,
        "returncode": r.returncode,
        "duration_s": r.duration_s,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "stdout_tail": r.stdout[-2000:],
        "stderr_tail": r.stderr[-2000:],
    }


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    finished = time.time()
    return CommandResult(
        command=cmd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_s=finished - started,
        started_at=started,
        finished_at=finished,
    )


def run_process(
    config_path: Path,
    model_name: str | None = None,
    python: str = sys.executable,
) -> CommandResult:
    """Run `mmore colpali process --config-file <path> [-m <model>]`."""
    cmd = [python, "-m", "mmore", "colpali", "process", "--config-file", str(config_path)]
    if model_name is not None:
        cmd += ["-m", model_name]
    return _run(cmd)


def run_index(config_path: Path, python: str = sys.executable) -> CommandResult:
    """Run `mmore colpali index --config-file <path>`."""
    cmd = [python, "-m", "mmore", "colpali", "index", "--config-file", str(config_path)]
    return _run(cmd)


def run_retrieve(
    config_path: Path,
    queries_file: Path | None = None,
    output_file: Path | None = None,
    python: str = sys.executable,
) -> CommandResult:
    """Run `mmore colpali retrieve --config-file <path> [-f <queries>] [-o <out>]`."""
    cmd = [python, "-m", "mmore", "colpali", "retrieve", "--config-file", str(config_path)]
    if queries_file is not None:
        cmd += ["-f", str(queries_file)]
    if output_file is not None:
        cmd += ["-o", str(output_file)]
    return _run(cmd)


def _make_mmore_qf(queries_file: Path) -> Path:
    """Convert SyntheticQuery JSONL to the plain-string-per-line format mmore expects.

    mmore's retrieve DataLoader passes each parsed JSON value directly to
    colpali_engine's process_queries(), which concatenates strings. Our JSONL
    has full SyntheticQuery dicts per line — mmore gets a dict instead of a str
    and raises TypeError. This converter extracts the 'question' field and
    writes one JSON-encoded string per line, which mmore can handle.
    """
    import json as _json
    out = queries_file.parent / (queries_file.stem + "_mmore.jsonl")
    lines = []
    for line in queries_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = _json.loads(line)
            q = obj.get("question") or obj.get("query") or str(obj)
        except Exception:
            q = line.strip()
        lines.append(_json.dumps(q))
    out.write_text("\n".join(lines) + "\n")
    return out


def _ensure_milvus_dir(index_config: Path) -> None:
    """Parse db_path from the index YAML and mkdir its parent (Milvus won't do it)."""
    try:
        import yaml  # type: ignore[import-not-found]
        cfg = yaml.safe_load(index_config.read_text())
        db_path = cfg.get("milvus", {}).get("db_path") or cfg.get("db_path")
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # best-effort; Milvus itself will error with a clear message if this fails


def run_pipeline(
    model_name: str,
    process_config: Path,
    index_config: Path,
    retrieve_config: Path,
    queries_file: Path,
    output_file: Path,
    python: str = sys.executable,
) -> MmoreRun:
    """Execute the full process → index → retrieve sequence for one model."""
    run = MmoreRun(model_name=model_name)
    run.process = run_process(process_config, model_name=model_name, python=python)
    if not run.process.succeeded:
        return run
    _ensure_milvus_dir(index_config)
    run.index = run_index(index_config, python=python)
    if not run.index.succeeded:
        return run
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mmore_qf = _make_mmore_qf(queries_file)
    run.retrieve = run_retrieve(retrieve_config, mmore_qf, output_file, python=python)
    return run


def save_run(run: MmoreRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2))


def format_command(cmd: list[str]) -> str:
    """Shell-quote a command for logging."""
    return " ".join(shlex.quote(c) for c in cmd)
