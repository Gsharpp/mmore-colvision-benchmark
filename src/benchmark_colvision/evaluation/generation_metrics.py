"""Generation-side metrics: RAGAS-style scores under an LLM judge.

RAGAS itself is imported lazily so that unit tests can run on a CPU-only host
without pulling its full dependency tree at import time. The public surface is
intentionally narrow: callers build `RagSample` objects and call `score()`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass
class RagSample:
    """One question/answer/context triple to evaluate."""

    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None


@dataclass
class GenerationMetricsResult:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    n_examples: int = 0
    judge_model: str | None = None
    per_sample: list[dict] = field(default_factory=list)


def _mean(xs: Iterable[float]) -> float | None:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def score_samples(
    samples: Sequence[RagSample],
    judge_model: str,
    metrics: Sequence[str] = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ),
    llm_callable=None,
) -> GenerationMetricsResult:
    """Score a batch of RAG samples with RAGAS.

    `llm_callable` is the LLM wrapper RAGAS will use as judge. When None, this
    function expects RAGAS to discover its own default — which only makes sense
    in an environment where the judge is already wired up. Tests should pass
    `llm_callable=FakeLLM(...)` and inject deterministic scores.
    """
    if not samples:
        return GenerationMetricsResult(n_examples=0, judge_model=judge_model)

    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas import metrics as ragas_metrics  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "ragas / datasets not installed — install the project with `uv sync` first"
        ) from e

    metric_map = {
        "faithfulness": ragas_metrics.faithfulness,
        "answer_relevancy": ragas_metrics.answer_relevancy,
        "context_precision": ragas_metrics.context_precision,
        "context_recall": ragas_metrics.context_recall,
    }
    selected = [metric_map[m] for m in metrics if m in metric_map]

    data = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
    }
    if any(s.reference is not None for s in samples):
        data["ground_truth"] = [s.reference or "" for s in samples]

    ds = Dataset.from_dict(data)
    result = evaluate(ds, metrics=selected, llm=llm_callable)
    scores = result.to_pandas().to_dict(orient="records") if hasattr(result, "to_pandas") else []
    aggregate = {
        m: _mean(r.get(m) for r in scores)
        for m in metric_map
        if m in metrics
    }
    return GenerationMetricsResult(
        faithfulness=aggregate.get("faithfulness"),
        answer_relevancy=aggregate.get("answer_relevancy"),
        context_precision=aggregate.get("context_precision"),
        context_recall=aggregate.get("context_recall"),
        n_examples=len(samples),
        judge_model=judge_model,
        per_sample=scores,
    )


def aggregate_results(
    results: Iterable[GenerationMetricsResult],
) -> GenerationMetricsResult:
    """Weighted mean across multiple `GenerationMetricsResult`s (e.g. per-seed)."""
    results = list(results)
    if not results:
        return GenerationMetricsResult()
    total_n = sum(r.n_examples for r in results) or 1

    def weighted(attr: str) -> float | None:
        num = 0.0
        denom = 0
        for r in results:
            v = getattr(r, attr)
            if v is None:
                continue
            num += v * r.n_examples
            denom += r.n_examples
        return num / denom if denom else None

    return GenerationMetricsResult(
        faithfulness=weighted("faithfulness"),
        answer_relevancy=weighted("answer_relevancy"),
        context_precision=weighted("context_precision"),
        context_recall=weighted("context_recall"),
        n_examples=total_n,
        judge_model=results[0].judge_model,
    )
