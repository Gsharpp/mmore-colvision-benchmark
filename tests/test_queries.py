"""Unit tests for the synthetic query pipeline (no real LLM calls)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmark_colvision.queries.ambiguity_filter import (
    filter_queries,
    parse_score,
)
from benchmark_colvision.queries.inverse_query_gen import (
    PageInput,
    build_prompt,
    generate_for_page,
    parse_completion,
)
from benchmark_colvision.queries.methodology_validation import correlate
from benchmark_colvision.queries.schema import QuerySet, SyntheticQuery, query_id_for


@dataclass
class FakeLLM:
    """Deterministic LLM stub: returns the next canned completion on every call."""

    responses: list[str]
    i: int = 0

    def complete(self, prompt: str, **kw) -> str:
        out = self.responses[self.i % len(self.responses)]
        self.i += 1
        return out


def test_build_prompt_truncates_long_text() -> None:
    page = PageInput(
        pdf_path="x.pdf",
        page_number=0,
        language="en",
        text="A" * 10000,
        has_figures=True,
    )
    prompt = build_prompt(page, n_queries=2)
    assert "figure" in prompt.lower()
    assert "AAAA" in prompt
    # We capped the page_text section to 3000 chars; the prompt is shorter than the raw input.
    assert len(prompt) < 10000


def test_parse_completion_extracts_json() -> None:
    raw = """Voici les questions:
    [{"question": "Quel diamètre?", "expected_answer": "5 mm", "requires_visual": true}]
    Fin."""
    items = parse_completion(raw)
    assert len(items) == 1
    assert items[0]["question"] == "Quel diamètre?"


def test_parse_completion_rejects_non_array() -> None:
    assert parse_completion("nothing here") == []
    assert parse_completion("[not json}") == []


def test_generate_for_page_filters_non_visual() -> None:
    page = PageInput(
        pdf_path="x.pdf", page_number=0, language="en", text="Cohort study.", has_figures=True
    )
    llm = FakeLLM(
        responses=[
            '[{"question":"q1","expected_answer":"a1","requires_visual":true},'
            '{"question":"q2","expected_answer":"a2","requires_visual":false}]'
        ]
    )
    out = generate_for_page(page, llm, n_queries=2, keep_visual_only=True)
    assert len(out) == 1
    assert out[0].question == "q1"


def test_generate_for_page_keeps_all_when_visual_filter_off() -> None:
    page = PageInput(
        pdf_path="x.pdf", page_number=0, language="en", text="...", has_figures=False
    )
    llm = FakeLLM(
        responses=[
            '[{"question":"q1","expected_answer":"a1","requires_visual":false}]'
        ]
    )
    out = generate_for_page(page, llm, n_queries=2, keep_visual_only=False)
    assert len(out) == 1


def test_parse_score_clamps_out_of_range() -> None:
    assert parse_score("0.92") == 0.92
    assert parse_score("Confiance: 1.5") == 1.0
    assert parse_score("-0.4") == 0.4  # we extract the first non-negative number
    assert parse_score("no number") == 0.0


def test_filter_queries_drops_below_threshold() -> None:
    queries = [
        SyntheticQuery(
            query_id=query_id_for("x.pdf", 0, 0),
            question="q1",
            expected_answer="a1",
            source_pdf="x.pdf",
            source_page=0,
            language="en",
        ),
        SyntheticQuery(
            query_id=query_id_for("x.pdf", 0, 1),
            question="q2",
            expected_answer="a2",
            source_pdf="x.pdf",
            source_page=0,
            language="en",
        ),
    ]
    page_text_for = {"x.pdf#page=0": "irrelevant for the fake LLM"}
    llm = FakeLLM(responses=["0.95", "0.3"])
    kept, report = filter_queries(queries, page_text_for, llm, threshold=0.8)
    assert len(kept) == 1
    assert kept[0].question == "q1"
    assert report.n_in == 2 and report.n_kept == 1
    assert queries[0].judge_score == 0.95
    assert queries[1].accepted is False


def test_queryset_jsonl_roundtrip(tmp_path) -> None:
    q = SyntheticQuery(
        query_id="q-abc-0",
        question="q?",
        expected_answer="a.",
        source_pdf="x.pdf",
        source_page=2,
        language="fr",
    )
    qs = QuerySet(name="t", language="fr", queries=[q])
    p = tmp_path / "qs.jsonl"
    qs.save_jsonl(p)
    loaded = QuerySet.load_jsonl(p, name="t", language="fr")
    assert len(loaded.queries) == 1
    assert loaded.queries[0].query_id == "q-abc-0"


def test_queryset_relevance_keys() -> None:
    q = SyntheticQuery(
        query_id="q-x-0",
        question="q?",
        expected_answer="a.",
        source_pdf="dir/x.pdf",
        source_page=3,
        language="en",
    )
    qs = QuerySet(name="t", language="en", queries=[q])
    rel = qs.relevance()
    assert rel == {"q-x-0": {"dir/x.pdf#page=3": 1.0}}


def test_correlate_perfect_positive() -> None:
    auto = [0.1, 0.4, 0.5, 0.7, 0.9]
    human = [0.15, 0.35, 0.55, 0.72, 0.88]
    rep = correlate(auto, human, subset_name="test")
    assert rep.pearson > 0.95
    assert rep.spearman > 0.95
    assert rep.passes_threshold is True


def test_correlate_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        correlate([0.1, 0.2], [0.1, 0.2, 0.3], subset_name="t")


def test_correlate_too_few_raises() -> None:
    with pytest.raises(ValueError):
        correlate([0.1], [0.2], subset_name="t")
