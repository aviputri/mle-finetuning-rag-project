"""
Test evaluation functionality.
"""

import pytest

from src.data import EvaluationCase
from src.evaluation import (
    contains_term, decision_correct, evaluate_retrieval, evaluate_system, faithful,
    relevance, render_markdown, retrieval_metrics, style_score, term_coverage,
)
from src.generation import Answer
from tests.helpers import FakeGenerator, FakeRetriever, make_hit

SUPPORTED = EvaluationCase(id="sup", question="Change address after packing?",
                           expected_sources=["shipping"], required_terms=["cannot", "carrier"],
                           forbidden_terms=["can always change"], should_abstain=False)
UNSUPPORTED = EvaluationCase(id="uns", question="Cafeteria menu?", expected_sources=[],
                             required_terms=[], forbidden_terms=[], should_abstain=True)


def answer(status, text="", citations=()):
    return Answer(system="rag-base", question="q", status=status, answer=text,
                  citations=list(citations))


def test_terms_match_whole_words_only():
    assert contains_term("You CANNOT change it", "cannot")          # case-insensitive
    assert not contains_term("Please note this", "not")             # no substring match
    assert term_coverage("It cannot change", ["cannot", "carrier"]) == 0.5
    assert term_coverage("anything", []) == 1.0


def test_retrieval_metrics_use_supported_cases_only():
    rows = [{"should_abstain": False, "rank": 1},
            {"should_abstain": False, "rank": 2},
            {"should_abstain": False, "rank": None},
            {"should_abstain": True, "rank": None}]               # ignored
    recall, mrr = retrieval_metrics(rows)
    assert recall == pytest.approx(2 / 3)
    assert mrr == pytest.approx((1 + 0.5 + 0) / 3)


def test_unsupported_case_rewards_only_abstention():
    good, bad = answer("abstained"), answer("answered", "Menu is online. [shipping]", ["shipping"])
    assert (relevance(UNSUPPORTED, good), faithful(UNSUPPORTED, good), decision_correct(UNSUPPORTED, good)) == (1.0, True, True)
    assert (relevance(UNSUPPORTED, bad), faithful(UNSUPPORTED, bad), decision_correct(UNSUPPORTED, bad)) == (0.0, False, False)


def test_supported_case_scoring():
    ok = answer("answered", "Support cannot change it; contact the carrier. [shipping]", ["shipping"])
    assert relevance(SUPPORTED, ok) == 1.0 and faithful(SUPPORTED, ok)
    forbidden = answer("answered", "Support can always change it. [shipping]", ["shipping"])
    assert not faithful(SUPPORTED, forbidden)
    wrong_source = answer("answered", "Support cannot change it. [refunds]", ["refunds"])
    assert not faithful(SUPPORTED, wrong_source)
    rejected = answer("rejected", "The generated answer failed citation validation.")
    assert relevance(SUPPORTED, rejected) == 0.0 and style_score(rejected) == 0.0
    assert not decision_correct(SUPPORTED, rejected)


def test_report_generation_end_to_end():
    retriever = FakeRetriever(by_question={
        SUPPORTED.question: [make_hit("shipping", 0.8)],
        UNSUPPORTED.question: [make_hit("subscriptions", 0.1)],
    })
    cases = [SUPPORTED, UNSUPPORTED]
    retrieval_rows = evaluate_retrieval(retriever, cases)
    system_rows = {
        "rag-base": evaluate_system("rag-base", cases, retriever, FakeGenerator("Yes | sure.")),
        "rag-lora": evaluate_system("rag-lora", cases, retriever,
                                    FakeGenerator("Support cannot change it; ask the carrier. [shipping]")),
    }
    report = render_markdown(retrieval_rows, system_rows, training=None)

    for heading in ("## Retrieval", "## System Comparison", "## Per-Case Comparison", "## Answers"):
        assert heading in report
    for case_id in ("sup", "uns"):
        assert f"| {case_id} | rag-base |" in report and f"| {case_id} | rag-lora |" in report
    assert "Yes \\| sure." in report        # pipes escaped so tables don't break
    lora_row = next(r for r in system_rows["rag-lora"] if r["id"] == "sup")
    assert lora_row["status"] == "answered" and lora_row["relevance"] == 1.0