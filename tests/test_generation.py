"""
Test generation functionality.
"""

import json

import pytest

from src import config
from src.generation import (
    REJECTED_MESSAGE, answer_question, citations_are_valid, citations_in, validate_adapter,
)
from src.prompts import FALLBACK_ANSWER
from tests.helpers import FakeGenerator, FakeRetriever, make_hit

SHIPPING = [make_hit("shipping", 0.8)]


@pytest.mark.parametrize("text, expected", [
    ("Support cannot change it after packing. [shipping]", True),
    ("Support cannot change it after packing.", False),               # no citation
    ("Support cannot change it. [shipping-policy]", False),           # invented ID
    ("Support cannot change it. [Shipping Policy]", False),           # malformed ID
    ("Refunds take 30 days. [refunds]", False),                       # valid ID, not retrieved
    ("[shipping]", False),                                            # citation only
])
def test_citation_validation(text, expected):
    assert citations_are_valid(text, SHIPPING) is expected


def test_citations_are_unique_and_ordered():
    assert citations_in("A [b] c [a] d [b]") == ["b", "a"]


def run(reply, hits=SHIPPING):
    generator = FakeGenerator(reply)
    answer = answer_question("Q?", "rag-base", FakeRetriever(hits), generator)
    return answer, generator


def test_valid_cited_answer_is_accepted():
    answer, generator = run("Support cannot change it. [shipping]")
    assert answer.status == "answered"
    assert answer.citations == ["shipping"]
    assert generator.calls == 1


def test_uncited_answer_is_rejected_but_raw_text_kept():
    answer, _ = run("Yes, support can change it.")
    assert answer.status == "rejected"
    assert answer.answer == REJECTED_MESSAGE                 # never shown to the user
    assert answer.raw_answer == "Yes, support can change it."


def test_citation_to_unretrieved_source_is_rejected():
    answer, _ = run("Refunds take 30 days. [refunds]")
    assert answer.status == "rejected"


def test_model_fallback_reply_counts_as_abstention():
    answer, _ = run(FALLBACK_ANSWER)
    assert answer.status == "abstained"
    assert answer.answer == FALLBACK_ANSWER


@pytest.mark.parametrize("hits", [[], [make_hit("shipping", config.MIN_SCORE - 0.01)]])
def test_weak_retrieval_abstains_without_calling_the_model(hits):
    answer, generator = run("Anything. [shipping]", hits=hits)
    assert answer.status == "abstained"
    assert answer.raw_answer is None
    assert generator.calls == 0


def test_missing_adapter_summary_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADAPTER_PATH", tmp_path)
    with pytest.raises(FileNotFoundError):
        validate_adapter()


@pytest.mark.parametrize("override, message", [
    ({"base_model": "some/other-model"}, "trained on"),
    ({"training_sha256": "0" * 64}, "Training data changed"),
])
def test_stale_adapter_is_rejected(tmp_path, monkeypatch, override, message):
    import hashlib
    summary = {
        "base_model": config.BASE_MODEL,
        "base_model_revision": config.BASE_MODEL_REVISION,
        "training_sha256": hashlib.sha256(config.TRAIN_FILE.read_bytes()).hexdigest(),
        **override,
    }
    (tmp_path / "training_summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(config, "ADAPTER_PATH", tmp_path)
    with pytest.raises(ValueError, match=message):
        validate_adapter()