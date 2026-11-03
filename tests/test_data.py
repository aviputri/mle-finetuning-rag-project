"""
Test data validation and chunking.
"""

import pytest

from src.data import chunk_documents, load_training_examples, validate_data


def test_supplied_data_is_valid(documents, examples, cases):
    validate_data(documents, examples, cases)            # must not raise
    assert {d.document_id for d in documents} == {
        "account-security", "incidents", "refunds",
        "response-style", "shipping", "subscriptions",
    }


def test_duplicate_evaluation_ids_are_rejected(documents, examples, cases):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_data(documents, examples, cases + [cases[0]])


def test_unknown_training_source_is_rejected(documents, examples, cases):
    bad = examples[0].model_copy(update={"reference_ids": ["does-not-exist"]})
    with pytest.raises(ValueError, match="unknown sources"):
        validate_data(documents, [bad] + examples[1:], cases)


def test_training_answer_cannot_cite_source_outside_its_context(documents, examples, cases):
    refund = next(e for e in examples if e.reference_ids == ["refunds", "response-style"])
    bad = refund.model_copy(update={"answer": "Some claim. [shipping]"})
    others = [e for e in examples if e.id != refund.id]
    with pytest.raises(ValueError, match="reference_ids"):
        validate_data(documents, others + [bad], cases)


def test_unknown_evaluation_source_is_rejected(documents, examples, cases):
    bad = cases[0].model_copy(update={"expected_sources": ["nope"]})
    with pytest.raises(ValueError, match="unknown sources"):
        validate_data(documents, examples, [bad] + cases[1:])


def test_evaluation_question_leaking_into_training_is_rejected(documents, examples, cases):
    # Upper-casing proves the check is case-insensitive
    leak = examples[0].model_copy(update={"id": "train-leak", "question": cases[0].question.upper()})
    with pytest.raises(ValueError, match="leaked"):
        validate_data(documents, examples + [leak], cases)


def test_malformed_training_line_reports_line_number(tmp_path, examples):
    path = tmp_path / "train.jsonl"
    path.write_text(examples[0].model_dump_json() + "\n" + '{"id": "broken"}\n')
    with pytest.raises(ValueError, match=":2"):
        load_training_examples(path)


def test_chunks_keep_source_metadata(documents):
    chunks = chunk_documents(documents)
    for doc in documents:
        doc_chunks = [c for c in chunks if c.document_id == doc.document_id]
        assert doc_chunks, f"{doc.document_id} produced no chunks"
        assert [c.position for c in doc_chunks] == list(range(len(doc_chunks)))
        for chunk in doc_chunks:
            assert chunk.chunk_id == f"{doc.document_id}-{chunk.position}"
            assert chunk.title == doc.title
            # chunk text really comes from this document (whitespace-normalized)
            assert " ".join(chunk.text.split()) in " ".join(doc.text.split())