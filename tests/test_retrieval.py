"""
Test retrieval functionality.
"""

from src import config
from src.retrieval import SearchHit, retrieve_or_abstain


def test_hits_keep_metadata_after_vector_storage(retriever, documents):
    hits = retriever.search("What is the usual standard shipping time?")
    assert len(hits) == config.TOP_K
    source_ids = {d.document_id for d in documents}
    for hit in hits:
        assert isinstance(hit, SearchHit)
        assert hit.document_id in source_ids
        assert hit.chunk_id.startswith(hit.document_id)
        assert hit.text and hit.title and hit.position >= 0
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)       # best match first


def test_supported_questions_rank_expected_source_first(retriever, cases):
    for case in (c for c in cases if not c.should_abstain):
        top = retriever.search(case.question)[0]
        assert top.document_id in case.expected_sources, case.id


def test_off_topic_questions_abstain_at_retrieval(retriever):
    for question in ("Where can I find next week's cafeteria menu?",
                     "Can I reserve a bicycle parking space near the office?"):
        assert retrieve_or_abstain(retriever, question) == []