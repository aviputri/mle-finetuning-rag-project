"""
Helpers for creating test fixtures.
"""

from src import config
from src.retrieval import SearchHit


def make_hit(document_id: str, score: float, text: str = "Policy text.") -> SearchHit:
    return SearchHit(chunk_id=f"{document_id}-0", document_id=document_id,
                     title=document_id.title(), position=0, text=text, score=score)


class FakeRetriever:
    """Returns fixed hits (optionally per question): no embedding model needed."""

    def __init__(self, hits=None, by_question=None):
        self.hits = hits or []
        self.by_question = by_question or {}

    def search(self, question, top_k=config.TOP_K):
        return self.by_question.get(question, self.hits)


class FakeGenerator:
    """Returns a fixed reply and counts calls: no language model needed."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def generate(self, question, hits):
        self.calls += 1
        return self.reply