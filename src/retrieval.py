"""
Retrieval of relevant chunks from the knowledge base, using a vector database and embeddings.
"""

from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from . import config
from .data import Chunk, chunk_documents, load_documents


class SearchHit(Chunk):
    """A retrieved chunk plus its similarity score (inherits all Chunk metadata)."""
    score: float          # cosine similarity: higher means closer


class Retriever:
    """One shared retriever, used by BOTH baseline RAG and hybrid RAG + LoRA."""

    collection = "northstar"

    def __init__(self) -> None:
        # Embedding model: turns text into a 384-number vector that captures meaning
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION)
        self.chunks = chunk_documents(load_documents(config.KB_DIR))
        self.client = QdrantClient(":memory:")    # real vector DB, running locally in memory

        # a) Embed every chunk. The title is prepended so a short paragraph still
        #    "knows" its topic (e.g. "Shipping Policy\nA delivery address can be...").
        texts = [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks]
        vectors = self._encode(texts)

        # b) Create a fresh collection sized to the embedding, compared by cosine similarity.
        #    Deleting first means a rebuild never leaves stale chunks from an older policy.
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1],
                distance=models.Distance.COSINE,
            ),
        )

        # c) Store one point per chunk: vector + the full chunk metadata as payload.
        #    uuid5 derives a stable ID from chunk_id, so the same chunk always gets the same ID.
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload=chunk.model_dump(),
                )
                for chunk, vector in zip(self.chunks, vectors, strict=True)
            ],
        )

    def _encode(self, texts: list[str]):
        """Encode texts as unit-length vectors (same settings for chunks and questions)."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, question: str, top_k: int = config.TOP_K) -> list[SearchHit]:
        """Return the top_k most similar chunks, best first."""
        query_vector = self._encode([question])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,       # we need the metadata back, not just IDs
        )
        # Rebuild typed hits from the stored payload -> document_id survives the round trip
        return [SearchHit(**point.payload, score=point.score) for point in response.points]


def retrieve_or_abstain(retriever: Retriever, question: str,
                        min_score: float = config.MIN_SCORE) -> list[SearchHit]:
    """Return evidence only if the best match is strong enough; [] means abstain."""
    hits = retriever.search(question)
    if not hits or hits[0].score < min_score:
        return []
    return hits