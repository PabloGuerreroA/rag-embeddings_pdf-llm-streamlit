"""
Persistence layer – Qdrant vector store.
Handles:
  - Collection creation / reset
  - Upserting document chunks with their embeddings
  - Semantic search (cosine similarity)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from business.document_processor import TextChunk
from config.settings import QdrantConfig


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search with its score."""
    text: str
    source: str
    chunk_index: int
    score: float


class VectorStore:
    """Thin wrapper around QdrantClient for our RAG use-case."""

    def __init__(self, config: QdrantConfig, embedding_dim: int):
        self._collection = config.collection_name
        self._dim = embedding_dim

        if config.use_in_memory:
            print("✅ Qdrant: modo en memoria")
            self._client = QdrantClient(":memory:")
        else:
            print(f"🔗 Qdrant: conectando a {config.url}")
            self._client = QdrantClient(
                url=config.url,
                api_key=config.api_key or None,
            )

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop and recreate the collection (e.g. when new docs are uploaded)."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._ensure_collection()

    def upsert_chunks(
        self, chunks: List[TextChunk], embeddings: List[List[float]]
    ) -> None:
        """Store *chunks* with their pre-computed *embeddings*."""
        assert len(chunks) == len(embeddings), "chunks and embeddings must align"

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Upload in batches of 64 to avoid request size limits
        BATCH = 64
        for i in range(0, len(points), BATCH):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + BATCH],
            )

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Return the *top_k* most similar chunks."""
        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            RetrievedChunk(
                text=hit.payload["text"],
                source=hit.payload.get("source", "unknown"),
                chunk_index=hit.payload.get("chunk_index", -1),
                score=hit.score,
            )
            for hit in hits
        ]

    def count(self) -> int:
        """Return the number of vectors currently stored."""
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                ),
            )
