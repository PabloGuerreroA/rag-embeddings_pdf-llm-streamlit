"""
Embeddings layer.
Wraps OpenAI's embedding API so the rest of the app never deals with
raw HTTP calls or response parsing.
"""
from __future__ import annotations

from typing import List

from openai import OpenAI

from config.settings import OpenAIConfig


class EmbeddingService:
    """Generates vector embeddings using the OpenAI Embeddings API."""

    def __init__(self, config: OpenAIConfig):
        self._model = config.embedding_model
        self._client = OpenAI(api_key=config.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single string."""
        cleaned = text.replace("\n", " ").strip()
        response = self._client.embeddings.create(
            model=self._model,
            input=cleaned,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts in one API call (OpenAI supports batching).
        Splits into chunks of 100 to stay within rate limits.
        """
        BATCH_SIZE = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = [t.replace("\n", " ").strip() for t in texts[i : i + BATCH_SIZE]]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            # API returns embeddings in the same order as input
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings
