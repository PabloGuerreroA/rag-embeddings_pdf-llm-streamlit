"""
RAG orchestration layer.
Combines:
  - EmbeddingService  → query embedding
  - VectorStore       → retrieval
  - OpenAI Chat API   → answer generation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from openai import OpenAI

from business.embeddings import EmbeddingService
from config.settings import OpenAIConfig
from persistence.vector_store import RetrievedChunk, VectorStore


@dataclass
class RAGResponse:
    """Everything the UI needs to render a Q&A result."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]


SYSTEM_PROMPT = (
    "Eres un asistente experto en análisis de documentos. "
    "Responde siempre en el mismo idioma que la pregunta del usuario. "
    "Sé preciso, conciso y profesional."
)

USER_PROMPT_TEMPLATE = """\
Utiliza ÚNICAMENTE el contexto proporcionado para responder la pregunta.
Si la respuesta no se encuentra en el contexto, indícalo claramente.
No inventes información.

--- CONTEXTO ---
{context}
--- FIN DEL CONTEXTO ---

Pregunta: {question}
"""


class RAGService:
    """End-to-end RAG: retrieve relevant chunks, then generate an answer."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        config: OpenAIConfig,
        top_k: int = 5,
    ):
        self._store = vector_store
        self._embeddings = embedding_service
        self._llm = OpenAI(api_key=config.api_key)
        self._chat_model = config.chat_model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        """Retrieve relevant chunks and generate an answer."""
        # 1. Embed the question
        query_vector = self._embeddings.embed_text(question)

        # 2. Retrieve top-k chunks from vector store
        chunks = self._store.search(query_vector, top_k=self._top_k)

        # 3. Build context string
        context = "\n\n---\n\n".join(
            f"[{c.source} – chunk {c.chunk_index}]\n{c.text}"
            for c in chunks
        )

        # 4. Call the LLM
        answer = self._generate_answer(question, context)

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_answer(self, question: str, context: str) -> str:
        user_message = USER_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        response = self._llm.chat.completions.create(
            model=self._chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content
