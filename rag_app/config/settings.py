"""
Application configuration – reads from environment variables or .env file.
All tuneable constants live here so nothing is hard-coded elsewhere.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OpenAIConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class QdrantConfig:
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = "rag_documents"
    # If no URL is provided, use in-memory Qdrant (great for demos)
    use_in_memory: bool = field(
        default_factory=lambda: not bool(os.getenv("QDRANT_URL", ""))
    )


@dataclass
class ChunkingConfig:
    max_words: int = 200
    overlap_words: int = 30


@dataclass
class AppConfig:
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    top_k_results: int = 5


# Singleton instance used throughout the app
settings = AppConfig()
