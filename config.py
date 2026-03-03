"""
config.py — Central configuration loaded from environment variables.

All other modules import `from config import cfg` and never call os.getenv directly.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Google AI
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    # LangSmith
    langsmith_tracing: str = field(default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false"))
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "llm-security-rag"))

    # Langfuse
    langfuse_secret_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    langfuse_public_key: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_host: str = field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))

    # Arize Phoenix
    phoenix_collector_endpoint: str = field(
        default_factory=lambda: os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
    )
    phoenix_project_name: str = field(
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "llm-security-rag")
    )

    # ChromaDB
    chroma_db_path: str = field(default_factory=lambda: os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "mitre_attack")
    )

    # Retrieval
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5"))
    )
    security_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("SECURITY_SCORE_THRESHOLD", "0.7"))
    )

    def validate(self) -> None:
        """Raise ValueError if any required configuration is missing."""
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required but not set. "
                "Copy .env.example to .env and add your Google AI API key."
            )


# Module-level singleton — import this everywhere
cfg = Config()
