"""Configuration: environment variables only (SPEC section 7)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings.

    Values come from the environment or a local ``.env`` file. Field names map to
    upper-case environment variables, e.g. ``MEDLEAF_DATA_DIR``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEDLEAF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    configs_dir: Path = Field(default=PROJECT_ROOT / "configs")
    corpus_version: str = Field(
        default="dev",
        description=(
            "Corpus version stamped on chunks. `medleaf index build` replaces the "
            "default with a hash of data/medicines.yaml so retrieval filters match "
            "the indexed corpus exactly."
        ),
    )
    user_agent: str = Field(
        default="medleaf-rag/0.1 (+https://github.com/NickLitharis/medleaf-rag)"
    )
    request_timeout_s: float = Field(default=60.0)
    max_section_tokens: int = Field(default=400)

    # --- Retrieval (M2). Dense e5 vectors plus Qdrant sparse IDF vectors. ---
    embed_model: str = Field(default="intfloat/multilingual-e5-base")
    embed_revision: str = Field(
        default="d128750597153bb5987e10b1c3493a34e5a4502a",
        description="Pinned Hugging Face commit; the index manifest records what ran.",
    )
    embed_device: str = Field(default="cpu")
    embed_batch_size: int = Field(default=32)
    embed_max_seq_length: int = Field(default=512)
    embed_cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "models")
    qdrant_url: str = Field(default="http://127.0.0.1:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="medleaf_chunks")
    dense_vector_name: str = Field(default="dense")
    sparse_vector_name: str = Field(default="sparse")
    prefetch_limit: int = Field(default=30)
    default_limit: int = Field(default=10)

    # --- Generation (M4). Any OpenAI-compatible chat API: OpenRouter, OpenAI, Ollama. ---
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_model: str = Field(default="openrouter/free")
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MEDLEAF_LLM_API_KEY", "OPENROUTER_API_KEY"),
        description="Read from the environment only; never commit it.",
    )
    llm_timeout_s: float = Field(default=30.0)
    ask_search_limit: int = Field(
        default=20, description="Chunks retrieved per question before picking the sources."
    )

    def pdf_path(self, slug: str, lang: str) -> Path:
        """Path of one downloaded PDF, matching ``ingest.download.download_medicine``."""
        return self.raw_dir / slug / f"{slug}_{lang}.pdf"

    @property
    def raw_dir(self) -> Path:
        """Directory holding downloaded PDFs (gitignored)."""
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        """Directory holding parsed chunk JSONL."""
        return self.data_dir / "parsed"

    @property
    def index_dir(self) -> Path:
        """Directory holding the built-index manifest."""
        return self.data_dir / "index"

    @property
    def index_manifest_file(self) -> Path:
        """Manifest describing the built Qdrant index."""
        return self.index_dir / "manifest.json"

    def parsed_path(self, slug: str, lang: str) -> Path:
        """Cache path for one parsed PDF, matching ``retrieval.pipeline``."""
        return self.parsed_dir / f"{slug}_{lang}.jsonl"

    @property
    def medicines_file(self) -> Path:
        """Committed corpus manifest."""
        return self.data_dir / "medicines.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
