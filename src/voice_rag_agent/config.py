from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)

    if raw is None or raw.strip() == "":
        return default

    return int(raw)


def _env_path(name: str, default: Path, root: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw) if raw else default

    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Settings:
    """Runtime settings kept dependency-free for local demos and tests."""

    root_dir: Path
    env: str
    data_dir: Path
    trace_dir: Path
    index_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_cache_entries: int
    enable_tracing: bool
    reasoner: str
    retriever: str
    ollama_base_url: str
    ollama_model: str
    openai_api_key: str
    openai_model: str
    assemblyai_api_key: str
    cartesia_api_key: str
    cartesia_voice_id: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "Settings":
        root = (root_dir or Path(os.getenv("VOICE_RAG_ROOT", Path.cwd()))).resolve()

        return cls(
            root_dir=root,
            env=os.getenv("VOICE_RAG_ENV", "local"),
            data_dir=_env_path("VOICE_RAG_DATA_DIR", Path("data/sample_docs"), root),
            trace_dir=_env_path("VOICE_RAG_TRACE_DIR", Path("traces"), root),
            index_dir=_env_path("VOICE_RAG_INDEX_DIR", Path("data/index_cache"), root),
            chunk_size=_env_int("VOICE_RAG_CHUNK_SIZE", 900),
            chunk_overlap=_env_int("VOICE_RAG_CHUNK_OVERLAP", 120),
            top_k=_env_int("VOICE_RAG_TOP_K", 4),
            max_cache_entries=_env_int("VOICE_RAG_MAX_CACHE_ENTRIES", 128),
            enable_tracing=_env_bool("VOICE_RAG_ENABLE_TRACING", True),
            reasoner=os.getenv("VOICE_RAG_REASONER", "template"),
            retriever=os.getenv("VOICE_RAG_RETRIEVER", "bm25"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", ""),
            assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY", ""),
            cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
            cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", ""),
            livekit_url=os.getenv("LIVEKIT_URL", ""),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
