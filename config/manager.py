"""Configuration manager for the RAG chatbot

Reads config/base.yaml as the foundation, overlays env-specific overrides
from config/dev.yaml or config/prod.yaml (controlled by APP_ENV env var),
then returns a frozen AppConfig dataclass. Missing keys fall back to
dataclass defaults.
"""

import os
from pathlib import Path
from typing import Optional

import yaml

from config.schema import (
    AppConfig, LLMConfig, ChunkingConfig, RetrievalConfig, DBConfig,
    TuningConfig, BlendConfig, StorageConfig,
)


_CONFIG_DIR = Path(__file__).resolve().parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = {}
    all_keys = set(base) | set(override)
    for key in all_keys:
        if key in base and key in override:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                merged[key] = _deep_merge(base[key], override[key])
            else:
                merged[key] = override[key]
        elif key in override:
            merged[key] = override[key]
        else:
            merged[key] = base[key]
    return merged


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _from_env(key: str, fallback: str) -> str:
    return os.getenv(key, fallback)


def _build_app_config(raw: dict) -> AppConfig:
    llm_raw = raw.get("llm", {})
    chunking_raw = raw.get("chunking", {})
    retrieval_raw = raw.get("retrieval", {})
    db_raw = raw.get("db", {})
    tuning_raw = raw.get("tuning", {})
    blend_raw = raw.get("blend", {})
    storage_raw = raw.get("storage", {})

    return AppConfig(
        llm=LLMConfig(
            strong_model=_from_env("STRONG_MODEL", llm_raw.get("strong_model", "Qwen/Qwen3-8B")),
            fast_model=_from_env("FAST_MODEL", llm_raw.get("fast_model", "Qwen/Qwen3-4B")),
            vision_model=_from_env("VISION_MODEL", llm_raw.get("vision_model", "Qwen/Qwen3-8B")),
            embedding_model=_from_env("EMBEDDING_MODEL", llm_raw.get("embedding_model", "BAAI/bge-small-en-v1.5")),
            server_url=_from_env("LLM_SERVER_URL", llm_raw.get("server_url", "http://127.0.0.1:8080/v1")),
            temperature=float(_from_env("LLM_TEMPERATURE", str(llm_raw.get("temperature", 0.1)))),
            max_tokens=int(_from_env("LLM_MAX_TOKENS", str(llm_raw.get("max_tokens", 2048)))),
            timeout=int(_from_env("LLM_TIMEOUT", str(llm_raw.get("timeout", 600)))),
        ),
        chunking=ChunkingConfig(
            max_tokens=int(_from_env("CHUNK_MAX_TOKENS", str(chunking_raw.get("max_tokens", 500)))),
            overlap_tokens=int(_from_env("CHUNK_OVERLAP_TOKENS", str(chunking_raw.get("overlap_tokens", 60)))),
            image_max_tokens=int(_from_env("CHUNK_IMAGE_MAX_TOKENS", str(chunking_raw.get("image_max_tokens", 2000)))),
            header_window_lines=int(_from_env("CHUNK_HEADER_WINDOW_LINES", str(chunking_raw.get("header_window_lines", 5)))),
            min_chunk_len=int(_from_env("CHUNK_MIN_LEN", str(chunking_raw.get("min_chunk_len", 5)))),
        ),
        retrieval=RetrievalConfig(
            text_k=int(_from_env("RETRIEVAL_TEXT_K", str(retrieval_raw.get("text_k", 15)))),
            image_k=int(_from_env("RETRIEVAL_IMAGE_K", str(retrieval_raw.get("image_k", 15)))),
            chunk_image_k=int(_from_env("RETRIEVAL_CHUNK_IMAGE_K", str(retrieval_raw.get("chunk_image_k", 15)))),
            top_k=int(_from_env("RETRIEVAL_TOP_K", str(retrieval_raw.get("top_k", 5)))),
            score_threshold=float(_from_env("RETRIEVAL_SCORE_THRESHOLD", str(retrieval_raw.get("score_threshold", 0.05)))),
            alpha=float(_from_env("RETRIEVAL_ALPHA", str(retrieval_raw.get("alpha", 0.5)))),
        ),
        tuning=TuningConfig(
            max_retries=int(_from_env("MAX_RETRIES", str(tuning_raw.get("max_retries", 3)))),
            retry_wait_seconds=int(_from_env("RETRY_WAIT_SECONDS", str(tuning_raw.get("retry_wait_seconds", 5)))),
            tokens_per_batch=int(_from_env("TOKENS_PER_BATCH", str(tuning_raw.get("tokens_per_batch", 3000)))),
            group_size=int(_from_env("GROUP_SIZE", str(tuning_raw.get("group_size", 5)))),
        ),
        blend=BlendConfig(
            low_weights=tuple(blend_raw.get("low_weights", [0.8, 0.2])),
            high_weights=tuple(blend_raw.get("high_weights", [0.2, 0.8])),
            ignore_image_threshold=float(blend_raw.get("ignore_image_threshold", 0.1)),
            high_image_threshold=float(blend_raw.get("high_image_threshold", 0.4)),
        ),
        storage=StorageConfig(
            upload_dir=_from_env("UPLOAD_DIR", storage_raw.get("upload_dir", "uploads")),
            image_dir=_from_env("IMAGE_DIR", storage_raw.get("image_dir", "images")),
        ),
        db=DBConfig(
            db_url=_from_env("DATABASE_URL", db_raw.get("db_url", "sqlite:///./chatbot.db")),
            postgres_user=_from_env("POSTGRES_USER", db_raw.get("postgres_user", "")),
            postgres_password=_from_env("POSTGRES_PASSWORD", db_raw.get("postgres_password", "")),
            postgres_db=_from_env("POSTGRES_DB", db_raw.get("postgres_db", "")),
        ),
        cors_origins=raw.get("cors_origins", ["http://localhost:8501"]),
        vector_db_path=_from_env("VECTOR_DB_PATH", raw.get("vector_db_path", "storage")),
        backend_api_url=_from_env("BACKEND_API_URL", raw.get("backend_api_url", "http://127.0.0.1:8000")),
    )


def get_config(env: Optional[str] = None) -> AppConfig:
    env = env or os.getenv("APP_ENV", "dev")

    base = _load_yaml(_CONFIG_DIR / "base.yaml")

    env_file = "prod.yaml" if env == "prod" else "dev.yaml"
    overrides = _load_yaml(_CONFIG_DIR / env_file)

    merged = _deep_merge(base, overrides)
    return _build_app_config(merged)
