from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PathConfig:
    data_url: str = ""
    artifact_dir: str = "artifacts"
    status_file: str = "status.txt"
    metrics_file: str = "metrics.json"


@dataclass(frozen=True)
class SchemaConfig:
    target_column: str = "target"
    column_names: List[str] = field(default_factory=list)
    column_types: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LLMConfig:
    strong_model: str = "Qwen/Qwen3-8B"
    fast_model: str = "Qwen/Qwen3-4B"
    vision_model: str = "Qwen/Qwen3-8B"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    server_url: str = "http://127.0.0.1:8080/v1"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 600


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 500
    overlap_tokens: int = 60
    image_max_tokens: int = 2000
    header_window_lines: int = 5
    min_chunk_len: int = 5


@dataclass(frozen=True)
class RetrievalConfig:
    text_k: int = 15
    image_k: int = 15
    chunk_image_k: int = 15
    top_k: int = 5
    score_threshold: float = 0.05
    alpha: float = 0.5


@dataclass(frozen=True)
class DBConfig:
    db_url: str = "sqlite:///./chatbot.db"
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_db: str = ""


@dataclass(frozen=True)
class TuningConfig:
    max_retries: int = 3
    retry_wait_seconds: int = 5
    tokens_per_batch: int = 3000
    group_size: int = 5


@dataclass(frozen=True)
class BlendConfig:
    low_weights: Tuple[float, float] = (0.8, 0.2)
    high_weights: Tuple[float, float] = (0.2, 0.8)
    ignore_image_threshold: float = 0.1
    high_image_threshold: float = 0.4


@dataclass(frozen=True)
class StorageConfig:
    upload_dir: str = "uploads"
    image_dir: str = "images"


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    db: DBConfig = field(default_factory=DBConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:8501"])
    vector_db_path: str = "storage"
    backend_api_url: str = "http://127.0.0.1:8000"
