from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.constants import DocType


@dataclass
class BaseMeta:
    doc_id: str
    title: str = ""
    source_type: str = ""
    doc_type: str = "General"


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    chunk_type: str
    style: str = "Normal"
    section_path: List[str] = field(default_factory=list)
    start_element_id: int = None
    end_element_id: int = None
    image_refs: Dict[int, str] = field(default_factory=dict)
    table_refs: Dict[int, str] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class UserRequest:
    query: str
    images: list = None
    documents: list = None
    audio: list = None


class Intent(str, Enum):
    GENERAL_CHAT = "GENERAL_CHAT"

    DOCUMENT_OVERVIEW = "DOCUMENT_OVERVIEW"

    DOCUMENT_QA = "DOCUMENT_QA"

    DOCUMENT_SECTION_EXPLAIN = "DOCUMENT_SECTION_EXPLAIN"
    DOCUMENT_SECTION_SUMMARIZE = "DOCUMENT_SECTION_SUMMARIZE"

    DOCUMENT_FULL_EXPLAIN = "DOCUMENT_FULL_EXPLAIN"
    DOCUMENT_SUMMARIZE = "DOCUMENT_SUMMARIZE"

    IMAGE_UNDERSTANDING = "IMAGE_UNDERSTANDING"
    IMAGE_SEARCH = "IMAGE_SEARCH"

    AUDIO_OVERVIEW = "AUDIO_OVERVIEW"
    AUDIO_TRANSCRIBE = "AUDIO_TRANSCRIBE"
    AUDIO_SUMMARIZE = "AUDIO_SUMMARIZE"
    AUDIO_QA = "AUDIO_QA"

    SEARCH_DOCUMENT = "SEARCH_DOCUMENT"

    COMPARE_DOCUMENTS = "COMPARE_DOCUMENTS"

    DOCUMENT_ACTION = "DOCUMENT_ACTION"

    UNKNOWN = "UNKNOWN"


@dataclass
class QueryContext:
    has_document: bool = False
    has_image: bool = False
    has_audio: bool = False

    num_documents: int = 0
    num_images: int = 0
    num_audio: int = 0

    has_stored_documents: bool = False
    has_stored_images: bool = False
    has_stored_audio: bool = False


@dataclass
class ActiveContext:
    has_file: bool = False
    active_documents: Set[str] = field(default_factory=set)
    active_images: Set[str] = field(default_factory=set)
    active_audio: Set[str] = field(default_factory=set)


@dataclass
class SessionContext:
    session_documents: Set[str] = field(default_factory=set)
    session_images: Set[str] = field(default_factory=set)
    session_audio: Set[str] = field(default_factory=set)


@dataclass
class ResolvedTargets:
    documents: List[str] = field(default_factory=list)

    images: List[str] = field(default_factory=list)

    confidence: float = 1.0


ChunkKey = Tuple[str, int]
MediaKey = Tuple[str, int]


class Data:
    docs: Dict[str, BaseMeta] = {}

    chunks: List[Chunk] = []

    chunk_by_key: Dict[ChunkKey, Chunk] = {}

    chunk_keys = set()

    doc_to_chunks: Dict[str, List[Chunk]] = {}

    image_to_chunks: Dict[MediaKey, Set[ChunkKey]] = {}

    table_to_chunks: Dict[MediaKey, Set[ChunkKey]] = {}


def _token_count(text: str) -> int:
    return len(text.split())


def ingest(
    chunks: List[Chunk],
    doc_meta: BaseMeta,
    img_to_ch: Dict[MediaKey, List[ChunkKey]],
    tbl_to_ch: Dict[MediaKey, List[ChunkKey]],
) -> None:

    Data.docs[doc_meta.doc_id] = doc_meta

    for chunk in chunks:
        chunk_key = (chunk.doc_id, chunk.chunk_index)
        if chunk_key not in Data.chunk_keys:
            Data.chunks.append(chunk)

            Data.chunk_by_key[chunk_key] = chunk
            Data.chunk_keys.add(chunk_key)

            Data.doc_to_chunks.setdefault(chunk.doc_id, []).append(chunk)

    if img_to_ch is not None:
        for image_key, chunk_keys in img_to_ch.items():
            Data.image_to_chunks.setdefault(image_key, set()).update(chunk_keys)

    if tbl_to_ch is not None:
        for table_key, chunk_keys in tbl_to_ch.items():
            Data.table_to_chunks.setdefault(table_key, set()).update(chunk_keys)
