import os
import faiss
import numpy as np
import pickle
from pathlib import Path

from core.logger import get_logger
from core.types import Data

logger = get_logger(__name__)


SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8001")

class Models:
    e5_model = None
    clip_model = None
    clip_preprocess = None
    reranker =None
    hf_api_key = None


class VectorStore:
    text_index = None
    text_meta = {}

    doc_image_index = None
    doc_image_meta = {}

    image_index = None
    image_meta = {}

    indexed_images =  set()
    indexed_chunks = set()
    indexed_doc_images = set()
    indexed_docs = set()




_initialized = False

def save(directory: str):
    initialize()
    os.makedirs(directory, exist_ok=True)
    if VectorStore.text_index:
        faiss.write_index(VectorStore.text_index, os.path.join(directory, "text.index"))
    if VectorStore.image_index:
        faiss.write_index(VectorStore.image_index, os.path.join(directory, "image.index"))
    if VectorStore.doc_image_index:                                       #
        faiss.write_index(VectorStore.doc_image_index, os.path.join(directory, "doc_image.index"))
    with open(os.path.join(directory, "meta.pkl"), "wb") as f:
        pickle.dump({
            "text_meta": VectorStore.text_meta,
            "image_meta": VectorStore.image_meta,
            "doc_image_meta": VectorStore.doc_image_meta,
            "doc_meta_store": Data.docs,
            "image_to_chunks": Data.image_to_chunks,
            "table_to_chunks": Data.table_to_chunks,
            "all_chunks": Data.chunks,
            "indexed_chunks": VectorStore.indexed_chunks,
            "indexed_images": VectorStore.indexed_images,
            "indexed_doc_images": VectorStore.indexed_doc_images,
            "indexed_docs": VectorStore.indexed_docs,
        }, f)
    logger.info("Vector store saved", extra={"path": directory})


def load(directory: str):
    initialize()
    tp = os.path.join(directory, "text.index")
    ip = os.path.join(directory, "image.index")
    dip = os.path.join(directory, "doc_image.index")
    mp = os.path.join(directory, "meta.pkl")

    if os.path.exists(tp):
        VectorStore.text_index = faiss.read_index(tp)
    if os.path.exists(ip):
        VectorStore.image_index = faiss.read_index(ip)
    if os.path.exists(dip):
        VectorStore.doc_image_index = faiss.read_index(dip)

    if os.path.exists(mp):
        with open(mp, "rb") as f:
            m = pickle.load(f)
        VectorStore.text_meta.update(m["text_meta"])
        VectorStore.image_meta.update(m["image_meta"])
        VectorStore.doc_image_meta.update(m.get("doc_image_meta", {}))
        Data.docs.update(m["doc_meta_store"])
        Data.image_to_chunks.update(m["image_to_chunks"])
        Data.table_to_chunks.update(m["table_to_chunks"])
        Data.chunks = m["all_chunks"]
        VectorStore.indexed_chunks.update(m.get("indexed_chunks", set()))
        VectorStore.indexed_images.update(m.get("indexed_images", set()))
        VectorStore.indexed_doc_images.update(m.get("indexed_doc_images", set()))
        VectorStore.indexed_docs.update(m.get("indexed_docs", set()))

        Data.chunk_by_key = {}
        Data.chunk_keys = set()
        Data.doc_to_chunks = {}
        for chunk in Data.chunks:
            key = (chunk.doc_id, chunk.chunk_index)
            Data.chunk_by_key[key] = chunk
            Data.chunk_keys.add(key)
            Data.doc_to_chunks.setdefault(chunk.doc_id, []).append(chunk)

    logger.info("Vector store loaded", extra={"path": directory})


def initialize():
    global _initialized
    if _initialized:
        return

    from llm.client import load_api_key
    load_api_key()

    _initialized = True
