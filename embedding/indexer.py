import numpy as np
import faiss
import requests
import os
from pathlib import Path
from typing import List

from core.mlflow_tracking import log_metrics
from core.types import Chunk, Data
from embedding.models import VectorStore, initialize, SERVER_URL


def get_image_id(path: Path):
    import hashlib
    return hashlib.md5(
        path.read_bytes()
    ).hexdigest()


def embed_text(texts: List[str], is_query: bool = False) -> np.ndarray:
    initialize()
    # print(f"Embedding {len(texts)} text chunks...", flush=True)
    # prefix  = "query: " if is_query else "passage: "
    # prefixed = [prefix + t for t in texts]
    # vecs = Models.e5_model.encode(prefixed, normalize_embeddings=True)
    # print("Text ingest complete.", flush=True)
    # return np.array(vecs, dtype="float32")
    response = requests.post(f"{SERVER_URL}/embed/text",
                             json={"texts":texts, "is_query":is_query})
    response.raise_for_status()
    return np.array(response.json()["embeddings"], dtype=np.float32)

def embed_image(image_paths: List[str]) -> np.ndarray:
    ####memory explosion issue: for example if someone upload 500 images
    # Todo Solve later
    initialize()
    # print(f"Processing {len(image_paths)} images through CLIP...", flush=True)
    # images = torch.stack([
    #     Models.clip_preprocess(Image.open(p).convert("RGB"))
    #     for p in image_paths
    # ])
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # images = images.to(device)
    # print(f"Sending images to device: {device}...", flush=True)
    # with torch.no_grad():
    #     vecs = Models.clip_model.encode_image(images)
    #     vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
    # print("Image ingest complete.", flush=True)
    # return vecs.cpu().numpy().astype("float32")

    response = requests.post(
        f"{SERVER_URL}/embed/open_clip/image",
        json={"image_paths": image_paths}
    )
    response.raise_for_status()
    return np.array(response.json()["embeddings"], dtype="float32")

def index_chunks(chunks: List[Chunk]):
    """Embed chunk texts and insert into the text FAISS index."""

    initialize()
    text_index = VectorStore.text_index
    text_meta = VectorStore.text_meta
    indexed_chunks= VectorStore.indexed_chunks

    new_chunks = [
        c for c in chunks
        if (c.doc_id, c.chunk_index) not in indexed_chunks
    ]
    if not new_chunks:
        return

    texts = [c.text for c in new_chunks]
    vecs  = embed_text(texts, is_query=False)       # (n, 768)

    if text_index is None:
        text_index = faiss.IndexFlatIP(vecs.shape[1])

    base = text_index.ntotal
    text_index.add(vecs)
    for i, chunk in enumerate(new_chunks):
        text_meta[base + i] = chunk
        indexed_chunks.add((chunk.doc_id, chunk.chunk_index))

    VectorStore.text_index = text_index
    VectorStore.text_meta = text_meta
    VectorStore.indexed_chunks = indexed_chunks
    log_metrics({"index_chunks_added": len(new_chunks), "index_total": text_index.ntotal})


def index_chunk_images(chunks: List[Chunk]):
    """Embed images from image_context chunks and insert into the image FAISS index."""
    initialize()

    image_index = VectorStore.doc_image_index
    image_meta = VectorStore.doc_image_meta
    indexed_images = VectorStore.indexed_doc_images

    paths, records = [], []
    for c in chunks:
        if not c.image_refs:
            continue
        for img_id, path in c.image_refs.items():
            if (c.doc_id, img_id) in indexed_images:
                continue
            if os.path.exists(path):
                paths.append(path)
                records.append({
                    "image_id":    img_id,
                    "path":        path,
                    "doc_id":      c.doc_id,
                    "chunk_index": c.chunk_index,
                    "section_path": c.section_path,
                    "source": "document"
                })

    if not paths:
        return

    vecs = embed_image(paths)                       # (m, 512)

    if image_index is None:
        image_index = faiss.IndexFlatIP(vecs.shape[1])

    base = image_index.ntotal
    image_index.add(vecs)
    for i, rec in enumerate(records):
        image_meta[base + i] = rec
        indexed_images.add((rec["doc_id"], rec["image_id"]))

    VectorStore.doc_image_index = image_index
    VectorStore.doc_image_meta = image_meta
    VectorStore.indexed_doc_images = indexed_images

def index_images(image_paths: List[str], doc_id:str):

    initialize()

    image_index = VectorStore.image_index
    image_meta = VectorStore.image_meta
    indexed_images = VectorStore.indexed_images

    records = []

    for path in image_paths:
        p = Path(path)

        image_id = get_image_id(p)

        key = (doc_id, image_id)

        if key in indexed_images:
            continue

        records.append({
            "image_id": image_id,
            "path": str(p),
            "doc_id": doc_id,
            "source": "upload"
        })

    vecs = embed_image(image_paths)

    if image_index is None:
        image_index = faiss.IndexFlatIP(vecs.shape[1])

    base = image_index.ntotal
    image_index.add(vecs)

    for i, rec in enumerate(records):
        image_meta[base + i] = rec
        indexed_images.add((rec["doc_id"], rec["image_id"]))

    VectorStore.image_index = image_index
    VectorStore.image_meta = image_meta
    VectorStore.indexed_images = indexed_images
