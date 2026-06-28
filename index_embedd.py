from sentence_transformers import SentenceTransformer
import open_clip
from sentence_transformers import CrossEncoder
import faiss
import torch
from typing import List, Dict
import numpy as np
from PIL import Image
from data_gathering import Chunk
from dotenv import load_dotenv
import os
import pickle
from data_gathering import Data
from pathlib import Path
import hashlib
import requests


SERVER_URL = "http://127.0.0.1:8000"


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


def get_image_id(path: Path):
    return hashlib.md5(
        path.read_bytes()
    ).hexdigest()


def set_environ():
    os.environ["HF_HOME"] = r"D:\models\huggingface"
    os.environ["TORCH_HOME"] = r"D:\models\torch_models"


def load_api_key():
    load_dotenv()
    hf_api_key = os.getenv("HUGGIN_FACE_API")
    Models.hf_api_key = hf_api_key


def initialize_models():

    if Models.e5_model is None:
        print("Loading E5 Model...", flush=True)
        Models.e5_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

    if Models.clip_model is None:
        print("Loading CLIP Model...", flush=True)
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
        )
        clip_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model.to(device)
        Models.clip_model = clip_model
        Models.clip_preprocess = clip_preprocess

    if Models.reranker is None:
        print("Loading Reranker...", flush=True)
        Models.reranker = CrossEncoder("BAAI/bge-reranker-base", trust_remote_code=True, device="cpu")


_initialized = False

def embed_text(texts: List[str], is_query: bool = False) -> np.ndarray:
    initialize()
    # print(f"Embedding {len(texts)} text chunks...", flush=True)
    # prefix  = "query: " if is_query else "passage: "
    # prefixed = [prefix + t for t in texts]
    # vecs = Models.e5_model.encode(prefixed, normalize_embeddings=True)
    # print("Text embedding complete.", flush=True)
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
    # print("Image embedding complete.", flush=True)
    # return vecs.cpu().numpy().astype("float32")

    response = requests.post(
        f"{SERVER_URL}/embed/image",
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

def index_images(image_paths: List[str]):

    initialize()

    image_index = VectorStore.image_index
    image_meta = VectorStore.image_meta
    indexed_images = VectorStore.indexed_images

    records = []

    for path in image_paths:
        p = Path(path)

        image_id = get_image_id(p)
        doc_id = "upload"

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



def save(directory: str):
    initialize()
    os.makedirs(directory, exist_ok=True)
    if VectorStore.text_index:
        faiss.write_index(VectorStore.text_index,  os.path.join(directory, "text.index"))
    if VectorStore.image_index:
        faiss.write_index(VectorStore.image_index, os.path.join(directory, "image.index"))
    with open(os.path.join(directory, "meta.pkl"), "wb") as f:
        pickle.dump({
            "text_meta":       VectorStore.text_meta,
            "image_meta":      VectorStore.image_meta,
            "doc_meta_store":  Data.docs,
            "image_to_chunks": Data.image_to_chunks,
            "table_to_chunks": Data.table_to_chunks,
            "all_chunks":     Data.chunks,
            "indexed_chunks":  VectorStore.indexed_chunks,
            "indexed_images":  VectorStore.indexed_images,
            "indexed_docs":    VectorStore.indexed_docs,
        }, f)
    print(f"saved to {directory}/")


def load(directory: str):
    initialize()

    tp = os.path.join(directory, "text.index")
    ip = os.path.join(directory, "image.index")
    mp = os.path.join(directory, "meta.pkl")

    if os.path.exists(tp):  text_index  = faiss.read_index(tp)
    if os.path.exists(ip):  image_index = faiss.read_index(ip)
    if os.path.exists(mp):
        with open(mp, "rb") as f:
            m = pickle.load(f)
        VectorStore.text_meta.update(m["text_meta"])
        VectorStore.image_meta.update(m["image_meta"])
        Data.docs.update(m["doc_meta_store"])
        Data.image_to_chunks.update(m["image_to_chunks"])
        Data.table_to_chunks.update(m["table_to_chunks"])
        Data.chunks = m["all_chunks"]
        VectorStore.indexed_chunks.update(m.get("indexed_chunks", set()))
        VectorStore.indexed_images.update(m.get("indexed_images", set()))
        VectorStore.indexed_docs.update(m.get("indexed_docs",   set()))
    print(f" loaded from {directory}/")


def initialize():
    global _initialized
    if _initialized:
        return

    set_environ()

    load_api_key()

    # print("Loading models... (this might take a minute)", flush=True)
    # print("Models loaded successfully!", flush=True)

    _initialized = True