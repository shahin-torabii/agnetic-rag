from sentence_transformers import SentenceTransformer
import open_clip
from sentence_transformers import CrossEncoder
import faiss
import torch
from typing import List, Dict
import numpy as np
from PIL import Image
from word_handler import Chunk, DocMeta
from dotenv import load_dotenv
import os

e5_model = SentenceTransformer("intfloat/multilingual-e5-base")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
reranker = CrossEncoder("BAAI/bge-reranker-base", trust_remote_code=True)

clip_model.eval()


load_dotenv()

hf_api_key = os.getenv("HUGGIN_FACE_API")


indexed_chunks: set = set()
indexed_images: set = set()
indexed_docs:   set = set()


text_index    = None
text_meta: Dict[int, Chunk] = {}


image_index   = None
image_meta: Dict[int, dict] = {}


doc_meta_store:  Dict[str, DocMeta]        = {}
image_to_chunks: Dict[tuple[str, int], List[Chunk]]      = {}
table_to_chunks: Dict[tuple[str, int], List[Chunk]]      = {}
all_chunks:      List[Chunk] = []




def embed_text(texts: List[str], is_query: bool = False) -> np.ndarray:

    prefix  = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]
    vecs = e5_model.encode(prefixed, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")

def embed_image(image_paths: List[str]) -> np.ndarray:
    images = torch.stack([
        clip_preprocess(Image.open(p).convert("RGB"))
        for p in image_paths
    ])                                              # shape (n, 3, 224, 224)

    with torch.no_grad():
        vecs = clip_model.encode_image(images)     # shape (n, 512)

    vecs = vecs /np.linalg.norm(vecs, keepdims=True)
    return vecs.cpu().numpy().astype("float32")



def index_chunks(chunks: List[Chunk]):
    """Embed chunk texts and insert into the text FAISS index."""
    global text_index


    new_chunks = [
        c for c in chunks
        if (c.doc_id, c.chunk_index) not in indexed_chunks
    ]
    if not new_chunks:
        return

    texts = [c.text for c in new_chunks]
    vecs  = embed_text(texts, is_query=False)       # (n, 768)

    if text_index is None:
        faiss.normalize_L2(vecs)
        text_index = faiss.IndexFlatIP(vecs.shape[1])

    base = text_index.ntotal
    text_index.add(vecs)
    for i, chunk in enumerate(new_chunks):
        text_meta[base + i] = chunk
        indexed_chunks.add((chunk.doc_id, chunk.chunk_index))


def index_images(chunks: List[Chunk]):
    """Embed images from image_context chunks and insert into the image FAISS index."""
    global image_index

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
                })

    if not paths:
        return

    vecs = embed_image(paths)                       # (m, 512)

    if image_index is None:
        faiss.normalize_L2(vecs)
        image_index = faiss.IndexFlatIP(vecs.shape[1])

    base = image_index.ntotal
    image_index.add(vecs)
    for i, rec in enumerate(records):
        image_meta[base + i] = rec
        indexed_images.add((rec["doc_id"], rec["image_id"]))