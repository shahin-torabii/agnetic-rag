from fastapi import FastAPI, HTTPException
import uvicorn
import torch
from sentence_transformers import SentenceTransformer,CrossEncoder
import open_clip
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image

app = FastAPI(title="models server")


print("Loading models ...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

e5_model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
clip_model.eval()

reranker = CrossEncoder("BAAI/bge-reranker-base", device=device, trust_remote_code=True)
print("All models loaded successfully and serving!", flush=True)


class TextEmbeddingRequest(BaseModel):
    texts: List[str]
    is_query: bool = False

class ImageEmbeddingRequest(BaseModel):
    image_paths: List[str]

class TextEmbeddingRequestOpenclip(BaseModel):
    query:str

class RerankRequest(BaseModel):
    query: str
    chunks: List[dict]


@app.post("/embed/text")
async def embed_text_endpoint(req:TextEmbeddingRequest):
    print(f"Embedding {len(req.texts)} text chunks...", flush=True)
    prefix  = "query: " if req.is_query else "passage: "
    prefixed = [prefix + t for t in req.texts]
    vecs = e5_model.encode(prefixed, normalize_embeddings=True)
    print("Text embedding complete.", flush=True)
    return {"embeddings": vecs.tolist()}


@app.post("/embed/open_clip/image")
async def embed_image_endpoint(req: ImageEmbeddingRequest):
    try:
        images = torch.stack([clip_preprocess(Image.open(p).convert("RGB")) for p in req.image_paths])
        images = images.to(device)
        with torch.no_grad():
            vecs = clip_model.encode_image(images)
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
        return {"embeddings": vecs.cpu().numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/open_clip/text")
async def embed_text_openclip_endpoint(req: TextEmbeddingRequestOpenclip):
    try:
        q_token = open_clip.tokenize([req.query])

        with torch.no_grad():
            q_vec = clip_model.encode_text(q_token)
        q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=-1)
        q_vec = q_vec.cpu().numpy().astype("float32").tolist()

        return {"embeddings": q_vec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
async def rerank_endpoint(req: RerankRequest):
    if not req.chunks:
        return {"chunks": []}

    try:

        pairs = [(req.query, chunk["text"]) for chunk in req.chunks]
        rerank_scores = reranker.predict(pairs)

        reranked = []
        for chunk, score in zip(req.chunks, rerank_scores):
            result = dict(chunk)
            result["score"] = round(float(score), 4)
            reranked.append(result)

        return {"chunks": reranked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    run_server()