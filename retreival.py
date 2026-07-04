import torch
from typing import List
from functools import lru_cache
import open_clip
from data_gathering import Data
from index_embedd import VectorStore, embed_text, embed_image, Models
import numpy as np
from LLM import HF_LLM
import requests
from index_embedd import embed_image
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser




IMPROVE_QUERY_SYSTEM_PROMPT = """You are a query expansion assistant for a multilingual RAG system.

Your goal is to improve retrieval recall by expanding the user's query with:
- Synonyms
- Alternative phrasings
- Related technical terminology
- Common document structure terms when appropriate
- Sequential references (first, second, final, etc.) when appropriate

Rules:
1. Keep the original query.
2. Preserve the original language. Never translate.
3. Add only highly plausible alternatives.
4. Do not invent facts or specific information.
5. Do not answer the query.
6. Do not explain your reasoning.
7. Output ONLY a comma-separated list.
8. Keep the expansion concise (typically 5-10 terms/phrases total).

Examples:

Query: آخرین بخش آزمایش
Output: آخرین بخش آزمایش, بخش پایانی, بخش آخر, قسمت نهایی, مراحل نهایی, بخش دوم

Query: نصب کتابخانه پایتون
Output: نصب کتابخانه پایتون, نصب پکیج پایتون, افزودن کتابخانه, راه اندازی کتابخانه, نصب وابستگی

Query: database connection error
Output: database connection error, database connectivity issue, connection failure, database access problem

Query: first step of installation
Output: first step of installation, installation beginning, setup start, initial installation step, step one"""

improve_query_prompt = ChatPromptTemplate.from_messages([
    ("system", IMPROVE_QUERY_SYSTEM_PROMPT),
    ("human", "User Query: {query}\n\nOptimized Search Query (Output only the terms):"),
])

improve_query_chain = improve_query_prompt | HF_LLM.fast_llm | StrOutputParser()


SERVER_URL = "http://127.0.0.1:8000"

BLEND_LOW_WEIGHTS  = (0.8, 0.2)
BLEND_HIGH_WEIGHTS = (0.2, 0.8)

IGNORE_IMAGE_THRESHOLD = 0.1
HIGH_IMAGE_THRESHOLD   = 0.4


STRUCTURAL_TERMS = {
    # Persian
    "بخش", "فصل", "مرحله", "گام",
    "اول", "اولین", "آخر", "آخرین",
    "پایانی", "نتیجه", "نتیجه گیری",
    "جمع بندی", "مقدمه",
    # English
    "chapter", "section", "part", "step",
    "first", "last", "final", "conclusion",
    "summary", "introduction",
}





def search_text(query: str, k: int = 5, oversample_factor: int = 4) -> List[dict]:
    text_index = VectorStore.text_index
    text_meta = VectorStore.text_meta
    if text_index is None or text_index.ntotal == 0:
        return []

    q_vec = embed_text([query], is_query=True)  #


    broad_k = min(k * oversample_factor, text_index.ntotal)
    scores, ids = text_index.search(q_vec, broad_k)

    results = []
    seen_texts = set()

    for score, fid in zip(scores[0], ids[0]):
        if fid == -1:
            continue
        chunk = text_meta[fid]


        if chunk.text in seen_texts:
            continue
        seen_texts.add(chunk.text)

        results.append({
            "faiss_score":  round(float(score), 4),
            "text":         chunk.text,
            "doc_id":       chunk.doc_id,
            "chunk_index":  chunk.chunk_index,
            "chunk_type":   chunk.chunk_type,
            "section_path": chunk.section_path,
            "image_refs":   chunk.image_refs,
            "table_refs":   chunk.table_refs,
        })


    return results


def search_chunk_image(query: str, k: int = 5) -> List[dict]:
    image_index = VectorStore.doc_image_index
    image_meta = VectorStore.doc_image_meta
    if image_index is None or image_index.ntotal == 0:
        return []

    response = requests.post(
        f"{SERVER_URL}/embed/open_clip/text",
        json={"query": query}
    )
    response.raise_for_status()
    q_vec = np.array(response.json()["embeddings"], dtype="float32")

    scores, ids = image_index.search(q_vec, k)

    results = []
    for score, fid in zip(scores[0], ids[0]):
        if fid == -1:
            continue
        meta = image_meta[fid]
        results.append({
            "score":        round(float(score), 4),
            "image_id":     meta["image_id"],
            "path":         meta["path"],
            "doc_id":       meta["doc_id"],
            "chunk_index":  meta["chunk_index"],
            "section_path": meta["section_path"],
        })
    return results



def search_image(query: str, k: int = 5) -> List[dict]:
    image_index = VectorStore.image_index
    image_meta = VectorStore.image_meta
    if image_index is None or image_index.ntotal == 0:
        return []
    # q_token = open_clip.tokenize([query])
    #
    # with torch.no_grad():
    #     q_vec = clip_model.encode_text(q_token)
    # q_vec =  torch.nn.functional.normalize(q_vec, p=2, dim=-1)
    # q_vec = q_vec.cpu().numpy().astype("float32")


    response = requests.post(
        f"{SERVER_URL}/embed/open_clip/text",
        json={"query": query}
    )
    response.raise_for_status()
    q_vec =  np.array(response.json()["embeddings"], dtype="float32")

    scores, ids = image_index.search(q_vec, k)

    results = []
    for score, fid in zip(scores[0], ids[0]):
        if fid == -1:
            continue
        meta = image_meta[fid]
        results.append({
            "score":        round(float(score), 4),
            "image_id":     meta["image_id"],
            "path":         meta["path"],
            "doc_id":       meta["doc_id"],
        })
    return results


def build_context(text_results:List[dict], image_results:List[dict]) ->str:
    parts = []

    for r in text_results:
        path = " > ".join(r["section_path"]) if r["section_path"] else r["doc_id"]
        parts.append(f"[{path}]\n{r['text']}")
    if image_results:
        for r in image_results:
              key = (r["doc_id"], r["image_id"])
              chunks  = Data.image_to_chunks.get(key,[])
              if chunks:
                  related = chunks[0]
                  path=">".join(r["section_path"] if r["section_path"] else r["doc_id"])
                  parts.append(f"[{path}| image_{r["image_id"]}]\n context: {related.text}")

    return "\n___\n".join(parts)


def rerank(query: str, chunks: List[dict],
           top_k: int = 5, threshold: float = 0.1) -> List[dict]:
    # if not chunks:
    #     return []
    #
    # pairs = [(query, chunk["text"]) for chunk in chunks]
    # rerank_scores = Models.reranker.predict(pairs)
    #
    # reranked = []
    # for chunk, score in zip(chunks, rerank_scores):
    #     result = dict(chunk)
    #     result["score"] = round(float(score), 4)
    #     reranked.append(result)

    response = requests.post(
        f"{SERVER_URL}/rerank",
        json={"query": query, "chunks": chunks}
    )
    response.raise_for_status()

    reranked = response.json()["chunks"]

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return [r for r in reranked if r["score"] > threshold][:top_k]


@lru_cache(maxsize=256)
def improve_query(query: str) -> str:
    """
    Cached — same query string won't trigger a second LLM call
    within the same session.
    """
    return improve_query_chain.invoke({"query": query}).strip()


def is_structural_query(query: str) -> bool:
    q = query.lower()
    return any(term.lower() in q for term in STRUCTURAL_TERMS)


def _chunk_to_dict(chunk) -> dict:
    return {
        "text":         chunk.text,
        "doc_id":       chunk.doc_id,
        "chunk_index":  chunk.chunk_index,
        "chunk_type":   chunk.chunk_type,
        "section_path": chunk.section_path,
        "image_refs":   chunk.image_refs,
        "table_refs":   chunk.table_refs,
    }


def retrieval_text(query: str, k: int = 3,
                   min_threshold: float = 0.08) -> List[dict]:

    primary_results  = search_text(query, k=2, oversample_factor=4)
    reranked_results = rerank(query=query, chunks=primary_results, threshold=-50)

    if reranked_results and reranked_results[0]["score"] >= min_threshold:
        print("✅ High confidence results matched!")
        return reranked_results

    print("⚠️  Low confidence — activating query expansion fallback...")
    new_query = improve_query(query)
    print(f"   Expanded query: '{new_query}'")

    fallback_candidates = search_text(new_query, k=k, oversample_factor=4)
    final_res = rerank(query=new_query, chunks=fallback_candidates,
                  top_k=k, threshold=0.0)

    return final_res


def search_and_rerank_chunk_images( query: str,rerank_query: str) -> tuple[list[dict], dict]:

    image_results = search_chunk_image(query, k=10)

    image_scores = {}

    for r in image_results:
        key = (r["doc_id"], r["chunk_index"])

        image_scores[key] = max(
            image_scores.get(key, 0.0),
            r["score"]
        )

    candidate_chunks = [
        _chunk_to_dict(chunk)
        for chunk in Data.chunks
        if (chunk.doc_id, chunk.chunk_index) in image_scores
    ]

    reranked = rerank(
        query=rerank_query,
        chunks=candidate_chunks
    )

    return reranked, image_scores


def retrieval_chunk_image(query: str, k: int = 3,
                    min_threshold: float = 0.08) -> List[dict]:

    reranked, image_scores = search_and_rerank_chunk_images(query, rerank_query=query)

    if not reranked or reranked[0]["score"] < min_threshold:
        print("⚠️  Low image confidence — activating query expansion fallback...")
        expanded_query          = improve_query(query)
        reranked, image_scores  = search_and_rerank_chunk_images(
            expanded_query, rerank_query=expanded_query
        )

    structural = is_structural_query(query)
    rw, iw_low, iw_high = 1.0, BLEND_LOW_WEIGHTS, BLEND_HIGH_WEIGHTS

    final_results = []
    for r in reranked:
        key          = (r["doc_id"], r["chunk_index"])
        image_score  = image_scores.get(key, 0.0)
        rerank_score = r["score"]

        if structural or image_score < IGNORE_IMAGE_THRESHOLD:
            final_score = rerank_score
        elif image_score < HIGH_IMAGE_THRESHOLD:
            final_score = iw_low[0] * rerank_score + iw_low[1] * image_score
        else:
            final_score = iw_high[0] * rerank_score + iw_high[1] * image_score

        result = dict(r)
        result["image_score"] = image_score
        result["final_score"] = round(final_score, 4)
        final_results.append(result)
    print(final_results)
    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    return final_results[:k]


##TODO for future and advanced versions: generate caption for images and use that as a reranker or use a multimodal reranker
def retrieval_image( query: str,k: int = 3,min_threshold: float = 0.08) -> list[dict]:

    original_results = search_image(query, k)

    if (
        not original_results
        or original_results[0]["score"] < min_threshold
    ):
        print(
            "⚠️ Low image confidence — activating query expansion fallback..."
        )

        expanded_query = improve_query(query)

        expanded_results = search_image(
            expanded_query,
            k
        )

        if (
                expanded_results
                and expanded_results[0]["score"]
                > original_results[0]["score"]
        ):
            return expanded_results

    return original_results



def retrieval( query: str,k: int = 3,min_threshold: float = 0.08,is_doc: bool = True):

    if not is_doc:
        return retrieval_image(query=query,k=k,min_threshold=min_threshold)

    text_results = retrieval_text(query,k,min_threshold)

    image_results = retrieval_chunk_image(query,k,min_threshold)
    seen = set()
    combined = []
    for r in text_results + image_results:

        key = (r["doc_id"], r["chunk_index"])

        if key not in seen:
            seen.add(key)
            combined.append(r)
    context = build_context(text_results=[r for r in combined if r in text_results ]
                            ,
        image_results=[
            r
            for r in combined
            if r in image_results
        ],
    )
    return context
