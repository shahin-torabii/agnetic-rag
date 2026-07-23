from typing import List, Tuple
import numpy as np
import requests

from config.manager import get_config
from core.logger import get_logger
from core.mlflow_tracking import log_metrics
from core.types import Data
from embedding.models import VectorStore, SERVER_URL

logger = get_logger(__name__)
from embedding.indexer import embed_text
from retriever.query_expansion import improve_query, is_structural_query, _chunk_to_dict, build_context


_cfg = get_config().retrieval


def search_text(query: str, k: int = None, oversample_factor: int = 4, allowed_doc_ids: list = None) -> List[dict]:
    if k is None:
        k = _cfg.text_k
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
        if allowed_doc_ids is not None and chunk.doc_id not in allowed_doc_ids:
            continue

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


def search_chunk_image(query: str, k: int = None, allowed_doc_ids: list = None) -> List[dict]:
    if k is None:
        k = _cfg.chunk_image_k
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
        if allowed_doc_ids is not None and meta.doc_id not in allowed_doc_ids:
            continue
        results.append({
            "score":        round(float(score), 4),
            "image_id":     meta["image_id"],
            "path":         meta["path"],
            "doc_id":       meta["doc_id"],
            "chunk_index":  meta["chunk_index"],
            "section_path": meta["section_path"],
        })
    return results



def search_image(query: str, k: int = None, allowed_doc_ids: list = None) -> List[dict]:
    if k is None:
        k = _cfg.image_k
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
        if allowed_doc_ids is not None and meta["doc_id"] not in allowed_doc_ids:
            continue
        results.append({
            "score":        round(float(score), 4),
            "image_id":     meta["image_id"],
            "path":         meta["path"],
            "doc_id":       meta["doc_id"],
        })
    return results




def rerank(query: str, chunks: List[dict],
           top_k: int = None, threshold: float = None) -> List[dict]:
    if top_k is None:
        top_k = _cfg.top_k
    if threshold is None:
        threshold = _cfg.score_threshold
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




def retrieval_text(query: str, k: int = None,
                   min_threshold: float = None, allowed_doc_ids: list = None) -> List[dict]:
    if k is None:
        k = _cfg.text_k
    if min_threshold is None:
        min_threshold = _cfg.score_threshold

    primary_results  = search_text(query, k=2, oversample_factor=4, allowed_doc_ids=allowed_doc_ids)
    reranked_results = rerank(query=query, chunks=primary_results, threshold=-50)

    if reranked_results and reranked_results[0]["score"] >= min_threshold:
        log_metrics({"retrieval_high_confidence": 1, "retrieval_count": len(reranked_results)})
        logger.info("High confidence results matched")
        return reranked_results

    new_query = improve_query(query)
    log_metrics({"retrieval_query_expansion": 1})
    logger.info("Activating query expansion fallback", extra={"expanded_query": new_query})

    fallback_candidates = search_text(new_query, k=k, oversample_factor=4, allowed_doc_ids=allowed_doc_ids)
    final_res = rerank(query=new_query, chunks=fallback_candidates,
                  top_k=k, threshold=0.0)
    log_metrics({"retrieval_fallback_count": len(final_res)})

    return final_res


def search_and_rerank_chunk_images( query: str,rerank_query: str, allowed_doc_ids: list = None) -> tuple:

    image_results = search_chunk_image(query, k=10, allowed_doc_ids=allowed_doc_ids)

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


def retrieval_chunk_image(query: str, k: int = None,
                    min_threshold: float = None, allowed_doc_ids: list = None) -> List[dict]:
    if k is None:
        k = _cfg.chunk_image_k
    if min_threshold is None:
        min_threshold = _cfg.score_threshold

    reranked, image_scores = search_and_rerank_chunk_images(query, rerank_query=query, allowed_doc_ids = allowed_doc_ids)

    if not reranked or reranked[0]["score"] < min_threshold:
        logger.info("Low image confidence — activating query expansion fallback")
        expanded_query          = improve_query(query)
        reranked, image_scores  = search_and_rerank_chunk_images(
            expanded_query, rerank_query=expanded_query, allowed_doc_ids = allowed_doc_ids
        )

    structural = is_structural_query(query)
    cfg_blend = get_config().blend
    rw, iw_low, iw_high = 1.0, cfg_blend.low_weights, cfg_blend.high_weights

    final_results = []
    for r in reranked:
        key          = (r["doc_id"], r["chunk_index"])
        image_score  = image_scores.get(key, 0.0)
        rerank_score = r["score"]

        if structural or image_score < cfg_blend.ignore_image_threshold:
            final_score = rerank_score
        elif image_score < cfg_blend.high_image_threshold:
            final_score = iw_low[0] * rerank_score + iw_low[1] * image_score
        else:
            final_score = iw_high[0] * rerank_score + iw_high[1] * image_score

        result = dict(r)
        result["image_score"] = image_score
        result["final_score"] = round(final_score, 4)
        final_results.append(result)
    logger.debug("Final blended results", extra={"count": len(final_results)})
    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    return final_results[:k]


##TODO for future and advanced versions: generate caption for images and use that as a reranker or use a multimodal reranker
def retrieval_image( query: str,k: int = None,min_threshold: float = None, allowed_doc_ids:list = None) -> list[dict]:
    if k is None:
        k = _cfg.image_k
    if min_threshold is None:
        min_threshold = _cfg.score_threshold

    original_results = search_image(query, k, allowed_doc_ids)

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


def retrieval(query:str, k :int = None, min_threshold: float = None, is_doc:bool = True, allowed_doc_ids: list = None):
    if k is None:
        k = _cfg.top_k
    if min_threshold is None:
        min_threshold = _cfg.score_threshold
    if not is_doc:
        return retrieval_image(query, k, min_threshold, allowed_doc_ids)
    text_results = retrieval_text(query, k, min_threshold, allowed_doc_ids)
    image_results = retrieval_chunk_image(query, k, min_threshold, allowed_doc_ids)
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
