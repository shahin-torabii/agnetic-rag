import json
import time
from functools import lru_cache
from typing import List

from config.manager import get_config
from core.constants import GROUP_SIZE, MAX_RETRIES, RETRY_WAIT_SECONDS, TOKENS_PER_BATCH
from core.types import BaseMeta, Chunk, Data
from llm.client import HF_LLM
from llm.prompts import (
    COMPARE_DOCUMENTS_TEMPLATE,
    CONTEXTUALIZE_SYSTEM_PROMPT,
    EXPLAIN_FINAL_TEMPLATE,
    EXPLAIN_PARTIAL_NO_QUERY_TEMPLATE,
    EXPLAIN_PARTIAL_WITH_QUERY_TEMPLATE,
    OVERVIEW_FINAL_TEMPLATE,
    OVERVIEW_PARTIAL_TEMPLATE,
    SUMMARIZE_FINAL_TEMPLATE,
    SUMMARIZE_PARTIAL_TEMPLATE,
)
from llm.vision import describe_images, send_images_to_vlm
from retriever.search import retrieval


def call_llm(
    llm, system_template: str, system_vars: dict, user_text: str, max_tokens=600
):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm.bind(max_tokens=max_tokens) | StrOutputParser()
    invoke_vars = {**system_vars, "input": user_text}

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return chain.invoke(invoke_vars)
        except Exception as e:
            last_error = e
            print(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_WAIT_SECONDS)
    raise last_error


image_description_cache = {}


def select_related_chunks(chunks, query=None):
    if query is None:
        return chunks
    q = query.lower()
    strict, fuzzy = [], []
    for c in chunks:
        if not c.section_path:
            continue
        joined = " ".join(c.section_path).lower()
        if any(q == s.lower() for s in c.section_path):
            strict.append(c)
        elif q in joined or joined in q:
            fuzzy.append(c)
    if strict:
        return strict
    if fuzzy:
        return fuzzy
    return retrieval(query, k=get_config().retrieval.text_k, is_doc=True)


def make_batches(chunks, max_tokens=TOKENS_PER_BATCH):
    batches, current_batch, current_tokens = [], [], 0
    for chunk in chunks:
        tokens = chunk.token_count or 0
        if current_batch and current_tokens + tokens > max_tokens:
            batches.append(current_batch)
            current_batch, current_tokens = [], 0
        current_batch.append(chunk)
        current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches


def batch_to_text(batch):
    parts = []
    for chunk in batch:
        if chunk.chunk_type == "image_context":
            images = list(chunk.image_refs.values())
            parts.extend(describe_images(images, context=chunk.text))
        else:
            parts.append(chunk.text)
    return "\n".join(parts)


def merge_in_groups(items, merge_fn, group_size=GROUP_SIZE):
    while len(items) > group_size:
        merged = []
        for i in range(0, len(items), group_size):
            merged.append(merge_fn(items[i : i + group_size]))
        items = merged
    return items


def summarize_partial(batch, query=None):
    text = batch_to_text(batch)
    if query:
        text = f"Query: {query}\n\n{text}"
    return call_llm(
        HF_LLM.fast_llm, SUMMARIZE_PARTIAL_TEMPLATE, {}, text, max_tokens=400
    )


def summarize_final(partial_summaries):
    text = "\n".join(partial_summaries)
    return call_llm(
        HF_LLM.strong_llm, SUMMARIZE_FINAL_TEMPLATE, {}, text, max_tokens=800
    )


def summarize_chunks(chunks, meta_data, full_summary=True, query=None):
    selected = (
        sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
        if full_summary
        else select_related_chunks(chunks, query)
    )
    batches = make_batches(selected)
    partials = [summarize_partial(b, query) for b in batches]
    if len(partials) == 1:
        return partials[0]
    partials = merge_in_groups(partials, summarize_final)
    return partials[0] if len(partials) == 1 else summarize_final(partials)


def _build_meta_context(meta):
    if meta is None:
        return ""
    return (
        f"Document Title: {meta.title}\n"
        f"Document Type: {meta.doc_type}\n"
        f"Source Type: {meta.source_type}\n"
    )


def overview_partial(batch):
    text = batch_to_text(batch)
    return call_llm(
        HF_LLM.fast_llm, OVERVIEW_PARTIAL_TEMPLATE, {}, text, max_tokens=550
    )


def overview_final(partial_overviews, meta=None):
    text = "\n".join(partial_overviews)
    meta_context = _build_meta_context(meta)
    return call_llm(
        HF_LLM.strong_llm,
        OVERVIEW_FINAL_TEMPLATE,
        {"meta_context": meta_context},
        text,
        max_tokens=800,
    )


def overview_func(chunks, meta=None, batch_size_tokens=TOKENS_PER_BATCH):
    batches = make_batches(chunks, max_tokens=batch_size_tokens)
    partials = [overview_partial(b) for b in batches]
    if len(partials) == 1:
        return partials[0]
    partials = merge_in_groups(partials, lambda g: overview_final(g, meta=meta))
    return partials[0] if len(partials) == 1 else overview_final(partials, meta=meta)


def explain_partial(batch, query=None):
    text = batch_to_text(batch)
    if query:
        return call_llm(
            HF_LLM.fast_llm,
            EXPLAIN_PARTIAL_WITH_QUERY_TEMPLATE,
            {"query": query},
            text,
            max_tokens=800,
        )
    return call_llm(
        HF_LLM.fast_llm, EXPLAIN_PARTIAL_NO_QUERY_TEMPLATE, {}, text, max_tokens=800
    )


def explain_final(partial_explanations, meta=None, query=None):
    text = "\n".join(partial_explanations)
    meta_context = _build_meta_context(meta)
    return call_llm(
        HF_LLM.strong_llm,
        EXPLAIN_FINAL_TEMPLATE,
        {"meta_context": meta_context},
        text,
        max_tokens=1700,
    )


def explain_doc(
    chunks,
    meta=None,
    full_explanation=False,
    query=None,
    batch_size_tokens=TOKENS_PER_BATCH,
):
    selected = (
        sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
        if full_explanation
        else select_related_chunks(chunks, query)
    )
    batches = make_batches(selected, max_tokens=batch_size_tokens)
    partials = [explain_partial(b, query) for b in batches]
    if len(partials) == 1:
        return partials[0]
    partials = merge_in_groups(
        partials, lambda g: explain_final(g, meta=meta, query=query)
    )
    return (
        partials[0]
        if len(partials) == 1
        else explain_final(partials, meta=meta, query=query)
    )


def compare_documents(target_files):
    docs_overviews = {}
    for file in target_files:
        related_chunks = Data.doc_to_chunks.get(file, [])
        meta = Data.docs.get(file)
        docs_overviews[file] = overview_func(chunks=related_chunks, meta=meta)

    comparison_text = "\n\n".join(
        f"\nDocument: {file}\n\nOverview:\n{overview}\n"
        for file, overview in docs_overviews.items()
    )
    return call_llm(
        HF_LLM.strong_llm,
        COMPARE_DOCUMENTS_TEMPLATE,
        {},
        comparison_text,
        max_tokens=900,
    )


def document_actions():
    pass


@lru_cache
def _get_rewrite_with_history_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    rewrite_with_history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            (
                "human",
                "Conversation history:\n{history}\n\nFollow-up question: {query}",
            ),
        ]
    )
    evaluator_chain = (
        rewrite_with_history_prompt
        | HF_LLM.fast_llm.bind(max_tokens=85)
        | StrOutputParser()
    )
    return evaluator_chain


def rewrite_query_with_history(query, history):
    return (
        _get_rewrite_with_history_chain()
        .invoke({"query": query, "history": history})
        .strip()
    )
