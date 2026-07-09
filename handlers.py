import time
import json
from retreival import retrieval
from data_gathering import Chunk, BaseMeta, Data
from LLM import encode_image_to_base64, HF_LLM

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5
TOKENS_PER_BATCH = 3000
GROUP_SIZE = 5



def call_llm(llm, system_template: str, system_vars: dict, user_text: str, max_tokens=600):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}"),
    ])
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

def send_images_to_vlm(image_paths, query, context=""):

    full_text = query if not context else f"{query}\n\nContext:\n{context}"
    content = [{"type": "text", "text": full_text}]
    for image_path in image_paths:
        img_b64 = encode_image_to_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    messages = [
        SystemMessage(content="You are an assistant that answers questions about images."),
        HumanMessage(content=content),
    ]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = HF_LLM.vision_llm.invoke(messages)
            return response.content
        except Exception as e:
            last_error = e
            print(f"VLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_WAIT_SECONDS)
    raise last_error

#
# def send_images_to_vlm(image_paths, query, context=""):
#
#     full_text = query if not context else f"{query}\n\nContext:\n{context}"
#
#     content = [{"type": "text", "text": full_text}]
#
#     for image_path in image_paths:
#         img_b64 = encode_image_to_base64(image_path)
#         content.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{img_b64}"
#             },
#         })
#
#     messages = [
#         {
#             "role": "system",
#             "content": "You are an assistant that answers questions about images."
#         },
#         {
#             "role": "user",
#             "content": content
#         }
#     ]
#
#     last_error = None
#
#     for attempt in range(MAX_RETRIES):
#         try:
#             response = HF_LLM.vision_llm.invoke(messages)
#             return response
#         except Exception as e:
#             last_error = e
#             print(f"VLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
#             time.sleep(RETRY_WAIT_SECONDS)
#
#     raise last_error


def describe_images(image_paths, context=""):
    descriptions = []
    for path in image_paths:
        if path not in image_description_cache:
            image_description_cache[path] = send_images_to_vlm(
                [path], "Explain this image using the provided context.", context
            )
        descriptions.append(image_description_cache[path])
    return descriptions


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
    return retrieval(query, k=15, is_doc=True)


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
            merged.append(merge_fn(items[i:i + group_size]))
        items = merged
    return items


SUMMARIZE_PARTIAL_TEMPLATE = """You are a focused summarization assistant.
Summarize ONLY the provided content.
Be concise and query-aware."""

SUMMARIZE_FINAL_TEMPLATE = """You are an expert summarizer.
You are given partial summaries of a document.

Merge them, remove redundancy, and produce one coherent final summary."""


def summarize_partial(batch, query=None):
    text = batch_to_text(batch)
    if query:

        text = f"Query: {query}\n\n{text}"
    return call_llm(HF_LLM.fast_llm, SUMMARIZE_PARTIAL_TEMPLATE, {}, text, max_tokens=400)


def summarize_final(partial_summaries):
    text = "\n".join(partial_summaries)
    return call_llm(HF_LLM.strong_llm, SUMMARIZE_FINAL_TEMPLATE, {}, text, max_tokens=800)


def summarize_chunks(chunks, meta_data, full_summary=True, query=None):
    selected = (sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
                if full_summary else select_related_chunks(chunks, query))
    batches = make_batches(selected)
    partials = [summarize_partial(b, query) for b in batches]
    if len(partials) == 1:
        return partials[0]
    partials = merge_in_groups(partials, summarize_final)
    return partials[0] if len(partials) == 1 else summarize_final(partials)



OVERVIEW_PARTIAL_TEMPLATE = """You are a document analyst.

You are given part of a document.

Identify:
1. Main topic
2. Important concepts
3. Purpose of this section
4. How this section contributes to the document

Do NOT summarize every detail. Be concise."""


OVERVIEW_FINAL_TEMPLATE = """You are a document analyst.

{meta_context}

You are given multiple section overviews from the same document.

Generate:
1. Document title/topic
2. Purpose
3. Main sections
4. Key concepts
5. Overall conclusion

Do NOT repeat information. Create a coherent high-level overview."""


def _build_meta_context(meta):
    if meta is None:
        return ""
    return (f"Document Title: {meta.title}\n"
            f"Document Type: {meta.doc_type}\n"
            f"Source Type: {meta.source_type}\n")


def overview_partial(batch):
    text = batch_to_text(batch)
    return call_llm(HF_LLM.fast_llm, OVERVIEW_PARTIAL_TEMPLATE, {}, text, max_tokens=550)


def overview_final(partial_overviews, meta=None):
    text = "\n".join(partial_overviews)
    meta_context = _build_meta_context(meta)
    return call_llm(
        HF_LLM.strong_llm, OVERVIEW_FINAL_TEMPLATE,
        {"meta_context": meta_context}, text, max_tokens=800,
    )


def overview_func(chunks, meta=None, batch_size_tokens=TOKENS_PER_BATCH):
    batches = make_batches(chunks, max_tokens=batch_size_tokens)
    partials = [overview_partial(b) for b in batches]
    if len(partials) == 1:
        return partials[0]
    partials = merge_in_groups(partials, lambda g: overview_final(g, meta=meta))
    return partials[0] if len(partials) == 1 else overview_final(partials, meta=meta)


EXPLAIN_PARTIAL_WITH_QUERY_TEMPLATE = """You are a teaching assistant.

Explain ONLY the content relevant to:
{query}

Your explanation should be educational, preserve important details,
explain concepts clearly, and avoid summarizing."""

EXPLAIN_PARTIAL_NO_QUERY_TEMPLATE = """You are a teaching assistant.

Explain the provided content. Teach the concepts, preserve important
details, explain terminology and relationships, and avoid merely
summarizing."""

EXPLAIN_FINAL_TEMPLATE = """You are a teaching assistant.

{meta_context}

You are given multiple explanations extracted from different sections
of the same document.

Your task is to merge them into a single coherent teaching document.

Requirements:
- Preserve important technical details.
- Preserve equations, definitions, examples, experimental findings,
  hyperparameters, assumptions, and conclusions.
- Explain relationships between concepts.
- Reorganize content logically when necessary.
- Remove only exact duplication.
- Do NOT shorten explanations merely for brevity.
- Do NOT summarize unless information is repeated.
- Prefer completeness over conciseness.

Your goal is to teach the document, not summarize it."""


def explain_partial(batch, query=None):
    text = batch_to_text(batch)
    if query:
        return call_llm(HF_LLM.fast_llm, EXPLAIN_PARTIAL_WITH_QUERY_TEMPLATE,
                         {"query": query}, text, max_tokens=800)
    return call_llm(HF_LLM.fast_llm, EXPLAIN_PARTIAL_NO_QUERY_TEMPLATE,
                     {}, text, max_tokens=800)


def explain_final(partial_explanations, meta=None, query=None):
    text = "\n".join(partial_explanations)
    meta_context = _build_meta_context(meta)
    return call_llm(
        HF_LLM.strong_llm, EXPLAIN_FINAL_TEMPLATE,
        {"meta_context": meta_context}, text, max_tokens=1700,
    )


def explain_doc(chunks, meta=None, full_explanation=False, query=None, batch_size_tokens=TOKENS_PER_BATCH):
    selected = (sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
                if full_explanation else select_related_chunks(chunks, query))
    batches = make_batches(selected, max_tokens=batch_size_tokens)
    partials = [explain_partial(b, query) for b in batches]
    if len(partials) == 1:
        return partials[0]
    partials = merge_in_groups(partials, lambda g: explain_final(g, meta=meta, query=query))
    return partials[0] if len(partials) == 1 else explain_final(partials, meta=meta, query=query)



COMPARE_DOCUMENTS_TEMPLATE = """You are an expert document comparison assistant.

Compare the provided documents. For each document identify: purpose, main
topics, important concepts.

Then provide:
1. Similarities
2. Differences
3. Strengths of each document
4. Unique contributions
5. Overall comparison

Be factual. Do not invent information."""


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
    return call_llm(HF_LLM.strong_llm, COMPARE_DOCUMENTS_TEMPLATE, {}, comparison_text, max_tokens=900)


def document_actions():
    pass


def rewrite_query_with_history(query, history):
    pass