import time
from retreival import retrieval
from data_gathering import Chunk, BaseMeta, Data
from LLM import encode_image_to_base64, HF_LLM


MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5

TOKENS_PER_BATCH = 3000

GROUP_SIZE = 5


def call_llm(model, system_prompt, user_text, max_tokens=600):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            }

            model_lower = model.lower()


            response = HF_LLM.client.chat.completions.create(**kwargs)

            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            print(
                f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            time.sleep(RETRY_WAIT_SECONDS)

    raise last_error


image_description_cache = {}


def send_images_to_vlm(image_paths, query, context=""):
    full_text = query
    if context:
        full_text = f"{query}\n\nContext:\n{context}"

    content = [{"type": "text", "text": full_text}]

    for image_path in image_paths:
        img_b64 = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = HF_LLM.client.chat.completions.create(
                model=HF_LLM.vision_model_name,
                temperature=0.2,
                max_tokens=500,

                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that answers questions about images.",
                    },
                    {"role": "user", "content": content},
                ],
            )
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            print(f"VLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_WAIT_SECONDS)

    raise last_error


def describe_images(image_paths, context=""):
    descriptions = []
    for path in image_paths:
        if path not in image_description_cache:
            image_description_cache[path] = send_images_to_vlm(
                [path],
                "Explain this image using the provided context.",
                context,
            )
        descriptions.append(image_description_cache[path])
    return descriptions



def select_related_chunks(chunks, query=None):
    if query is None:
        return chunks

    q = query.lower()
    strict = []
    fuzzy = []

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

    batches = []
    current_batch = []
    current_tokens = 0

    for chunk in chunks:
        tokens = chunk.token_count or 0

        if current_batch and current_tokens + tokens > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

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
            group = items[i:i + group_size]
            merged.append(merge_fn(group))
        items = merged

    return items


def summarize_partial(batch, query=None):
    text = batch_to_text(batch)
    if query:
        text = f"Query: {query}\n\n{text}"

    system_prompt = """
You are a focused summarization assistant.
Summarize ONLY the provided content.
Be concise and query-aware.
"""
    summary = call_llm(HF_LLM.fast_model_name, system_prompt, text, max_tokens=400)
    return summary


def summarize_final(partial_summaries):
    text = "\n".join(partial_summaries)

    system_prompt = """
You are an expert summarizer.
You are given partial summaries of a document.

Merge them, remove redundancy, and produce one coherent final summary.
"""
    final_summary = call_llm(HF_LLM.strong_model_name, system_prompt, text, max_tokens=800)
    return final_summary


def summarize_chunks(chunks, meta_data, full_summary=True, query=None):
    if full_summary:
        selected_chunks = sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
    else:
        selected_chunks = select_related_chunks(chunks, query)

    batches = make_batches(selected_chunks)

    partial_summaries = [summarize_partial(batch, query) for batch in batches]

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    partial_summaries = merge_in_groups(partial_summaries, summarize_final)
    for p in partial_summaries:
        print(p)
        print("\n\n")
        print("__"*40)

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    final_summary = summarize_final(partial_summaries)
    return final_summary


def overview_partial(batch):
    text = batch_to_text(batch)

    system_prompt = """
You are a document analyst.

You are given part of a document.

Identify:
1. Main topic
2. Important concepts
3. Purpose of this section
4. How this section contributes to the document

Do NOT summarize every detail. Be concise.
"""
    overview = call_llm(HF_LLM.fast_model_name, system_prompt, text, max_tokens=550)
    return overview


def overview_final(partial_overviews, meta=None):
    text = "\n".join(partial_overviews)

    meta_context = ""
    if meta is not None:
        meta_context = f"""
Document Title: {meta.title}
Document Type: {meta.doc_type}
Source Type: {meta.source_type}
"""

    system_prompt = f"""
You are a document analyst.

{meta_context}

You are given multiple section overviews from the same document.

Generate:
1. Document title/topic
2. Purpose
3. Main sections
4. Key concepts
5. Overall conclusion

Do NOT repeat information. Create a coherent high-level overview.
"""
    final_overview = call_llm(HF_LLM.strong_model_name, system_prompt, text, max_tokens=800)
    return final_overview


def overview_func(chunks, meta=None, batch_size_tokens=TOKENS_PER_BATCH):

    batches = make_batches(chunks, max_tokens=batch_size_tokens)

    partial_overviews = [overview_partial(batch) for batch in batches]

    if len(partial_overviews) == 1:
        return partial_overviews[0]

    partial_overviews = merge_in_groups(
        partial_overviews,
        lambda group: overview_final(group, meta=meta),
    )

    if len(partial_overviews) == 1:
        return partial_overviews[0]

    return overview_final(partial_overviews, meta=meta)


def explain_partial(batch, query=None):
    text = batch_to_text(batch)

    if query:
        system_prompt = f"""
You are a teaching assistant.

Explain ONLY the content relevant to:
{query}

Your explanation should be educational, preserve important details,
explain concepts clearly, and avoid summarizing.
"""
    else:
        system_prompt = """
You are a teaching assistant.

Explain the provided content. Teach the concepts, preserve important
details, explain terminology and relationships, and avoid merely
summarizing.
"""

    return call_llm(HF_LLM.fast_model_name, system_prompt, text, max_tokens=600)


def explain_final(partial_explanations, meta=None, query=None):
    text = "\n".join(partial_explanations)

    meta_context = ""
    if meta is not None:
        meta_context = f"""
Document Title: {meta.title}
Document Type: {meta.doc_type}
Source Type: {meta.source_type}
"""

    system_prompt = f"""
You are a teaching assistant.

{meta_context}

You are given multiple explanations from different parts of the same
document. Create ONE coherent explanation: organize logically, remove
redundancy, preserve technical details, and connect concepts across
sections.
"""
    response = call_llm(HF_LLM.strong_model_name, system_prompt, text, max_tokens=1700)

    return response


def explain_doc(chunks, meta=None, full_explanation=False, query=None,
                 batch_size_tokens=TOKENS_PER_BATCH):
    if full_explanation:
        selected_chunks = sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
    else:
        selected_chunks = select_related_chunks(chunks, query)

    batches = make_batches(selected_chunks, max_tokens=batch_size_tokens)

    partial_explanations = [explain_partial(batch, query) for batch in batches]

    if len(partial_explanations) == 1:
        return partial_explanations[0]

    partial_explanations = merge_in_groups(
        partial_explanations,
        lambda group: explain_final(group, meta=meta, query=query),
    )

    for p in partial_explanations:
        print(p)
        print("\n\n")
        print("__"*40)

    if len(partial_explanations) == 1:
        return partial_explanations[0]

    return explain_final(partial_explanations, meta=meta, query=query)


def compare_documents(target_files):
    docs_overviews = {}

    for file in target_files:
        related_chunks = Data.doc_to_chunks.get(file, [])
        meta = Data.docs.get(file)
        docs_overviews[file] = overview_func(chunks=related_chunks, meta=meta)

    comparison_input = [
        f"\nDocument: {file}\n\nOverview:\n{overview}\n"
        for file, overview in docs_overviews.items()
    ]
    comparison_text = "\n\n".join(comparison_input)

    system_prompt = """
You are an expert document comparison assistant.

Compare the provided documents. For each document identify: purpose, main
topics, important concepts.

Then provide:
1. Similarities
2. Differences
3. Strengths of each document
4. Unique contributions
5. Overall comparison

Be factual. Do not invent information.
"""
    return call_llm(HF_LLM.strong_model_name, system_prompt, comparison_text, max_tokens=900)


def document_actions():
    pass