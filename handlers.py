from collections import defaultdict
from LLM import HF_LLM, encode_image_to_base64
from retreival import retrieval
from typing import List
from data_gathering import Chunk , BaseMeta, Data


def send_images_to_vlm(image_paths: list[str],query: str, context:str = ""):
    content = [
        {
            "type": "text",
            "text": query,
            "cotext":context

        }
    ]

    for image_path in image_paths:
        img_b64 = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            }
        )
    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.vision_model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content":
                "You are an assistant that answers questions about images."
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.choices[0].message.content


def select_related_chunks(chunks, query: str =  None):


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


def compare_documents(target_files):

    docs_overviews = {}

    for file in target_files:

        related_chunks = (
            Data.doc_to_chunks.get(file, [])
        )

        meta = Data.docs.get(file)

        overview = overview_func(
            chunks=related_chunks,
            meta=meta
        )

        docs_overviews[file] = overview

    comparison_input = []

    for file, overview in docs_overviews.items():
        comparison_input.append(
            f"""
    Document: {file}

    Overview:
    {overview}
    """
        )

    comparison_text = "\n\n".join(comparison_input)

    system_prompt = """
    You are an expert document comparison assistant.

    Compare the provided documents.

    For each document identify:

    - purpose
    - main topics
    - important concepts

    Then provide:

    1. Similarities
    2. Differences
    3. Strengths of each document
    4. Unique contributions
    5. Overall comparison

    Be factual.
    Do not invent information.
    """

    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": comparison_text
            }
        ]
    )

    comparison =  response.choices[0].message.content
    return comparison

def document_actions():
    pass


def llm_summarizer(batch, mode="partial", query=None):

    image_descriptions = []
    text_chunks = []
    if mode in ["partial", "section"]:

        for chunk in batch:

            if chunk.chunk_type == "image_context":

                images = list(chunk.image_refs.values())
                context = chunk.text

                img_query = f"""
    Explain these images using context:
    {context}
    """

                desc = send_images_to_vlm(images, img_query)
                image_descriptions.append(desc)

            else:
                text_chunks.append(chunk.text)

        combined_text = "\n".join(text_chunks + image_descriptions)

        if query:
            combined_text = f"Query: {query}\n\n{combined_text}"

        system_prompt = """
    You are a focused summarization assistant.
    Summarize ONLY the provided content.
    Be concise and query-aware.
    """

    elif mode == "full":

        for chunk in batch:

            if chunk.chunk_type == "image_context":

                images = list(chunk.image_refs.values())
                context = chunk.text

                img_query = f"""
    Explain these images using context:
    {context}
    """

                desc = send_images_to_vlm(images, img_query)
                image_descriptions.append(desc)

            else:
                text_chunks.append(chunk.text)

        combined_text = "\n".join(text_chunks + image_descriptions)

        system_prompt = """
    You are a document-level summarizer.
    Summarize the entire document comprehensively.
    """

    elif mode == "final":


        combined_text = "\n".join(batch)

        system_prompt = """
    You are an expert summarizer.
    You are given partial summaries of a document.

    Your task:
    - merge them
    - remove redundancy
    - produce a coherent final summary
    """

    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_text}
        ]
    )

    return response.choices[0].message.content


def summarize_chunks(chunks: List[Chunk], meta_data: List[BaseMeta] , full_summary: bool = True, query: str = None):

    if full_summary:
        selected_chunks = sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
    else:
        selected_chunks = select_related_chunks(chunks, query)

    summaries = []

    for i in range(0, len(selected_chunks), 5):
        batch = selected_chunks[i:i + 5]

        summaries.append(
            llm_summarizer(batch, mode="partial", query=query)
        )

    return llm_summarizer(summaries, mode="final")


def llm_overview( batch, mode: str = "partial", meta=None ):

    if mode == "partial":

        image_descriptions = []
        text_chunks = []

        for chunk in batch:

            if chunk.chunk_type == "image_context":

                images = list(chunk.image_refs.values())

                context = chunk.text

                image_description = send_images_to_vlm(
                    image_paths=images,
                    query=(
                        "Explain the provided images using the provided context."
                    ),
                    context=context,
                )

                image_descriptions.append(image_description)

            else:
                text_chunks.append(chunk.text)

        combined_text = "\n".join(
            text_chunks + image_descriptions
        )

        system_prompt = """
You are a document analyst.

You are given part of a document.

Identify:

1. Main topic
2. Important concepts
3. Purpose of this section
4. How this section contributes to the document

Do NOT summarize every detail.

Be concise.
"""

    elif mode == "final":

        # batch is List[str]
        combined_text = "\n".join(batch)

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

You are given multiple section overviews
from the same document.

Generate:

1. Document title/topic
2. Purpose
3. Main sections
4. Key concepts
5. Overall conclusion

Do NOT repeat information.

Create a coherent high-level overview.
"""

    else:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": combined_text
            }
        ]
    )

    return response.choices[0].message.content


def overview_func(
    chunks,
    meta,
    batch_size: int = 5
):

    partial_overviews = []

    for start_idx in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            start_idx:
            start_idx + batch_size
        ]

        partial_overview = llm_overview(batch=batch, mode="partial")

        partial_overviews.append(partial_overview)

    final_overview = llm_overview(
        batch=partial_overviews,
        mode="final",
        meta=meta
    )

    return final_overview


def llm_explainer(batch, mode="partial", meta=None, query=None ):

    if mode == "partial":

        image_descriptions = []
        text_chunks = []

        for chunk in batch:

            if chunk.chunk_type == "image_context":

                images = list(chunk.image_refs.values())

                context = chunk.text

                image_description = send_images_to_vlm(
                    image_paths=images,
                    query=(
                        "Explain the provided images using the provided context."
                    ),
                    context=context
                )

                image_descriptions.append(image_description)

            else:
                text_chunks.append(chunk.text)

        combined_text = "\n".join(
            text_chunks + image_descriptions)

        if query:

            system_prompt = f"""
You are a teaching assistant.

Explain ONLY the content relevant to:

{query}

Your explanation should:

- be educational
- preserve important details
- explain concepts clearly
- avoid summarizing
"""
        else:

            system_prompt = """
You are a teaching assistant.

Explain the provided content.

Your explanation should:

- teach the concepts
- preserve important details
- explain terminology
- explain relationships
- avoid merely summarizing
"""

    elif mode == "final":

        combined_text = "\n".join(batch)

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

You are given multiple explanations
from different parts of the same document.

Create ONE coherent explanation.

Requirements:

- organize logically
- remove redundancy
- preserve technical details
- make the explanation educational
- connect concepts across sections
"""

    else:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": combined_text
            }
        ]
    )

    return response.choices[0].message.content


def explain_doc(chunks, meta, full_explanation=False, query=None, batch_size=5):


    if full_explanation:
        selected_chunks = sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
    else:
        selected_chunks = select_related_chunks(chunks, query)

    partial_explanations = []

    for start_idx in range(0, len(selected_chunks), batch_size):

        batch = selected_chunks[start_idx: start_idx + batch_size]

        explanation = llm_explainer(batch=batch, mode="partial", query=query)

        partial_explanations.append(explanation)

    final_explanation = llm_explainer(batch=partial_explanations,mode="final",meta=meta,query=query
    )

    return final_explanation