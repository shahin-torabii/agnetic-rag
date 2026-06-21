from query_router import  Intent, handle_request, ActiveContext
from data_gathering import UserRequest, BaseMeta
from LLM import HF_LLM, encode_image_to_base64
from index_embedd import index_images, VectorStore, get_image_id, index_chunks, index_chunk_images
from typing import List
from retreival import retrieval
from pathlib import Path
from chunking_handlers.audio_hanlder import process_audio
from chunking_handlers.non_digital_pdf_handler import process_scanned_pdf
from chunking_handlers.digital_pdf_handler import process_pdf
from chunking_handlers.word_handler import process_docx
from chunking_handlers.pp_handler import process_pptx
from chunking_handlers.excel_handler import process_excel
from data_gathering import ingest, Chunk
import puremagic
from rapidfuzz import fuzz
from dataclasses import dataclass, field
from data_gathering import Data
import json
from collections import defaultdict


AUDIO_TRANSCRIPT = None

CURRENT_FILE_SIGNALS = {

    # English
    "this",
    "these",
    "attached",
    "uploaded",

    "this file",
    "this document",
    "this pdf",

    # Persian
    "این",
    "این فایل",
    "این سند",
    "این پی دی اف",
    "این pdf",

    "فایل آپلود شده",
    "سند آپلود شده",

    "فایلی که آپلود کردم",
    "فایلی که فرستادم"
}


ALL_FILES_SIGNALS = {

    # English
    "all uploaded files",
    "all files",
    "all documents",

    "every file",
    "every document",

    # Persian
    "همه فایل ها",
    "همه فایل‌ها",

    "تمام فایل ها",
    "تمام فایل‌ها",

    "همه اسناد",
    "تمام اسناد"
}


ORDINAL_SIGNALS = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,

    "اول": 0,
    "دوم": 1,
    "سوم": 2,
    "چهارم": 3,
    "پنجم": 4
}


LAST_SIGNALS = {
    "last",
    "latest",

    "آخر",
    "آخری",
    "آخرین"
}


@dataclass
class ResolvedTargets:

    documents: List[str] = field(default_factory=list)

    images: List[str] = field(default_factory=list)

    confidence: float = 1.0



def normalize_name(name: str) -> str:

    name = Path(name).name.lower()

    if "." in name:
        name = ".".join(name.split(".")[:-1])

    return name.strip()



def resolve_ordinals(query: str) -> list[str]:

    q = query.lower()

    active_docs = list(
        ActiveContext.active_documents + ActiveContext.active_audio
    )

    matched = []

    for signal, idx in ORDINAL_SIGNALS.items():

        if signal in q:

            if idx < len(active_docs):

                matched.append(
                    active_docs[idx]
                )

    if any(
        signal in q
        for signal in LAST_SIGNALS
    ):

        if active_docs:

            matched.append(
                active_docs[-1]
            )

    return list(set(matched))



def extract_document_mentions(
    query: str,
    threshold: int = 85
) -> list[str]:

    q = query.lower()

    matches = []

    for doc_id, meta in Data.docs.items():

        title = normalize_name(
            meta.title
        )

        if title in q:

            matches.append(doc_id)
            continue

        score = fuzz.partial_ratio(
            title,
            q
        )
        if score >= threshold:
            matches.append(doc_id)

    return list(set(matches))


def contains_signal(
    query: str,
    signals: set[str]
) -> bool:

    q = query.lower()

    return any(
        signal.lower() in q
        for signal in signals
    )


def resolve_targets(
    query: str
) -> ResolvedTargets:

    result = ResolvedTargets()

    q = query.lower()


    explicit_docs = extract_document_mentions(
        query
    )

    if explicit_docs:

        result.documents.extend(
            explicit_docs
        )


    if contains_signal(
        q,
        CURRENT_FILE_SIGNALS
    ):

        result.documents.extend(
            ActiveContext.active_documents
        )
        result.documents.extend(ActiveContext.active_audio)

        result.images.extend(
            ActiveContext.active_images
        )


    if contains_signal(
        q,
        ALL_FILES_SIGNALS
    ):

        result.documents.extend(
            Data.docs.keys()
        )


    result.documents = list(
        set(result.documents)
    )

    result.images = list(
        set(result.images)
    )


    if (
        not result.documents
        and not result.images
    ):

        result.confidence = 0.3

    else:

        result.confidence = 0.9

    return result


def build_candidate_context():

    docs = []

    for doc_id, meta in Data.docs.items():

        docs.append(
            {
                "doc_id": doc_id,
                "title": meta.title,
                "doc_type": getattr(
                    meta,
                    "doc_type",
                    "document"
                )
            }
        )

    return docs


def llm_reference_resolver(
    query: str
) -> ResolvedTargets:

    candidates = build_candidate_context()

    current_uploads = {
        "documents": list(
            ActiveContext.active_documents +  ActiveContext.active_audio
        ),
        "images": list(
            ActiveContext.active_images
        ),

    }

    system_prompt = """
You are a file reference resolver.

Your task:

Determine which files the user is referring to.

The user may refer to:

- currently uploaded files
- previously uploaded files
- both

You may ONLY return files that exist in the provided candidate lists.

Return ONLY valid JSON.

Schema:

{
  "documents": [],
  "images": [],
  "confidence": 0.0
}
"""

    user_prompt = f"""
User Query:
{query}

Current Uploads:
{json.dumps(current_uploads, ensure_ascii=False, indent=2)}

Available Files:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Examples:

Query:
"Compare the first two documents"

Output:
{{
  "documents": ["doc_a.pdf", "doc_b.pdf"],
  "images": [],
  "confidence": 0.95
}}

Query:
"Summarize the transformer paper"

Output:
{{
  "documents": ["transformer_survey.pdf"],
  "images": [],
  "confidence": 0.9
}}

Return JSON only.
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
                "content": user_prompt
            }
        ]
    )

    content = response.choices[0].message.content

    try:

        parsed = json.loads(content)

        return ResolvedTargets(
            documents=parsed.get(
                "documents",
                []
            ),
            images=parsed.get(
                "images",
                []
            ),

            confidence=parsed.get(
                "confidence",
                0.0
            )
        )

    except Exception:

        return ResolvedTargets(
            confidence=0.0
        )

def resolve(request:UserRequest):

    query = request.query
    resolved = resolve_targets(query)


    if (
            resolved.confidence < 0.6
            and ActiveContext.active_documents
    ):

        ordinal_docs = resolve_ordinals(
            query
        )

        if ordinal_docs:
            resolved.documents.extend(
                ordinal_docs
            )

            resolved.documents = list(
                set(resolved.documents)
            )

            resolved.confidence = 0.95

    if resolved.confidence < 0.6:
        resolved = llm_reference_resolver(
            query
        )

    return resolved

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



def detect_file_type(file_path):
    path = Path(file_path)

    suffix, mime = None, None

    if path.exists():
        suffix = path.suffix
        mime = puremagic.from_file(str(path), mime=True)

    return mime, suffix, path


def get_doc_chunks(file_path):
    mime, suffix, path = detect_file_type(file_path)
    if mime == "application/pdf":
        #pdf
        chunks, meta, img_idx, tbl_idx = process_pdf(file_path)
        if chunks == []:
            return process_scanned_pdf(file_path)

        return  chunks, meta, img_idx, tbl_idx

    if mime in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        #word
        return process_docx(file_path)


    if mime in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel"
    ]:
        return process_excel(file_path)

    if mime in [
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint"
    ]:

        return process_pptx(file_path)
    # if mime.startswith("video/"):
    #     return "Video"

    return "invalid"



def ingest_image(image_paths:List[str]):
    not_indexed_images = []
    for im_path in image_paths:
        p = Path(im_path)

        image_id = get_image_id(p)
        doc_id = "upload"
        key = (doc_id, image_id)
        if not key in VectorStore.indexed_images:
            not_indexed_images.append(im_path)

    index_images(not_indexed_images)

def index_to_faiss(chunks:List[Chunk]):
    index_chunks(chunks)
    index_chunk_images(chunks)


def handle_image(intent, request, target_files):
    ##TODO image path are not necesaarily from request
    image_paths = [image for image in target_files.images] if target_files.current_images else []



    match intent:
        case Intent.IMAGE_SEARCH:
            if len(request.images) <4:
                response = send_images_to_vlm(image_paths, request)
                print(response)
            else:
                result = retrieval(request.query, k = 5 , is_doc=False)
                ##TODO send results along query to llm for final result
        case Intent.IMAGE_UNDERSTANDING:
            response = send_images_to_vlm(image_paths, request)
            print(response)

        case _:
            raise ValueError(
                f"Unhandled image intent: {intent}"
            )

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



def select_summary_chunks(chunks, query: str =  None):


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


def summarize_chunks(chunks: List[Chunk], meta_data: BaseMeta , full_summary: bool = True, query: str = None):
    if full_summary:

        selected_chunks = sorted(chunks, key=lambda x: (x.doc_id, x.chunk_index))
    else:
        selected_chunks = select_summary_chunks(chunks, query)

    summaries = []

    for i in range(0, len(selected_chunks), 5):
        batch = selected_chunks[i:i + 5]

        summaries.append(
            llm_summarizer(batch, mode="partial", query=query)
        )

    return llm_summarizer(summaries, mode="final")


def overview_func(chunks, meta):
    pass


def explain_doc(chunks, meta, full_explanation = False, query = None):
    pass



def handle_audio(intent, request ,target_files):


    transcripts = AUDIO_TRANSCRIPT

    related_chunks = []
    meta_audio = []

    for doc_id in target_files.documents:

        related_chunks.extend(
            Data.doc_to_chunks.get(doc_id, [])
        )

        if doc_id in Data.docs:
            meta_audio.append(
                Data.docs[doc_id]
            )
    match intent:

        case Intent.AUDIO_TRANSCRIBE:
            return "\n".join(transcripts)

        case Intent.AUDIO_QA:
            result = retrieval(query=request.query,k=10,is_doc=True)
            # return answer_with_context(
            #     query=request.query,
            #     context=context
            # )

        case Intent.AUDIO_SUMMARIZE:
            summary  = summarize_chunks(chunks=related_chunks,meta=meta_audio)
        case Intent.AUDIO_OVERVIEW:
            overview = overview_func(related_chunks, meta_audio)
        case _:
            return "\n".join(transcripts)


def compare_documents():
    pass


def document_actions():
    pass


def handle_document(intent, request, target_files):



    related_chunks = []
    related_docs = []

    for doc_id in target_files.documents:


        related_chunks.extend(
            Data.doc_to_chunks.get(doc_id, [])
        )


        if doc_id in Data.docs:
            related_docs.append(
                Data.docs[doc_id]
            )

    match intent:
        case Intent.DOCUMENT_SUMMARIZE:
            summary = summarize_chunks(related_chunks, related_docs,Full_summary=True)
        case Intent.DOCUMENT_SECTION_SUMMARIZE:
            summary = summarize_chunks(related_chunks, related_docs, Full_summary=False, query= request.query)
        case Intent.DOCUMENT_QA | Intent.SEARCH_DOCUMENT:
            result = retrieval(query=request.query, k=10, is_doc=True)
            # return answer_with_context(
            #     query=request.query,
            #     context=context
            # )
        case Intent.DOCUMENT_OVERVIEW:
            overview = overview_func(related_chunks, related_docs)
        case Intent.DOCUMENT_FULL_EXPLAIN:
            explain_doc(related_chunks, related_docs, full_explanation=True)
        case Intent.DOCUMENT_SECTION_EXPLAIN:
            explain_doc(related_chunks, related_docs, full_explanation=False, query= request.query)
        case Intent.COMPARE_DOCUMENTS:
            compare_documents()
        case Intent.DOCUMENT_ACTION:
            document_actions()
        case _:
            raise ValueError(
                f"Unhandled intent: {intent}"
            )



def handle_general(intent, request):
    ##TODO change this to a better handler
    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.model_name,
        messages=[
            {
                "role": "user",
                "content": request.query
            }
        ]
    )
    return response.choices[0].message.content


def handle_uploads(request:UserRequest):
    if ActiveContext.has_file:
        if ActiveContext.active_documents is not None and len(ActiveContext.active_documents) > 0:

            for doc_path in request.documents:
                chunks, meta, img_idx, tbl_idx = get_doc_chunks(doc_path)


                ingest(chunks, meta, img_to_ch=img_idx, tbl_to_ch=tbl_idx)
                index_to_faiss(chunks)

        if ActiveContext.active_audio is not None and len(ActiveContext.active_audio) > 0:

            transcripts = []
            for audio_path in ActiveContext.active_audio:
                chunks, transcript, meta = process_audio(audio_path)

                ingest(
                    chunks=chunks,
                    doc_meta=meta,
                    img_to_ch=None,
                    tbl_to_ch=None
                )

                transcripts.append(transcript)

                index_to_faiss(chunks)

            AUDIO_TRANSCRIPT = transcripts

    if ActiveContext.active_images is not None and len(ActiveContext.active_images) > 0:
            image_paths = [image_path for image_path in ActiveContext.active_images]
            ingest_image(image_paths)


def handle_query(request: UserRequest):

    intent, ctx = handle_request(request)
    handle_uploads(request)
    target_files = resolve(request)

    match intent:

        case (Intent.IMAGE_SEARCH | Intent.IMAGE_UNDERSTANDING):
            return handle_image(intent, request, target_files)

        case (Intent.AUDIO_QA| Intent.AUDIO_SUMMARIZE| Intent.AUDIO_TRANSCRIBE | Intent.AUDIO_OVERVIEW):
            return handle_audio(intent, request, target_files)

        case (
            Intent.GENERAL_CHAT| Intent.UNKNOWN ):
            return handle_general(intent, request, target_files)

        case _:
            return handle_document(intent, request, target_files)




# def answer_with_context(
#     query: str,
#     context: str
# ):
#
#     response = HF_LLM.client.chat.completions.create(
#         model=HF_LLM.model_name,
#         temperature=0.2,
#         messages=[
#             {
#                 "role": "system",
#                 "content":
#                 "Answer only from the provided context."
#             },
#             {
#                 "role": "user",
#                 "content":
#                 f"""
# Question:
# {query}
#
# Context:
# {context}
# """
#             }
#         ]
#     )
#
#     return response.choices[0].message.content