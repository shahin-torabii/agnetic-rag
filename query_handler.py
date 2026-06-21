from query_router import  Intent, handle_request, ActiveContext
from data_gathering import UserRequest
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


CURRENT_UPLOAD_SIGNALS = {

    # English
    "this",
    "these",
    "attached",
    "uploaded",
    "current",
    "above",

    "this file",
    "this document",
    "this pdf",
    "this image",
    "this audio",

    # Persian
    "این",
    "این فایل",
    "این سند",
    "این pdf",
    "این پی دی اف",

    "فایل آپلود شده",
    "سند آپلود شده",

    "فایلی که آپلود کردم",
    "فایلی که فرستادم",

    "تصویر آپلود شده",
    "عکس آپلود شده",

    "صدای آپلود شده",
    "فایل صوتی آپلود شده"
}


ALL_UPLOAD_SIGNALS = {

    # English
    "all uploaded",
    "all files",
    "all documents",
    "all images",
    "all audios",

    "every file",
    "every document",

    # Persian
    "همه فایل ها",
    "همه فایل‌ها",

    "تمام فایل ها",
    "تمام فایل‌ها",

    "همه اسناد",
    "تمام اسناد",

    "همه تصاویر",
    "تمام تصاویر",

    "همه عکس ها",
    "همه عکس‌ها",

    "همه فایل های صوتی",
    "تمام فایل های صوتی"
}


PREVIOUS_FILE_SIGNALS = {

    # English
    "previous file",
    "previous document",
    "earlier file",
    "earlier document",

    "uploaded before",
    "previously uploaded",

    "last uploaded",
    "last document",

    # Persian
    "فایل قبلی",
    "سند قبلی",

    "گزارش قبلی",

    "فایلی که قبلا آپلود کردم",
    "فایلی که قبلاً آپلود کردم",

    "سندی که قبلا فرستادم",
    "سندی که قبلاً فرستادم",

    "قبلی"
}


MULTI_FILE_SIGNALS = {

    # English
    "both",
    "compare",
    "comparison",
    "versus",
    "vs",

    # Persian
    "هر دو",
    "مقایسه",
    "در مقایسه با"
}


@dataclass
class ResolvedReferences:

    current_upload_relevant: bool = False

    previous_upload_relevant: bool = False

    use_all_current_uploads: bool = False

    current_documents: List[str] = field(default_factory=list)
    current_images: List[str] = field(default_factory=list)
    current_audio: List[str] = field(default_factory=list)

    referenced_documents: List[str] = field(default_factory=list)
    referenced_images: List[str] = field(default_factory=list)
    referenced_audio: List[str] = field(default_factory=list)

    confidence: float = 1.0



def normalize_name(name: str) -> str:

    name = Path(name).name.lower()

    if "." in name:
        name = ".".join(name.split(".")[:-1])

    return name.strip()


def extract_document_mentions(
    query: str,
    threshold: int = 85
) -> list[str]:

    q = query.lower()

    matches = []

    for doc_id, meta in Data.docs.items():

        title = normalize_name(meta.title)

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


def references_current_upload(
    query: str
) -> bool:

    return contains_signal(
        query,
        CURRENT_UPLOAD_SIGNALS
    )


def references_all_uploads(
    query: str
) -> bool:

    return contains_signal(
        query,
        ALL_UPLOAD_SIGNALS
    )


def references_previous_uploads(
    query: str
) -> bool:

    return contains_signal(
        query,
        PREVIOUS_FILE_SIGNALS
    )


def references_multiple_files(
    query: str
) -> bool:

    return contains_signal(
        query,
        MULTI_FILE_SIGNALS
    )


def resolve_references(
    query: str
) -> ResolvedReferences:

    result = ResolvedReferences()



    current_docs = list(
        ActiveContext.active_documents
    )

    current_images = list(
        ActiveContext.active_images
    )

    current_audio = list(
        ActiveContext.active_audio
    )

    has_current_uploads = (
        len(current_docs) > 0
        or len(current_images) > 0
        or len(current_audio) > 0
    )



    current_signal = references_current_upload(
        query
    )

    previous_signal = references_previous_uploads(
        query
    )

    all_signal = references_all_uploads(
        query
    )

    multi_signal = references_multiple_files(
        query
    )


    explicit_docs = extract_document_mentions(
        query
    )


    result.current_upload_relevant = (
        has_current_uploads
        and current_signal
    )

    result.previous_upload_relevant = (
        previous_signal
        or len(explicit_docs) > 0
    )

    result.use_all_current_uploads = (
        has_current_uploads
        and all_signal
    )


    if result.current_upload_relevant:

        if result.use_all_current_uploads:

            result.current_documents = current_docs
            result.current_images = current_images
            result.current_audio = current_audio

        else:



            if len(current_docs) == 1:
                result.current_documents = current_docs

            if len(current_images) == 1:
                result.current_images = current_images

            if len(current_audio) == 1:
                result.current_audio = current_audio


            elif len(current_docs) > 1:

                current_doc_names = {
                    normalize_name(x): x
                    for x in current_docs
                }

                for doc_name in current_doc_names:

                    if doc_name in query.lower():

                        result.current_documents.append(
                            current_doc_names[doc_name]
                        )

                #
                # If none matched and query says
                # "compare", "both", etc.
                #

                if (
                    not result.current_documents
                    and multi_signal
                ):
                    result.current_documents = current_docs

    #
    # Historical references
    #

    current_names = {
        normalize_name(x)
        for x in current_docs
    }

    for doc_id in explicit_docs:

        meta = Data.docs[doc_id]

        if (
            normalize_name(meta.title)
            not in current_names
        ):
            result.referenced_documents.append(
                doc_id
            )

    #
    # Confidence
    #

    if (
        not result.current_documents
        and not result.referenced_documents
        and not result.current_images
        and not result.current_audio
    ):
        result.confidence = 0.3

    elif (
        result.referenced_documents
        or result.current_documents
    ):
        result.confidence = 0.9

    return result





def build_candidate_context():

    docs = []

    for doc_id, meta in Data.docs.items():

        docs.append(
            {
                "doc_id": doc_id,
                "title": meta.title,
                "type": getattr(
                    meta,
                    "doc_type",
                    "document"
                )
            }
        )

    return docs


def llm_reference_resolver(
    query: str
) -> ResolvedReferences:

    candidates = build_candidate_context()

    current_uploads = {
        "documents": list(
            ActiveContext.active_documents
        ),
        "images": list(
            ActiveContext.active_images
        ),
        "audio": list(
            ActiveContext.active_audio
        )
    }

    system_prompt = """
You are a file reference resolver.

Your task:

1. Determine whether the user refers to:
   - currently uploaded files
   - previously uploaded files
   - both

2. Determine which files are being referenced.

3. Only use file names that exist
   in the provided candidate list.

4. Return valid JSON only.

Output schema:

{
  "current_upload_relevant": bool,
  "use_all_current_uploads": bool,

  "current_documents": [],
  "current_images": [],
  "current_audio": [],

  "referenced_documents": [],
  "referenced_images": [],
  "referenced_audio": [],

  "confidence": float
}
"""

    user_prompt = f"""
User Query:
{query}

Current Uploads:
{json.dumps(current_uploads, ensure_ascii=False, indent=2)}

Available Files:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

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

        return ResolvedReferences(
            **parsed
        )

    except Exception:

        return ResolvedReferences(
            confidence=0.0
        )


def resolve(request:UserRequest):
    resolved = resolve_references(request.query)

    if (
            resolved.confidence < 0.6
            or (
            not resolved.current_documents
            and not resolved.referenced_documents
            and not resolved.current_images
            and not resolved.referenced_images
            and not resolved.current_audio
            and not resolved.referenced_audio
    )
    ):
        llm_result = llm_reference_resolver(
            request.query
        )

        if llm_result.confidence > resolved.confidence:
            resolved = llm_result

    return resolved


def send_images_to_vlm(image_paths: list[str],query: str):
    content = [
        {
            "type": "text",
            "text": query
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
    image_paths = [image for image in target_files.current_images] if target_files.current_images else []
    image_paths.append(image for image in target_files.referenced_images) if target_files.referenced_images else image_paths


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
def summarize_chunks(chunks, meta_data, Full_summary = True, query = None):
    pass




def overview_func(chunks, meta):
    pass


def explain_doc(chunks, meta, full_explanation = False, query = None):
    pass



def handle_audio(intent, request ,target_files):

    chunks = ActiveContext.active_files_chunks["audio"]["audio_chunks"]
    transcripts  = ActiveContext.active_files_chunks["audio"]["transcripts"]
    meta_audio = ActiveContext.active_files_chunks["audio"]["meta_audio"]

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
            summary  = summarize_chunks(chunks=chunks,meta=meta_audio)
        case Intent.AUDIO_OVERVIEW:
            overview = overview_func(chunks, meta_audio)
        case _:
            return "\n".join(transcripts)


def compare_documents():
    pass


def document_actions():
    pass


def handle_document(intent, request, target_files):

    cur_docs_chunks = ActiveContext.active_files_chunks["document"]["doc_chunks"]
    cur_docs_meta = ActiveContext.active_files_chunks["document"]["doc_meta"]
    cur_docs_img_idx = ActiveContext.active_files_chunks["document"]["doc_img_idx"]
    cur_docs_tbl_idx = ActiveContext.active_files_chunks["document"]["doc_tbl_idx"]

    related_chunks = []
    related_docs = []
    related_files = (
            target_files.current_documents +
            target_files.referenced_documents
    )

    for doc_id in related_files:


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


            for audio_path in ActiveContext.audio:
                chunks, transcript, meta = process_audio(audio_path)

                ingest(
                    chunks=chunks,
                    doc_meta=meta,
                    img_to_ch=None,
                    tbl_to_ch=None
                )

                index_to_faiss(chunks)

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