from query_router import Intent, handle_request, ActiveContext
from data_gathering import UserRequest, ingest
from index_embedd import index_images, VectorStore, get_image_id, index_chunks, index_chunk_images
from pathlib import Path
from chunking_handlers.audio_hanlder import process_audio
from chunking_handlers.non_digital_pdf_handler import process_scanned_pdf
from chunking_handlers.digital_pdf_handler import process_pdf
from chunking_handlers.word_handler import process_docx
from chunking_handlers.pp_handler import process_pptx
from chunking_handlers.excel_handler import process_excel
import puremagic
from data_gathering import Data
from handlers import *
from resolver import resolve

AUDIO_TRANSCRIPT = None


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
        print("digital pdf")
        #pdf
        chunks, meta, img_idx, tbl_idx = process_pdf(file_path)
        if chunks == []:
            print("scanned pdf")
            return process_scanned_pdf(file_path)

        return chunks, meta, img_idx, tbl_idx

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


def ingest_image(image_paths: List[str]):
    not_indexed_images = []
    for im_path in image_paths:
        p = Path(im_path)

        image_id = get_image_id(p)
        doc_id = "upload"
        key = (doc_id, image_id)
        if not key in VectorStore.indexed_images:
            not_indexed_images.append(im_path)

    index_images(not_indexed_images)


def index_to_faiss(chunks: List[Chunk]):
    index_chunks(chunks)
    index_chunk_images(chunks)


def handle_image(intent, request, target_files):
    ##TODO image path are not necesaarily from request
    image_paths = [image for image in target_files.images] if target_files.current_images else []

    match intent:
        case Intent.IMAGE_SEARCH:
            if len(request.images) < 4:
                response = send_images_to_vlm(image_paths, request)
                print(response)
            else:
                result = retrieval(request.query, k=5, is_doc=False)
                ##TODO send results along query to llm for final result
        case Intent.IMAGE_UNDERSTANDING:
            response = send_images_to_vlm(image_paths, request)
            print(response)

        case _:
            raise ValueError(
                f"Unhandled image intent: {intent}"
            )


def handle_audio(intent, request, target_files):
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
            audio_transcript =  "\n".join(transcripts)

        case Intent.AUDIO_QA:
            result = retrieval(query=request.query, k=10, is_doc=True)

        case Intent.AUDIO_SUMMARIZE:
            summary = summarize_chunks(related_chunks, meta_audio, full_summary=True)

        case Intent.AUDIO_OVERVIEW:
            overview = overview_func(related_chunks, meta_audio)

        case _:
            return "\n".join(transcripts)


def handle_document(intent, request, target_files):
    related_chunks = {}
    related_docs = {}

    for doc_id in target_files.documents:
        related_chunks[doc_id] = (Data.doc_to_chunks.get(doc_id, []))

        if doc_id in Data.docs:
            related_docs[doc_id] = (Data.docs[doc_id])

    match intent:
        case Intent.DOCUMENT_SUMMARIZE:
            summaries = []
            for doc_id in target_files.documents:
                summary = summarize_chunks(related_chunks[doc_id], related_docs[doc_id], full_summary=True)
                summaries.append(summary)
        case Intent.DOCUMENT_SECTION_SUMMARIZE:
            summaries = []
            for doc_id in target_files.documents:
                summary = summarize_chunks(related_chunks[doc_id], related_docs[doc_id], full_summary=False, query=request.query)

        case Intent.DOCUMENT_QA | Intent.SEARCH_DOCUMENT:
            result = retrieval(query=request.query, k=10, is_doc=True)

        case Intent.DOCUMENT_OVERVIEW:
            over_views = []
            for doc_id in target_files.documents:
                overview = overview_func(related_chunks[doc_id], related_docs[doc_id])
                over_views.append(overview)

        case Intent.DOCUMENT_FULL_EXPLAIN:
            explanations = []
            for doc_id in target_files.documents:
                explanation = explain_doc(related_chunks[doc_id], related_docs[doc_id], full_explanation=True)
                explanations.append(explanation)
            print("\n\n".join(explanations))

        case Intent.DOCUMENT_SECTION_EXPLAIN:
            explanations = []
            for doc_id in target_files.documents:
                explanation = explain_doc(related_chunks[doc_id], related_docs[doc_id], full_explanation=False, query=request.query)
                explanations.append(explanation)
            print("\n\n".join(explanations))

        case Intent.COMPARE_DOCUMENTS:
            compare_documents(target_files.documents)

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


def handle_uploads(request: UserRequest):
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

    print("intent:",intent)
    print("docs:", target_files)

    match intent:

        case (Intent.IMAGE_SEARCH | Intent.IMAGE_UNDERSTANDING):
            return handle_image(intent, request, target_files)

        case (Intent.AUDIO_QA | Intent.AUDIO_SUMMARIZE | Intent.AUDIO_TRANSCRIBE | Intent.AUDIO_OVERVIEW):
            return handle_audio(intent, request, target_files)

        case (
        Intent.GENERAL_CHAT | Intent.UNKNOWN ):
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
