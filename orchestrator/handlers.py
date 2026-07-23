import time
from pathlib import Path
from typing import List

from analysis.handlers import (
    compare_documents,
    describe_images,
    document_actions,
    explain_doc,
    overview_func,
    send_images_to_vlm,
    summarize_chunks,
)
from config.manager import get_config

from core.logger import get_logger

logger = get_logger(__name__)
from core.types import Chunk, Data, Intent, UserRequest, ingest
from embedding.indexer import (
    get_image_id,
    index_chunk_images,
    index_chunks,
    index_images,
)
from embedding.models import VectorStore, save
from ingest.handlers.audio_hanlder import process_audio
from ingest.loader import get_doc_chunks
from llm.client import HF_LLM
from repositories.document import create_session_document, create_user_document
from retriever.search import retrieval
from router.classifier import make_doc_id

AUDIO_TRANSCRIPT = None
PERSIST_DIR = get_config().vector_db_path


def ingest_image(image_paths: List[str], user_id: str):
    not_indexed_images = []
    for im_path in image_paths:
        p = Path(im_path)

        image_id = get_image_id(p)
        doc_id = f"{user_id}:upload"
        key = (doc_id, image_id)
        if not key in VectorStore.indexed_images:
            not_indexed_images.append(im_path)

    doc_id = f"{user_id}:upload"
    index_images(not_indexed_images, doc_id)


def index_to_faiss(chunks: List[Chunk]):
    index_chunks(chunks)
    index_chunk_images(chunks)


def final_rag_llm(query: str, context: str) -> str:
    system_prompt = """
You are a helpful assistant.
Answer ONLY using the provided context.
If context is insufficient, say you don't know.
Be precise, clear, and structured.
"""

    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.strong_model_name,  # or strong model
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Question:
{query}

Context:
{context}
""",
            },
        ],
    )

    return response.choices[0].message.content


def build_context(chunks):
    return "\n\n".join([f"[{c.doc_id}:{c.chunk_index}]\n{c.text}" for c in chunks])


def handle_image(intent, request, target_files, k=None):
    if k is None:
        k = get_config().retrieval.image_k
    ##TODO image path are not necesaarily from request
    image_paths = (
        [image for image in target_files.images] if target_files.images else []
    )

    match intent:
        case Intent.IMAGE_SEARCH:
            if len(request.images) < 4:
                response = send_images_to_vlm(image_paths, request.query)
                return response
            else:
                result = retrieval(
                    request.query,
                    k=k,
                    is_doc=False,
                    allowed_doc_ids=set(target_files.images),
                )
                context = build_context(result)
                final_answer = final_rag_llm(request.query, context)
                return final_answer

        case Intent.IMAGE_UNDERSTANDING:
            response = send_images_to_vlm(image_paths, request.query)
            return response

        case _:
            raise ValueError(f"Unhandled image intent: {intent}")


def handle_audio(intent, request, target_files, k=None):
    if k is None:
        k = get_config().retrieval.top_k
    transcripts = AUDIO_TRANSCRIPT

    related_chunks = []
    meta_audio = []

    for doc_id in target_files.documents:
        related_chunks.extend(Data.doc_to_chunks.get(doc_id, []))

        if doc_id in Data.docs:
            meta_audio.append(Data.docs[doc_id])
    match intent:
        case Intent.AUDIO_TRANSCRIBE:
            audio_transcript = "\n".join(transcripts)
            return audio_transcript

        case Intent.AUDIO_QA:
            result = retrieval(
                query=request.query,
                k=k,
                is_doc=True,
                allowed_doc_ids=set(target_files.documents),
            )
            context = build_context(result)
            final_answer = final_rag_llm(request.query, context)
            return final_answer

        case Intent.AUDIO_SUMMARIZE:
            summary = summarize_chunks(related_chunks, meta_audio, full_summary=True)
            return summary

        case Intent.AUDIO_OVERVIEW:
            overview = overview_func(related_chunks, meta_audio)
            return overview

        case _:
            return "\n".join(transcripts)


def handle_document(intent, request, target_files, k=None):
    if k is None:
        k = get_config().retrieval.top_k
    related_chunks = {}
    related_docs = {}

    for doc_id in target_files.documents:
        related_chunks[doc_id] = Data.doc_to_chunks.get(doc_id, [])

        if doc_id in Data.docs:
            related_docs[doc_id] = Data.docs[doc_id]

    match intent:
        case Intent.DOCUMENT_SUMMARIZE:
            summaries = []
            for doc_id in target_files.documents:
                summary = summarize_chunks(
                    related_chunks[doc_id], related_docs[doc_id], full_summary=True
                )
                summaries.append(summary)
            logger.debug("Document summaries generated", extra={"count": len(summaries)})
            return "\n\n".join(summaries)

        case Intent.DOCUMENT_SECTION_SUMMARIZE:
            summaries = []
            for doc_id in target_files.documents:
                summary = summarize_chunks(
                    related_chunks[doc_id],
                    related_docs[doc_id],
                    full_summary=False,
                    query=request.query,
                )
                summaries.append(summary)

            return "\n\n".join(summaries)

        case Intent.DOCUMENT_QA | Intent.SEARCH_DOCUMENT:
            result = retrieval(
                query=request.query,
                k=k,
                is_doc=True,
                allowed_doc_ids=set(target_files.documents),
            )
            logger.debug("Retrieval result", extra={"chunks_count": len(result) if result else 0})
            context = build_context(result)
            final_answer = final_rag_llm(request.query, context)
            return final_answer

        case Intent.DOCUMENT_OVERVIEW:
            over_views = []
            for doc_id in target_files.documents:
                overview = overview_func(related_chunks[doc_id], related_docs[doc_id])
                over_views.append(overview)
            logger.debug("Document overviews generated", extra={"count": len(over_views)})
            return "\n\n".join(over_views)

        case Intent.DOCUMENT_FULL_EXPLAIN:
            explanations = []
            for doc_id in target_files.documents:
                explanation = explain_doc(
                    related_chunks[doc_id], related_docs[doc_id], full_explanation=True
                )
                explanations.append(explanation)
            logger.debug("Full explanations generated", extra={"count": len(explanations)})
            return "\n\n".join(explanations)

        case Intent.DOCUMENT_SECTION_EXPLAIN:
            explanations = []
            for doc_id in target_files.documents:
                explanation = explain_doc(
                    related_chunks[doc_id],
                    related_docs[doc_id],
                    full_explanation=False,
                    query=request.query,
                )
                explanations.append(explanation)
            logger.debug("Section explanations generated", extra={"count": len(explanations)})
            return "\n\n".join(explanations)

        case Intent.COMPARE_DOCUMENTS:
            compare_result = compare_documents(target_files.documents)
            return compare_result

        case Intent.DOCUMENT_ACTION:
            document_actions()

        case _:
            raise ValueError(f"Unhandled intent: {intent}")


def handle_multimodal(intent, request, target_files, k=None):
    if k is None:
        k = get_config().retrieval.top_k

    audios = []

    for doc_id in target_files.documents:
        if doc_id not in Data.docs:
            continue

        meta = Data.docs[doc_id]
        if getattr(meta, "doc_type", "").lower() == "audio":
            audios.append(doc_id)
        else:
            pass

    images = list(target_files.images)

    related_doc_chunks = {}
    related_doc_docs = {}

    for doc_id in target_files.documents:
        meta = Data.docs.get(doc_id)

        if meta is not None and getattr(meta, "doc_type", "").lower() == "audio":
            continue

        related_doc_chunks[doc_id] = Data.doc_to_chunks.get(doc_id, [])

        if doc_id in Data.docs:
            related_doc_docs[doc_id] = Data.docs[doc_id]

    match intent:
        case Intent.DOCUMENT_SUMMARIZE:
            summaries = []
            for doc_id in related_doc_chunks:
                summary = summarize_chunks(
                    related_doc_chunks[doc_id],
                    related_doc_docs[doc_id],
                    full_summary=True,
                )
                summaries.append(summary)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_SUMMARIZE, request, audios)

            result = {"docs": summaries, "images": image_res, "audio": audio_res}
            return result

        case Intent.DOCUMENT_SECTION_SUMMARIZE:
            summaries = []

            for doc_id in related_doc_chunks:
                summary = summarize_chunks(
                    related_doc_chunks[doc_id],
                    related_doc_docs[doc_id],
                    full_summary=False,
                    query=request.query,
                )
                summaries.append(summary)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_SUMMARIZE, request, audios)

            result = {
                "docs": "\n\n".join(summaries),
                "images": image_res,
                "audio": audio_res,
            }
            return result

        case Intent.DOCUMENT_QA | Intent.SEARCH_DOCUMENT:
            doc_res = retrieval(
                query=request.query,
                k=k,
                is_doc=True,
                allowed_doc_ids=set(target_files.documents),
            )
            doc_context = build_context(doc_res)
            final_doc_answer = final_rag_llm(request.query, doc_context)

            image_res = handle_image(Intent.IMAGE_SEARCH, request, images, k)
            im_context = build_context(image_res)
            final_im_answer = final_rag_llm(request.query, im_context)

            audio_res = handle_audio(Intent.AUDIO_QA, request, audios, k)
            audio_context = build_context(audio_res)
            final_audio_answer = final_rag_llm(request.query, audio_context)

            results = {
                "docs": final_doc_answer,
                "images": final_im_answer,
                "audio": final_audio_answer,
            }
            return results

        case Intent.DOCUMENT_OVERVIEW:
            overviews = []

            for doc_id in related_doc_chunks:
                overview = overview_func(
                    related_doc_chunks[doc_id], related_doc_docs[doc_id]
                )

                overviews.append(overview)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_OVERVIEW, request, audios)

            results = {
                "docs": "\n\n".join(overviews),
                "images": image_res,
                "audio": audio_res,
            }
            return results

        case Intent.DOCUMENT_FULL_EXPLAIN:
            explanations = []

            for doc_id in related_doc_chunks:
                explanation = explain_doc(
                    related_doc_chunks[doc_id],
                    related_doc_docs[doc_id],
                    full_explanation=True,
                )

                explanations.append(explanation)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_OVERVIEW, request, audios)

            result = {
                "docs": "\n\n".join(explanations),
                "images": image_res,
                "audio": audio_res,
            }
            return result

        case Intent.DOCUMENT_SECTION_EXPLAIN:
            explanations = []

            for doc_id in related_doc_chunks:
                explanation = explain_doc(
                    related_doc_chunks[doc_id],
                    related_doc_docs[doc_id],
                    full_explanation=False,
                    query=request.query,
                )

                explanations.append(explanation)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_SUMMARIZE, request, audios)

            result = {
                "docs": "\n\n".join(explanation),
                "images": image_res,
                "audio": audio_res,
            }
            return result

        case Intent.COMPARE_DOCUMENTS:
            return compare_documents(list(related_doc_chunks.keys()))

        case Intent.DOCUMENT_ACTION:
            return document_actions()

        case _:
            raise ValueError(f"Unhandled intent: {intent}")


def handle_general(intent, request):
    ##TODO change this to a better handler
    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.model_name, messages=[{"role": "user", "content": request.query}]
    )
    return response.choices[0].message.content


def handle_uploads(
    request: UserRequest,
    active_ctx,
    session_id: str,
    user_id: str = "1",
    db_session=None,
):
    logger.info(
        "Handling uploads",
        extra={
            "has_file": active_ctx.has_file,
            "docs_count": len(request.documents) if request.documents else 0,
            "imgs_count": len(request.images) if request.images else 0,
        },
    )

    global AUDIO_TRANSCRIPT

    if not active_ctx.has_file:
        return

    if request.documents:
        for doc_path in request.documents:
            doc_id = make_doc_id(user_id, doc_path)

            if doc_id not in Data.docs:
                chunks, meta, img_idx, tbl_idx = get_doc_chunks(doc_path)
                meta.doc_id = doc_id
                for c in chunks:
                    c.doc_id = doc_id

                ingest(chunks, meta, img_to_ch=img_idx, tbl_to_ch=tbl_idx)
                index_to_faiss(chunks)

            if db_session is not None:
                meta = Data.docs.get(doc_id)
                create_user_document(
                    db_session,
                    user_id,
                    doc_id,
                    Path(doc_path).name,
                    title=getattr(meta, "title", "") if meta else "",
                    doc_type=getattr(meta, "doc_type", "GENERAL")
                    if meta
                    else "GENERAL",
                    kind="document",
                )
                create_session_document(
                    db_session, session_id, user_id, doc_id, kind="document"
                )

    if request.audio:
        transcripts = []
        for audio_path in request.audio:
            doc_id = make_doc_id(user_id, audio_path)

            if doc_id not in Data.docs:
                chunks, transcript, meta = process_audio(audio_path)
                meta.doc_id = doc_id
                for c in chunks:
                    c.doc_id = doc_id

                ingest(chunks=chunks, doc_meta=meta, img_to_ch=None, tbl_to_ch=None)
                transcripts.append(transcript)
                index_to_faiss(chunks)

            if db_session is not None:
                meta = Data.docs.get(doc_id)
                create_user_document(
                    db_session,
                    user_id,
                    doc_id,
                    Path(audio_path).name,
                    title=getattr(meta, "title", "") if meta else "",
                    doc_type=getattr(meta, "doc_type", "GENERAL")
                    if meta
                    else "GENERAL",
                    kind="audio",
                )
                create_session_document(
                    db_session, session_id, user_id, doc_id, kind="audio"
                )
        if transcripts:
            AUDIO_TRANSCRIPT = transcripts

    if request.images:
        ingest_image(list(request.images), user_id=user_id)
        if db_session is not None:
            for img_path in request.images:
                doc_id = make_doc_id(user_id, img_path)
                create_user_document(
                    db_session, user_id, doc_id, Path(img_path).name, kind="image"
                )
                create_session_document(
                    db_session, session_id, user_id, doc_id, kind="image"
                )

    save(PERSIST_DIR)


def handle_query(request: UserRequest):
    logger.info("Entering query handler", extra={"query": request.query[:50]})
    from router.classifier import handle_request

    intent, ctx = handle_request(request)
    logger.info("Query classified", extra={"intent": intent.value if intent else None})
    handle_uploads(request)
    from resolve.resolver import resolve

    target_files = resolve(request)
    logger.info("Files resolved", extra={"docs": len(target_files.documents), "images": len(target_files.images)})

    match intent:
        case Intent.IMAGE_SEARCH | Intent.IMAGE_UNDERSTANDING:
            return handle_image(intent, request, target_files)

        case (
            Intent.AUDIO_QA
            | Intent.AUDIO_SUMMARIZE
            | Intent.AUDIO_TRANSCRIBE
            | Intent.AUDIO_OVERVIEW
        ):
            return handle_audio(intent, request, target_files)

        case Intent.GENERAL_CHAT | Intent.UNKNOWN:
            return handle_general(intent, request)

        case _:
            if request.audio or request.images:
                return handle_multimodal(intent, request, target_files)
            else:
                return handle_document(intent, request, target_files)
