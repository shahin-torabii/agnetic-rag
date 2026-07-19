from query_router import Intent, handle_request, ActiveContext
from data_gathering import UserRequest, ingest
from index_embedd import index_images, VectorStore, get_image_id, index_chunks, index_chunk_images, save
from pathlib import Path
from chunking_handlers.audio_hanlder import process_audio
from chunking_handlers.non_digital_pdf_handler import process_scanned_pdf
from chunking_handlers.digital_pdf_handler import process_pdf
from chunking_handlers.word_handler import process_docx
from chunking_handlers.pp_handler import process_pptx
from chunking_handlers.excel_handler import process_excel
import puremagic
from handlers import *
from resolver import resolve
from typing import List
from models import UserDocument, SessionDocument
from handlers import make_doc_id

AUDIO_TRANSCRIPT = None
PERSIST_DIR = "storage"

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


def ingest_image(image_paths: List[str], user_id:str):
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
        model=HF_LLM.strong_model_name,   # or strong model
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
"""
            }
        ]
    )

    return response.choices[0].message.content


def build_context(chunks):
    return "\n\n".join([
        f"[{c.doc_id}:{c.chunk_index}]\n{c.text}"
        for c in chunks
    ])


def handle_image(intent, request, target_files, k = 5):
    ##TODO image path are not necesaarily from request
    image_paths = [image for image in target_files.images] if target_files.images else []

    match intent:
        case Intent.IMAGE_SEARCH:
            if len(request.images) < 4:
                response = send_images_to_vlm(image_paths, request.query)
                return response
            else:
                context = retrieval(request.query, k=k, is_doc=False,  allowed_doc_ids=set(target_files.images))
                final_answer = final_rag_llm(request.query, context)
                return final_answer

        case Intent.IMAGE_UNDERSTANDING:
            response = send_images_to_vlm(image_paths, request.query)
            return response

        case _:
            raise ValueError(
                f"Unhandled image intent: {intent}"
            )


def handle_audio(intent, request, target_files, k = 10):
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
            return audio_transcript

        case Intent.AUDIO_QA:
            context = retrieval(query=request.query, k=k, is_doc=True,  allowed_doc_ids=set(target_files.documents))
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


def handle_document(intent, request, target_files, k = 10):
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
            print("\n\n".join(summaries))
            print("\n\n\n")
            print("-" * 50)
            print("final summary")
            print(summaries)
            return "\n\n".join(summaries)

        case Intent.DOCUMENT_SECTION_SUMMARIZE:
            summaries = []
            for doc_id in target_files.documents:
                summary = summarize_chunks(related_chunks[doc_id], related_docs[doc_id], full_summary=False, query=request.query)
                summaries.append(summary)

            return "\n\n".join(summaries)

        case Intent.DOCUMENT_QA | Intent.SEARCH_DOCUMENT:
            context = retrieval(query=request.query, k=k, is_doc=True,  allowed_doc_ids=set(target_files.documents))
            print("\n\n")
            print(context)
            final_answer = final_rag_llm(request.query, context)
            return final_answer

        case Intent.DOCUMENT_OVERVIEW:
            over_views = []
            for doc_id in target_files.documents:
                overview = overview_func(related_chunks[doc_id], related_docs[doc_id])
                over_views.append(overview)
            print("\n\n\n")
            print("-" * 50)
            print("final explanation")
            print("\n\n".join(over_views))
            return "\n\n".join(over_views)

        case Intent.DOCUMENT_FULL_EXPLAIN:
            explanations = []
            for doc_id in target_files.documents:
                explanation = explain_doc(related_chunks[doc_id], related_docs[doc_id], full_explanation=True)
                explanations.append(explanation)
            print("\n\n\n")
            print("-"*50)
            print("final explanation")
            print("\n\n".join(explanations))
            return "\n\n".join(explanations)

        case Intent.DOCUMENT_SECTION_EXPLAIN:
            explanations = []
            for doc_id in target_files.documents:
                explanation = explain_doc(related_chunks[doc_id], related_docs[doc_id], full_explanation=False, query=request.query)
                explanations.append(explanation)
            print("\n\n".join(explanations))
            return "\n\n".join(explanations)

        case Intent.COMPARE_DOCUMENTS:
            compare_result = compare_documents(target_files.documents)
            return compare_result

        case Intent.DOCUMENT_ACTION:
            document_actions()

        case _:
            raise ValueError(
                f"Unhandled intent: {intent}"
            )


def handle_multimodal(intent, request, target_files, k = 10):

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

        if (meta is not None and getattr(meta, "doc_type", "").lower() == "audio"):
            continue

        related_doc_chunks[doc_id] = ( Data.doc_to_chunks.get(doc_id, []))

        if doc_id in Data.docs:
            related_doc_docs[doc_id] = (Data.docs[doc_id] )

    match intent:

        case Intent.DOCUMENT_SUMMARIZE:
            summaries = []
            for doc_id in related_doc_chunks:

                summary = summarize_chunks(related_doc_chunks[doc_id], related_doc_docs[doc_id], full_summary=True)
                summaries.append(summary)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_SUMMARIZE, request, audios)

            result = {
                "docs": summaries,
                "images": image_res,
                "audio": audio_res
            }
            return result

        case Intent.DOCUMENT_SECTION_SUMMARIZE:

            summaries = []

            for doc_id in related_doc_chunks:

                summary = summarize_chunks(related_doc_chunks[doc_id], related_doc_docs[doc_id], full_summary=False,
                                           query=request.query)
                summaries.append(summary)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING,request,images)

            audio_res = handle_audio(Intent.AUDIO_SUMMARIZE, request, audios)

            result = {
                "docs": "\n\n".join(summaries),
                "images": image_res,
                "audio": audio_res
            }
            return result

        case Intent.DOCUMENT_QA | Intent.SEARCH_DOCUMENT:

            doc_context = retrieval(query=request.query,k=k, is_doc=True,  allowed_doc_ids=set(target_files.documents))
            final_doc_answer = final_rag_llm(request.query, doc_context)

            final_im_answer = handle_image(Intent.IMAGE_SEARCH, request, images, k)

            final_audio_answer = handle_audio(Intent.AUDIO_QA, request, audios, k)

            results ={
                "docs": final_doc_answer,
                "images": final_im_answer,
                "audio": final_audio_answer
            }
            return results

        case Intent.DOCUMENT_OVERVIEW:

            overviews = []

            for doc_id in related_doc_chunks:

                overview = overview_func(related_doc_chunks[doc_id], related_doc_docs[doc_id])

                overviews.append(overview)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_OVERVIEW, request, audios)

            results =  {
                "docs": "\n\n".join(overviews),
                "images": image_res,
                "audio": audio_res
            }
            return results

        case Intent.DOCUMENT_FULL_EXPLAIN:

            explanations = []

            for doc_id in related_doc_chunks:

                explanation = explain_doc(related_doc_chunks[doc_id], related_doc_docs[doc_id],
                    full_explanation=True
                )

                explanations.append(explanation)

            image_res = handle_image( Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_OVERVIEW, request, audios)

            result =  {
                "docs": "\n\n".join(explanations),
                "images": image_res,
                "audio": audio_res
            }
            return result

        case Intent.DOCUMENT_SECTION_EXPLAIN:

            explanations = []

            for doc_id in related_doc_chunks:

                explanation = explain_doc(related_doc_chunks[doc_id], related_doc_docs[doc_id],
                    full_explanation=False,
                    query=request.query
                )

                explanations.append(explanation)

            image_res = handle_image(Intent.IMAGE_UNDERSTANDING, request, images)

            audio_res = handle_audio(Intent.AUDIO_SUMMARIZE, request, audios)

            result =  {
                "docs": "\n\n".join(explanation),
                "images": image_res,
                "audio": audio_res
            }
            return result

        case Intent.COMPARE_DOCUMENTS:

            return compare_documents(
                list(related_doc_chunks.keys())
            )

        case Intent.DOCUMENT_ACTION:

            return document_actions()

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


def _record_user_document(db, user_id, doc_id, filename, kind, meta=None):
    existing = db.query(UserDocument).filter(UserDocument.doc_id == doc_id).first()
    if existing:
        return
    user_doc = UserDocument(
        user_id=user_id, doc_id=doc_id, filename=filename, kind=kind,
        title=getattr(meta , "title", " ") if meta else " ",
        doc_type=getattr(meta, "doc_type", "GENERAL") if meta else "GENERAL",
    )
    db.add(user_doc)
    db.commit()
    db.refresh(user_doc)


def _record_session_document(db_session, session_id, user_id, doc_id, kind):
    existing = db_session.query(SessionDocument).filter(
        SessionDocument.session_id == session_id, SessionDocument.doc_id == doc_id
    ).first()
    if existing:
        return
    db_session.add(SessionDocument(session_id=session_id, user_id=user_id, doc_id=doc_id, kind=kind))
    db_session.commit()


def handle_uploads(request: UserRequest, active_ctx, session_id: str, user_id: str = "1", db_session=None):
    print(f"[handle_uploads] has_file={active_ctx.has_file} docs={request.documents} imgs={request.images}")

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
                _record_user_document(db_session, user_id, doc_id, Path(doc_path).name, "document", Data.docs.get(doc_id))
                _record_session_document(db_session, session_id, user_id, doc_id, "document")

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
                _record_user_document(db_session, user_id, doc_id, Path(audio_path).name, "audio", Data.docs.get(doc_id))
                _record_session_document(db_session, session_id, user_id, doc_id, "audio")
        if transcripts:
            AUDIO_TRANSCRIPT = transcripts

    if request.images:
        ingest_image(list(request.images), user_id=user_id)
        if db_session is not None:
            for img_path in request.images:
                doc_id = make_doc_id(user_id, img_path)
                _record_user_document(db_session, user_id, doc_id, Path(img_path).name, "image")
                _record_session_document(db_session, session_id, user_id, doc_id, "image")

    save(PERSIST_DIR)


def handle_query(request: UserRequest):
    print("enter the handler")
    intent, ctx = handle_request(request)
    print("intent done")

    print("intent:",intent)
    handle_uploads(request)
    print("uploade done")
    target_files = resolve(request)
    print("targets: ", target_files)
    print("resolve done")

    print("docs:", target_files)

    match intent:

        case (Intent.IMAGE_SEARCH | Intent.IMAGE_UNDERSTANDING):
            return handle_image(intent, request, target_files)

        case (Intent.AUDIO_QA | Intent.AUDIO_SUMMARIZE | Intent.AUDIO_TRANSCRIBE | Intent.AUDIO_OVERVIEW):
            return handle_audio(intent, request, target_files)

        case (Intent.GENERAL_CHAT | Intent.UNKNOWN ):
            return handle_general(intent, request)

        case _:
            if request.audio or request.images:
                return handle_multimodal(intent, request, target_files)
            else:
                return handle_document(intent, request, target_files)
