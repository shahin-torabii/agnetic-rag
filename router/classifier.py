from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm.prompts import ROUTER_SYSTEM_PROMPT
from core.logger import get_logger
from core.constants import (
    EN_ACTION,
    EN_AUDIO_SUMMARIZE,
    EN_AUDIO_TRANSCRIBE,
    EN_COMPARE,
    EN_EXPLAIN,
    EN_OVERVIEW,
    EN_SEARCH,
    EN_SECTION,
    EN_SUMMARIZE,
    FA_ACTION,
    FA_AUDIO_SUMMARIZE,
    FA_AUDIO_TRANSCRIBE,
    FA_COMPARE,
    FA_EXPLAIN,
    FA_OVERVIEW,
    FA_SEARCH,
    FA_SECTION,
    FA_SUMMARIZE,
)
from core.types import (
    ActiveContext,
    Data,
    Intent,
    QueryContext,
    SessionContext,
    UserRequest,
)
from embedding.models import VectorStore
from llm.client import HF_LLM

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_router_chain():
    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", "Query: {query}\n\nContext:\n{context}"),
        ]
    )

    router_chain = (
        router_prompt
        | HF_LLM.fast_llm.bind(max_tokens=20, temperature=0)
        | StrOutputParser()
    )

    return router_chain


def contains_any(query: str, keywords: list[str]) -> bool:
    query = query.lower().strip()

    return any(keyword.lower().strip() in query for keyword in keywords)


def classify_query(query: str, ctx: QueryContext) -> Intent:
    q = query.lower().strip()

    has_any_documents = ctx.has_document or ctx.has_stored_documents

    has_any_images = ctx.has_image or ctx.has_stored_images

    has_any_audio = ctx.has_audio or ctx.has_stored_audio
    if has_any_images and not has_any_documents and not has_any_audio:
        if contains_any(q, EN_SEARCH) or contains_any(q, FA_SEARCH):
            return Intent.IMAGE_SEARCH

        return Intent.IMAGE_UNDERSTANDING

    if has_any_audio and not has_any_documents and not has_any_images:
        if contains_any(q, EN_AUDIO_TRANSCRIBE) or contains_any(q, FA_AUDIO_TRANSCRIBE):
            return Intent.AUDIO_TRANSCRIBE

        if contains_any(q, EN_OVERVIEW) or contains_any(q, FA_OVERVIEW):
            return Intent.AUDIO_OVERVIEW

        if contains_any(q, EN_AUDIO_SUMMARIZE) or contains_any(q, FA_AUDIO_SUMMARIZE):
            return Intent.AUDIO_SUMMARIZE

        return Intent.AUDIO_QA

    if ctx.num_documents >= 2:
        if contains_any(q, EN_COMPARE) or contains_any(q, FA_COMPARE):
            return Intent.COMPARE_DOCUMENTS

    if has_any_documents:
        if contains_any(q, EN_OVERVIEW) or contains_any(q, FA_OVERVIEW):
            return Intent.DOCUMENT_OVERVIEW

        if contains_any(q, EN_SEARCH) or contains_any(q, FA_SEARCH):
            return Intent.SEARCH_DOCUMENT

        if contains_any(q, EN_ACTION) or contains_any(q, FA_ACTION):
            return Intent.DOCUMENT_ACTION

        if contains_any(q, EN_SECTION) or contains_any(q, FA_SECTION):
            if contains_any(q, EN_EXPLAIN) or contains_any(q, FA_EXPLAIN):
                return Intent.DOCUMENT_SECTION_EXPLAIN

            if contains_any(q, EN_SUMMARIZE) or contains_any(q, FA_SUMMARIZE):
                return Intent.DOCUMENT_SECTION_SUMMARIZE

        if contains_any(q, EN_EXPLAIN) or contains_any(q, FA_EXPLAIN):
            return Intent.DOCUMENT_FULL_EXPLAIN

        if contains_any(q, EN_SUMMARIZE) or contains_any(q, FA_SUMMARIZE):
            return Intent.DOCUMENT_SUMMARIZE

        return Intent.DOCUMENT_QA

    return Intent.UNKNOWN


def llm_router(query: str, ctx) -> Intent:
    context_text = f"""has_image: {ctx.has_image}
has_document: {ctx.has_document}
has_audio: {ctx.has_audio}
has_stored_documents: {ctx.has_stored_documents}
has_stored_images: {ctx.has_stored_images}
has_stored_audio: {ctx.has_stored_audio}
num_documents: {ctx.num_documents}
num_images: {ctx.num_images}
num_audio: {ctx.num_audio}"""

    raw = get_router_chain().invoke({"query": query, "context": context_text})
    query_class = raw.strip().split()[0].replace(".", "")

    try:
        return Intent(query_class)
    except ValueError:
        return Intent.GENERAL_CHAT


def route_query(request: UserRequest):
    has_image = request.images is not None and len(request.images) > 0

    has_document = request.documents is not None and len(request.documents) > 0

    has_audio = request.audio is not None and len(request.audio) > 0

    ctx = QueryContext(
        has_image=has_image,
        has_document=has_document,
        has_audio=has_audio,
        num_images=len(request.images) if request.images else 0,
        num_documents=len(request.documents) if request.documents else 0,
        num_audio=len(request.audio) if request.audio else 0,
        has_stored_documents=(len(Data.docs) > 0),
        has_stored_images=(
            VectorStore.image_index is not None and VectorStore.image_index.ntotal > 0
        ),
        has_stored_audio=(
            any(getattr(doc, "doc_type", None) == "audio" for doc in Data.docs.values())
        ),
    )

    intent = classify_query(request.query, ctx)

    if intent == Intent.UNKNOWN:
        intent = llm_router(request.query, ctx)

    return intent, ctx


def make_doc_id(user_id: str, filename_or_path: str) -> str:
    from pathlib import Path

    return f"{user_id}:{Path(filename_or_path).name}"


def build_active_context(request: UserRequest, user_id: str) -> ActiveContext:
    has_docs = bool(request.documents)
    has_images = bool(request.images)
    has_audio = bool(request.audio)

    ctx = ActiveContext(has_file=has_docs or has_images or has_audio)
    if has_audio:
        ctx.active_audio = {make_doc_id(user_id, a) for a in request.audio}
    if has_images:
        ctx.active_images = {make_doc_id(user_id, i) for i in request.images}
    if has_docs:
        ctx.active_documents = {make_doc_id(user_id, d) for d in request.documents}
    return ctx


def build_session_context(session_id: str, user_id: str, db) -> SessionContext:
    from database.models import SessionDocument

    rows = (
        db.query(SessionDocument)
        .filter(
            SessionDocument.session_id == session_id, SessionDocument.user_id == user_id
        )
        .all()
    )
    ctx = SessionContext()
    for r in rows:
        if r.kind == "document":
            ctx.session_documents.add(r.doc_id)
        elif r.kind == "image":
            ctx.session_images.add(r.doc_id)
        elif r.kind == "audio":
            ctx.session_audio.add(r.doc_id)
    return ctx


def handle_request(request: UserRequest, user_id: str):
    logger.info("Classifying request", extra={"query": request.query[:50]})
    intent, ctx = route_query(request)
    logger.info("Routing complete", extra={"intent": intent.value if intent else None})

    return intent, ctx
