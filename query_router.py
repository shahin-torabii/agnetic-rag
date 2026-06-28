from enum import Enum
from dataclasses import dataclass
from data_gathering import UserRequest
from LLM import HF_LLM
from data_gathering import Data
from index_embedd import VectorStore
from typing import List, Set, Dict
from pathlib import Path


class Intent(str, Enum):

    GENERAL_CHAT = "GENERAL_CHAT"

    DOCUMENT_OVERVIEW = "DOCUMENT_OVERVIEW"

    DOCUMENT_QA = "DOCUMENT_QA"

    DOCUMENT_SECTION_EXPLAIN = "DOCUMENT_SECTION_EXPLAIN"
    DOCUMENT_SECTION_SUMMARIZE = "DOCUMENT_SECTION_SUMMARIZE"

    DOCUMENT_FULL_EXPLAIN = "DOCUMENT_FULL_EXPLAIN"
    DOCUMENT_SUMMARIZE = "DOCUMENT_SUMMARIZE"

    IMAGE_UNDERSTANDING = "IMAGE_UNDERSTANDING"
    IMAGE_SEARCH = "IMAGE_SEARCH"

    AUDIO_OVERVIEW = "AUDIO_OVERVIEW"
    AUDIO_TRANSCRIBE = "AUDIO_TRANSCRIBE"
    AUDIO_SUMMARIZE = "AUDIO_SUMMARIZE"
    AUDIO_QA = "AUDIO_QA"

    SEARCH_DOCUMENT = "SEARCH_DOCUMENT"

    COMPARE_DOCUMENTS = "COMPARE_DOCUMENTS"

    DOCUMENT_ACTION = "DOCUMENT_ACTION"

    UNKNOWN = "UNKNOWN"




@dataclass
class QueryContext:

    has_document: bool = False
    has_image: bool = False
    has_audio: bool = False

    num_documents: int = 0
    num_images: int = 0
    num_audio: int = 0

    has_stored_documents: bool = False
    has_stored_images: bool = False
    has_stored_audio: bool = False


class ActiveContext:

    has_file :bool = False

    active_documents: Set[str] = set()

    active_images: Set[str] = set()

    active_audio: Set[str] = set()

    active_files_chunks :Dict[str, dict] = {"document": {}, "audio":{}}



FA_EXPLAIN = [
    "توضیح",
    "توضیح بده",
    "شرح بده",
    "تفسیر کن",
]

FA_SUMMARIZE = [
    "خلاصه",
    "خلاصه کن",
    "جمع بندی",
    "جمع‌بندی",
]

FA_SEARCH = [
    "پیدا کن",
    "جستجو",
    "بگرد",
    "کجا",
    "کجاست",
]

FA_COMPARE = [
    "مقایسه",
    "فرق",
    "تفاوت",
]


FA_SECTION = [
    "بخش",
    "فصل",
    "قسمت",
    "مقدمه",
    "نتایج",
    "نتیجه گیری",
    "نتیجه‌گیری",
    "روش",
    "روش شناسی",
    "روش‌شناسی",
]

FA_ACTION = [
    "ترجمه",
    "یادداشت",
    "فلش کارت",
    "فلش‌کارت",
    "سوال",
    "تمرین",
    "quiz",
]


EN_OVERVIEW = [
    "what is this about",
    "what is this pdf about",
    "what is this file about",
    "overview",
    "give me an overview",
    "high level overview",
    "main idea",
    "main topic",
    "topic",
    "subject",
    "what does it discuss",
    "what does this discuss",
    "what is being discussed",
]

FA_OVERVIEW = [
    "درباره چیست",
    "در مورد چیست",
    "موضوع چیست",
    "موضوع فایل چیست",
    "موضوع این فایل چیست",
    "موضوع این سند چیست",
    "موضوع این pdf چیست",
    "این فایل درباره چیست",
    "این سند درباره چیست",
    "این مقاله درباره چیست",
    "این صوت درباره چیست",
    "مرور کلی",
    "نمای کلی",
    "دید کلی",
    "ایده اصلی",
    "موضوع اصلی",
]


EN_EXPLAIN = [
    "explain",
    "describe",
    "interpret",
]

EN_SUMMARIZE = [
    "summarize",
    "summary",
    "tldr",
]

EN_SEARCH = [
    "find",
    "search",
    "locate",
    "show me",
]

EN_COMPARE = [
    "compare",
    "difference",
]

EN_SECTION = [
    "section",
    "chapter",
    "part",
    "methodology",
    "results",
    "discussion",
    "conclusion",
    "introduction",
    "background",
    "related work",
]
EN_ACTION = [
    "translate",
    "notes",
    "flashcards",
    "quiz",
    "questions",
]


EN_AUDIO_TRANSCRIBE = [
    "transcribe",
    "speech to text",
    "convert audio to text",
]

FA_AUDIO_TRANSCRIBE = [
    "رونویسی",
    "تبدیل صوت به متن",
    "پیاده سازی صوت",
]

EN_AUDIO_SUMMARIZE = [
    "summarize audio",
    "summarize recording",
    "summarize meeting",
]

FA_AUDIO_SUMMARIZE = [
    "خلاصه فایل صوتی",
    "خلاصه جلسه",
    "خلاصه کن",
]


def contains_any(query: str, keywords: list[str]) -> bool:
    query = query.lower().strip()

    return any(
        keyword.lower().strip() in query
        for keyword in keywords
    )


def classify_query(query: str, ctx: QueryContext) -> Intent:


    q = query.lower().strip()

    has_any_documents = (
            ctx.has_document
            or ctx.has_stored_documents
    )

    has_any_images = (
            ctx.has_image
            or ctx.has_stored_images
    )

    has_any_audio = (
            ctx.has_audio
            or ctx.has_stored_audio
    )
    if (
            has_any_images
            and not has_any_documents
            and not has_any_audio
    ):
        if (
                contains_any(q, EN_SEARCH)
                or contains_any(q, FA_SEARCH)
        ):
            return Intent.IMAGE_SEARCH

        return Intent.IMAGE_UNDERSTANDING

    if (
            has_any_audio
            and not has_any_documents
            and not has_any_images
    ):

        if(
                contains_any(q, EN_AUDIO_TRANSCRIBE)
                or contains_any(q, FA_AUDIO_TRANSCRIBE)
        ):
            return Intent.AUDIO_TRANSCRIBE

        if (
                contains_any(q, EN_OVERVIEW)
                or contains_any(q, FA_OVERVIEW)
        ):
            return Intent.AUDIO_OVERVIEW

        if (
                contains_any(q, EN_AUDIO_SUMMARIZE)
                or contains_any(q, FA_AUDIO_SUMMARIZE)
        ):
            return Intent.AUDIO_SUMMARIZE

        return Intent.AUDIO_QA

    if (
            ctx.num_documents >= 2
            or len(Data.docs) >= 2
    ):

        if (
            contains_any(q, EN_COMPARE)
            or contains_any(q, FA_COMPARE)
        ):
            return Intent.COMPARE_DOCUMENTS

    if has_any_documents:

        if (
                contains_any(q, EN_OVERVIEW)
                or contains_any(q, FA_OVERVIEW)
        ):
            return Intent.DOCUMENT_OVERVIEW

        if (
                contains_any(q, EN_SEARCH)
                or contains_any(q, FA_SEARCH)
        ):
            return Intent.SEARCH_DOCUMENT

        if (
                contains_any(q, EN_ACTION)
                or contains_any(q, FA_ACTION)
        ):
            return Intent.DOCUMENT_ACTION

        if (
                contains_any(q, EN_SECTION)
                or contains_any(q, FA_SECTION)
        ):
            if (
                    contains_any(q, EN_EXPLAIN)
                    or contains_any(q, FA_EXPLAIN)
            ):
                return Intent.DOCUMENT_SECTION_EXPLAIN

            if (
                    contains_any(q, EN_SUMMARIZE)
                    or contains_any(q, FA_SUMMARIZE)
            ):
                return Intent.DOCUMENT_SECTION_SUMMARIZE

        if (
                contains_any(q, EN_EXPLAIN)
                or contains_any(q, FA_EXPLAIN)
        ):
            return Intent.DOCUMENT_FULL_EXPLAIN

        if (
                contains_any(q, EN_SUMMARIZE)
                or contains_any(q, FA_SUMMARIZE)
        ):
            return Intent.DOCUMENT_SUMMARIZE


        return Intent.DOCUMENT_QA

    return Intent.UNKNOWN


def llm_router(query: str, ctx) -> Intent:


    client = HF_LLM.client
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    system_prompt = """
You are an intent classifier for a multimodal RAG system.

Classify the request into EXACTLY ONE of the following classes:

GENERAL_CHAT

DOCUMENT_OVERVIEW
DOCUMENT_QA
DOCUMENT_SECTION_EXPLAIN
DOCUMENT_FULL_EXPLAIN
DOCUMENT_SUMMARIZE

IMAGE_UNDERSTANDING
IMAGE_SEARCH

AUDIO_OVERVIEW
AUDIO_TRANSCRIBE
AUDIO_SUMMARIZE
AUDIO_QA

SEARCH_DOCUMENT

COMPARE_DOCUMENTS

DOCUMENT_ACTION

Rules:

- Return ONLY the class name.
- No explanation.
- No markdown.
- No extra text.
-Classify ONLY the user intent.
Do not decide:
- chunking
- retrieval strategy
- summarization strategy
- document size handling
"""


    user_prompt = f"""
    Query: {query}

    Context:

    has_image: {ctx.has_image}
    has_document: {ctx.has_document}
    has_audio: {ctx.has_audio}

    has_stored_documents: {ctx.has_stored_documents}
    has_stored_images: {ctx.has_stored_images}
    has_stored_audio: {ctx.has_stored_audio}

    num_documents: {ctx.num_documents}
    num_images: {ctx.num_images}
    num_audio: {ctx.num_audio}
    """

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=20,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    query_class = response.choices[0].message.content.strip().split()[0].replace(".", "")

    try:
        intent = Intent(query_class)
    except ValueError:
        intent = Intent.GENERAL_CHAT

    return intent



def route_query(request: UserRequest):

    has_image = (
        request.images is not None
        and len(request.images) > 0
    )

    has_document = (
        request.documents is not None
        and len(request.documents) > 0
    )

    has_audio = (
        request.audio is not None
        and len(request.audio) > 0
    )

    ctx = QueryContext(
        has_image=has_image,
        has_document=has_document,
        has_audio=has_audio,

        num_images=len(request.images)
        if request.images else 0,

        num_documents=len(request.documents)
        if request.documents else 0,

        num_audio=len(request.audio)
        if request.audio else 0,

        has_stored_documents=(
                len(Data.docs) > 0
        ),

        has_stored_images=(
                VectorStore.image_index is not None
                and VectorStore.image_index.ntotal > 0
        ),

        has_stored_audio=(
            any(
                getattr(doc, "doc_type", None) == "audio"
                for doc in Data.docs.values()
            )
        )
    )

    intent = classify_query(
        request.query,
        ctx
    )

    if intent == Intent.UNKNOWN:
        intent = llm_router(
            request.query,
            ctx
        )

    return intent, ctx



def manage_active_context(request:UserRequest):

    ActiveContext.active_images = set()
    ActiveContext.active_documents = set()
    ActiveContext.active_audio = set()
    ActiveContext.has_file = False

    has_docs = (
            request.documents is not None
            and len(request.documents) > 0
    )

    has_images = (
            request.images is not None
            and len(request.images) > 0
    )

    has_audio = (
            request.audio is not None
            and len(request.audio) > 0
    )

    if ( has_docs or has_images or has_audio ):
        ActiveContext.has_file = True

        if has_audio:
            audios = [Path(audio).name for audio in request.audio]
            ActiveContext.active_audio.update(audios)

        if has_images:
            ims = [Path(image).name for image in request.images]
            ActiveContext.active_images.update(ims)

        if has_docs:
            docs = [Path(doc).name for doc in request.documents]
            ActiveContext.active_documents.update(docs)




def handle_request(request: UserRequest):
    print("enter the handle request")
    intent , ctx = route_query(request)
    print("finish routing")
    manage_active_context(request)

    return intent, ctx