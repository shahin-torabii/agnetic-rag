from enum import Enum
from dataclasses import dataclass
from data_gathering import UserRequest
from LLM import HF_LLM

class Intent(str, Enum):
    GENERAL_CHAT = "GENERAL_CHAT"

    DOCUMENT_OVERVIEW = "DOCUMENT_OVERVIEW"
    DOCUMENT_QA = "DOCUMENT_QA"
    DOCUMENT_SECTION_EXPLAIN = "DOCUMENT_SECTION_EXPLAIN"
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
    "chapter",
    "section",
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
    "methodology",
    "results",
    "conclusion",
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

    if ctx.has_image and not ctx.has_document and not ctx.has_audio:

        if (
                contains_any(q, EN_SEARCH)
                or contains_any(q, FA_SEARCH)
        ):
            return Intent.IMAGE_SEARCH

        return Intent.IMAGE_UNDERSTANDING

    if ctx.has_audio and not ctx.has_document and not ctx.has_image:

        if (
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


    if ctx.num_documents >= 2:

        if (
            contains_any(q, EN_COMPARE)
            or contains_any(q, FA_COMPARE)
        ):
            return Intent.COMPARE_DOCUMENTS

    if ctx.has_document:

        if (
                contains_any(q, EN_OVERVIEW)
                or contains_any(q, FA_OVERVIEW)
        ):
            return Intent.DOCUMENT_OVERVIEW

        if (
                contains_any(q, EN_SUMMARIZE)
                or contains_any(q, FA_SUMMARIZE)
        ):
            return Intent.DOCUMENT_SUMMARIZE

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
                contains_any(q, EN_EXPLAIN)
                or contains_any(q, FA_EXPLAIN)
        ):
            return Intent.DOCUMENT_FULL_EXPLAIN

        return Intent.DOCUMENT_QA

    return Intent.UNKNOWN


def llm_router(query: str, ctx) -> Intent:


    client = HF_LLM.client
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    system_prompt = """
You are an intent classifier for a multimodal RAG system.

Classify the request into EXACTLY ONE of the following classes:

GENERAL_CHAT

DOCUMENT_QA
DOCUMENT_SECTION_EXPLAIN
DOCUMENT_FULL_EXPLAIN
DOCUMENT_SUMMARIZE

IMAGE_UNDERSTANDING
IMAGE_SEARCH

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
num_documents: {ctx.num_documents}
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