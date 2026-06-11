from openai import OpenAI
from enum import Enum
import re
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

class Intent(str, Enum):
    GENERAL_CHAT = "GENERAL_CHAT"

    DOCUMENT_QA ="DOCUMENT_QA"
    DOCUMENT_SECTION_EXPLAIN = "DOCUMENT_SECTION_EXPLAIN"
    DOCUMENT_FULL_EXPLAIN = "DOCUMENT_FULL_EXPLAIN"
    DOCUMENT_SUMMARIZE = "DOCUMENT_SUMMARIZE"

    IMAGE_EXPLAIN = "IMAGE_EXPLAIN"
    IMAGE_QA = "IMAGE_QA"

    SEARCH_DOCUMENT = "SEARCH_DOCUMENT"

    COMPARE_DOCUMENTS = "COMPARE_DOCUMENTS"

    DOCUMENT_ACTION = "DOCUMENT_ACTION"

    UNKNOWN = "UNKNOWN"


@dataclass
class QueryContext:
    has_document: bool = False
    has_image: bool = False
    num_documents: int = 0
    num_pages: Optional[int] = None


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

def contains_any(query:str, keywords:List[str]) -> bool:
    query = query.lower().strip()

    contains = any(k.lower().strip() in query for k in keywords)
    return  contains

def classify_query(query: str, ctx: QueryContext) -> Intent:
    q = query.lower().strip()


    if ctx.has_image and not ctx.has_document:

        if (
            contains_any(q, EN_EXPLAIN)
            or contains_any(q, FA_EXPLAIN)
        ):
            return Intent.IMAGE_EXPLAIN

        return Intent.IMAGE_QA


    if ctx.num_documents >= 2:
        if (
            contains_any(q, EN_COMPARE)
            or contains_any(q, FA_COMPARE)
        ):
            return Intent.COMPARE_DOCUMENTS

    if ctx.has_document:

        # summarize document
        if (
            contains_any(q, EN_SUMMARIZE)
            or contains_any(q, FA_SUMMARIZE)
        ):
            return Intent.DOCUMENT_SUMMARIZE

        # explain whole document
        if (
            contains_any(q, EN_EXPLAIN)
            or contains_any(q, FA_EXPLAIN)
        ):
            if not (
                contains_any(q, EN_SECTION)
                or contains_any(q, FA_SECTION)
            ):
                return Intent.DOCUMENT_FULL_EXPLAIN


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
            contains_any(q, EN_SEARCH)
            or contains_any(q, FA_SEARCH)
        ):
            return Intent.SEARCH_DOCUMENT


        if (
            contains_any(q, EN_ACTION)
            or contains_any(q, FA_ACTION)
        ):
            return Intent.DOCUMENT_ACTION

        return Intent.DOCUMENT_QA


    return Intent.GENERAL_CHAT

from openai import OpenAI


VALID_CLASSES = {
    "GENERAL_CHAT",
    "DOCUMENT_QA",
    "DOCUMENT_SECTION_EXPLAIN",
    "DOCUMENT_FULL_EXPLAIN",
    "DOCUMENT_SUMMARIZE",
    "IMAGE_EXPLAIN",
    "IMAGE_QA",
    "SEARCH_DOCUMENT",
    "COMPARE_DOCUMENTS",
    "DOCUMENT_ACTION",
    "HIERARCHICAL_SUMMARIZATION",
}


def llm_router(query: str, ctx, hf_api_key: str) -> str:

    client = OpenAI(
        api_key=hf_api_key,
        base_url="https://router.huggingface.co/v1",
    )

    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    system_prompt = """
You are an intent classifier for a multimodal RAG system.

Classify the request into EXACTLY ONE of the following classes:

GENERAL_CHAT

DOCUMENT_QA
DOCUMENT_SECTION_EXPLAIN
DOCUMENT_FULL_EXPLAIN
DOCUMENT_SUMMARIZE

IMAGE_EXPLAIN
IMAGE_QA

SEARCH_DOCUMENT

COMPARE_DOCUMENTS

DOCUMENT_ACTION

HIERARCHICAL_SUMMARIZATION

Rules:

- Return ONLY the class name.
- No explanation.
- No markdown.
- No extra text.

HIERARCHICAL_SUMMARIZATION should be selected when:
- the user asks to explain or summarize an entire large document
- document pages > 25

DOCUMENT_FULL_EXPLAIN should be selected when:
- the user asks to explain the whole document
- document pages <= 25
"""

    user_prompt = f"""
Query: {query}

Context:
has_image: {ctx.has_image}
has_document: {ctx.has_document}
num_documents: {ctx.num_documents}
num_pages: {ctx.num_pages}
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

    query_class = response.choices[0].message.content.strip()

    if query_class not in VALID_CLASSES:
        return "GENERAL_CHAT"

    return query_class


def route_query(query:str, ctx : QueryContext):
    intent = classify_query(query, ctx)

    if intent == Intent.UNKNOWN:
        intent = llm_router(query, ctx)