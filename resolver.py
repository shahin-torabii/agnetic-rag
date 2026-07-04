from LLM import HF_LLM
from query_router import ActiveContext
from dataclasses import dataclass, field
from typing import List
from data_gathering import Data, UserRequest
import json
from rapidfuzz import fuzz
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


REFERENCE_RESOLVER_SYSTEM_PROMPT = """You are a file reference resolver.

Your task:
Determine which files the user is referring to.

The user may refer to:
- currently uploaded files
- previously uploaded files
- both

You may ONLY return files that exist in the provided candidate lists.

Return ONLY valid JSON.

Schema:
{{
  "documents": [],
  "images": [],
  "confidence": 0.0
}}"""


resolver_prompt = ChatPromptTemplate.from_messages([
    ("system", REFERENCE_RESOLVER_SYSTEM_PROMPT),
    ("human", "{user_prompt}"),
])

resolver_chain = resolver_prompt | HF_LLM.strong_llm.bind(temperature=0) | JsonOutputParser()


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


def contains_signal(query: str, signals: set[str]) -> bool:

    q = query.lower()

    return any(signal.lower() in q
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


def llm_reference_resolver(query: str) -> ResolvedTargets:
    candidates = build_candidate_context()

    current_uploads = {
        "documents": list(
            ActiveContext.active_documents | ActiveContext.active_audio
        ),
        "images": list(ActiveContext.active_images),
    }

    user_prompt = f"""User Query:
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

Return JSON only."""

    try:
        parsed = resolver_chain.invoke({"user_prompt": user_prompt})
        return ResolvedTargets(
            documents=parsed.get("documents", []),
            images=parsed.get("images", []),
            confidence=parsed.get("confidence", 0.0),
        )
    except Exception:
        return ResolvedTargets(confidence=0.0)

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

