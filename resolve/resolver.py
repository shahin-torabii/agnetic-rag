from functools import lru_cache
from typing import List
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rapidfuzz import fuzz
from pathlib import Path

from llm.client import HF_LLM
from core.types import Data, UserRequest, ActiveContext, SessionContext, ResolvedTargets
from llm.prompts import REFERENCE_RESOLVER_SYSTEM_PROMPT
from core.constants import CURRENT_FILE_SIGNALS, ALL_FILES_SIGNALS, ORDINAL_SIGNALS, LAST_SIGNALS
from sqlalchemy.orm import Session


def user_doc_ids(user_id: str, db_session) -> list[str]:
    from database.models import UserDocument
    rows = db_session.query(UserDocument.doc_id).filter(UserDocument.user_id == user_id).all()
    return [r[0] for r in rows]


def normalize_name(name: str) -> str:
    name = Path(name).name.lower()

    if "." in name:
        name = ".".join(name.split(".")[:-1])

    return name.strip()


def resolve_ordinals(query: str, active_ctx, session_ctx) -> list[str]:
    q = query.lower()
    active_docs = list(active_ctx.active_documents | active_ctx.active_audio)
    if not active_docs:
        active_docs = list(session_ctx.session_documents | session_ctx.session_audio)
    matched = []
    for signal, idx in ORDINAL_SIGNALS.items():
        if signal in q and idx < len(active_docs):
            matched.append(active_docs[idx])
    if any(signal in q for signal in LAST_SIGNALS) and active_docs:
        matched.append(active_docs[-1])
    return list(set(matched))


def extract_document_mentions(query: str, user_id : str, db: Session ,threshold: int = 85) -> list[str]:
    q = query.lower()
    matches = []

    for doc_id in user_doc_ids(user_id, db):
        meta = Data.docs.get(doc_id)

        if meta is None:
            continue
        title = normalize_name(meta.title)
        if title in q:
            matches.append(doc_id)
            continue
        if fuzz.partial_ratio(title, q) >= threshold:
            matches.append(doc_id)
    return list(set(matches))


def contains_signal(query: str, signals: set[str]) -> bool:
    q = query.lower()

    return any(signal.lower() in q for signal in signals)


def resolve_targets(query: str, active_ctx: ActiveContext, session_ctx: SessionContext, user_id: str, db_session) -> ResolvedTargets:
    result = ResolvedTargets()
    q = query.lower()

    explicit_docs = extract_document_mentions(query, user_id, db_session)
    if explicit_docs:
        result.documents.extend(explicit_docs)

    if active_ctx.active_documents or active_ctx.active_images or active_ctx.active_audio:
        result.documents.extend(active_ctx.active_documents)
        result.documents.extend(active_ctx.active_audio)
        result.images.extend(active_ctx.active_images)
    else:
        result.documents.extend(session_ctx.session_documents)
        result.documents.extend(session_ctx.session_audio)
        result.images.extend(session_ctx.session_images)

    if contains_signal(q, ALL_FILES_SIGNALS):
        result.documents.extend(user_doc_ids(user_id, db_session))  # user scope: every chat

    result.documents = list(set(result.documents))
    result.images = list(set(result.images))
    result.confidence = 0.9 if (result.documents or result.images) else 0.3
    return result


def build_candidate_context(user_id: str, db: Session):
    from database.models import UserDocument
    rows = db.query(UserDocument).filter(UserDocument.user_id == user_id).all()
    docs = [{"doc_id": r.doc_id, "title": r.title, "doc_type": r.doc_type} for r in rows]
    return docs


@lru_cache(maxsize=1)
def get_resolver_chain():
    resolver_prompt = ChatPromptTemplate.from_messages([
        ("system", REFERENCE_RESOLVER_SYSTEM_PROMPT),
        ("human", "{user_prompt}"),
    ])

    resolver_chain = resolver_prompt | HF_LLM.strong_llm.bind(temperature=0) | JsonOutputParser()
    return resolver_chain


import json

def llm_reference_resolver(query: str, active_ctx: ActiveContext, user_id: str, db: Session) -> ResolvedTargets:
    candidates = build_candidate_context(user_id, db)

    current_uploads = {
        "documents": list(active_ctx.active_documents | active_ctx.active_audio),
        "images": list(active_ctx.active_images),
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
        parsed = get_resolver_chain().invoke({"user_prompt": user_prompt})
        return ResolvedTargets(
            documents=parsed.get("documents", []),
            images=parsed.get("images", []),
            confidence=parsed.get("confidence", 0.0),
        )
    except Exception:
        return ResolvedTargets(confidence=0.0)



def resolve(request: UserRequest, active_ctx, session_ctx,user_id: str, db: Session) -> ResolvedTargets:
    query = request.query
    resolved = resolve_targets(query, active_ctx, session_ctx,user_id, db)

    has_scope = bool(active_ctx.active_documents or session_ctx.session_documents)
    if resolved.confidence < 0.6 and has_scope:
        ordinal_docs = resolve_ordinals(query, active_ctx, session_ctx)
        if ordinal_docs:
            resolved.documents.extend(ordinal_docs)
            resolved.documents = list(set(resolved.documents))
            resolved.confidence = 0.95

    if resolved.confidence < 0.6:
        resolved = llm_reference_resolver(query, active_ctx, user_id, db)

    return resolved
