import re
from typing import List

from sqlalchemy.orm import Session

from analysis.handlers import rewrite_query_with_history
from core.constants import REFERENTIAL_PATTERN
from core.types import UserRequest
from graph.builder import run_agent
from repositories.message import create_message, get_recent_messages


def need_history_context(query: str, history_exists: bool) -> bool:
    if not history_exists:
        return False
    return len(query.strip().split()) <= 8 and bool(
        re.search(pattern=REFERENTIAL_PATTERN, string=query, flags=re.IGNORECASE)
    )


def load_recent_history(db: Session, session_id: str, limit: int = 8) -> List:
    return get_recent_messages(db, session_id, limit)


def save_message(db: Session, session_id: str, role: str, content: str):
    create_message(db, session_id, role, content)


def chat(
    db: Session,
    user_id: str,
    session_id: str,
    query: str,
    documents=None,
    images=None,
    audio=None,
) -> str:
    history = load_recent_history(db, session_id)
    save_message(db, session_id, role="User", content=query)

    effective_query = query
    if need_history_context(query, len(history) > 0):
        context = "\n\n".join(f"{m.role}: {m.content}" for m in history)
        effective_query = rewrite_query_with_history(query, context)

    request = UserRequest(
        query=effective_query,
        documents=documents or [],
        images=images or [],
        audio=audio or [],
    )

    result = run_agent(request, user_id=user_id, session_id=session_id, db=db)

    answer = result if isinstance(result, str) else str(result)
    save_message(db, session_id, role="Assistant", content=answer)
    return answer
