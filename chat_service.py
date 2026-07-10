import re
from typing import List

from sqlalchemy.orm import Session
from models import Message
from handlers import rewrite_query_with_history
from agent_graph import run_agent
from data_gathering import UserRequest


REFERENTIAL_PATTERN = r"\b(it|its|that|this|them|those|again|the (first|second|third|last|previous|other) one)\b"


def need_history_context(query: str, history_exists: bool) -> bool:

    if not history_exists:
        return False

    is_need = len(query.strip().split()) <= 8 and bool(re.search(pattern=REFERENTIAL_PATTERN, string=query, flags=re.IGNORECASE))
    return  is_need


def load_recent_history(db: Session, session_id: str, limit: int = 8)-> List[Message]:

    messages = (db.query(Message).filter(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(limit))

    return list(reversed(messages))

def save_message(db: Session, session_id: str, role: str, content: str):

    db.add(Message(session_id = session_id , role = role, content = content))
    db.commit()


def chat(db: Session, user_id: str, session_id: str, query: str, documents=None, images=None, audio=None) -> str:
    history = load_recent_history(db, session_id)
    save_message(db, session_id, role="User", content=query)

    effective_query = query
    if need_history_context(query, len(history)> 0):
        context = "\n\n".join(f"{m.role}: {m.content}" for m in history)
        effective_query = rewrite_query_with_history(query, context)

    request = UserRequest(
        query= effective_query,
        documents= documents or [],
        images= images or [],
        audio = audio or []
    )

    result = run_agent(request, user_id= user_id, session_id=session_id, db=db)

    answer = result if isinstance(result, str) else str(result)
    save_message(db, session_id, role="Assistant", content=answer)
    return answer