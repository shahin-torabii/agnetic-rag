from sqlalchemy.orm import Session

from database.models import ChatSession, Message, SessionDocument


def create_session(db: Session, user_id: str, title: str = "New chat") -> ChatSession:
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_by_id(db: Session, session_id: str, user_id: str) -> ChatSession | None:
    return (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )


def list_sessions(db: Session, user_id: str):
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )


def delete_session(db: Session, session_id: str, user_id: str):
    session = get_session_by_id(db, session_id, user_id)
    if session is None:
        return None
    db.query(SessionDocument).filter(SessionDocument.session_id == session_id).delete()
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.delete(session)
    db.commit()
    return session


def create_message(db: Session, session_id: str, role: str, content: str) -> Message:
    msg = Message(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_messages_by_session(db: Session, session_id: str):
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at)
        .all()
    )


def get_recent_messages(db: Session, session_id: str, limit: int = 8):
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(messages))
