import uuid
from datetime import datetime, UTC
from sqlalchemy import Column, String, Text, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from database.engine import Base


def gen_id():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


class UserDocument(Base):
    __tablename__ = "user_documents"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    doc_id = Column(String, nullable=False, index=True,)
    filename = Column(String, nullable=False)
    title = Column(String, default="")
    path = Column(String, nullable=True)
    doc_type = Column(String, default="GENERAL")
    kind = Column(String, default="document")
    uploaded_in_session = Column(String, ForeignKey("chat_sessions.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


class SessionDocument(Base):
    __tablename__ = "session_documents"

    id = Column(String, primary_key=True, default=gen_id)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    doc_id = Column(String, nullable=False, index=True)
    kind = Column(String, default="document")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    __table_args__ = (UniqueConstraint("session_id", "doc_id", name="uq_session_doc"),)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, default="New chat")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=gen_id)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)     ###ai/human
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
