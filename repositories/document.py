import os
import shutil
from typing import Optional

from sqlalchemy.orm import Session

from core.constants import UPLOAD_DIR
from database.models import SessionDocument, UserDocument


def create_user_document(
    db: Session,
    user_id: str,
    doc_id: str,
    filename: str,
    title: str = "",
    path: str = "",
    doc_type: str = "GENERAL",
    kind: str = "document",
    uploaded_in_session: Optional[str] = None,
) -> UserDocument:
    existing = db.query(UserDocument).filter(UserDocument.doc_id == doc_id).first()
    if existing:
        return existing
    row = UserDocument(
        user_id=user_id,
        doc_id=doc_id,
        filename=filename,
        title=title,
        path=path,
        doc_type=doc_type,
        kind=kind,
        uploaded_in_session=uploaded_in_session,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_user_documents(db: Session, user_id: str):
    return db.query(UserDocument).filter(UserDocument.user_id == user_id).all()


def create_session_document(
    db: Session, session_id: str, user_id: str, doc_id: str, kind: str = "document"
) -> SessionDocument:
    existing = (
        db.query(SessionDocument)
        .filter(
            SessionDocument.session_id == session_id,
            SessionDocument.doc_id == doc_id,
        )
        .first()
    )
    if existing:
        return existing
    row = SessionDocument(
        session_id=session_id,
        user_id=user_id,
        doc_id=doc_id,
        kind=kind,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_session_documents(db: Session, session_id: str):
    db.query(SessionDocument).filter(SessionDocument.session_id == session_id).delete()
    db.commit()


def save_uploaded_file(user_id: str, filename: str, file) -> str:
    user_upload_dir = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    dest = os.path.join(user_upload_dir, filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest
