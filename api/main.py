from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from config.manager import get_config
from core.logger import get_logger
from core.schemas import (
    ChatRequest,
    ChatResponse,
    MessageOut,
    RegisterRequest,
    TokenResponse,
)
from database.engine import Base, engine, get_db
from embedding.models import initialize, load, save
from llm.client import HF_LLM, initialize_hf_llm
from repositories.document import list_user_documents, save_uploaded_file
from repositories.message import (
    create_session,
    delete_session,
    get_messages_by_session,
    get_session_by_id,
    list_sessions,
)
from repositories.user import authenticate_user, create_user, get_user_by_username
from services.auth import creat_access_token, get_current_user
from services.chat import chat

logger = get_logger(__name__)
config = get_config()

Base.metadata.create_all(bind=engine)


def _migrate():
    with engine.connect() as conn:
        cols = [
            row[1] for row in conn.execute(text("PRAGMA table_info(user_documents)"))
        ]
        if "path" not in cols:
            conn.execute(text("ALTER TABLE user_documents ADD COLUMN path VARCHAR"))
        if "title" not in cols:
            conn.execute(
                text("ALTER TABLE user_documents ADD COLUMN title VARCHAR DEFAULT ''")
            )
        if "doc_type" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE user_documents ADD COLUMN doc_type VARCHAR DEFAULT 'GENERAL'"
                )
            )
        conn.commit()


_migrate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    initialize_hf_llm()
    load(config.vector_db_path)
    logger.info("LLM initialized", extra={"model": str(HF_LLM.fast_llm)})
    yield
    save(config.vector_db_path)
    logger.info("Shutting down")


app = FastAPI(title="agentic-rag chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/auth/register", response_model=TokenResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_username(db, request.username):
        raise HTTPException(400, "Username already taken")
    user = create_user(db, request.username, request.password)
    return TokenResponse(access_token=creat_access_token(user.id))


@app.post("/auth/login", response_model=TokenResponse)
def login(request: RegisterRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=creat_access_token(user.id))


@app.post("/upload")
def upload_file(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    dest = save_uploaded_file(current_user.id, file.filename, file)
    return {"path": dest}


@app.post("/chat", response_model=ChatResponse)
def chat_req(
    request: ChatRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if request.session_id:
        session = get_session_by_id(db, request.session_id, current_user.id)
        if session is None:
            raise HTTPException(404, "Session not found")
        session_id = session.id
    else:
        session = create_session(db, current_user.id, title=request.query[:30])
        session_id = session.id

    answer = chat(
        db=db,
        user_id=current_user.id,
        session_id=session_id,
        query=request.query,
        documents=request.documents,
        images=request.images,
        audio=request.audio,
    )
    return ChatResponse(session_id=session_id, answer=answer)


@app.get("/chats/{session_id}/messages", response_model=list[MessageOut])
def get_messages(
    session_id: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = get_session_by_id(db, session_id, current_user.id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return get_messages_by_session(db, session_id)


@app.get("/chats")
def list_chats(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = list_sessions(db, current_user.id)
    return [
        {"session_id": s.id, "title": s.title, "created_at": s.created_at.isoformat()}
        for s in sessions
    ]


@app.delete("/chats/{session_id}")
def delete_chat(
    session_id: str, current_user=Depends(get_current_user), db=Depends(get_db)
):
    session = delete_session(db, session_id, current_user.id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {"deleted": session_id}


@app.get("/documents")
def list_documents(
    current_user=Depends(get_current_user), db: Session = Depends(get_db)
):
    rows = list_user_documents(db, current_user.id)
    return [
        {"doc_id": r.doc_id, "filename": r.filename, "kind": r.kind, "path": r.path}
        for r in rows
    ]


if __name__ == "__main__":
    initialize()
    initialize_hf_llm()
    logger.info("LLM initialized (direct run)")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
