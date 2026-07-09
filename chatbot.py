import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from models import Message, User, ChatSession
from schemas import *
import shutil
from database import get_db, engine, Base
from chat_service import chat
from auth import *

Base.metadata.create_all(bind=engine)
app = FastAPI(title="agentic-rag chatbot")

UPLOAD_DIR = "uploads"


@app.post("/auth/register", response_model=TokenResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(400, "Username already taken")

    user = User(username=request.username, hashed_password=hash_password(request.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = TokenResponse(access_token=creat_access_token(user.id))
    return token


@app.post("/auth/login", response_model=TokenResponse)
def login(request: RegisterRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")

    return TokenResponse(access_token=creat_access_token(user.id))


@app.post("/upload")
def upload_file(file: UploadFile = File(...), current_user = Depends(get_current_user)):
    user_upload_dir = os.path.join(UPLOAD_DIR, current_user.id)
    os.makedirs(user_upload_dir, exist_ok=True)
    dest = os.path.join(user_upload_dir, file.filename)

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"path": dest}


@app.post("/chat", response_model=ChatResponse)
def chat_req(request: ChatRequest, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    if request.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == request.session_id,
                                               ChatSession.user_id == current_user.id).first()
        if session is None:
            raise HTTPException(404, "Session not found")
        session_id = session.id

    else:
        session = ChatSession(user_id=current_user.id, title=request.query[:30])
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.id

    answer = chat(db = db, user_id=current_user.id, session_id=session_id,
                        query=request.query, documents=request.documents,
                        images=request.images, audio=request.audio,
                        )

    return ChatResponse(session_id=session_id, answer= answer)


@app.get("/chats/{session_id}/messages", response_model=list[MessageOut])
def get_messages(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if session is None:
        raise HTTPException(404, "Session not found")
    return db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at).all()







