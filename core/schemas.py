from pydantic import BaseModel
from typing import List, Optional


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    session_id : Optional[str] = None     #None -> new chat
    query: str
    documents: Optional[List[str]] = None
    images: Optional[List[str]] = None
    audio: Optional[List[str]] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class MessageOut(BaseModel):
    role: str
    content: str

    class Config:
        from_attributes = True
