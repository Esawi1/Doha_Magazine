"""Pydantic models for API requests/responses"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ========== Chat Models ==========

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000)
    session_id: Optional[str] = None


class Source(BaseModel):
    title: str
    url: str
    content: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str
    message_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None

class Message(BaseModel):
    """Message stored in Cosmos DB (simplified)"""
    id: str
    role: Literal["user", "assistant"]
    text: str
    feedback: Optional[Literal["up", "down"]] = None


class Feedback(BaseModel):
    """Feedback stored in Cosmos DB"""
    id: str
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = None
    created_utc: str


class TTSRequest(BaseModel):
    """Text-to-Speech request"""
    text: str = Field(..., min_length=1, max_length=2000)


 

