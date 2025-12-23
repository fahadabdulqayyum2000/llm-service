# app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)

class SourceItem(BaseModel):
    source: str

class ChatResponse(BaseModel):
    reply: str
    model: str
    sources: Optional[List[SourceItem]] = None
