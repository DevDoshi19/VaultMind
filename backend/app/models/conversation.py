from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional


class Message(BaseModel):
    """One single message inside a conversation's embedded messages list."""

    role: str                      # "user" or "ai"
    content: str
    confidence_score: Optional[float] = None
    cached: Optional[bool] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Conversation(BaseModel):
    """Shape of one document in the 'conversations' MongoDB collection."""

    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[Message] = Field(default_factory=list)