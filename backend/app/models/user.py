from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timezone
import uuid


class User(BaseModel):
    """
    Shape of one document in the 'users' MongoDB collection.
    This is what gets validated before saving, and what we get back when reading.
    """

    email: EmailStr
    hashed_password: str
    api_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    