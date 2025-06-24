from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserModel(BaseModel):
    id: Optional[str] = Field(alias="_id")
    username: str
    discord_id: str
    joined_at: datetime
    roles: List[str] = []
    is_banned: bool = False