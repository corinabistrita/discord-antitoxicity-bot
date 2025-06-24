from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ServerModel(BaseModel):
    id: Optional[str] = Field(alias="_id")
    server_id: str
    name: str
    created_at: datetime
    owner_id: str
    region: Optional[str] = None
    member_count: int = 0
