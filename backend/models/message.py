from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId
from datetime import datetime

class ToxicityScores(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float

class MessageModel(BaseModel):
    id: Optional[str] = Field(alias="_id")
    message_id: str = Field(..., regex=r"^[0-9]{17,19}$")
    server_id: str = Field(..., regex=r"^[0-9]{17,19}$")
    user_id: str = Field(..., regex=r"^[0-9]{17,19}$")
    content: str
    created_at: datetime
    toxicity_scores: ToxicityScores
    moderated: bool = False
    moderation_action_id: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat(),
        }
