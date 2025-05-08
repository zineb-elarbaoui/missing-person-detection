from pydantic import BaseModel, Field, EmailStr 
from typing import List, Optional
from datetime import datetime
from bson import ObjectId


# Custom Pydantic-compatible ObjectId field
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


class PersonModel(BaseModel):
    name: str
    age: int
    description: Optional[str] = None
    
# 1. User model
class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    name: str
    email: EmailStr
    fcm_token: Optional[str]
    role: str  # "user" or "admin"

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# 2. Missing Person model
class MissingPersonModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    full_name: str
    age: int
    image_url: str
    embedding: List[float]
    uploader_id: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# 3. Match Report model
class MatchModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    person_id: PyObjectId
    matched_by: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    location: str
    image_url: str
    notified: bool = False

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# 4. Notification log model (optional)
class NotificationModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    user_id: PyObjectId
    message: str
    sent_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

