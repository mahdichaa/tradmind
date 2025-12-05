
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator
from typing import Optional

from uuid import UUID

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    username: str = Field(min_length=3, max_length=100)
    first_name: Optional[str] = Field(default=None, max_length=100)
    last_name: Optional[str]  = Field(default=None, max_length=100)
    avatar_url: Optional[HttpUrl] = None
    timezone: Optional[str] = None  # will default to 'UTC' server-side
    
class UserOut(BaseModel):
    user_id:UUID
    email: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    email_verified: Optional[bool] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    model_config = {
        "from_attributes": True  # allow .from_orm / model dump from SQLAlchemy
    }
    
class SessionOut(BaseModel):
    session_id: UUID
    device_type: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    is_active: bool
    expires_at: datetime
    refresh_expires_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

class ChangePasswordIn(BaseModel):
    current_password: str = Field(..., min_length=6, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def new_vs_current(cls, v, info):
        # We’ll re-check in service with current password, but keep basic validation here
        return v

class UpdateUserIn(BaseModel):
    # Allowed mutable fields (adapt as you wish)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    username: Optional[str] = Field(None, max_length=100)
    avatar_url: Optional[str] = Field(None, max_length=500)
    timezone: Optional[str] = Field(None, max_length=50)
    
class MeOut(BaseModel):
    user: UserOut
    session: SessionOut

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LogoutOut(BaseModel):
    revoked: bool
    revoked_at: datetime

class ForgotPasswordIn(BaseModel):
    email: EmailStr

class ResetPasswordIn(BaseModel):
    selector: str = Field(..., min_length=8, max_length=64)
    token: str = Field(..., min_length=16, max_length=256)
    new_password: str = Field(..., min_length=8, max_length=128)

class LoginIn(BaseModel):
    username_or_email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class VerifyEmailRequestIn(BaseModel):
    email: EmailStr

class VerifyEmailConfirmIn(BaseModel):
    selector: str = Field(..., min_length=8, max_length=64)
    token: str = Field(..., min_length=16, max_length=256)    


class UserForLogsOut(BaseModel):
    user_id: UUID
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: Optional[str] = None
    role: Optional[str] = None     # if your enum is str-based
    status: Optional[str] = None   # if your enum is str-based
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True