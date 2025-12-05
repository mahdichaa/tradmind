from pydantic import BaseModel, Field, EmailStr

class UpdateUserRole(BaseModel):
    role: str = Field(..., examples=["user", "Admin"])

class UpdateUserStatus(BaseModel):
    status: str = Field(..., examples=["active", "suspended", "deleted"])

# Admin create/update payloads
class AdminCreateUserIn(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=256)
    first_name: str | None = Field(default=None, max_length=100)
    last_name: str | None = Field(default=None, max_length=100)
    timezone: str | None = Field(default="UTC", max_length=50)
    role: str = Field(default="user", examples=["user", "admin"])

class AdminUpdateUserIn(BaseModel):
    email: EmailStr | None = None
    username: str | None = Field(default=None, max_length=100)
    first_name: str | None = Field(default=None, max_length=100)
    last_name: str | None = Field(default=None, max_length=100)
    timezone: str | None = Field(default=None, max_length=50)
    role: str | None = Field(default=None, examples=["user", "admin"])
    status: str | None = Field(default=None, examples=["active", "suspended", "deleted"])
    # Optional password change by admin
    password: str | None = Field(default=None, min_length=8, max_length=256)
