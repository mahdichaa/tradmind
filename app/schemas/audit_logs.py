# app/schemas/audit_log.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any
from uuid import UUID

from app.schemas.auth import UserForLogsOut


class AuditLogOut(BaseModel):
    log_id: UUID
    user_id: Optional[UUID]
    user: Optional[UserForLogsOut] = None          # ← full nested user
    action: str
    entity_type: Optional[str]
    entity_id: Optional[UUID]
    values: Optional[Any]
    ip_address: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True