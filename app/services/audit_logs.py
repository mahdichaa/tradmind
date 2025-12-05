# app/services/audit_log_service.py
from typing import Any, Dict, Optional, Union
from uuid import UUID
from sqlalchemy.orm import Session

from app.repositories.audit_logs import AuditLogRepository
from app.models.audit_logs import AuditLog


class AuditLogService:
    def __init__(self, db: Session):
        self.session = db
        self.repo = AuditLogRepository(db)

    def create(
        self,
        *,
        user_id: Optional[Union[str, UUID]],
        action: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[Union[str, UUID]] = None,
        values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        commit: bool = False,
    ) -> AuditLog:
        return self.repo.create_log(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            values=values,
            ip_address=ip_address,
            commit=commit,
        )
