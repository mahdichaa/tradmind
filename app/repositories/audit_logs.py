# app/repositories/audit_log_repository.py
from typing import Any, Dict, Optional, Union
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.audit_logs import AuditLog
from app.repositories.base import BaseRepository  # adjust import if needed


class AuditLogRepository(BaseRepository[AuditLog]):
    def __init__(self, session: Session):
        super().__init__(session, AuditLog)

    def create_log(
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
        # Delegate to BaseRepository.create with validated columns
        data: Dict[str, Any] = {
            "user_id": user_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "values": values or {},
            "ip_address": ip_address,
        }
        return super().create(data, commit=commit)
    
    def list_with_user(
        self,
        where: Optional[dict] = None,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_desc: bool = True,
    ) -> list[AuditLog]:
        stmt = select(AuditLog).options(selectinload(AuditLog.user))

        where_expr = self._build_where(where) if where else None
        if where_expr is not None:
            stmt = stmt.where(where_expr)

        # sort by latest first by default
        stmt = stmt.order_by(AuditLog.created_at.desc() if order_desc else AuditLog.created_at.asc())

        if isinstance(limit, int) and limit >= 0:
            stmt = stmt.limit(limit)
        if isinstance(offset, int) and offset > 0:
            stmt = stmt.offset(offset)

        return self.session.execute(stmt).scalars().all()
