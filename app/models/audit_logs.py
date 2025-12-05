# app/models/audit_log.py
from sqlalchemy import Column, String, DateTime, Index, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("idx_audit_user", "user_id"),
    )

    log_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    # null + ON DELETE SET NULL per DDL
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(UUID(as_uuid=True))
    values = Column(JSONB)               # stores arbitrary JSON payload
    ip_address = Column(String(45))      # IPv4/IPv6 printable length
    created_at = Column(DateTime, server_default=func.now())

    # Relationship to User; creates User.audit_logs backref without editing User class
    user = relationship(
        "User",
        backref="audit_logs",
        lazy="selectin",
        foreign_keys=[user_id],
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<AuditLog log_id={self.log_id} user_id={self.user_id} action={self.action}>"
