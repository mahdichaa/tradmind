import uuid
from sqlalchemy import Column, String, DateTime, Index, ForeignKey, Boolean, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        Index("idx_sessions_active", "user_id", "is_active"),
    )

    session_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    token = Column(String(500), unique=True, nullable=False)
    refresh_token = Column(String(500))       # (legacy/unused raw) keep nullable if you already created it
    # NEW: secure refresh fields
    refresh_token_hash = Column(String(128), nullable=False)      # sha256 hex
    refresh_expires_at = Column(DateTime, nullable=False)
    rotated_at = Column(DateTime)  
    ip_address = Column(String(45))
    user_agent = Column(String)
    device_type = Column(String(50))
    is_active = Column(Boolean, server_default=text("TRUE"))
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    last_activity = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="sessions", lazy="selectin")
