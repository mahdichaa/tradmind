from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.base import Base


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    prt_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    selector = Column(String(32), nullable=False, index=True)
    verifier_hash = Column(String(255), nullable=False)
    requested_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True))
    ip_address = Column(String(64))
    user_agent = Column(String(256))
    is_used = Column(Boolean, server_default=text("FALSE"), nullable=False)
    reason = Column(String(32), server_default="forgot", nullable=False)

    user = relationship("User", backref="password_reset_tokens")
