import uuid
from sqlalchemy import Boolean, Column, String, DateTime, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base

from app.models.enums import UserStatus, UserRole

class User(Base):
    __tablename__ = "users"
    __table_args__ = (Index("idx_users_email", "email"),)

    user_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    username = Column(String(100), unique=True)
    avatar_url = Column(String(500))
    timezone = Column(String(50), server_default=text("'UTC'"))
    email_verified = Column(Boolean, server_default=text("FALSE"), nullable=False)
    status = Column(UserStatus, nullable=False, server_default=text("'active'"))
    role = Column(UserRole, nullable=False, server_default=text("'user'"))
    google_sub = Column(String(255), unique=True)
    

    paypal_payer_id = Column(String(255))

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime)

    # relationships (string targets to avoid circular imports)
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan", lazy="selectin")
    subscriptions = relationship("UserSubscription", back_populates="user", lazy="selectin")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan", lazy="selectin")
    analyses = relationship("ChartAnalysis", back_populates="user", cascade="all, delete-orphan", lazy="selectin")
