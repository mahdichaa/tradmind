import uuid
from sqlalchemy import Column, String, DateTime, Numeric, ForeignKey, Index, Boolean, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base

from app.models.enums  import ProviderType, SubscriptionStatus

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"
    __table_args__ = (Index("idx_user_subscriptions_user", "user_id"),)

    subscription_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    plan_id = Column(UUID(as_uuid=True), ForeignKey("subscription_plans.plan_id", ondelete="SET NULL"))

    provider = Column(ProviderType, nullable=False)
    provider_subscription_id = Column(String(255))
    status = Column(SubscriptionStatus, nullable=False, server_default=text("'unpaid'"))
    amount = Column(Numeric(10, 2), nullable=False)


    started_at = Column(DateTime, server_default=func.now())
    current_period_end = Column(DateTime)
    cancel_at = Column(DateTime)
    cancel_reason = Column(String)
    receipt_url = Column(String(500))
    error_message = Column(String)
    renews_automatically = Column(Boolean, server_default=text("TRUE"))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="subscriptions", lazy="selectin")
    plan = relationship("SubscriptionPlan", back_populates="subscriptions", lazy="selectin")
