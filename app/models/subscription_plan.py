import uuid
from sqlalchemy import Boolean, Column, String, DateTime, Numeric, Index, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base

from app.models.enums  import BillingInterval

class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"
    __table_args__ = (Index("uix_plans_name", "name", unique=True),)

    plan_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    name = Column(String(100), nullable=False, unique=True)
    description = Column(String)
    price = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(10), server_default=text("'USD'"))
    billing_interval = Column(BillingInterval, nullable=False)
    features = Column(JSONB, server_default=text("'{}'::jsonb"))
    is_active = Column(Boolean, server_default=text("TRUE"))
    paddle_price_id = Column(String)
    paypal_plan_id = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())

    subscriptions = relationship("UserSubscription", back_populates="plan", lazy="selectin")
