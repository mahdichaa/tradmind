# schemas/subscription.py
from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class SubscriptionItem(BaseModel):
    subscription_id: UUID
    user_id: UUID
    plan_id: Optional[UUID]
    plan_name: Optional[str] = None
    provider: str
    provider_subscription_id: Optional[str]
    status: str
    amount: float
    started_at: Optional[datetime]
    current_period_end: Optional[datetime]
    cancel_at: Optional[datetime]
    cancel_reason: Optional[str]
    receipt_url: Optional[str]
    error_message: Optional[str]
    renews_automatically: bool
    created_at: datetime
    updated_at: datetime

class SubscriptionsResponse(BaseModel):
    items: list[SubscriptionItem]
    total_count: int
    total_amount: float
    active_count: int
    canceled_count: int

class SubscriptionSchema(BaseModel):
    subscription_id: UUID
    user_id: UUID
    plan_id: Optional[UUID]
    plan_name: Optional[str]
    provider: str
    status: str
    amount: float
    started_at: datetime
    current_period_end: Optional[datetime]
    cancel_at: Optional[datetime]
    cancel_reason: Optional[str]
    receipt_url: Optional[str]
    error_message: Optional[str]
    renews_automatically: bool