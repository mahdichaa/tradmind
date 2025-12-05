from pydantic import BaseModel, Field, EmailStr

class AdminCreateSubscriptionPlan(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=100)
    price: float  # Changed from int to float to accept decimal prices
    currency: str | None = Field(default="USD", max_length=10)
    billing_interval: str
    features: dict
    is_active: bool | None = Field(default=True)
    paddle_price_id: str | None = Field(default=None)  # Added for Paddle integration
    paypal_plan_id: str | None = Field(default=None)  # Added for PayPal integration
    
