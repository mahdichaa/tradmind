from sqlalchemy import Enum

UserStatus          = Enum("active", "suspended", "deleted", name="user_status", native_enum=False)
UserRole            = Enum("user", "admin", name="user_role", native_enum=False)

BillingInterval     = Enum("month", "year", name="billing_interval", native_enum=False)
ProviderType        = Enum("stripe", "paypal", name="provider_type", native_enum=False)
SubscriptionStatus  = Enum("trialing", "active", "past_due", "canceled", "expired","unpaid",
                           name="subscription_status", native_enum=False)

TradeOutcome        = Enum("WIN", "LOSS", "SUGGESTED", "NOT_TAKEN","PENDING", name="trade_outcome", native_enum=False)
TradeSide           = Enum("SWING", "SCALP","BOTH", name="trade_side", native_enum=False)
AnalysisStatus      = Enum("PENDING", "COMPLETED", "FAILED", name="analysis_status", native_enum=False)
TradeSource         = Enum("AI", "MANUAL", name="trade_source", native_enum=False)
