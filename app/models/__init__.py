# app/models/__init__.py
from app.models.base import Base

# Import every module that defines mapped classes 👇
from app.models.user import User
from app.models.session import Session
from app.models.subscription_plan import SubscriptionPlan
from app.models.user_subscription import UserSubscription
from app.models.trade import Trade
from app.models.chart_analysis import ChartAnalysis
from app.models.password_reset import PasswordResetToken
from app.models.email_verification import EmailVerificationToken
from app.models.audit_logs import AuditLog
from app.models.ai_config import AIConfig
