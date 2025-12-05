from fastapi import APIRouter

from app.api.endpoints import health
from app.api.endpoints import auth
from app.api.endpoints import chart_analysis
from app.api.endpoints import trade
from app.api.endpoints import subscription
from app.api.endpoints import admin
from app.api.endpoints import webhook
from app.api.endpoints import subscription_plan
from app.api.endpoints import subscription
from app.api.endpoints import system

router = APIRouter()
router.include_router(health.router, prefix="/health", tags=["_meta"])
router.include_router(auth.router, prefix="/auth", tags=["auth"]) 
router.include_router(chart_analysis.router, prefix="/chart", tags=["chart"]) 
router.include_router(trade.router, prefix="/trade-journal", tags=["trade-journal"]) 
router.include_router(subscription.router, prefix="/subscription", tags=["subscription"]) 
router.include_router(webhook.router, prefix="/webhook", tags=["webhook"]) 
router.include_router(admin.router, prefix="/admin", tags=["admin"]) 
router.include_router(subscription_plan.router, prefix="/subscription-plan", tags=["subscription-plan"]) 
router.include_router(system.router, prefix="/system", tags=["system"]) 
