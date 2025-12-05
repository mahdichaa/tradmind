import os
from fastapi import FastAPI, HTTPException, Depends, Path
from app.database.session import get_db
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from fastapi import APIRouter
from app.services.subscription_plan import SubscriptionPlanService
from app.services.user_subscription import UserSubscriptionService
from app.services.user import UserService
from app.services.admin import AdminService
from fastapi import Request
import requests
from app.core.deps import get_current_user
from app.schemas.common import Envelope
from app.schemas.subscription_plan import AdminCreateSubscriptionPlan
from app.core.deps import  get_current_user_and_session

def require_admin(pair = Depends(get_current_user_and_session)):
    user, _ = pair
    if user.role != "admin":
        raise HTTPException(403, "Admins only")
    return user

router = APIRouter(prefix="", tags=["subscription-plan"])

@router.post("")
async def create_subscription_plan(body:AdminCreateSubscriptionPlan,admin = Depends(require_admin), db: Session = Depends(get_db)):
    sps=SubscriptionPlanService(db)
    subscription_plan=sps.create(body)
    return Envelope(status="ok",code=201,message="subscription plan was created",data=subscription_plan)

@router.patch("/{plan_id}/disactivate")
async def disactivate(plan_id: str, admin = Depends(require_admin), db: Session = Depends(get_db)):
    sps=SubscriptionPlanService(db)
    updated=sps.disactivate(plan_id)
    return Envelope(status="ok",code=200,message="subscription plan was diactivated",data=updated)

@router.patch("/{plan_id}/activate")
async def activate(plan_id: str,admin = Depends(require_admin), db: Session = Depends(get_db)):
    sps=SubscriptionPlanService(db)
    updated=sps.activate(plan_id)
    return Envelope(status="ok",code=201,message="subscription plan was activated",data=updated)


@router.get("")
async def get_all(db: Session = Depends(get_db)):
    sps=SubscriptionPlanService(db)
    plans=sps.get_all()
    return Envelope(status="ok",code=200,data=plans)



