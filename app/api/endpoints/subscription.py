import os
import requests
from fastapi import FastAPI, HTTPException, Depends, Path, APIRouter, Request
from app.database.session import get_db
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.services.subscription_plan import SubscriptionPlanService
from app.services.user_subscription import UserSubscriptionService
from app.services.user import UserService
from app.services.admin import AdminService
from dotenv import load_dotenv
from app.core.deps import get_current_user, get_current_user_and_session
from app.schemas.common import Envelope
from datetime import datetime, timezone
from typing import Optional
from app.repositories.user_subscription import subscriptionRepository


load_dotenv()

# Paypal keys
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET")
PAYPAL_BASE_URL = os.getenv("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")

# Frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173/subscribe")

router = APIRouter(prefix="", tags=["subscription"], dependencies=[Depends(get_current_user)])


def get_paypal_access_token():
    """Generate PayPal access token"""
    auth_response = requests.post(
        f"{PAYPAL_BASE_URL}/v1/oauth2/token",
        headers={"Accept": "application/json", "Accept-Language": "en_US"},
        data={"grant_type": "client_credentials"},
        auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
    )
    auth_response.raise_for_status()
    return auth_response.json()["access_token"]


@router.delete("/cancel/{subscription_id}")
async def cancel_user_subscription(
    subscription_id: str = Path(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ps = SubscriptionPlanService(db)
    us = UserService(db)
    uss = UserSubscriptionService(db)

    subscription = uss.get_by_criteria({'subscription_id': subscription_id})
    if not subscription:
        raise HTTPException(status_code=404, detail="subscription not found")
    if str(user.user_id) != str(subscription.user_id) and user.role != "admin":
        raise HTTPException(status_code=401, detail='unauthorized')

    if subscription.provider == "paddle":
        # Paddle subscription cancellation will be handled via webhook
        # For now, just mark as canceled in our database
        uss.update_by_criteria(
            {"subscription_id": subscription_id},
            {"status": "canceled"}
        )
    elif subscription.provider == "paypal":
        raise HTTPException(status_code=409, detail="PayPal subscription cancellation not implemented yet")
    else:
        raise HTTPException(status_code=400, detail="Unknown payment provider")
    return Envelope(status="ok", code=200)


@router.post("/paddle/confirm-subscription")
async def confirm_paddle_subscription(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Confirm Paddle subscription after successful checkout.
    Called by frontend after Paddle checkout completes.
    """
    try:
        body = await request.json()
        plan_id = body.get("plan_id")
        subscription_id = body.get("subscription_id")  # Paddle subscription ID

        if not plan_id or not subscription_id:
            raise HTTPException(status_code=400, detail="Missing plan_id or subscription_id")

        plan_service = SubscriptionPlanService(db)
        user_subscription_service = UserSubscriptionService(db)

        plan = plan_service.get_by_id(plan_id)
        if not plan.is_active:
            raise HTTPException(status_code=409, detail="This plan is not active")

        # Create or update subscription record
        # The webhook will update the full details when it arrives
        payload_match = {
            "user_id": user.user_id,
            "provider_subscription_id": subscription_id,
        }
        payload_update = {
            "provider": "paddle",
            "user_id": user.user_id,
            "plan_id": plan.plan_id,
            "provider_subscription_id": subscription_id,
            "status": "active",  # Will be updated by webhook
            "amount": plan.price,
        }
        user_subscription_service.create_or_update(payload_match, payload_update)

        return {"status": "ok", "message": "Subscription confirmed"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# TODO: add try and catch
@router.post("/paypal/create-order/{subscription_id}")
async def create_order(subscription_id: str = Path(...), db: Session = Depends(get_db)):
    plan_service = SubscriptionPlanService(db)
    plan = plan_service.get_by_id(subscription_id)
    access_token = get_paypal_access_token()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    body = {
        "intent": "CAPTURE",
        "purchase_units": [
            {"amount": {"currency_code": plan.currency, "value": f"{plan.price}"}}
        ],
    }
    response = requests.post(
        f"{PAYPAL_BASE_URL}/v2/checkout/orders",
        json=body,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


# TODO: add try and catch
@router.post("/paypal/capture-order/{order_id}")
async def capture_order(order_id: str = Path(...)):
    access_token = get_paypal_access_token()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.post(
        f"{PAYPAL_BASE_URL}/v2/checkout/orders/{order_id}/capture",
        headers=headers,
    )
    response.raise_for_status()
    return response.json()


@router.post("/paypal/confirm-subscription")
async def confirm_paypal_subscription(
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Confirm PayPal subscription after successful checkout.
    Called by frontend after PayPal checkout completes.
    """
    try:
        body = await request.json()
        plan_id = body.get("plan_id")
        subscription_id = body.get("subscription_id")  # PayPal subscription ID

        if not plan_id or not subscription_id:
            raise HTTPException(status_code=400, detail="Missing plan_id or subscription_id")

        plan_service = SubscriptionPlanService(db)
        user_subscription_service = UserSubscriptionService(db)

        plan = plan_service.get_by_id(plan_id)
        if not plan.is_active:
            raise HTTPException(status_code=409, detail="This plan is not active")

        # Create or update subscription record
        payload_match = {
            "user_id": user.user_id,
            "provider_subscription_id": subscription_id,
        }
        payload_update = {
            "provider": "paypal",
            "user_id": user.user_id,
            "plan_id": plan.plan_id,
            "provider_subscription_id": subscription_id,
            "status": "active",
            "amount": plan.price,
        }
        user_subscription_service.create_or_update(payload_match, payload_update)

        return {"status": "ok", "message": "Subscription confirmed"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
def get_my_subscription_status(
    pair=Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _session = pair
    repo = subscriptionRepository(db)

    # Pick latest active/trialing subscription; fallback to any latest record
    items = repo.find(
        where={
            "and": [
                {"user_id": str(user.user_id)},
                {"status": {"in": ["active", "trialing", "past_due"]}},
            ]
        },
        order_by=[("current_period_end", "desc")],
        take=1,
    )
    sub = items[0] if items else None

    # Compute plan name
    plan_name: Optional[str] = None
    if sub and getattr(sub, "plan", None) is not None:
        plan_name = getattr(sub.plan, "name", None)

    # Compute days left until current_period_end
    days_left: Optional[int] = None
    period_end = getattr(sub, "current_period_end", None) if sub else None
    now = datetime.now(timezone.utc)
    if period_end:
        try:
            if period_end.tzinfo is None:
                period_end = period_end.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        delta = period_end - now
        days_left = max(0, int(delta.days))

    return {
        "has_subscription": bool(sub is not None),
        "status": getattr(sub, "status", None) if sub else None,
        "plan_name": plan_name or "Free",
        "days_left": days_left if days_left is not None else 0,
        "current_period_end": period_end.isoformat() if period_end else None,
        "renews_automatically": bool(getattr(sub, "renews_automatically", False)) if sub else False,
    }