import os
import json
import hmac
import hashlib
from fastapi import HTTPException, Depends
from app.database.session import get_db
from sqlalchemy.orm import Session
from fastapi import APIRouter
from app.services.subscription_plan import SubscriptionPlanService
from app.services.user_subscription import UserSubscriptionService
from app.services.user import UserService
from dotenv import load_dotenv
from fastapi import Request
from datetime import datetime

load_dotenv()

# Paddle webhook secret
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET")

router = APIRouter(prefix="", tags=["webhook"])


def verify_paddle_webhook(signature: str, raw_body: bytes) -> bool:
    """
    Verify Paddle webhook signature.
    Paddle sends signature in format: ts=timestamp;h1=signature
    """
    if not PADDLE_WEBHOOK_SECRET:
        print("WARNING: PADDLE_WEBHOOK_SECRET not configured")
        return False
    
    try:
        # Parse signature header
        sig_parts = {}
        for part in signature.split(';'):
            key, value = part.split('=', 1)
            sig_parts[key] = value
        
        timestamp = sig_parts.get('ts')
        received_signature = sig_parts.get('h1')
        
        if not timestamp or not received_signature:
            return False
        
        # Create signed payload: timestamp + ":" + raw_body
        signed_payload = f"{timestamp}:{raw_body.decode('utf-8')}"
        
        # Compute HMAC
        expected_signature = hmac.new(
            PADDLE_WEBHOOK_SECRET.encode('utf-8'),
            signed_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(expected_signature, received_signature)
    
    except Exception as e:
        print(f"Paddle signature verification error: {e}")
        return False


@router.post('/paddle')
async def paddle_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handle Paddle webhook notifications.
    Paddle Billing API v2 webhook events.
    """
    # Get raw body for signature verification
    raw_body = await request.body()
    signature = request.headers.get("paddle-signature")
    
    # Verify webhook signature
    if not verify_paddle_webhook(signature, raw_body):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    
    # Parse event
    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    event_type = event.get("event_type")
    event_data = event.get("data")
    
    if not event_type or not event_data:
        raise HTTPException(status_code=400, detail="Missing event_type or data")
    
    usv = UserService(db)
    sps = SubscriptionPlanService(db)
    uss = UserSubscriptionService(db)
    
    # Handle subscription.created event
    if event_type == "subscription.created":
        subscription_id = event_data.get("id")
        customer_id = event_data.get("customer_id")
        
        # Find user by email (Paddle doesn't have a customer_id we store)
        # We'll match by subscription_id from the confirm endpoint
        existing_sub = uss.get_by_criteria({"provider_subscription_id": subscription_id})
        
        if existing_sub:
            # Update with full details from webhook
            price_id = event_data.get("items", [{}])[0].get("price", {}).get("id")
            plan = sps.get_by_criteria({"paddle_price_id": price_id})
            
            if plan:
                current_period_end_str = event_data.get("current_billing_period", {}).get("ends_at")
                current_period_end = datetime.fromisoformat(current_period_end_str.replace('Z', '+00:00')) if current_period_end_str else None
                
                unit_amount = float(event_data.get("items", [{}])[0].get("price", {}).get("unit_price", {}).get("amount", 0)) / 100.0
                status = event_data.get("status", "active")
                
                uss.update_by_criteria(
                    {"provider_subscription_id": subscription_id},
                    {
                        "status": status,
                        "current_period_end": current_period_end,
                        "amount": unit_amount,
                    }
                )
    
    # Handle subscription.updated event
    elif event_type == "subscription.updated":
        subscription_id = event_data.get("id")
        current_period_end_str = event_data.get("current_billing_period", {}).get("ends_at")
        current_period_end = datetime.fromisoformat(current_period_end_str.replace('Z', '+00:00')) if current_period_end_str else None
        status = event_data.get("status", "active")
        
        uss.update_by_criteria(
            {"provider_subscription_id": subscription_id, "provider": "paddle"},
            {
                "status": status,
                "current_period_end": current_period_end,
            }
        )
    
    # Handle subscription.canceled event
    elif event_type == "subscription.canceled":
        subscription_id = event_data.get("id")
        uss.update_by_criteria(
            {"provider_subscription_id": subscription_id, "provider": "paddle"},
            {"status": "canceled"}
        )
    
    # Handle transaction.completed (payment succeeded)
    elif event_type == "transaction.completed":
        subscription_id = event_data.get("subscription_id")
        if subscription_id:
            # Update subscription status
            uss.update_by_criteria(
                {"provider_subscription_id": subscription_id, "provider": "paddle"},
                {"status": "active"}
            )
    
    return json.dumps({"success": True})
