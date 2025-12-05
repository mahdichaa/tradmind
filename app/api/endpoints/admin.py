from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from app.core.deps import get_current_user_and_session
from app.core.security import _client_ip
from app.schemas.common import Envelope
from app.database.session import get_db
from sqlalchemy.orm import Session
from app.schemas.audit_logs import AuditLogOut
from app.schemas.ip_blocklist import BlockItemIn
from app.schemas.user_subscription import  SubscriptionSchema, SubscriptionsResponse
from app.services.admin import AdminService
from app.schemas.common import Envelope
from app.schemas.auth import UserOut
from pydantic import BaseModel
from app.schemas.user import UpdateUserRole, UpdateUserStatus, AdminCreateUserIn, AdminUpdateUserIn
from app.repositories.audit_logs import AuditLogRepository
from app.schemas.ai_settings import AIConfigOut, AIConfigUpdateIn

def require_admin(pair = Depends(get_current_user_and_session)):
    user, _ = pair
    if user.role != "admin":
        raise HTTPException(403, "Admins only")
    return user

router = APIRouter(prefix="", tags=["admin"] , dependencies=[Depends(require_admin)])


from app.services.chart_analysis import ChartAnalysisService
from app.models.chart_analysis import ChartAnalysis

@router.get("/analyses")
def list_all_analyses(
    admin = Depends(require_admin),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="PENDING | COMPLETED | FAILED"),
    date_from: Optional[str] = Query(None, description="ISO date or datetime"),
    date_to: Optional[str] = Query(None, description="ISO date or datetime (exclusive)"),
    user_id: Optional[str] = Query(None, description="Filter by user_id"),
    email_substr: Optional[str] = Query(None, description="Filter by email substring, case-insensitive"),
):
    # local lite date parser (aligns with /chart/analyses behavior)
    def _parse_dt(s: Optional[str]):
        if not s:
            return None
        try:
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                from datetime import date
                d = date.fromisoformat(s)
                from datetime import time as _t, datetime as _dt
                return _dt.combine(d, _t.min)
            from datetime import datetime as _dt
            if s.endswith("Z"):
                return _dt.fromisoformat(s[:-1] + "+00:00")
            return _dt.fromisoformat(s)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Invalid datetime format: {s}")

    svc = ChartAnalysisService(db)
    rows = svc.list_filtered_any(
        status=status,
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        limit=limit,
        offset=offset,
        user_id=user_id,
        email_substr=email_substr,
    )

    items = []
    for a in rows:
        items.append({
            "analysis_id": str(a.analysis_id),
            "created_at": a.created_at,
            "updated_at": a.updated_at,
            "symbol": a.symbol,
            "timeframe": a.timeframe,
            "chart_image_url": a.chart_image_url,
            "chart_image_data": getattr(a, 'chart_image_data', None),
            "status": a.status,
            "error_message": a.error_message,
            "direction": a.direction,
            "market_trend": a.market_trend,
            "pattern": a.pattern,
            "confidence_score": a.confidence_score,
            "trading_type": ((a.ai_request or {}).get("user_inputs") or {}).get("trading_type") if getattr(a, 'ai_request', None) else None,
            "outcome": getattr(getattr(a, 'trade', None), 'outcome', None),
            "suggested_entry_price": a.suggested_entry_price,
            "suggested_stop_loss": a.suggested_stop_loss,
            "suggested_take_profit": a.suggested_take_profit,
            "suggested_risk_reward": a.suggested_risk_reward,
            "suggested_position_size": a.suggested_position_size,
            # include user info hint for admin views
            "user_id": str(a.user_id) if getattr(a, 'user_id', None) else None,
        })
    return {"items": items, "limit": limit, "offset": offset}

@router.get("/users")
def list_users(admin = Depends(require_admin), db: Session = Depends(get_db)):
    svc = AdminService(db)
    data = svc.get_all_users_with_stats()
    return Envelope(status="ok", code=200, message="Users list", data=data)

@router.post("/users")
def create_user(body: AdminCreateUserIn, request: Request, admin = Depends(require_admin), db: Session = Depends(get_db)):
    svc = AdminService(db)
    try:
        user = svc.create_user(body)
        # audit log
        AuditLogRepository(db).create_log(
            user_id=admin.user_id,
            action="POST",
            entity_type="User",
            entity_id=user.user_id,
            values={"username": user.username, "email": user.email},
            ip_address=_client_ip(request),
            commit=True,
        )
        return Envelope(status="ok", code=200, message="User created",
                        data=UserOut.model_validate(user, from_attributes=True))
    except HTTPException as e:
        return Envelope(status="error", code=200, message="Create failed",
                        error={"detail": e.detail, "status_code": e.status_code})

@router.get("/user/{user_id}")
def get_user(user_id: str, admin = Depends(require_admin), db: Session = Depends(get_db)):
    svc = AdminService(db)
    try:
        user = svc.get_user(user_id)
        return Envelope(status="ok", code=200, message="User",
                        data=UserOut.model_validate(user, from_attributes=True))
    except HTTPException as e:
        return Envelope(status="error", code=200, message="Get user failed",
                        error={"detail": e.detail, "status_code": e.status_code})

@router.patch("/user/{user_id}")
def update_user(user_id: str, body: AdminUpdateUserIn, request: Request, admin = Depends(require_admin), db: Session = Depends(get_db)):
    svc = AdminService(db)
    try:
        user = svc.update_user(user_id, body)
        # audit log
        AuditLogRepository(db).create_log(
            user_id=admin.user_id,
            action="PATCH",
            entity_type="User",
            entity_id=user.user_id,
            values={k: v for k, v in body.model_dump(exclude_none=True).items()},
            ip_address=_client_ip(request),
            commit=True,
        )
        return Envelope(status="ok", code=200, message="User updated",
                        data=UserOut.model_validate(user, from_attributes=True))
    except HTTPException as e:
        return Envelope(status="error", code=200, message="Update failed",
                        error={"detail": e.detail, "status_code": e.status_code})

@router.patch("/update_role/{user_id}")
def change_role(
    user_id: str,
    body: UpdateUserRole,
    request: Request,
    admin = Depends(require_admin),
    db: Session = Depends(get_db)
):
    svc = AdminService(db)
    try:
        svc.change_role(user_id, body.role)
        AuditLogRepository(db).create_log(
            user_id=admin.user_id,
            action="PATCH",
            entity_type="User",
            entity_id=user_id,
            values={"role": body.role},
            ip_address=_client_ip(request),
            commit=True,
        )
        return Envelope(status="ok", code=200, message="Role updated", data={})
    except HTTPException as e:
        return Envelope(status="error", code=200, message="Role update failed",
                        error={"detail": e.detail, "status_code": e.status_code})

@router.patch("/update_status/{user_id}")
def change_status(
    user_id: str,
    body: UpdateUserStatus,
    request: Request,
    admin = Depends(require_admin),
    db: Session = Depends(get_db)
):
    svc = AdminService(db)
    try:
        svc.change_status(user_id, body.status)
        AuditLogRepository(db).create_log(
            user_id=admin.user_id,
            action="PATCH",
            entity_type="User",
            entity_id=user_id,
            values={"status": body.status},
            ip_address=_client_ip(request),
            commit=True,
        )
        return Envelope(status="ok", code=200, message="Status updated", data={})
    except HTTPException as e:
        return Envelope(status="error", code=200, message="Status update failed",
                        error={"detail": e.detail, "status_code": e.status_code})

@router.delete("/user/{user_id}")
def delete_user(
    user_id: str,
    request: Request,
    admin = Depends(require_admin),
    db: Session = Depends(get_db)
):
    svc = AdminService(db)
    try:
        svc.delete_user(user_id)
        AuditLogRepository(db).create_log(
            user_id=admin.user_id,
            action="DELETE",
            entity_type="User",
            entity_id=user_id,
            values={},
            ip_address=_client_ip(request),
            commit=True,
        )
        return Envelope(status="ok", code=200, message="User deleted", data={})
    except HTTPException as e:
        return Envelope(status="error", code=200, message="Delete failed",
                        error={"detail": e.detail, "status_code": e.status_code})

@router.get("/payments")
def list_payments(admin = Depends(require_admin), db: Session = Depends(get_db)):
    service = AdminService(db)
    return service.get_all_subscriptions()


# @router.get("/subscription/{user_id}", response_model=SubscriptionsResponse)
# def get_subscriptions_by_user(user_id: str, db: Session = Depends(get_db),admin = Depends(require_admin)):
#     service = AdminService(db)
#     return service.get_subscriptions_by_user(user_id)
@router.get("/Audit_Logs", response_model=list[AuditLogOut])
def list_audit_logs(
    db: Session = Depends(get_db),
    entity_type: Optional[str] = Query(None, description="Filter by entity type, e.g. 'ChartAnalysis'"),
    action: Optional[str] = Query(None, description="Filter by action, e.g. 'STARTED', 'ERROR', 'DB_ERROR'"),
    start_date: Optional[datetime] = Query(None, description="Start of date range (YYYY-MM-DD)"),
    end_date: Optional[datetime] = Query(None, description="End of date range (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=200, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    order_desc: bool = Query(True, description="Latest first (default)"),
):
    """
    Admin endpoint to list audit logs with filters and pagination.
    """
    svc = AdminService(db)
    logs = svc.list_logs(
        entity_type=entity_type,
        action=action,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        order_desc=order_desc,
    )
    return logs

class EmailBlockItem(BaseModel):
    email: str

@router.get("/blocked-emails", response_model=list[str])
def list_blocked_emails(db: Session = Depends(get_db)):
    svc = AdminService(db)
    return svc.list_blocked_emails()

@router.post("/blocked-emails", status_code=204)
def add_blocked_email(item: EmailBlockItem, db: Session = Depends(get_db)):
    svc = AdminService(db)
    svc.add_blocked_email(item.email)
    return

@router.delete("/blocked-emails", status_code=204)
def remove_blocked_email(item: EmailBlockItem, db: Session = Depends(get_db)):
    svc = AdminService(db)
    svc.remove_blocked_email(item.email)
    return

@router.get("/blocked-ips", response_model=list[str])
def list_blocked_ips(db: Session = Depends(get_db),):
    svc = AdminService(db)
    return svc.list_blocked_ips()

@router.post("/blocked-ips", status_code=204)
def add_blocked_ip(item: BlockItemIn,db: Session = Depends(get_db)):
    svc = AdminService(db)

    try:
        svc.add_blocked_ip(item.item)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid IP/CIDR: {e}")
    return

@router.delete("/blocked-ips", status_code=204)
def remove_blocked_ip(item: BlockItemIn,db: Session = Depends(get_db)):
    svc = AdminService(db)

    svc.remove_blocked_ip(item.item)
    return

@router.get("/blocked-ips/check")
def check_ip(ip: str = Query(..., description="IP to test"),db: Session = Depends(get_db),):
    svc = AdminService(db)

    return {"ip": ip, "blocked": svc.is_ip_blocked(ip)}

# -------- AI Settings (OpenRouter) ---------
@router.get("/ai-settings", response_model=AIConfigOut)
def get_ai_settings(db: Session = Depends(get_db)):
    """Get current OpenRouter AI configuration"""
    from app.repositories.ai_config import AIConfigRepository
    
    repo = AIConfigRepository(db)
    config = repo.get_or_create()
    
    # Mask API key for security
    masked_key = None
    if config.openrouter_api_key:
        key = config.openrouter_api_key
        masked_key = f"••••{key[-4:]}" if len(key) >= 4 else "••••"
    
    # Build masked keys array for multi-key support
    masked_keys = []
    if config.openrouter_api_keys:
        for i, key_obj in enumerate(config.openrouter_api_keys):
            key = key_obj.get("key", "")
            masked = f"••••{key[-4:]}" if len(key) >= 4 else "••••"
            masked_keys.append({
                "index": i,
                "masked_key": masked,
                "label": key_obj.get("label", f"API Key {i + 1}"),
                "is_active": key_obj.get("is_active", True)
            })
    
    return AIConfigOut(
        id=config.id,
        openrouter_api_key_masked=masked_key,
        api_keys=masked_keys,  # Add multi-key array
        selected_model=config.selected_model,
        risk_defaults=config.risk_defaults or {}
    )


@router.put("/ai-settings", response_model=AIConfigOut)
def update_ai_settings(body: AIConfigUpdateIn, db: Session = Depends(get_db)):
    """Update OpenRouter AI configuration"""
    from app.repositories.ai_config import AIConfigRepository
    
    repo = AIConfigRepository(db)
    
    # Update configuration
    config = repo.update_config(
        openrouter_api_key=body.openrouter_api_key,
        selected_model=body.selected_model,
        risk_defaults=body.risk_defaults
    )
    
    # Mask API key for response
    masked_key = None
    if config.openrouter_api_key:
        key = config.openrouter_api_key
        masked_key = f"••••{key[-4:]}" if len(key) >= 4 else "••••"
    
    return AIConfigOut(
        id=config.id,
        openrouter_api_key_masked=masked_key,
        selected_model=config.selected_model,
        risk_defaults=config.risk_defaults or {}
    )


@router.get("/ai-settings/models")
def list_available_models(db: Session = Depends(get_db)):
    """Fetch available models from OpenRouter API"""
    from app.repositories.ai_config import AIConfigRepository
    from app.core.openrouter_client import get_openrouter_client
    from app.schemas.ai_settings import OpenRouterModel
    
    repo = AIConfigRepository(db)
    config = repo.get_or_create()
    
    # Get active API keys (supports both legacy and multi-key)
    api_keys = config.get_active_keys()
    
    if not api_keys:
        raise HTTPException(
            status_code=400,
            detail="OpenRouter API key not configured. Please add an API key first."
        )
    
    try:
        # Use first active key to fetch models
        client = get_openrouter_client(api_keys[0])
        models = client.list_models()
        return {"data": models}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch models from OpenRouter: {str(e)}"
        )


@router.get("/ai-settings/credits")
def get_api_keys_credits(db: Session = Depends(get_db)):
    """Get credit balance for all configured API keys"""
    from app.repositories.ai_config import AIConfigRepository
    from app.core.openrouter_client import get_all_credits
    
    repo = AIConfigRepository(db)
    config = repo.get_or_create()
    
    api_keys = config.get_active_keys()
    
    if not api_keys:
        return {"keys": [], "total_credits": 0}
    
    try:
        credits_info = get_all_credits(api_keys)
        
        # Enhance with labels from config
        for i, credit_data in enumerate(credits_info):
            key_info = config.get_key_info(i)
            if key_info:
                credit_data["label"] = key_info.get("label", f"API Key {i + 1}")
                credit_data["is_active"] = key_info.get("is_active", True)
            else:
                credit_data["label"] = f"API Key {i + 1}"
                credit_data["is_active"] = True
        
        # Handle None values in credits calculation
        total_credits = sum(c.get("credits", 0) or 0 for c in credits_info)
        
        return {
            "keys": credits_info,
            "total_credits": total_credits
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch credits: {str(e)}"
        )


@router.post("/ai-settings/api-keys")
def add_api_key(
    body: dict,
    db: Session = Depends(get_db)
):
    """Add a new API key to the configuration"""
    from app.repositories.ai_config import AIConfigRepository
    
    api_key = body.get("api_key")
    label = body.get("label")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    
    repo = AIConfigRepository(db)
    config = repo.add_api_key(api_key, label)
    
    # Return masked keys
    masked_keys = []
    if config.openrouter_api_keys:
        for i, key_obj in enumerate(config.openrouter_api_keys):
            key = key_obj.get("key", "")
            masked = f"••••{key[-4:]}" if len(key) >= 4 else "••••"
            masked_keys.append({
                "index": i,
                "masked_key": masked,
                "label": key_obj.get("label", f"API Key {i + 1}"),
                "is_active": key_obj.get("is_active", True)
            })
    
    return {
        "id": config.id,
        "api_keys": masked_keys,
        "selected_model": config.selected_model,
        "risk_defaults": config.risk_defaults or {}
    }


@router.delete("/ai-settings/api-keys/{index}")
def remove_api_key(
    index: int,
    db: Session = Depends(get_db)
):
    """Remove API key at specified index"""
    from app.repositories.ai_config import AIConfigRepository
    
    repo = AIConfigRepository(db)
    
    try:
        config = repo.remove_api_key(index)
        
        # Return updated masked keys
        masked_keys = []
        if config.openrouter_api_keys:
            for i, key_obj in enumerate(config.openrouter_api_keys):
                key = key_obj.get("key", "")
                masked = f"••••{key[-4:]}" if len(key) >= 4 else "••••"
                masked_keys.append({
                    "index": i,
                    "masked_key": masked,
                    "label": key_obj.get("label", f"API Key {i + 1}"),
                    "is_active": key_obj.get("is_active", True)
                })
        
        return {
            "id": config.id,
            "api_keys": masked_keys,
            "selected_model": config.selected_model,
            "risk_defaults": config.risk_defaults or {}
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/ai-settings/api-keys/reorder")
def reorder_api_keys(
    body: dict,
    db: Session = Depends(get_db)
):
    """Reorder API keys based on new priority"""
    from app.repositories.ai_config import AIConfigRepository
    
    new_order = body.get("new_order")
    
    if not isinstance(new_order, list):
        raise HTTPException(status_code=400, detail="new_order must be a list of indices")
    
    repo = AIConfigRepository(db)
    
    try:
        config = repo.reorder_api_keys(new_order)
        
        # Return updated masked keys
        masked_keys = []
        if config.openrouter_api_keys:
            for i, key_obj in enumerate(config.openrouter_api_keys):
                key = key_obj.get("key", "")
                masked = f"••••{key[-4:]}" if len(key) >= 4 else "••••"
                masked_keys.append({
                    "index": i,
                    "masked_key": masked,
                    "label": key_obj.get("label", f"API Key {i + 1}"),
                    "is_active": key_obj.get("is_active", True)
                })
        
        return {
            "id": config.id,
            "api_keys": masked_keys,
            "selected_model": config.selected_model,
            "risk_defaults": config.risk_defaults or {}
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
