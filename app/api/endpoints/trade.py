from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime, time, timedelta

from app.core.deps import get_current_user_and_session
from app.database.session import get_db
from app.schemas.trade import TradeCreateManualRequest, TradeUpdateRequest
from app.services.trade import TradeService

router = APIRouter(prefix="", tags=["trade-journal"] , dependencies=[Depends(get_current_user_and_session)])

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1] + "+00:00")
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            d = datetime.fromisoformat(s)  # will be date-only 00:00
            return d
        return datetime.fromisoformat(s)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid datetime format: {s}")

def _normalize_range(df: Optional[datetime], dt: Optional[datetime]) -> tuple[Optional[datetime], Optional[datetime]]:
    if dt and dt.tzinfo is None and dt.time() == time.min:
        dt = dt + timedelta(days=1)  # make date-only end exclusive
    return df, dt

# --------- POST /ai/trades (manual creation; analysis_id optional) ----------
@router.post("/trades")
def create_trade(
    trade_in: TradeCreateManualRequest,
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _session = pair
    svc = TradeService(db)

    payload = trade_in.model_dump(exclude_none=True)
    # Normalize empty string analysis_id -> None
    if "analysis_id" in payload and isinstance(payload["analysis_id"], str) and payload["analysis_id"].strip() == "":
        payload["analysis_id"] = None

    trade = svc.create_manual(user_id=str(user.user_id), payload=payload)
    db.commit()

    # Response can still include suggested_* keys (they’ll be null)
    return {
        "trade_id": str(trade.trade_id),
        "user_id": str(trade.user_id),
        "analysis_id": str(trade.analysis_id) if trade.analysis_id else None,
        "source": trade.source,                # MANUAL
        "trade_date": trade.trade_date,
        "symbol": trade.symbol,
        "trade_type": trade.trade_type,

        "entry_time": trade.entry_time,
        "entry_price": trade.entry_price,
        "exit_time": trade.exit_time,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,

        "outcome": trade.outcome,
        "profit_loss": trade.profit_loss,
        "profit_percent": trade.profit_percent,
        "trading_notes": trade.trading_notes,
        "review_notes": trade.review_notes,

        "created_at": trade.created_at,
        "updated_at": trade.updated_at,
    }
# --------- GET /ai/trades (list the user's trade journal) ----------
@router.get("/trades")
def list_trades(
    pair=Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    analysis_id: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    user, _session = pair
    svc = TradeService(db)

    df = _parse_dt(date_from)
    dt = _parse_dt(date_to)
    df, dt = _normalize_range(df, dt)

    rows = svc.list_owned(
        user_id=str(user.user_id),
        symbol=symbol,
        source=source,
        outcome=outcome,
        analysis_id=analysis_id,
        date_from=df,
        date_to=dt,
        limit=limit,
        offset=offset,
    )

    items = []
    for t in rows:
        items.append({
            "trade_id": str(t.trade_id),
            "user_id": str(t.user_id),
            "analysis_id": str(t.analysis_id) if t.analysis_id else None,
            "source": t.source,
            "trade_date": t.trade_date,
            "symbol": t.symbol,
            "trade_type": t.trade_type,

            "suggested_direction": t.suggested_direction,
            "suggested_entry_price": t.suggested_entry_price,
            "suggested_stop_loss": t.suggested_stop_loss,
            "suggested_take_profit": t.suggested_take_profit,
            "suggested_risk_reward": t.suggested_risk_reward,
            "suggested_position_size": t.suggested_position_size,

            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "quantity": t.quantity,

            "outcome": t.outcome,
            "profit_loss": t.profit_loss,
            "profit_percent": t.profit_percent,
            "trading_notes": t.trading_notes,
            "review_notes": t.review_notes,

            "created_at": t.created_at,
            "updated_at": t.updated_at,
        })

    return {"items": items, "limit": limit, "offset": offset}

# --------- GET /ai/trades/{trade_id} (single trade) ----------
@router.get("/trades/{trade_id}")
def get_trade(
    trade_id: str = Path(...),
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _session = pair
    svc = TradeService(db)
    try:
        t = svc.get_owned_required(user_id=str(user.user_id), trade_id=trade_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Trade not found")

    return {
        "trade_id": str(t.trade_id),
        "user_id": str(t.user_id),
        "analysis_id": str(t.analysis_id) if t.analysis_id else None,
        "source": t.source,
        "trade_date": t.trade_date,
        "symbol": t.symbol,
        "trade_type": t.trade_type,

        "suggested_direction": t.suggested_direction,
        "suggested_entry_price": t.suggested_entry_price,
        "suggested_stop_loss": t.suggested_stop_loss,
        "suggested_take_profit": t.suggested_take_profit,
        "suggested_risk_reward": t.suggested_risk_reward,
        "suggested_position_size": t.suggested_position_size,

        "entry_time": t.entry_time,
        "entry_price": t.entry_price,
        "exit_time": t.exit_time,
        "exit_price": t.exit_price,
        "quantity": t.quantity,

        "outcome": t.outcome,
        "profit_loss": t.profit_loss,
        "profit_percent": t.profit_percent,
        "trading_notes": t.trading_notes,
        "review_notes": t.review_notes,

        "created_at": t.created_at,
        "updated_at": t.updated_at,
    }
@router.patch("/trades/{trade_id}")
def update_trade(
    trade_id: str,
    req: TradeUpdateRequest,
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _ = pair
    svc = TradeService(db)

    try:
        updated = svc.update_trade(
            user_id=str(user.user_id),
            trade_id=trade_id,
            data=req.model_dump(exclude_none=True),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "trade_id": str(updated.trade_id),
        "outcome": updated.outcome,
        "message": "Trade updated successfully"
    }

@router.delete("/trades/{trade_id}")
def delete_trade(
    trade_id: str = Path(...),
    pair = Depends(get_current_user_and_session),
    db = Depends(get_db)
):
    user, _ = pair
    svc = TradeService(db)
    return svc.delete_trade(trade_id=trade_id, user_id=str(user.user_id))
