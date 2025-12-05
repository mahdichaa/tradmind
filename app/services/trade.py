from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.chart_analysis import ChartAnalysis
from app.models.trade import Trade
from app.repositories.trade import TradeRepository


FINAL_STATES = {"WIN", "LOSS"}

class TradeService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = TradeRepository(db)

    def create_from_analysis_snapshot(self, analysis: ChartAnalysis) -> Trade:
        trade = self.repo.create_from_analysis_snapshot(analysis)
        self.db.flush()
        return trade
    
    def create_manual(self, *, user_id: str, payload: Dict[str, Any]) -> Trade:
        t = self.repo.create_manual(user_id=user_id, payload=payload)
        self.db.flush()
        return t

    def list_owned(
        self,
        *,
        user_id: str,
        symbol: Optional[str],
        source: Optional[str],
        outcome: Optional[str],
        analysis_id: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
        limit: int,
        offset: int,
    ) -> List[Trade]:
        return self.repo.list_owned(
            user_id=user_id,
            symbol=symbol,
            source=source,
            outcome=outcome,
            analysis_id=analysis_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )

    def get_owned_required(self, *, user_id: str, trade_id) -> Trade:
        obj = self.repo.get_owned(user_id=user_id, trade_id=trade_id)
        if not obj:
            raise ValueError("trade not found")
        return obj

    def get_owned_required(self, user_id: str, trade_id: str) -> Trade:
        trade = self.repo.find_one(where={
            "trade_id": trade_id,
            "user_id": user_id
        })
        if not trade:
            raise ValueError("Trade not found or unauthorized")
        return trade

    # ------------------------------------------------------------
    # MAIN UPDATE LOGIC
    # ------------------------------------------------------------
    def update_trade(self, user_id: str, trade_id: str, data: dict):
        trade = self.get_owned_required(user_id, trade_id)

        # ------------------------------------------------------------
        # 1. Final trade (WIN/LOSE) → only notes allowed
        # ------------------------------------------------------------
        if trade.outcome in FINAL_STATES and str(getattr(trade, "source", "")).upper() != "MANUAL":
            # For AI/completed trades: allow only notes and outcome correction
            allowed = {"trading_notes", "review_notes", "outcome"}
            blocked_keys = [k for k in data.keys() if k not in allowed]

            if blocked_keys:
                raise ValueError(
                    f"Cannot modify completed trades (blocked: {blocked_keys})"
                )
            updated = self.repo.update(where=trade, data=data)
            self.db.commit()
            self.db.refresh(updated)
            return updated

        # ------------------------------------------------------------
        # 2. Update normal fields
        # ------------------------------------------------------------
        updated = self.repo.update(where=trade, data=data)

        # Reload the object (repo.update returns the instance)
        trade = updated

        # ------------------------------------------------------------
        # 3. Determine the correct outcome after update
        # ------------------------------------------------------------
        if "outcome" in data:
            # Respect explicit outcome change from client
            trade.outcome = data["outcome"]
        else:
            has_entry = trade.entry_time and trade.entry_price and trade.quantity
            has_exit = trade.exit_time and trade.exit_price
            has_profit = trade.profit_loss is not None or trade.profit_percent is not None

            # SUGGESTED or PENDING → PENDING if entry complete
            if has_entry and not has_exit:
                trade.outcome = "PENDING"

            # All fields completed → finalize outcome
            elif has_entry and has_exit and has_profit:
                trade.outcome = "WIN" if trade.profit_loss and trade.profit_loss > 0 else "LOSS"

        # ------------------------------------------------------------
        # 4. Save changes
        # ------------------------------------------------------------
        self.db.commit()
        self.db.refresh(trade)
        return trade 
    def delete_trade(self, trade_id: str, user_id: str):
        # Make sure the trade belongs to the user
        trade = self.repo.find_one(where={"trade_id": trade_id, "user_id": user_id})
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        self.repo.delete(trade, commit=True)
        return {"status": "success", "message": f"Trade {trade_id} deleted"}
