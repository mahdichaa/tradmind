from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Any, Dict
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.repositories.base import BaseRepository
from app.models.trade import Trade
from app.models.chart_analysis import ChartAnalysis

def _to_decimal(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None

class TradeRepository(BaseRepository[Trade]):
    def __init__(self, session: Session):
        super().__init__(session, Trade)

    @staticmethod
    def _clamp_text(value: Optional[str], limit: int) -> Optional[str]:
        if value is None:
            return None
        s = str(value)
        return s if len(s) <= limit else s[:limit]

    def get_by_analysis_id(self, analysis_id) -> Optional[Trade]:
        return self.find_one(where={"analysis_id": analysis_id})

    def create_from_analysis_snapshot(self, analysis: ChartAnalysis) -> Trade:
        """
        Create a SUGGESTED trade from the populated analysis snapshot.
        If it already exists (one-to-one), return it.
        """
        existing = self.get_by_analysis_id(analysis.analysis_id)
        if existing:
            return existing

        ai_inputs = analysis.ai_request.get("user_inputs", {}) if analysis.ai_request else {}
        ai_risk = analysis.ai_response.get("risk_management", {}) if analysis.ai_response else {}
        final_rec = analysis.ai_response.get("final_recommendation", {}) if analysis.ai_response else {}
        fundamentals = analysis.ai_response.get("fundamentals_news", {}) if analysis.ai_response else {}

        # Ensure non-null, properly sized symbol to satisfy NOT NULL constraint
        symbol_value = analysis.symbol or ai_inputs.get("symbol") or "UNKNOWN"
        symbol_value = self._clamp_text(symbol_value, 20)

        # Fallbacks for executed-like fields (kept optional)
        entry_price = _to_decimal(ai_risk.get("entry_price")) or _to_decimal(final_rec.get("entry")) or _to_decimal(fundamentals.get("current_price"))
        position_size = _to_decimal(ai_risk.get("position_size"))
        if position_size is None:
            # compute simple estimate if we have balance/risk% and stop_loss
            try:
                balance = Decimal(str(ai_inputs.get("account_balance")))
                risk_pct = Decimal(str(ai_inputs.get("risk_per_trade_percent")))
                sl = _to_decimal(ai_risk.get("stop_loss")) or _to_decimal(final_rec.get("stop_loss"))
                if entry_price is not None and sl is not None and balance is not None and risk_pct is not None:
                    risk_amount = balance * (risk_pct / Decimal("100"))
                    dist = abs(Decimal(entry_price) - Decimal(sl))
                    if dist > 0:
                        position_size = risk_amount / dist
            except Exception:
                position_size = None

        trade = self.create({
                "user_id": analysis.user_id,
                "analysis_id": analysis.analysis_id,
                "symbol":  symbol_value,
                "source": "AI",
                "outcome": "SUGGESTED",

                # 🔹 Suggested snapshot from analysis
                "suggested_direction": self._clamp_text(analysis.direction, 20),
                "suggested_entry_price": _to_decimal(analysis.suggested_entry_price) or entry_price,
                "suggested_stop_loss": _to_decimal(analysis.suggested_stop_loss) or _to_decimal(ai_risk.get("stop_loss")) or _to_decimal(final_rec.get("stop_loss")),
                "suggested_take_profit": _to_decimal(analysis.suggested_take_profit) or _to_decimal(ai_risk.get("take_profit")),
                "suggested_risk_reward": _to_decimal(analysis.suggested_risk_reward),
                "suggested_position_size": _to_decimal(analysis.suggested_position_size) or position_size,

                # 🔹 NEW extracted
                "trade_type": ai_inputs.get("trading_type"),  # ✅ from user inputs (Swing, Scalping...)
                
                # 🔹 From AI risk mgmt (executed fields)
                "entry_price": entry_price,
                "quantity": position_size,

                # 🔹 Initially empty
                "entry_time": None,
                "exit_time": None,
                "exit_price": None,
                "profit_loss": None,
                "profit_percent": None,

                # 🔹 AI Notes
                "trading_notes": None,
                "review_notes": None,


            # Executed fields remain NULL intentionally
        }, commit=False)

        return trade
        
    def get_owned(self, *, user_id: str, trade_id) -> Optional[Trade]:
        return (
            self.session.query(self.model)
            .filter(self.model.trade_id == trade_id, self.model.user_id == user_id)
            .one_or_none()
        )

    def list_owned(
        self,
        *,
        user_id: str,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        outcome: Optional[str] = None,
        analysis_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Trade]:
        q = self.session.query(self.model).filter(self.model.user_id == user_id)

        if analysis_id:
            q = q.filter(self.model.analysis_id == analysis_id)
        if symbol:
            # Case-insensitive contains match on symbol for flexible search
            q = q.filter(self.model.symbol.ilike(f"%{str(symbol)}%"))
        if source:
            q = q.filter(func.upper(self.model.source) == str(source).upper())
        if outcome and outcome != "All":
            q = q.filter(func.upper(self.model.outcome) == str(outcome).upper())
        if date_from:
            q = q.filter(self.model.trade_date >= date_from)
        if date_to:
            q = q.filter(self.model.trade_date < date_to)

        q = q.order_by(self.model.trade_date.desc(), self.model.created_at.desc())
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        return q.all()

    # app/repositories/trade_repository.py
    def create_manual(self, *, user_id: str, payload: Dict[str, Any]) -> Trade:
        analysis_id = payload.get("analysis_id")
        if isinstance(analysis_id, str) and analysis_id.strip() == "":
            analysis_id = None

        # Extract raw fields
        entry_time = payload.get("entry_time")
        entry_price = _to_decimal(payload.get("entry_price"))
        exit_time = payload.get("exit_time")
        exit_price = _to_decimal(payload.get("exit_price"))
        quantity = _to_decimal(payload.get("quantity"))
        profit_loss = _to_decimal(payload.get("profit_loss"))
        profit_percent = _to_decimal(payload.get("profit_percent"))

        # ------------------------------------------------------------
        # AUTO-DETERMINE OUTCOME
        # ------------------------------------------------------------
        has_entry = bool(entry_price) and bool(quantity)
        has_profit = bool(exit_price) and bool(profit_loss)

        outcome = "NOT_TAKEN"
        if has_entry and not has_profit:
            outcome = "PENDING"
        elif has_entry and has_profit:
            outcome = "WIN" if (profit_loss and profit_loss > 0) else "LOSS"

        # Common fields payload
        # Mode: database enum expects SWING|SCALP|BOTH. Accept either payload.mode or trade_type for compatibility.
        raw_mode = (payload.get("mode") or payload.get("trade_type") or "").upper()
        allowed_modes = {"SWING", "SCALP", "BOTH"}
        db_trade_type = raw_mode if raw_mode in allowed_modes else None

        data: Dict[str, Any] = {
            "user_id": user_id,
            "analysis_id": analysis_id,
            "source": "MANUAL",
            "trade_date": payload.get("trade_date"),
            "symbol": self._clamp_text(payload.get("symbol") or "UNKNOWN", 20),
            "trade_type": db_trade_type,

            # Executed fields
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "quantity": quantity,

            # Outcome (computed)
            "outcome": outcome,

            # Profit & notes
            "profit_loss": profit_loss,
            "profit_percent": profit_percent,
            "trading_notes": payload.get("trading_notes"),
            "review_notes": payload.get("review_notes"),

            # For manual trades, allow saving position side as suggested_direction (LONG/SHORT)
            "suggested_direction": (str(payload.get("suggested_direction")).upper() if payload.get("suggested_direction") else (
                str(payload.get("trade_type")).upper() if payload.get("trade_type") in ["LONG", "SHORT"] else None
            )),
            "suggested_entry_price": None,
            "suggested_stop_loss": None,
            "suggested_take_profit": None,
            "suggested_risk_reward": None,
            "suggested_position_size": None,
        }

        # If an analysis_id is provided and already has a trade, update it instead of violating UNIQUE
        if analysis_id:
            existing = self.get_by_analysis_id(analysis_id)
            if existing:
                # Ensure ownership
                if str(existing.user_id) != str(user_id):
                    # Different owner → better to reject; caller should not link foreign analysis
                    from fastapi import HTTPException
                    raise HTTPException(status_code=403, detail="Analysis belongs to a different user")
                updated = self.update(where=existing, data=data, commit=False)
                return updated

        # Otherwise create a new manual trade
        trade = self.create(data, commit=False)
        return trade
