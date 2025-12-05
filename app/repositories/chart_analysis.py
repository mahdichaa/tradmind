# app/repositories/chart_analysis_repo.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSONB
from app.repositories.base import BaseRepository
from app.models.chart_analysis import ChartAnalysis
from app.models.user import User
from app.models.trade import Trade

def _to_decimal(v):
    if v is None: return None
    try: return Decimal(str(v))
    except Exception: return None

class ChartAnalysisRepository(BaseRepository[ChartAnalysis]):
    def __init__(self, session: Session):
        super().__init__(session, ChartAnalysis)

    @staticmethod
    def _clamp_text(value, limit: int):
        if value is None:
            return None
        s = str(value)
        return s if len(s) <= limit else s[:limit]

    # create PENDING using BaseRepository.create(commit=False)
    def create_pending(
        self,
        *,
        user_id,
        symbol: Optional[str],
        timeframe: Optional[str],
        chart_image_url: str = "",
        ai_model: str,
        ai_request: Dict[str, Any],
        ocr_vendor: Optional[str] = None,
        ocr_text: Optional[Dict[str, Any]] = None,
        chart_image_data: Optional[str] = None,
    ) -> ChartAnalysis:
        return self.create({
            "user_id": user_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "chart_image_url": chart_image_url or "",
            "chart_image_data": chart_image_data,
            "ocr_vendor": ocr_vendor,
            "ocr_text": ocr_text,
            "ai_model": ai_model,
            "ai_request": ai_request,
            "status": "PENDING",
        }, commit=False)

    def get_by_id(self, analysis_id) -> Optional[ChartAnalysis]:
        return self.session.get(ChartAnalysis, analysis_id)

    # generic partial update by id
    def patch_by_id(self, analysis_id, fields: Dict[str, Any]) -> Optional[ChartAnalysis]:
        return self.update_one(where={"analysis_id": analysis_id}, data=fields, commit=False)

    # mark completed using BaseRepository.update(where=obj, data=...)
    def mark_completed(self, obj: ChartAnalysis, result: dict, extractable: dict) -> ChartAnalysis:
        ai   = (result.get("ai_analysis") or {})
        tech = (ai.get("technical_analysis") or {})
        rm   = (ai.get("risk_management") or {})
        sug  = (ai.get("trade_suggestion") or {})

        raw_signal = (sug.get("signal") or "").upper()
        if "BUY" in raw_signal or "LONG" in raw_signal:
            direction = "LONG"
        elif "SELL" in raw_signal or "SHORT" in raw_signal:
            direction = "SHORT"
        else:
            direction = None

        fields = {
            "status": "COMPLETED",
            "ocr_text": extractable,                  # ← exactly your column
            "ai_response": ai,                        # ← exactly your column

            # clamp text fields to DB column sizes to avoid StringDataRightTruncation
            "direction": self._clamp_text(direction, 50),
            "market_trend": self._clamp_text(ai.get("trend_direction"), 20),
            "pattern": self._clamp_text(tech.get("pattern_detected"), 100),

            "confidence_score": _to_decimal(ai.get("confidence_score")),
            "insights_json": ai.get("ai_insights") or [],

            "suggested_entry_price":   _to_decimal(rm.get("entry_price")),
            "suggested_stop_loss":     _to_decimal(rm.get("stop_loss")),
            "suggested_take_profit":   _to_decimal(rm.get("take_profit")),
            "suggested_risk_reward":   _to_decimal(rm.get("reward_risk_ratio")),
            "suggested_position_size": _to_decimal(rm.get("position_size")),
            #fundamentals_news
            #final_recommendation
            #"per_strategy"
            "error_message": None,
        }
        return self.update(where=obj, data=fields, commit=False)

    def list_filtered(
        self,
        *,
        user_id: str,
        status: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
        trading_type: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> List[ChartAnalysis]:
        q = (
            self.session.query(self.model)
            .outerjoin(Trade, Trade.analysis_id == self.model.analysis_id)
            .filter(self.model.user_id == user_id)
        )

        if status:
            q = q.filter(self.model.status == status)
        if date_from:
            q = q.filter(self.model.created_at >= date_from)
        if date_to:
            q = q.filter(self.model.created_at < date_to)
        if trading_type:
            try:
                q = q.filter(self.model.ai_request[('user_inputs')][('trading_type')].astext == trading_type)
            except Exception:
                pass
        if outcome:
            q = q.filter(Trade.outcome == outcome)

        q = q.order_by(self.model.created_at.desc())
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        return q.all()

    def list_filtered_any(
        self,
        *,
        status: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        email_substr: Optional[str] = None,
        trading_type: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> List[ChartAnalysis]:
        q = self.session.query(self.model).outerjoin(Trade, Trade.analysis_id == self.model.analysis_id)
        if user_id:
            q = q.filter(self.model.user_id == user_id)
        if email_substr:
            q = q.join(User, self.model.user_id == User.user_id).filter(User.email.ilike(f"%{email_substr}%"))
        if status:
            q = q.filter(self.model.status == status)
        if date_from:
            q = q.filter(self.model.created_at >= date_from)
        if date_to:
            q = q.filter(self.model.created_at < date_to)
        if trading_type:
            try:
                q = q.filter(self.model.ai_request[('user_inputs')][('trading_type')].astext == trading_type)
            except Exception:
                pass
        if outcome:
            q = q.filter(Trade.outcome == outcome)
        q = q.order_by(self.model.created_at.desc())
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        return q.all()

    def get_owned(self, *, user_id: str, analysis_id) -> Optional[ChartAnalysis]:
        return (
            self.session.query(self.model)
            .filter(self.model.analysis_id == analysis_id, self.model.user_id == user_id)
            .one_or_none()
        )

    def delete_owned_by_id(self, *, user_id: str, analysis_id) -> int:
        # hard delete using WhereExpr (safe with your BaseRepository)
        return self.delete(where={"analysis_id": analysis_id, "user_id": user_id}, commit=False)
