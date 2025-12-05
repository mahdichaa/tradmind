# app/services/chart_analysis_service.py
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
import os
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.repositories.chart_analysis import ChartAnalysisRepository
from app.models.chart_analysis import ChartAnalysis



def deep_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safe nested getter: deep_get(d, "ai_analysis.risk_management.entry_price").
    Returns `default` if any level is missing or not a dict.
    """
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def to_decimal(value: Any) -> Optional[Decimal]:
    """Convert numbers to Decimal safely for Numeric columns; return None if invalid."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None

class ChartAnalysisService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ChartAnalysisRepository(db)

    
    def start_analysis(
        self,
        *,
        user_id: str,
        symbol: Optional[str],
        timeframe: str,
        chart_image_url: str,
        user_inputs: Dict[str, Any],
        ocr_vendor: Optional[str] = None,
        ocr_text: Optional[Dict[str, Any]] = None,
        chart_image_data: Optional[str] = None,
    ) -> ChartAnalysis:
        ai_request = {
            "user_inputs": user_inputs,
            "notes": "gemini unified request",
        }
        return self.repo.create_pending(
            user_id=user_id,
            symbol=symbol,
            timeframe=timeframe,
            chart_image_url=chart_image_url,
            ai_model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            ai_request=ai_request,
            ocr_vendor=ocr_vendor,
            ocr_text=ocr_text,
            chart_image_data=chart_image_data,
        )

    # def complete_analysis(self, analysis: ChartAnalysis, *, gemini_result: Dict[str, Any]) -> ChartAnalysis:
    #     """
    #     Mark analysis as COMPLETED and persist both the unified JSON and extraction subset.
    #     (This is a thin wrapper; for fine-grained control use write_result().)
    #     """
    #     extraction = gemini_result.get("extraction") or {}
    #     updated = self.repo.mark_completed(analysis, gemini_result, extraction)
    #     self.db.commit()
    #     return updated

    def fail_analysis(self, analysis: ChartAnalysis, error: str) -> None:
        """
        Mark analysis as FAILED and persist the error message so admins can review it.
        """
        # Persist failure instead of hard-deleting for auditability
        self.repo.patch_by_id(analysis.analysis_id, {
            "status": "FAILED",
            "error_message": (error or "unknown error")[:500],
        })
        self.db.commit()

    def write_result(self, analysis: ChartAnalysis, gemini_json: Dict[str, Any]) -> ChartAnalysis:
        """
        Persist the unified JSON and mapped summary fields, and set status=COMPLETED.
        Delegates to the repository to keep a single source of truth.
        """
        extraction = gemini_json.get("extraction") or {}
        updated = self.repo.mark_completed(analysis, gemini_json, extraction)
        self.db.flush()  # leave commit to the caller/endpoint
        return updated
    
    def list_filtered(
         self,
         *,
         user_id: str,
         status: Optional[str],
         date_from: Optional[datetime],
         date_to: Optional[datetime],
         limit: int,
         offset: int,
         trading_type: Optional[str] = None,
         outcome: Optional[str] = None,
     ) -> List[ChartAnalysis]:
         return self.repo.list_filtered(
             user_id=user_id,
             status=status,
             date_from=date_from,
             date_to=date_to,
             limit=limit,
             offset=offset,
             trading_type=trading_type,
             outcome=outcome,
         )

    def list_filtered_any(
         self,
         *,
         status: Optional[str],
         date_from: Optional[datetime],
         date_to: Optional[datetime],
         limit: int,
         offset: int,
         user_id: Optional[str] = None,
         email_substr: Optional[str] = None,
         trading_type: Optional[str] = None,
         outcome: Optional[str] = None,
     ) -> List[ChartAnalysis]:
         return self.repo.list_filtered_any(
             status=status,
             date_from=date_from,
             date_to=date_to,
             limit=limit,
             offset=offset,
             user_id=user_id,
             email_substr=email_substr,
             trading_type=trading_type,
             outcome=outcome,
         )

    def get_owned_required(self, *, user_id: str, analysis_id) -> ChartAnalysis:
        obj = self.repo.get_owned(user_id=user_id, analysis_id=analysis_id)
        if not obj:
            raise ValueError("analysis not found")
        return obj

    def delete_owned_by_id(self, *, user_id: str, analysis_id) -> int:
        rows = self.repo.delete_owned_by_id(user_id=user_id, analysis_id=analysis_id)
        self.db.commit()
        return rows
