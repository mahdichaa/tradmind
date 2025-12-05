from __future__ import annotations
from typing import Literal, Optional
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field

class TradeCreateManualRequest(BaseModel):
    # Linkage (optional)
    analysis_id: Optional[str] = Field(
        None, description="Optional analysis UUID; leave null for a pure manual trade."
    )

    # Meta
    source: Optional[Literal["MANUAL", "AI"]] = Field(
        "MANUAL", description="Defaults to MANUAL for this endpoint."
    )
    trade_date: Optional[datetime] = Field(
        None, description="Defaults to now if omitted."
    )

    # Instrument + execution
    symbol: str = Field(..., description="Instrument symbol, e.g. EURUSD")
    trade_type: Optional[Literal["LONG", "SHORT"]] = Field(
        None, description="Executed direction if known (LONG or SHORT)."
    )
    # Also accept explicit direction for clarity; backend stores this in suggested_direction
    suggested_direction: Optional[Literal["LONG", "SHORT"]] = Field(
        None, description="Position side LONG/SHORT; stored as suggested_direction."
    )

    # Executed fields (MANUAL focus)
    entry_time: Optional[datetime] = None
    entry_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None

    # Outcome & notes
    outcome: Optional[str] = Field(
        "NOT_TAKEN", description="Outcome label; default NOT_TAKEN."
    )
    profit_loss: Optional[Decimal] = None
    profit_percent: Optional[Decimal] = None
    trading_notes: Optional[str] = None
    review_notes: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": None,
                "source": "MANUAL",
                "trade_date": "2025-11-09T10:30:00Z",
                "symbol": "EURUSD",
                "trade_type": "LONG",
                "entry_time": "2025-11-09T10:31:00Z",
                "entry_price": 1.0716,
                "exit_time": "2025-11-09T12:00:00Z",
                "exit_price": 1.0742,
                "quantity": 1000,
                "outcome": "NOT_TAKEN",
                "profit_loss": 2.60,
                "profit_percent": 0.24,
                "trading_notes": "Manual trade.",
                "review_notes": None
            }
        }

class TradeUpdateRequest(BaseModel):
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: Optional[float] = None
    profit_loss: Optional[float] = None
    profit_percent: Optional[float] = None
    trading_notes: Optional[str] = None
    review_notes: Optional[str] = None
    outcome: Optional[str] = None
    suggested_direction: Optional[Literal["LONG","SHORT"]] = None
