import uuid
from pydantic import field_validator
from sqlalchemy import Column, String, DateTime, Numeric, Index, ForeignKey, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base

from app.models.enums  import TradeOutcome, TradeSide, TradeSource

class Trade(Base):
    __tablename__ = "trade_journal"
    __table_args__ = (
        Index("idx_trade_user_date", "user_id", "trade_date"),
        Index("idx_trade_symbol_date", "symbol", "trade_date"),
        UniqueConstraint("analysis_id", name="uq_trade_analysis_id"),  # enforce 1-1 with analysis
    )

    trade_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)

    analysis_id = Column(UUID(as_uuid=True), ForeignKey("chart_analyses.analysis_id", ondelete="SET NULL"))

    source = Column(TradeSource, server_default=text("'AI'"))
    trade_date = Column(DateTime, server_default=func.now())

    symbol = Column(String(20), nullable=False)
    trade_type = Column(TradeSide)  # executed direction

    # Suggested snapshot from AI
    suggested_direction = Column(String(20))
    suggested_entry_price = Column(Numeric(18, 6))
    suggested_stop_loss = Column(Numeric(18, 6))
    suggested_take_profit = Column(Numeric(18, 6))
    suggested_risk_reward = Column(Numeric(10, 2))
    suggested_position_size = Column(Numeric(20, 8))

    # Executed fields
    entry_time = Column(DateTime)
    entry_price = Column(Numeric(18, 6))
    exit_time = Column(DateTime)
    exit_price = Column(Numeric(18, 6))
    quantity = Column(Numeric(20, 8))

    outcome = Column(TradeOutcome, server_default=text("'NOT_TAKEN'"))
    @field_validator("quantity", "profit_loss", "profit_percent", mode="before")
    def empty_string_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v
    profit_loss = Column(Numeric(18, 6))
    profit_percent = Column(Numeric(6, 2))
    trading_notes = Column(String)
    review_notes = Column(String)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())

    analysis = relationship("ChartAnalysis", back_populates="trade", uselist=False, lazy="selectin")
    user = relationship("User", back_populates="trades", lazy="selectin")
