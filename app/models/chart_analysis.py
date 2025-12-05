import uuid
from sqlalchemy import Column, String, DateTime, Numeric, Index, ForeignKey, text, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base

from app.models.enums import AnalysisStatus, TradeSide

class ChartAnalysis(Base):
    __tablename__ = "chart_analyses"
    __table_args__ = (
        Index("idx_chart_analyses_user", "user_id"),
        Index("idx_chart_status", "status"),
        Index("idx_chart_symbol_tf", "symbol", "timeframe"),
    )

    analysis_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())

    symbol = Column(String(20))
    timeframe = Column(String(20))
    chart_image_url = Column(String(500), nullable=False)
    # Base64 data URL of the uploaded chart image (e.g., "data:image/png;base64,....")
    chart_image_data = Column(Text, nullable=True)

    ocr_vendor = Column(String(50))
    ocr_text = Column(JSONB)
    ai_model = Column(String(100))
    ai_request = Column(JSONB)
    ai_response = Column(JSONB)
    status = Column(AnalysisStatus, nullable=False, server_default=text("'PENDING'"))
    error_message = Column(String)

    direction = Column(String(50))
    market_trend = Column(String(20))
    pattern = Column(String(100))
    confidence_score = Column(Numeric(5, 2))
    insights_json = Column(JSONB, server_default=text("'[]'::jsonb"))

    suggested_entry_price = Column(Numeric(18, 6))
    suggested_stop_loss = Column(Numeric(18, 6))
    suggested_take_profit = Column(Numeric(18, 6))
    suggested_risk_reward = Column(Numeric(10, 2))
    suggested_position_size = Column(Numeric(20, 8))

    # one-to-one (convention) to Trade; enforced by UNIQUE on trade.analysis_id
    trade = relationship("Trade", back_populates="analysis", uselist=False, lazy="selectin")
    user = relationship("User", back_populates="analyses", lazy="selectin")
