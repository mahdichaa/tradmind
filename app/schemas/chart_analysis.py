
from typing import List
from pydantic import BaseModel, Field










class UserInputs(BaseModel):
    trading_type: str = Field(..., description="Trading style: Swing, Scalp, Intraday, Day Trading, Position")
    account_balance: float = Field(..., description="Account Balance in USD", gt=0)
    risk_per_trade_percent: float = Field(..., description="Risk Per Trade in %", gt=0, le=100)
    stop_loss_points: float = Field(..., description="Stop Loss in Points", gt=0)
    take_profit_points: float = Field(..., description="Take Profit in Points", gt=0)
    chart_timeframe: str = Field(..., description="Chart timeframe: 1-Min, 5-Min, 15-Min, 1-Hour, 4-Hour, Daily, Weekly")


class TechnicalAnalysis(BaseModel):
    support_level: float = Field(..., description="AI-estimated support level")
    resistance_level: float = Field(..., description="AI-estimated resistance level")
    price_range: List[float] = Field(..., description="Overall visible price range [min, max]")
    key_zone: List[float] = Field(..., description="Key decision zone for entry/rejection")
    pattern_detected: str = Field(..., description="Recognized chart pattern")
    signal_strength: float = Field(..., description="Confidence (0-1) of pattern detection", ge=0, le=1)


class RiskManagement(BaseModel):
    risk_amount_usd: float = Field(..., description="Amount of money at risk per trade")
    entry_price: float = Field(..., description="Entry price (latest visible price)")
    stop_loss: float = Field(..., description="Stop-loss price level")
    take_profit: float = Field(..., description="Take-profit price level")
    reward_risk_ratio: float = Field(..., description="Reward/Risk ratio")
    position_size: float = Field(..., description="Calculated position size")


class TradeSuggestion(BaseModel):
    signal: str = Field(..., description="Trade direction: BUY, SELL, or NEUTRAL")
    entry_recommendation: str = Field(..., description="Entry strategy recommendation")
    stop_loss_recommendation: str = Field(..., description="Suggested stop-loss placement")
    take_profit_recommendation: str = Field(..., description="Suggested take-profit levels")
    ai_confidence: float = Field(..., description="Confidence level (0-1)", ge=0, le=1)


class AIAnalysis(BaseModel):
    trend_summary: str = Field(..., description="AI-generated summary of chart")
    trend_direction: str = Field(..., description="Overall trend classification")
    volatility: str = Field(..., description="Volatility level: Low, Medium, High")
    momentum: str = Field(..., description="Momentum strength")
    market_sentiment: str = Field(..., description="Overall market sentiment")
    technical_analysis: TechnicalAnalysis
    risk_management: RiskManagement
    trade_suggestion: TradeSuggestion
    ai_insights: List[str] = Field(..., description="List of AI-generated insights")
    risk_label: str = Field(..., description="Risk label: Low Risk, Moderate Risk, High Risk")
    confidence_score: float = Field(..., description="Global confidence (0-1)", ge=0, le=1)


class FinalRecommendation(BaseModel):
    decision: str = Field(..., description="Final decision: BUY, SELL, or HOLD")
    summary: str = Field(..., description="Reasoning behind the decision")
    alert: str = Field(..., description="Alert or tip before trade execution")


class ChartAnalysisResponse(BaseModel):
    user_inputs: UserInputs
    ai_analysis: AIAnalysis
    final_recommendation: FinalRecommendation


class OCRData(BaseModel):
    extracted_text: str
    price_levels: List[float]
    time_labels: List[str]