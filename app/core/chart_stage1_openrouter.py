# app/core/chart_stage1_openrouter.py
"""
OpenRouter-based chart analysis functions.
Replaces Gemini-based functions with OpenRouter SDK.
"""
import json
import os
import re
from typing import Dict, Any
from fastapi import HTTPException
import cv2
import numpy as np
import pytesseract
import time
import random

from app.schemas.chart_analysis import OCRData, RiskManagement, UserInputs
from app.core.openrouter_client import (
    get_openrouter_client,
    quick_inspect_with_retry,
    analyze_with_retry
)
from app.core.price_rules import classify_symbol
from app.repositories.ai_config import AIConfigRepository
from app.database.session import get_db


def extract_text_with_ocr(image_bytes: bytes) -> OCRData:
    """
    Extract text, price levels, and time labels from chart using Tesseract OCR.
    (Same as before - no changes needed)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image for OCR")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    
    price_levels = []
    time_labels = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            price = float(line.replace(',', ''))
            if 100 < price < 1000000:
                price_levels.append(price)
        except ValueError:
            pass
        
        if any(char.isdigit() and ':' in line for char in line):
            time_labels.append(line)
    
    return OCRData(
        extracted_text=text,
        price_levels=sorted(set(price_levels)),
        time_labels=time_labels[:20]
    )


def preprocess_image(image_bytes: bytes) -> tuple[bytes, np.ndarray]:
    """
    Preprocess for speed: resize to <=1280px long side and compress JPEG (quality=70).
    Returns processed JPEG bytes and original color image.
    (Same as before - no changes needed)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > 1280:
        scale = 1280.0 / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img

    success, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not success:
        raise ValueError("Failed to encode image")

    processed_bytes = buffer.tobytes()

    return processed_bytes, img


def calculate_risk_management(user_inputs: UserInputs, entry_price: float, signal: str) -> RiskManagement:
    """
    Calculate risk management metrics based on user inputs, entry price, and trade direction.
    (Same as before - no changes needed)
    """
    risk_amount_usd = user_inputs.account_balance * (user_inputs.risk_per_trade_percent / 100)
    
    if signal.upper() == "BUY":
        stop_loss_price = entry_price - user_inputs.stop_loss_points
        take_profit_price = entry_price + user_inputs.take_profit_points
    else:
        stop_loss_price = entry_price + user_inputs.stop_loss_points
        take_profit_price = entry_price - user_inputs.take_profit_points
    
    risk_distance = abs(entry_price - stop_loss_price)
    reward_distance = abs(entry_price - take_profit_price)
    
    reward_risk_ratio = round(reward_distance / risk_distance, 2) if risk_distance > 0 else 0.0
    
    position_size = round(risk_amount_usd / risk_distance, 4) if risk_distance > 0 else 0.0
    
    return RiskManagement(
        risk_amount_usd=round(risk_amount_usd, 2),
        entry_price=entry_price,
        stop_loss=round(stop_loss_price, 2),
        take_profit=round(take_profit_price, 2),
        reward_risk_ratio=reward_risk_ratio,
        position_size=position_size
    )


def quick_inspect_with_openrouter(
    image_bytes: bytes,
    ocr_text: str,
    api_key: str = None,
    api_keys: list = None,
    model: str = None
) -> tuple[Dict[str, Any], int]:
    """
    Fast AI-driven quick-inspect using OpenRouter with automatic retry.
    Returns tuple of (result dict, successful_key_index).
    
    Args:
        image_bytes: Image data
        ocr_text: OCR text hint
        api_key: Single API key (legacy, for backward compatibility)
        api_keys: List of API keys for retry logic
        model: Model ID to use
    
    Returns:
        Tuple of (result dict, key index used)
    """
    try:
        # Use multi-key retry if available, otherwise fall back to single key
        if api_keys:
            result, key_index = quick_inspect_with_retry(
                image_bytes=image_bytes,
                api_keys=api_keys,
                ocr_text=ocr_text,
                model=model
            )
            return result, key_index
        elif api_key:
            # Legacy single-key path
            client = get_openrouter_client(api_key)
            result = client.quick_inspect(image_bytes, ocr_text, model=model)
            return result, 0
        else:
            raise ValueError("Either api_key or api_keys must be provided")
    except Exception as e:
        print(f"[OpenRouter] Quick inspect failed: {e}")
        raise HTTPException(status_code=503, detail=f"Quick-inspect AI unavailable: {str(e)}")


def analyze_chart_with_openrouter(
    image_bytes: bytes,
    user_inputs: UserInputs,
    ocr_data: OCRData,
    api_key: str = None,
    api_keys: list = None,
    model: str = None
) -> tuple[Dict[str, Any], int]:
    """
    One-shot extraction + analysis with OpenRouter with automatic retry.
    Returns tuple of (result dict, successful_key_index).
    
    Args:
        image_bytes: Image data
        user_inputs: User trading inputs
        ocr_data: OCR extracted data
        api_key: Single API key (legacy, for backward compatibility)
        api_keys: List of API keys for retry logic
        model: Model ID to use
    
    Returns:
        Tuple of (result dict, key index used)
    """
    # Prepare optional fundamentals context string if provided by caller
    fundamentals_summary = ""
    try:
        _fs = (user_inputs or {}).get("_fundamentals_summary")
        if isinstance(_fs, str) and _fs.strip():
            fundamentals_summary = _fs.strip()
    except Exception:
        fundamentals_summary = ""

    # Resolve price decimals and point value for prompt and rounding
    try:
        dec = int((user_inputs or {}).get("_price_decimals") or 2)
        pv = float((user_inputs or {}).get("_point_value") or 1.0)
        if not (user_inputs or {}).get("_asset_type_resolved"):
            meta = classify_symbol((user_inputs or {}).get("symbol"), (user_inputs or {}).get("asset_type"))
            if isinstance(meta, dict):
                dec = int(meta.get("decimals") or dec)
                pv = float(meta.get("point_value") or pv)
    except Exception:
        dec = 2
        pv = 1.0

    # Build comprehensive prompt for chart analysis
    head = f"""
You are an advanced multi-strategy trading analysis engine.

You will receive: (1) a chart image, (2) user inputs, (3) OCR hints.
Your job is to:
  • Apply 6–10 strategies (Price Action/SR, Supply–Demand, Chart Patterns, Fibonacci, Gann-style balance, MAs/Trend filters,
    Oscillators/Divergences, Volume/Volatility, Multi-timeframe context, and a Fundamentals/News layer).
  • For each strategy: explain what you see, give a bias (bullish/bearish/neutral), and you MUST suggest a concrete entry/SL/TP (no nulls),
    with a confidence score 0–100%.
  • Combine all strategies into ONE consolidated trade idea or say "No Trade" if confluence is weak.
  • ALWAYS include a risk disclaimer at the end.
  • Before responding, quickly check the latest market context for the symbol (price & news). THIS IS SO IMPORTANT YOU NEED TO HAVE SOME INFORMATION ABOUT THE MARKET.

# INSTRUMENT CONTEXT
- symbol: {user_inputs.get('symbol', 'unknown')}
- asset_type: {user_inputs.get('asset_type', 'unknown')}
- chart_timeframe: {user_inputs.get('chart_timeframe')}
- trading_mode: {user_inputs.get('trading_type')}
- risk_preference: {user_inputs.get('risk_profile', 'normal')}

# PRICE SCALE (critical)
- decimals: {dec}
- point_value_per_point: {pv}
- expected_price_range: {user_inputs.get('_price_range_hint')}

# USER INPUTS (for risk math)
- account_balance: {user_inputs.get('account_balance')}
- risk_per_trade_percent: {user_inputs.get('risk_per_trade_percent')}
- stop_loss_points: {user_inputs.get('stop_loss_points')}
- take_profit_points: {user_inputs.get('take_profit_points')}

# OCR HINTS (optional)
- detected_price_levels: {ocr_data.get('price_levels')}
- time_labels: {ocr_data.get('time_labels')}

# FUNDAMENTALS CONTEXT (from API)
{fundamentals_summary}

# METADATA RULES
- metadata.source must be the platform/watermark if visible (tradingview, metatrader, binance, coinbase, bybit, kucoin, investing.com, yahoo finance, bloomberg, coinmarketcap, coingecko). If unknown, set "".
- metadata.chart_style is the chart TYPE (e.g., "candlestick", "line", "bar", "heikin ashi"). If uncertain, default to "candlestick".
- metadata.timeframe should match the on-chart timeframe or fallback to user_inputs.chart_timeframe (e.g., "M1","H1","D1").
- metadata.current_price must be numeric; derive from visible labels or OCR price levels when necessary.

# TECHNICAL OUTPUT REQUIREMENTS (must populate)
- ai_analysis.technical_analysis.support_level: number (key support)
- ai_analysis.technical_analysis.resistance_level: number (key resistance)
- ai_analysis.technical_analysis.price_range: [low, high] ALWAYS present
- ai_analysis.technical_analysis.key_zone: [low, high] ALWAYS present (same as price_range when no better refinement)
- ai_analysis.technical_analysis.pattern_detected: concise Title Case label
- ai_analysis.technical_analysis.patterns: array of strings
- ai_analysis.technical_analysis.pattern_explanation: short text
- Add "Pattern: <name>" into ai_analysis.key_factors when a pattern is detected
Fallback logic:
- If exact levels are not visible, infer low/high from the visible extremes or OCR price_levels (min/max).
- If only a consolidation/range is visible, set price_range/key_zone accordingly and label the pattern appropriately (e.g., "Rectangle/Range" or "Symmetrical Triangle").

# PATTERN DETECTION (prefer a concrete label over 'None')
Use the following taxonomy and pick the closest visible structure. Avoid "No Pattern Detected" unless absolutely none apply after careful inspection.
a) Reversal
   - Bullish: Double Bottom, Inverse Head and Shoulders, Falling Wedge, Morning Star, Hammer, Bullish Engulfing
   - Bearish: Double Top, Head and Shoulders, Rising Wedge, Evening Star, Shooting Star, Bearish Engulfing
b) Continuation
   - Bull Flag, Bear Flag, Pennant (bull/bear), Ascending Triangle (bullish), Descending Triangle (bearish),
     Symmetrical Triangle (neutral), Cup and Handle (bullish)
c) Neutral/Indecision
   - Symmetrical Triangle, Rectangle/Range, Doji, Inside Bars, Spinning Top
d) Candlestick (single/multi)
   - Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star, Piercing Pattern, Dark Cloud Cover, Doji, Spinning Top
e) Market Structure (SMC/ICT)
   - BOS, CHOCH, Liquidity Sweep, Order Blocks, Fair Value Gaps (FVG)

# EXECUTION RULES
1) STRICTLY return ONE JSON object (no code fences, no extra prose).
2) Keep all numeric fields numeric; DO NOT leave entry/stop_loss/take_profit null in any section.
3) Provide both per-strategy analysis and a final_trade block (or "no_trade").
4) Include a fundamentals_news block with relevant headlines and events.
5) Always output ai_analysis.technical_analysis with support_level, resistance_level, price_range, key_zone, and pattern fields as specified above.
6) Round prices to {dec} decimals. Keep confidence in 0..100.
7) Trade Reasoning: Include final_recommendation.reasoning_markdown (and optionally ai_analysis.reasoning_markdown) as short, user-friendly Markdown with:
   - A one-line quote: "> Primary signal: BUY/SELL/NO_TRADE • Confidence XX%"
   - "## 1) Market Context" with 2–3 bullets (trend, sentiment, volatility)
   - "## 2) Technical Factors" bullets (Pattern, Support, Resistance, Key Zone, Range)
   - "## 3) Risk Management" bullets (Entry, Stop Loss, Take Profit, R:R)
   - "## 4) Action" one bullet explaining the plan or "No strong trade recommendation".
   Keep it concise (5–8 bullets total), readable, and based ONLY on extracted/derived numbers.
8) Decision policy and consistency:
   - Prefer a concrete trade: set trade_suggestion.signal to BUY or SELL when confidence is reasonable (>= 35 by default).
   - Use NO_TRADE only if confidence is low (< 35) or confluence is insufficient; when NO_TRADE, provide a conditional breakout plan and explain why (e.g., low confidence, range-bound).
   - final_recommendation.decision must match the signal: BUY -> enter_long, SELL -> enter_short, NO_TRADE -> no_trade.
   - Reasoning must reflect the final plan exactly (signal and levels), not a different scenario.
9) Level sanity:
   - Ensure coherent ordering: BUY => stop_loss < entry < take_profit; SELL => take_profit < entry < stop_loss.
   - Avoid non-positive or implausible levels; when user deltas yield invalid values, derive SL/TP from visible structure (support/resistance/key_zone) and adjust to nearest plausible levels, then recalculate R:R.
""".strip()

    prompt = (
        f"STRICT OUTPUT: Return only a single valid JSON object (no code fences, no comments). "
        f"Never leave entry/stop_loss/take_profit null. If exact values are not visible, infer them using user inputs and the risk math. "
        f"Round prices to {dec} decimals. Keep confidence in 0..100."
    )

    try:
        # Use multi-key retry if available, otherwise fall back to single key
        if api_keys:
            result, key_index = analyze_with_retry(
                image_bytes=image_bytes,
                prompt=prompt,
                api_keys=api_keys,
                model=model,
                system_instruction=head,
                temperature=0.2,
                max_tokens=4096
            )
            return result, key_index
        elif api_key:
            # Legacy single-key path
            client = get_openrouter_client(api_key)
            result = client.generate_content(
                image_bytes=image_bytes,
                prompt=prompt,
                system_instruction=head,
                model=model,
                temperature=0.2,
                max_tokens=4096
            )
            return result, 0
        else:
            raise ValueError("Either api_key or api_keys must be provided")
    except Exception as e:
        print(f"[OpenRouter] Chart analysis failed: {e}")
        raise HTTPException(status_code=503, detail=f"AI analysis unavailable: {str(e)}")
