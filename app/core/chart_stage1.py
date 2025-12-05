

import json
import os
import re
from typing import Dict, Any
from fastapi import HTTPException
import cv2
import numpy as np
from openai import OpenAI
import pytesseract
import time
import random

from app.schemas.chart_analysis import OCRData, RiskManagement, UserInputs
from google.genai import types
from google import genai
from app.core.ai_settings import AISettingsStore
from app.core.price_rules import classify_symbol

# Optional xAI config (not required for current path). Avoid KeyError and never hard-code secrets.
XAI_API_KEY = os.environ.get("XAI_API_KEY")
XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
GROK_MODEL  = os.environ.get("GROK_MODEL", "grok-4-fast-reasoning")

AI_TEMPERATURE = float(os.environ.get("AI_TEMPERATURE", "0.2"))
AI_TIMEOUT_SECONDS = int(os.environ.get("AI_TIMEOUT_SECONDS", "45"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # optional; runtime settings may override

# xAI client is created only if a key is configured; otherwise remains None.
grok_client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL) if XAI_API_KEY else None
# Gemini client will be created per-attempt using configured keys

def extract_text_with_ocr(image_bytes: bytes) -> OCRData:
    """
    Extract text, price levels, and time labels from chart using Tesseract OCR.
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
    
    For BUY (long): SL below entry, TP above entry
    For SELL (short): SL above entry, TP below entry
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


def quick_inspect_with_gemini(image_bytes: bytes, ocr_text: str) -> Dict[str, Any]:
    """
    Fast AI-driven quick-inspect using Gemini. Returns a compact JSON with:
      - symbol_guess           (exactly as printed on chart, preserve separators/case)
      - asset_type_guess       ("forex" | "crypto" | "index" | "stock" | "commodity" | "other")
      - source_guess           (e.g., TradingView, Binance, MetaTrader, etc.) or null
      - is_relevant            (true if the image is a trading chart/graph)
      - confidence             (0..100)
      - excerpt                (<= 120 chars short phrase)
    OCR text is provided only as optional hint; the model must prefer the visible on-chart text.
    """
    # Build a strong instruction for vision extraction (concise to reduce latency)
    head = """
You are a vision model that inspects a single image and returns ONLY a compact JSON summary.

IMPORTANT:
- Detect the primary instrument symbol EXACTLY AS PRINTED on the chart (preserve separators like "/", "-", ":" and original case).
- Do NOT guess a normalized symbol unless it is visibly printed; prefer on-image text over watermarks or side legends.
- Classify the asset type as one of: forex | crypto | index | stock | commodity | other.
- Detect the platform/source if a credible watermark/logo is visible (e.g., TradingView, Binance, Coinbase, Bybit, KuCoin, Investing.com, Yahoo Finance, Bloomberg, MetaTrader).
- Set is_relevant=true ONLY if this looks like a trading/chart/graph image (axes, candles/lines, OHLC, indicators, price/time scales, etc.)
- confidence: integer 0..100 for your extraction quality.
- excerpt: a short, readable single-line phrase (<= 120 chars), e.g. "BTC/USDT H1 crypto via TradingView".
- Return a single JSON object with keys exactly: symbol_guess, asset_type_guess, source_guess, is_relevant, confidence, excerpt.
- No code fences. No extra text.
""".strip()

    # Provide OCR hint text to help disambiguation but emphasize preference to visible on-chart symbol
    hint = (
        "OCR_HINT (optional, do not overfit; prefer on-chart text):\n"
        + (ocr_text or "")[:4000]
    )

    # Minimal response schema to keep SDK strictness low but still encourage structure
    schema_out = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "symbol_guess": types.Schema(type=types.Type.STRING),
            "asset_type_guess": types.Schema(type=types.Type.STRING),
            "source_guess": types.Schema(type=types.Type.STRING),
            "is_relevant": types.Schema(type=types.Type.BOOLEAN),
            "confidence": types.Schema(type=types.Type.NUMBER),
            "excerpt": types.Schema(type=types.Type.STRING),
        },
        required=["is_relevant", "confidence"]
    )

    settings = AISettingsStore().get()
    selected = settings.get("selected_model")
    chain = settings.get("model_chain") or []
    model_chain = ([selected] if selected else []) + [m for m in chain if m and m != selected]
    if not model_chain:
        models_env = os.environ.get(
            "GEMINI_MODELS",
            "gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash-lite",
        )
        model_chain = [m.strip() for m in models_env.split(",") if m.strip()]
    model_chain = [m for m in model_chain if m][:3]

    api_keys = (settings.get("api_keys") or ([GEMINI_API_KEY] if GEMINI_API_KEY else []))[:3]
    if not api_keys:
        raise HTTPException(status_code=500, detail="Gemini quick-inspect failed: No API keys configured")

    def _detect_mime(b: bytes) -> str:
        if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xD8:
            return "image/jpeg"
        if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    def _json_object_candidates(text: str) -> list[str]:
        spans = []
        stack = []
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(i)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start != -1:
                        spans.append(text[start:i+1])
                        start = -1
        spans.sort(key=len, reverse=True)
        return spans

    def _clean_and_parse(s: str) -> dict:
        x = s
        x = re.sub(r"```[a-zA-Z]*", "", x)
        x = x.replace("```", "")
        x = re.sub(r"//.*?$", "", x, flags=re.MULTILINE)
        x = re.sub(r"/\*.*?\*/", "", x, flags=re.DOTALL)
        x = re.sub(r",\s*(?=[}\]])", "", x)
        x = x.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
        x = re.sub(r"\bTrue\b", "true", x)
        x = re.sub(r"\bFalse\b", "false", x)
        x = re.sub(r"\bNone\b", "null", x)
        x = x.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        if '"' not in x and "'" in x:
            x = x.replace("'", '"')
        x = ''.join(ch for ch in x if ch.isprintable() or ch in ['\n','\r','\t'])
        x = re.sub(r",\s*(?=[}\]])", "", x)
        return json.loads(x)

    attempt_limit = int(os.environ.get("AI_MAX_ATTEMPTS", "4"))
    attempt_idx = 0
    last_err = None

    for key_index, api_key in enumerate(api_keys):
        for model_name in model_chain:
            if attempt_idx >= attempt_limit:
                break
            attempt_idx += 1
            try:
                client = genai.Client(api_key=api_key)
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type=_detect_mime(image_bytes)),
                            types.Part(text=head + "\n\n" + hint),
                        ],
                    )
                ]
                cfg_base = dict(response_mime_type="application/json", max_output_tokens=512, temperature=AI_TEMPERATURE)

                # Try without schema first (some SDKs fail hard on strict schema)
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**cfg_base),
                    )
                except Exception:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(response_schema=schema_out, **cfg_base),
                    )

                # Use parsed when available
                parsed = getattr(resp, "parsed", None)
                if isinstance(parsed, dict):
                    parsed["_model_used"] = model_name
                    parsed["_api_key_index"] = key_index
                    return parsed

                raw_text = getattr(resp, "text", None)
                if not raw_text and getattr(resp, "candidates", None):
                    # salvage any text from candidates
                    parts = []
                    try:
                        for c in resp.candidates:
                            for p in getattr(c, "content", {}).parts:
                                if getattr(p, "text", None):
                                    parts.append(p.text)
                    except Exception:
                        pass
                    raw_text = "".join(parts) if parts else None
                if not raw_text:
                    raise ValueError("Empty response")

                try:
                    result = json.loads(raw_text)
                except Exception:
                    # try to salvage JSON blocks
                    blocks = []
                    start = raw_text.find("{"); end = raw_text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        blocks.append(raw_text[start:end+1])
                    blocks.extend(_json_object_candidates(raw_text))
                    parsed_dict = None
                    for b in blocks:
                        try:
                            cand = _clean_and_parse(b)
                            if isinstance(cand, dict):
                                parsed_dict = cand; break
                        except Exception:
                            continue
                    if parsed_dict is None:
                        parsed_dict = _clean_and_parse(raw_text)
                    result = parsed_dict

                if isinstance(result, dict):
                    result["_model_used"] = model_name
                    result["_api_key_index"] = key_index
                    return result
            except Exception as e:
                last_err = e
                msg = str(e)
                # brief backoff on quota
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    try:
                        base_ms = float(os.environ.get("AI_BACKOFF_BASE_MS", "600.0"))
                        jitter_ms = float(os.environ.get("AI_BACKOFF_JITTER_MS", "400.0"))
                    except Exception:
                        base_ms, jitter_ms = 600.0, 400.0
                    delay = (base_ms + random.uniform(0.0, jitter_ms)) / 1000.0
                    time.sleep(delay)
                continue

    raise HTTPException(status_code=503, detail=f"Quick-inspect AI unavailable: {str(last_err) if last_err else 'no response'}")


def analyze_chart_with_gemini(image_bytes: bytes, user_inputs: UserInputs, ocr_data: OCRData) -> Dict[str, Any]:
    """
    One-shot extraction + analysis with Gemini (gemini-2.5-flash).
    Returns a single JSON object that includes the full extraction block and the final analysis.
    """

    # Prepare optional fundamentals context string if provided by caller
    fundamentals_summary = ""
    try:
        _fs = (user_inputs or {}).get("_fundamentals_summary")
        if isinstance(_fs, str) and _fs.strip():
            fundamentals_summary = _fs.strip()
    except Exception:
        fundamentals_summary = ""

    # ---- Build prompt safely: dynamic head (f-string) + plain schema string (no f-string) ----
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
  • Before responding, quickly check the latest market context for the symbol (price & news). THIS IS SO IMPORTANT YOU NEED TO HAVE SOE INFORMATION ABOUT THE MARKET .

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

# FUNDAMENTALS USAGE
- Use the fundamentals context above to influence per-strategy biases and the consolidated plan.
- Prefer headlines/events that directly affect the instrument/timeframe (e.g., CPI, FOMC, earnings, regulatory news for crypto).
- If headlines suggest elevated risk, reflect this in volatility and risk label, and adjust the plan narrative accordingly.

# METADATA RULES
- Do not set metadata.source to the instrument/symbol or company name. Source should be the platform/watermark if visible
  (TradingView, MetaTrader, Binance, Coinbase, Bybit, KuCoin, Investing.com, Yahoo Finance, Bloomberg, CoinMarketCap, etc.).
  If not visible, set source to "" (empty string), not the instrument.
- metadata.chart_style is the chart TYPE (e.g., "candlestick", "line", "bar", "heikin ashi", "renko"), not the timeframe.
  If uncertain, default to "candlestick".
- Provide metadata.timeframe using the on-image timeframe or fallback to user_inputs.chart_timeframe (e.g., "M1","H1","D1").
- Keep metadata.current_price numeric. If no label is visible, infer from visible levels or OCR hints.

# EXECUTION RULES
1) STRICTLY return ONE JSON object (no code fences, no extra prose).
2) Keep all numeric fields numeric; DO NOT leave entry/stop_loss/take_profit null in any section. If exact levels are not visible,
   infer them using the risk math helpers below with the best current price from extraction.metadata.current_price or OCR price levels.
3) Provide both:
   • per-strategy analysis (with bias, entry/SL/TP if possible, confidence 0–100),
   • a final_trade block (or "no_trade").
4) Include a fundamentals_news block (GROUND THIS IN THE FUNDAMENTALS CONTEXT ABOVE; DO NOT HALLUCINATE):
   - Select up to 3 headlines most relevant to the proposed trade idea; format each string as "Title — Source — YYYY-MM-DD".
   - Select up to 3 upcoming high/medium impact events for the instrument/timeframe; format each string as "YYYY-MM-DD HH:MM — Event — Country — Impact".
   - If no credible context, set "news_status": "unavailable".
5) Risk math helpers (from user inputs):
   - risk_amount_usd = account_balance * (risk_per_trade_percent / 100)
   - entry_price = extraction.metadata.current_price (if visible) else best visible price
   - price_delta_per_point = point_value_per_point
   - sl_delta = stop_loss_points * price_delta_per_point
   - tp_delta = take_profit_points * price_delta_per_point
   - For BUY:  SL = entry_price - sl_delta; TP = entry_price + tp_delta
   - For SELL: SL = entry_price + sl_delta; TP = entry_price - tp_delta
   - reward_risk_ratio = tp_delta / sl_delta
   - position_size = risk_amount_usd / sl_delta
6) Pattern detection:
   Use the following taxonomy and pick the closest visible structure. Prefer a concrete pattern over "None".
   a) Reversal Patterns
      - Bullish: Double Bottom, Inverse Head and Shoulders, Falling Wedge, Morning Star (candlestick), Hammer (candlestick), Bullish Engulfing
      - Bearish: Double Top, Head and Shoulders, Rising Wedge, Evening Star (candlestick), Shooting Star (candlestick), Bearish Engulfing
   b) Continuation Patterns
      - Bull Flag, Bear Flag, Pennant (bullish/bearish), Ascending Triangle (bullish), Descending Triangle (bearish),
        Symmetrical Triangle (neutral breakout-based), Cup and Handle (bullish)
   c) Neutral/Indecision
      - Symmetrical Triangle, Rectangle/Range, Doji (candlestick), Inside Bars, Spinning Top
   d) Candlestick (individual/multi-candle)
      - Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star, Piercing Pattern, Dark Cloud Cover, Doji, Spinning Top, Inside Bar
   e) Volume-based confirmations
      - Breakout with increasing volume, volume/price divergences, climax candles
   f) Market Structure (SMC/ICT)
      - Break of Structure (BOS), Change of Character (CHOCH), Liquidity Sweep/Stop Hunt, Order Blocks, Fair Value Gaps (FVG)

   Populate ai_analysis.technical_analysis.pattern_detected with a concise, Title Case label
   (or a ' | '-separated combination when multiple are clearly present). Also set
   ai_analysis.technical_analysis.patterns (array of strings) and add "Pattern: <name>" into ai_analysis.key_factors.
   Avoid "No Pattern Detected" unless absolutely none apply after careful inspection.

   If only a triangle/flag/wedge-like consolidation is visible, label it accordingly (e.g., "Symmetrical Triangle", "Bull Flag").
6) Final recommendation quality:
   - final_recommendation.summary must include: the main trigger (breakout/pullback and price), invalidation level, why this setup (which strategies agree), expected time horizon, and a brief management plan (partial profits, trailing, or move SL to BE when RR>=1).
   - If decision is "no_trade", include a conditional plan (e.g., Long above R after retest with SL below R; Short below S with SL above S) and a note about what would invalidate the scenario.
""".strip()

    # Enforce strict JSON output; keep task text concise and put long instructions in system_instruction
    prompt = (
        f"STRICT OUTPUT: Return only a single valid JSON object (no code fences, no comments). "
        f"Never leave entry/stop_loss/take_profit null. If exact values are not visible, infer them using user inputs and the risk math. "
        f"Round prices to {dec} decimals. Keep confidence in 0..100."
    )

    # Try multiple Gemini models as fallback chain.
    settings = AISettingsStore().get()
    selected = settings.get("selected_model")
    chain = settings.get("model_chain") or []
    model_chain = ([selected] if selected else []) + [m for m in chain if m and m != selected]
    if not model_chain:
        models_env = os.environ.get(
            "GEMINI_MODELS",
            "gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash-lite",
        )
        model_chain = [m.strip() for m in models_env.split(",") if m.strip()]
    print(f"[Gemini] Model chain: {model_chain}")
    # Cap to max 3 models and keys (policy)
    model_chain = [m for m in model_chain if m][:3]
    api_keys = (settings.get("api_keys") or ([GEMINI_API_KEY] if GEMINI_API_KEY else []))[:3]
    if not api_keys:
        raise HTTPException(status_code=500, detail="Gemini analysis failed: No API keys configured")

    # Define an output schema to have Gemini produce strictly valid JSON
    schema_out = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "extraction": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "metadata": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "source": types.Schema(type=types.Type.STRING),
                            "screenshot_type": types.Schema(type=types.Type.STRING),
                            "chart_style": types.Schema(type=types.Type.STRING),
                            "time_axis": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                            "price_axis": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
                            "timestamp_range": types.Schema(type=types.Type.STRING),
                            "prev_close": types.Schema(type=types.Type.NUMBER),
                            "current_price": types.Schema(type=types.Type.NUMBER),
                            "data_confidence": types.Schema(type=types.Type.NUMBER),
                        },
                    ),
                    "chart_data": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "price_scale_min": types.Schema(type=types.Type.NUMBER),
                            "price_scale_max": types.Schema(type=types.Type.NUMBER),
                            "visible_trend_points": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "time": types.Schema(type=types.Type.STRING),
                                        "price": types.Schema(type=types.Type.NUMBER),
                                    },
                                ),
                            ),
                            "approx_swing_high": types.Schema(type=types.Type.NUMBER),
                            "approx_swing_low": types.Schema(type=types.Type.NUMBER),
                            "price_change": types.Schema(type=types.Type.NUMBER),
                            "price_change_percent": types.Schema(type=types.Type.NUMBER),
                        },
                    ),
                    "trend_indicators": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "direction": types.Schema(type=types.Type.STRING),
                            "volatility_level": types.Schema(type=types.Type.STRING),
                            "momentum_strength": types.Schema(type=types.Type.STRING),
                            "pattern_type": types.Schema(type=types.Type.STRING),
                        },
                    ),
                    "annotations_detected": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "has_prev_close_label": types.Schema(type=types.Type.BOOLEAN),
                            "has_current_price_label": types.Schema(type=types.Type.BOOLEAN),
                            "no_indicators_detected": types.Schema(type=types.Type.BOOLEAN),
                        },
                    ),
                    "risk_parameters": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "account_balance": types.Schema(type=types.Type.NUMBER),
                            "risk_per_trade": types.Schema(type=types.Type.NUMBER),
                            "stop_loss_points": types.Schema(type=types.Type.NUMBER),
                            "take_profit_points": types.Schema(type=types.Type.NUMBER),
                        },
                    ),
                },
            ),
                    "user_inputs": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "symbol": types.Schema(type=types.Type.STRING),
                    "asset_type": types.Schema(type=types.Type.STRING),
                    "trading_type": types.Schema(type=types.Type.STRING),
                    "risk_profile": types.Schema(type=types.Type.STRING),
                    "account_balance": types.Schema(type=types.Type.NUMBER),
                    "risk_per_trade_percent": types.Schema(type=types.Type.NUMBER),
                    "stop_loss_points": types.Schema(type=types.Type.NUMBER),
                    "take_profit_points": types.Schema(type=types.Type.NUMBER),
                    "chart_timeframe": types.Schema(type=types.Type.STRING),
                },
            ),
            "per_strategy": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "name": types.Schema(type=types.Type.STRING),
                        "explanation": types.Schema(type=types.Type.STRING),
                        "bias": types.Schema(type=types.Type.STRING),
                        "entry_zone": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
                        "stop_loss": types.Schema(type=types.Type.NUMBER),
                        "take_profits": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
                        "confidence_percent": types.Schema(type=types.Type.NUMBER),
                    },
                ),
            ),
            "fundamentals_news": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "news_status": types.Schema(type=types.Type.STRING),
                    "current_price": types.Schema(type=types.Type.NUMBER),
                    "fundamental_bias": types.Schema(type=types.Type.STRING),
                    "key_headlines": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                    "upcoming_events": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                },
            ),
            "ai_analysis": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "trend_summary": types.Schema(type=types.Type.STRING),
                    "trend_direction": types.Schema(type=types.Type.STRING),
                    "volatility": types.Schema(type=types.Type.STRING),
                    "momentum": types.Schema(type=types.Type.STRING),
                    "market_sentiment": types.Schema(type=types.Type.STRING),
                    "technical_analysis": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "support_level": types.Schema(type=types.Type.NUMBER),
                            "resistance_level": types.Schema(type=types.Type.NUMBER),
                            "price_range": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
                            "key_zone": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
                            "pattern_detected": types.Schema(type=types.Type.STRING),
                            "patterns": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                            "pattern_explanation": types.Schema(type=types.Type.STRING),
                            "signal_strength": types.Schema(type=types.Type.NUMBER),
                        },
                    ),
                    "risk_management": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "risk_amount_usd": types.Schema(type=types.Type.NUMBER),
                            "entry_price": types.Schema(type=types.Type.NUMBER),
                            "stop_loss": types.Schema(type=types.Type.NUMBER),
                            "take_profit": types.Schema(type=types.Type.NUMBER),
                            "reward_risk_ratio": types.Schema(type=types.Type.NUMBER),
                            "position_size": types.Schema(type=types.Type.NUMBER),
                        },
                    ),
                    "trade_suggestion": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "signal": types.Schema(type=types.Type.STRING),
                            "entry_recommendation": types.Schema(type=types.Type.STRING),
                            "stop_loss_recommendation": types.Schema(type=types.Type.STRING),
                            "take_profit_recommendation": types.Schema(type=types.Type.STRING),
                            "ai_confidence": types.Schema(type=types.Type.NUMBER),
                        },
                    ),
                    "ai_insights": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                    "risk_label": types.Schema(type=types.Type.STRING),
                    "confidence_score": types.Schema(type=types.Type.NUMBER),
                    "reasoning_brief": types.Schema(type=types.Type.STRING),
                    "key_factors": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                },
            ),
            "final_recommendation": types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "decision": types.Schema(type=types.Type.STRING),
                    "summary": types.Schema(type=types.Type.STRING),
                    "entry": types.Schema(type=types.Type.NUMBER),
                    "entry_zone": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
                    "stop_loss": types.Schema(type=types.Type.NUMBER),
                    "take_profits": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER)),
                    "rr_estimates": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "tp1": types.Schema(type=types.Type.NUMBER),
                            "tp2": types.Schema(type=types.Type.NUMBER),
                            "tp3": types.Schema(type=types.Type.NUMBER),
                        },
                    ),
                    "time_horizon": types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "scalping": types.Schema(type=types.Type.STRING),
                            "swing": types.Schema(type=types.Type.STRING),
                        },
                    ),
                },
            ),
            "disclaimer": types.Schema(type=types.Type.STRING),
        },
        # keep requirements minimal to reduce model failures while ensuring a usable payload
        required=["ai_analysis", "final_recommendation", "disclaimer"],
    )

    # Helper to call Gemini with or without response_schema. Some SDK builds may raise
    # JSON decode errors when the model output is not perfectly compliant with the schema.
    # In that case, we retry without a schema and salvage JSON ourselves.
    def _gemini_generate(gemini_client, model_name: str, image_bytes: bytes, head: str, prompt: str, schema_out, use_schema: bool):
        def _detect_mime(b: bytes) -> str:
            if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xD8:
                return "image/jpeg"
            if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
                return "image/webp"
            return "image/jpeg"
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=_detect_mime(image_bytes)),
                    types.Part(text=head + "\n\n" + prompt),
                ],
            )
        ]
        cfg = dict(response_mime_type="application/json", max_output_tokens=1536, temperature=AI_TEMPERATURE)
        if use_schema:
            return gemini_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(response_schema=schema_out, **cfg),
            )
        else:
            return gemini_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**cfg),
            )

    # Utility: extract all top-level JSON object substrings using brace matching
    def _json_object_candidates(text: str) -> list[str]:
        spans = []
        stack = []
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(i)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start != -1:
                        spans.append(text[start:i+1])
                        start = -1
        # return longer candidates first
        spans.sort(key=len, reverse=True)
        return spans

    last_err: Exception | None = None
    # Attempt plan: rotate through keys x models in priority order, max 6 attempts
    attempt_limit = int(os.environ.get("AI_MAX_ATTEMPTS", "6"))
    attempt_idx = 0
    for key_index, api_key in enumerate(api_keys):
        for model_name in model_chain:
            if attempt_idx >= attempt_limit:
                break
            attempt_idx += 1
            try:
                print(f"[Gemini] Trying key #{key_index} with model: {model_name}")
                gemini_client = genai.Client(api_key=api_key)
                # Detect mime type for the uploaded image (jpeg/png/webp) to avoid decode errors
                def _detect_mime(b: bytes) -> str:
                    if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xD8:
                        return "image/jpeg"
                    if len(b) >= 8 and b[:8] == b"\x89PNG\r\n\x1a\n":
                        return "image/png"
                    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
                        return "image/webp"
                    return "image/jpeg"

                try:
                    # Try without schema first to avoid SDK strict JSON decode issues
                    response = _gemini_generate(
                        gemini_client,
                        model_name,
                        image_bytes,
                        head,
                        prompt,
                        schema_out,
                        False,
                    )
                except Exception as no_schema_err:
                    print(f"[Gemini] No-schema call failed on {model_name}: {no_schema_err}. Retrying with schema...")
                    response = _gemini_generate(
                        gemini_client,
                        model_name,
                        image_bytes,
                        head,
                        prompt,
                        schema_out,
                        True,
                    )

                # Prefer structured parse if SDK provides it
                parsed_obj = getattr(response, "parsed", None)
                if parsed_obj:
                    result = parsed_obj if isinstance(parsed_obj, dict) else getattr(parsed_obj, "to_dict", lambda: None)()
                    if isinstance(result, dict):
                        result["_model_used"] = model_name
                        result["_api_key_index"] = key_index
                        return result

                if not getattr(response, "text", None):
                    raise ValueError("Gemini returned empty response")

                # Prefer the SDK's text join; if missing, collect parts manually
                raw_text = getattr(response, "text", None)
                if not raw_text and getattr(response, "candidates", None):
                    parts = []
                    try:
                        for c in response.candidates:
                            for p in getattr(c, "content", {}).parts:
                                if getattr(p, "text", None):
                                    parts.append(p.text)
                    except Exception:
                        pass
                    raw_text = "".join(parts) if parts else None
                if not raw_text:
                    raise ValueError("Gemini returned empty response body")
                try:
                    result = json.loads(raw_text)
                except Exception:
                    # Attempt to salvage valid JSON if the model added stray text around the object
                    start = raw_text.find("{")
                    end = raw_text.rfind("}")
                    candidate_blocks = []
                    if start != -1 and end != -1 and end > start:
                        candidate_blocks.append(raw_text[start:end+1])
                    # Also try all balanced JSON object spans
                    candidate_blocks.extend(_json_object_candidates(raw_text))

                    def _clean_and_parse(s: str) -> dict:
                        x = s
                        # Strip code fences and comments
                        x = re.sub(r"```[a-zA-Z]*", "", x)
                        x = x.replace("```", "")
                        x = re.sub(r"//.*?$", "", x, flags=re.MULTILINE)
                        x = re.sub(r"/\*.*?\*/", "", x, flags=re.DOTALL)
                        # Remove trailing commas before } or ]
                        x = re.sub(r",\s*(?=[}\]])", "", x)
                        # Replace non-JSON tokens
                        x = x.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
                        # Normalize booleans/None/quotes
                        x = re.sub(r"\bTrue\b", "true", x)
                        x = re.sub(r"\bFalse\b", "false", x)
                        x = re.sub(r"\bNone\b", "null", x)
                        x = x.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
                        # Quote unquoted keys
                        x = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', x)
                        # If only single quotes across the doc, flip to double
                        if '"' not in x and "'" in x:
                            x = x.replace("'", '"')
                        # Remove stray control chars and trailing commas again
                        x = ''.join(ch for ch in x if ch.isprintable() or ch in ['\n','\r','\t'])
                        x = re.sub(r",\s*(?=[}\]])", "", x)
                        return json.loads(x)

                    parsed = None
                    for block in candidate_blocks:
                        try:
                            cand = _clean_and_parse(block)
                            # Ensure it looks like our payload
                            if isinstance(cand, dict) and ("ai_analysis" in cand or "final_recommendation" in cand):
                                parsed = cand
                                break
                        except Exception:
                            continue
                    if parsed is None:
                        # last attempt: try the whole raw_text
                        parsed = _clean_and_parse(raw_text)
                    result = parsed
                # annotate which model produced the result for observability
                if isinstance(result, dict):
                    result["_model_used"] = model_name
                    result["_api_key_index"] = key_index
                return result
            except Exception as e:  # try next model
                last_err = e
                msg = str(e)
                print(f"[Gemini] Model {model_name} failed: {msg}. Trying next...")
                # Small backoff on quota/429 errors before trying next model
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    # Exponential backoff with jitter; scales with attempt index
                    try:
                        base_ms = float(os.environ.get("AI_BACKOFF_BASE_MS", "800.0"))
                        factor = float(os.environ.get("AI_BACKOFF_FACTOR", "1.6"))
                        jitter_ms = float(os.environ.get("AI_BACKOFF_JITTER_MS", "400.0"))
                        max_ms = float(os.environ.get("AI_BACKOFF_MAX_MS", "6000.0"))
                    except Exception:
                        base_ms, factor, jitter_ms, max_ms = 800.0, 1.6, 400.0, 6000.0
                    scale = (factor ** max(0, attempt_idx - 1))
                    jitter = random.uniform(0.0, jitter_ms)
                    delay = min(max_ms, max(200.0, base_ms * scale + jitter)) / 1000.0
                    time.sleep(delay)
                continue

    # If all models failed, return a safe fallback JSON instead of 500 to avoid breaking UX
    msg = f"All Gemini models failed: {str(last_err)}" if last_err else "All Gemini models failed"

    def _safe_num(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    acc = _safe_num((user_inputs or {}).get("account_balance"), 0.0)
    rpp = _safe_num((user_inputs or {}).get("risk_per_trade_percent"), 0.0)
    slp = _safe_num((user_inputs or {}).get("stop_loss_points"), 0.0)
    tpp = _safe_num((user_inputs or {}).get("take_profit_points"), 0.0)

    # Heuristic levels from OCR (median as current price, min/max as key zone)
    cur_price = None
    key_low = None
    key_high = None
    try:
        levels = []
        if isinstance(ocr_data, dict):
            pl = (ocr_data or {}).get("price_levels") or []
            levels = [float(x) for x in pl if isinstance(x, (int, float))]
        if levels:
            levels_sorted = sorted(levels)
            cur_price = float(levels_sorted[len(levels_sorted)//2])
            key_low = float(levels_sorted[0])
            key_high = float(levels_sorted[-1])
    except Exception:
        pass

    # Compute simple RM numbers if we have a price and distances
    rm_entry = cur_price
    if rm_entry is not None and slp > 0 and tpp > 0:
        sl_delta = slp * pv
        tp_delta = tpp * pv
        sl_buy = round(rm_entry - sl_delta, dec)
        tp_buy = round(rm_entry + tp_delta, dec)
        risk_amount = round(acc * (rpp / 100.0), 2) if acc and rpp else 0.0
        risk_dist = abs(rm_entry - sl_buy)
        reward_dist = abs(tp_buy - rm_entry)
        rr = round((reward_dist / risk_dist), 2) if risk_dist else None
        pos_size = round((risk_amount / risk_dist), 4) if risk_dist else None
    else:
        sl_buy = None
        tp_buy = None
        risk_amount = round(acc * (rpp / 100.0), 2) if acc and rpp else 0.0
        rr = None
        pos_size = None

    # Build a more helpful fallback payload
    strategies = []
    if rm_entry is not None and sl_buy is not None and tp_buy is not None:
        strategies.append({
            "name": "Price Action — Breakout Plan",
            "explanation": (
                f"Range and breakout context. If price breaks above {key_high:.2f} on retest, consider longs; "
                f"if it loses {key_low:.2f} on retest, consider shorts." if key_low and key_high else
                "Wait for breakout of clear structure; trade the retest with tight invalidation."
            ),
            "bias": "neutral",
            "entry_zone": [round(rm_entry, 2)],
            "stop_loss": sl_buy,
            "take_profits": [tp_buy],
            "confidence_percent": 20,
        })
        # Mean reversion alternative
        strategies.append({
            "name": "Mean Reversion — Range Play",
            "explanation": (
                f"Fade extremes inside {key_low:.2f}–{key_high:.2f} with tight stops and quick targets." if key_low and key_high else
                "Fade short-term extensions back to the median with reduced size."
            ),
            "bias": "neutral",
            "entry_zone": [round(rm_entry, 2)],
            "stop_loss": round(rm_entry + slp, 2) if slp else sl_buy,
            "take_profits": [round(rm_entry - tpp, 2)] if tpp else [tp_buy] if tp_buy is not None else [],
            "confidence_percent": 15,
        })

    fallback = {
        "extraction": {
            "metadata": {
                "source": (user_inputs or {}).get("symbol") or "",
                "chart_style": (user_inputs or {}).get("chart_timeframe") or "",
                "current_price": cur_price,
            },
            "chart_data": {},
            "trend_indicators": {},
            "annotations_detected": {},
            "risk_parameters": {},
        },
        "user_inputs": user_inputs,
        "per_strategy": strategies,
        "fundamentals_news": {"news_status": "unavailable"},
        "ai_analysis": {
            "trend_summary": "Unavailable",
            "trend_direction": "NEUTRAL",
            "volatility": "UNKNOWN",
            "momentum": "UNKNOWN",
            "market_sentiment": "NEUTRAL",
            "technical_analysis": {
                "support_level": key_low,
                "resistance_level": key_high,
                "price_range": ([key_low, key_high] if (key_low is not None and key_high is not None) else []),
                "key_zone": ([key_low, key_high] if (key_low is not None and key_high is not None) else []),
                "pattern_detected": None,
                "signal_strength": 0.2,
            },
            "risk_management": {
                "risk_amount_usd": risk_amount,
                "entry_price": round(rm_entry, dec) if rm_entry is not None else None,
                "stop_loss": sl_buy,
                "take_profit": tp_buy,
                "reward_risk_ratio": rr,
                "position_size": pos_size,
            },
            "trade_suggestion": {
                "signal": "NO_TRADE",
                "entry_recommendation": (
                    f"Wait for breakout: Long above {key_high:.2f} on retest; Short below {key_low:.2f} on retest."
                    if key_low is not None and key_high is not None else
                    "Wait for a clear breakout and retest before entering."
                ),
                "stop_loss_recommendation": "Place SL beyond invalidation level (structure-based).",
                "take_profit_recommendation": "Scale out at next key levels; trail when RR>=1.",
                "ai_confidence": 0,
            },
            "ai_insights": [],
            "risk_label": "error",
            "confidence_score": 0,
            "reasoning_brief": f"AI output could not be parsed. {msg}",
            "key_factors": [],
        },
        "final_recommendation": {
            "decision": "no_trade",
            "summary": (
                f"Fallback plan: trade breakouts of {key_low:.2f}/{key_high:.2f} with retests; manage risk per inputs."
                if key_low is not None and key_high is not None else
                "Fallback plan: no conclusive AI output; wait for structure and manage risk per inputs."
            ),
            "entry": round(rm_entry, dec) if rm_entry is not None else None,
            "entry_zone": [round(rm_entry, dec)] if rm_entry is not None else [],
            "stop_loss": sl_buy,
            "take_profits": [tp_buy] if tp_buy is not None else [],
            "rr_estimates": {"tp1": None, "tp2": None, "tp3": None},
            "time_horizon": {"scalping": "N/A", "swing": "N/A"},
        },
        "disclaimer": "Automated fallback: AI response was not parseable; values are heuristic.",
        "_model_used": None,
        "_api_key_index": None,
        "_fallback": True,
        "_error": msg,
    }
    return fallback
