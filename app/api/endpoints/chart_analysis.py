# app/routers/chart_analysis_router.py
from datetime import date, datetime
import traceback
from typing import Optional
from fastapi import APIRouter, Depends, File, Path, Query, Request, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os
import  time
from sqlalchemy.exc import DataError, IntegrityError, ProgrammingError
import base64
import re
from app.core.price_rules import classify_symbol, points_to_price_delta, round_price
from app.core.deps import get_current_user_and_session
from app.core.logs_utils import compact_values
from app.database.session import get_db
from app.repositories.trade import TradeRepository
from app.services.audit_logs import AuditLogService
from app.services.chart_analysis import ChartAnalysisService
from app.core.chart_stage1_openrouter import (
    analyze_chart_with_openrouter,
    preprocess_image,
    extract_text_with_ocr,
    quick_inspect_with_openrouter
)
from app.repositories.ai_config import AIConfigRepository
from app.services.fundamentals import get_fundamentals
from app.schemas.common import Envelope

router = APIRouter(prefix="", tags=["chart"])

@router.get("/risk-defaults")
def get_risk_defaults(pair = Depends(get_current_user_and_session), db: Session = Depends(get_db)):
    """Get risk defaults from database configuration"""
    repo = AIConfigRepository(db)
    config = repo.get_or_create()
    return config.risk_defaults or {}

@router.post("/quick-inspect")
async def quick_inspect_chart(
    image: UploadFile = File(...),
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    """
    Fast pre-upload inspection using OpenRouter.
    Returns: symbol_guess, asset_type_guess, source_guess, is_relevant, confidence, excerpt, image_data
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await image.read()
    processed, _ = preprocess_image(raw)
    
    # Get AI config from database
    repo = AIConfigRepository(db)
    config = repo.get_or_create()
    
    # Get all active API keys for retry logic
    api_keys = config.get_active_keys()
    
    if not api_keys:
        raise HTTPException(
            status_code=400,
            detail="OpenRouter API key not configured. Please configure in admin settings."
        )

    # Primary path: fast AI extraction using OpenRouter with retry
    try:
        ai, key_index = quick_inspect_with_openrouter(
            processed,
            "",
            api_keys=api_keys,
            model=config.selected_model
        )
        data_url = f"data:image/jpeg;base64,{base64.b64encode(processed).decode('utf-8')}"
        symbol_guess = ai.get("symbol_guess")
        asset_type_guess = ai.get("asset_type_guess")
        source_guess = ai.get("source_guess")
        is_relevant = bool(ai.get("is_relevant"))
        try:
            confidence = int(ai.get("confidence") or 0)
        except Exception:
            confidence = 0
        excerpt = (ai.get("excerpt") or "")[:120]
        return {
            "symbol_guess": symbol_guess,
            "asset_type_guess": asset_type_guess,
            "source_guess": source_guess,
            "is_relevant": is_relevant,
            "confidence": confidence,
            "excerpt": excerpt,
            "extra_notes": (f"source: {str(source_guess).lower()}, {datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}" if source_guess else ""),
            "image_data": data_url,
        }
    except Exception:
        # Fall back to OCR-based heuristics below
        pass

    try:
        ocr = extract_text_with_ocr(processed)
        text = getattr(ocr, "extracted_text", "") or ""
        price_levels = getattr(ocr, "price_levels", []) or []
        time_labels = getattr(ocr, "time_labels", []) or []
    except Exception:
        text, price_levels, time_labels = "", [], []

    low = (text or "").lower()
    original_text = text or ""

    # Extract source/platform guess (expanded)
    tokens = [t for t in re.split(r"[^a-z0-9/]+", low) if t]
    def has(*keys):
        s = set(tokens)
        return any(k in s or k in low for k in keys)

    source_guess = None
    if has("tradingview"): source_guess = "TradingView"
    elif has("binance"): source_guess = "Binance"
    elif has("coinbase"): source_guess = "Coinbase"
    elif has("bybit"): source_guess = "Bybit"
    elif has("kucoin"): source_guess = "KuCoin"
    elif has("investing","investing.com"): source_guess = "Investing.com"
    elif has("yahoo","yahoo finance"): source_guess = "Yahoo Finance"
    elif has("bloomberg"): source_guess = "Bloomberg"
    elif has("metatrader","mt4","mt5"): source_guess = "MetaTrader"

    # Exact symbol extraction: detect and return as printed in the image
    # We scan the OCR text (case-sensitive) and select the highest-priority match.
    symbol_guess = None
    asset_type_guess = None

    def find_symbol_candidates(s: str) -> list[tuple[str, int]]:
        # Return list of (match, score)
        cands: list[tuple[str,int]] = []
        patterns: list[tuple[re.Pattern, int]] = [
            # Exchange prefix e.g., TRADINGVIEW:BTCUSDT or NASDAQ:TSLA
            (re.compile(r"\b[A-Z]{2,10}:[A-Za-z0-9\.\-\/]{2,15}\b"), 100),
            # Forex pairs (EUR/USD, EURUSD)
            (re.compile(r"\b[A-Z]{3}\/[A-Z]{3,4}\b"), 90),
            (re.compile(r"\b[A-Z]{6,7}\b"), 80),
            # Crypto common forms (BTCUSDT, BTC/USD, ETH-USD)
            (re.compile(r"\b[A-Z0-9]{2,10}[\/\-]?(?:USD|USDT|USDC|EUR|BTC|ETH)\b"), 85),
            # Index tickers
            (re.compile(r"\b(?:US30|US100|NAS100|NASDAQ|SPX|SP-?500|DAX|DE40|UK100|FTSE100|JP225|NIKKEI|HK50)\b", re.IGNORECASE), 80),
            # Metals / commodities
            (re.compile(r"\b(?:XAUUSD|XAGUSD|WTI|BRENT|GOLD|SILVER)\b", re.IGNORECASE), 80),
            # Stocks (1–6 letters with optional dot)
            (re.compile(r"\b[A-Z]{1,6}(?:\.[A-Z])?\b"), 60),
        ]
        for pat, base in patterns:
            for m in pat.finditer(s):
                txt = m.group(0)
                # slight boost if near "O H L C" or timeframe marker on same line
                boost = 0
                line_start = s.rfind("\n", 0, m.start()) + 1
                line_end = s.find("\n", m.end())
                if line_end == -1: line_end = len(s)
                line = s[line_start:line_end]
                if re.search(r"\b(O|Open)\b.*\b(H|High)\b.*\b(L|Low)\b.*\b(C|Close)\b", line, re.IGNORECASE):
                    boost += 10
                if re.search(r"\b(?:M1|M5|M15|M30|H1|H4|D1|W1|1m|5m|15m|30m|1h|4h|1d|1w)\b", line, re.IGNORECASE):
                    boost += 5
                cands.append((txt, base + boost))
        # Deduplicate preserving best score
        best: dict[str,int] = {}
        for txt, sc in cands:
            best[txt] = max(best.get(txt, 0), sc)
        # Sort by score desc, then length desc to prefer multi-token forms like EUR/USD
        return sorted(best.items(), key=lambda x: (x[1], len(x[0])), reverse=True)

    cands = find_symbol_candidates(original_text)
    if cands:
        symbol_guess = cands[0][0]  # keep exact as rendered in OCR text

        # Normalize for asset classification (strip exchange prefixes and separators)
        norm = symbol_guess
        if ":" in norm:
            norm = norm.split(":", 1)[1]
        norm = norm.replace("/", "").replace("-", "").upper()
        try:
            meta = classify_symbol(norm, None)
            asset_type_guess = meta.get("asset_type")
        except Exception:
            asset_type_guess = None

    # Fallback asset type from heuristics if symbol not found
    if asset_type_guess is None:
        if has("eurusd","gbpusd","usdjpy","audusd","usdchf","usdcad","eur","usd","jpy","gbp","aud","cad","chf"):
            asset_type_guess = "forex"
        elif has("btc","bitcoin","eth","bnb","sol","usdt","usdc"):
            asset_type_guess = "crypto"
        elif has("nas100","nasdaq","us30","dow","spx","sp500","dax","de40","ftse","uk100","nikkei","jp225","hk50"):
            asset_type_guess = "index"
        elif has("xau","gold","xauusd","xagusd","silver","oil","brent","wti"):
            asset_type_guess = "commodity"
        elif has("tesla","tsla","apple","aapl","nvidia","nvda","microsoft","msft"):
            asset_type_guess = "stock"

    # Detect timeframe token for excerpt
    timeframe_guess = None
    m_tf = re.search(r"\b(?:M1|M5|M15|M30|H1|H4|D1|W1|1m|5m|15m|30m|1h|4h|1d|1w)\b", original_text, re.IGNORECASE)
    if m_tf:
        timeframe_guess = m_tf.group(0).upper()

    # Relevance: chart-like signals (axes/ohlc/indicators/price/time labels/platform/price levels)
    chart_terms = ["open","high","low","close","ohlc","volume","candl","rsi","macd","ema","sma","boll","fibo","retest","breakout"]
    platform_tokens = ["tradingview","binance","coinbase","bybit","kucoin","investing","yahoo","bloomberg","metatrader","mt4","mt5"]
    signals = 0
    if any(t in low for t in chart_terms): signals += 1
    if any(pt in low for pt in platform_tokens): signals += 1
    if len(price_levels) >= 2: signals += 1
    if len(time_labels) >= 1: signals += 1
    if symbol_guess: signals += 1
    is_relevant = signals >= 2

    # Confidence scoring
    confidence = 40
    if is_relevant: confidence += 20
    if symbol_guess: confidence += 15
    if asset_type_guess: confidence += 10
    if timeframe_guess: confidence += 5
    confidence = int(min(confidence, 99))

    # Build small data URL for immediate preview on the client
    data_url = f"data:image/jpeg;base64,{base64.b64encode(processed).decode('utf-8')}"

    # One‑line readable excerpt (<= 120 chars) for quick inspect
    parts = []
    if symbol_guess: parts.append(str(symbol_guess))
    if timeframe_guess: parts.append(str(timeframe_guess))
    if asset_type_guess: parts.append(str(asset_type_guess))
    if source_guess: parts.append(f"via {source_guess}")
    if not parts:
        parts = ["Trading chart detected"]
    excerpt = " ".join(parts)[:120]
    return {
        "symbol_guess": symbol_guess,
        "asset_type_guess": asset_type_guess,
        "source_guess": source_guess,
        "is_relevant": is_relevant,
        "confidence": confidence,
        "excerpt": excerpt,
        "extra_notes": (f"source: {str(source_guess).lower()}, {datetime.utcnow().strftime('%Y-%m-%d %H:%MZ')}" if source_guess else ""),
        "image_data": data_url,
    }
def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    """
    Accepts ISO strings like:
      - "2025-11-08" (date-only)             -> 2025-11-08 00:00:00
      - "2025-11-08T13:45:00"
      - "2025-11-08T13:45:00Z"               -> treated as UTC
      - "2025-11-08T13:45:00+01:00"

    Returns None if s is falsy. Raises 422 if the format is invalid.
    """
    if not s:
        return None
    try:
        # Handle trailing 'Z' (UTC)
        if s.endswith("Z"):
            return datetime.fromisoformat(s[:-1] + "+00:00")

        # Date-only (YYYY-MM-DD)
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            d = date.fromisoformat(s)
            return datetime.combine(d, time.min)

        # Full ISO with optional offset
        return datetime.fromisoformat(s)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid datetime format: {s}")


@router.post("/analyze")
async def analyze_chart_endpoint(
    request: Request,
    image: UploadFile = File(...),
    symbol: Optional[str] = Form(None),
    timeframe: str = Form(...),
    chart_image_url: str = Form(""),
    # legacy field kept for backward-compat; may be null when using `mode`
    trading_type: Optional[str] = Form(None),
    # new unified inputs per spec
    asset_type: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    risk_profile: Optional[str] = Form(None),
    max_risk_per_trade_percent: Optional[float] = Form(None),
    user_bias: Optional[str] = Form(None),
    extra_notes: Optional[str] = Form(None),

    account_balance: str = Form(...),
    risk_per_trade_percent: str = Form(...),
    stop_loss_points: str = Form(...),
    take_profit_points: str = Form(...),
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _session = pair
    audit = AuditLogService(db)

    # 0) Validate input + log if invalid
    if image.content_type and not image.content_type.startswith("image/"):
        audit.create(
            user_id=user.user_id,
            action="VALIDATION_ERROR",
            entity_type="ChartAnalysis",
            entity_id=None,
            values=compact_values({
                "reason": "Non-image upload",
                "content_type": image.content_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "user_agent": request.headers.get("user-agent"),
            }),
            ip_address=(request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)),
            commit=False,
        )
        db.commit()
        raise HTTPException(status_code=400, detail="File must be an image")

    # locale number parser (accepts "0,25" or "6,00" etc.)
    def _num_locale(v):
        try:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                cleaned = v.replace(" ", "").replace(",", ".")
                return float(cleaned)
        except Exception:
            return None
        return None

    ab = _num_locale(account_balance)
    rpp = _num_locale(risk_per_trade_percent)
    slp = _num_locale(stop_loss_points)
    tpp = _num_locale(take_profit_points)

    # unify mode/trading_type; default SWING when missing
    _mode = (mode or trading_type or "SWING").upper()
    if _mode in ("SCALPING", "SCALP"): _mode = "SCALP"
    if _mode in ("SWING", "SWING_TRADING"): _mode = "SWING"
    if _mode in ("BOTH", "BOTH_MODES"): _mode = "BOTH"

    user_inputs = {
        "symbol": symbol,
        "asset_type": asset_type,
        "trading_type": _mode,
        "risk_profile": (risk_profile or "normal"),
        "max_risk_per_trade_percent": max_risk_per_trade_percent,
        "user_bias": user_bias,
        "extra_notes": extra_notes,
        "account_balance": ab,
        "risk_per_trade_percent": rpp,
        "stop_loss_points": slp,
        "take_profit_points": tpp,
        "chart_timeframe": timeframe,
    }

    chart_service = ChartAnalysisService(db)
    trade_repo = TradeRepository(db)

    # 1) Read and preprocess image early; run OCR for better hints
    raw = await image.read()
    processed, _ = preprocess_image(raw)
    # Build base64 data URL from original upload for storage/display
    mime = image.content_type or "image/png"
    base64_data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"
    try:
        ocr_model = extract_text_with_ocr(processed)
        # Convert Pydantic model to plain dict for JSONB column + prompt hints
        ocr_payload = (
            ocr_model.model_dump() if hasattr(ocr_model, "model_dump") else (
                dict(ocr_model) if isinstance(ocr_model, dict) else {}
            )
        )
        ocr_vendor = "tesseract"
    except Exception:
        ocr_payload = {}
        ocr_vendor = None

    # Infer source from watermark/logo tokens (OCR) and inject into extra_notes as "source: coinmarketcap, time"
    try:
        low_txt = ""
        if isinstance(ocr_payload, dict):
            low_txt = str(ocr_payload.get("extracted_text") or "").lower()
        tokens = [t for t in re.split(r"[^a-z0-9/]+", low_txt) if t]

        def _has(*keys: str) -> bool:
            s = set(tokens)
            return any(k in s or k in low_txt for k in keys)

        detected_source = None
        if _has("coinmarketcap", "cmc"):
            detected_source = "coinmarketcap"
        elif _has("tradingview"):
            detected_source = "tradingview"
        elif _has("binance"):
            detected_source = "binance"
        elif _has("coinbase"):
            detected_source = "coinbase"
        elif _has("bybit"):
            detected_source = "bybit"
        elif _has("kucoin"):
            detected_source = "kucoin"
        elif _has("coingecko"):
            detected_source = "coingecko"
        elif _has("investing", "investing.com"):
            detected_source = "investing.com"
        elif _has("yahoo", "yahoo finance"):
            detected_source = "yahoo finance"
        elif _has("bloomberg"):
            detected_source = "bloomberg"
        elif _has("metatrader", "mt4", "mt5"):
            detected_source = "metatrader"

        if detected_source:
            detected_time = datetime.utcnow().strftime("%Y-%m-%d %H:%MZ")
            extra_line = f"source: {detected_source}, {detected_time}"
            prev = user_inputs.get("extra_notes")
            if isinstance(prev, str) and prev.strip():
                user_inputs["extra_notes"] = f"{prev} | {extra_line}"
            else:
                user_inputs["extra_notes"] = extra_line
    except Exception:
        # Do not fail flow if detection fails
        pass

    # 2) Start analysis (persist first)
    norm_symbol = symbol or user_inputs.get("symbol")

    # Symbol classification for price precision and point conversion
    try:
        meta = classify_symbol(norm_symbol, asset_type)
        user_inputs["_asset_type_resolved"] = meta.get("asset_type")
        user_inputs["_price_decimals"] = meta.get("decimals")
        user_inputs["_point_value"] = meta.get("point_value")
        user_inputs["_price_range_hint"] = meta.get("price_range")
    except Exception:
        pass

    # Fetch fundamentals early (for reuse and consistency)
    try:
        fundamentals = get_fundamentals(norm_symbol or "", timeframe)
    except Exception:
        fundamentals = None

    analysis = chart_service.start_analysis(
        user_id=str(user.user_id),
        symbol=norm_symbol,
        timeframe=timeframe,
        chart_image_url=chart_image_url,
        user_inputs=user_inputs,
        ocr_vendor=ocr_vendor,
        ocr_text=ocr_payload,
        chart_image_data=base64_data_url,
    )
    db.commit()  # ensure analysis row exists, get its UUID

    # Log: STARTED
    audit.create(
        user_id=user.user_id,
        action="STARTED",
        entity_type="ChartAnalysis",
        entity_id=analysis.analysis_id,
        values=compact_values({
            "symbol": symbol,
            "timeframe": timeframe,
            "chart_image_url": bool(chart_image_url),
            "inputs": user_inputs,
            "user_agent": request.headers.get("user-agent"),
        }),
        ip_address=(request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)),
        commit=False,
    )

    try:
        # 3) Run Gemini using preprocessed image + OCR

        # Build a concise fundamentals summary for the model prompt (optional)
        fundamentals_summary = ""
        try:
            if fundamentals:
                bias = fundamentals.get("fundamental_bias")
                vol = fundamentals.get("volatility_risk")
                ev = fundamentals.get("upcoming_events") or []
                hd = fundamentals.get("news_headlines") or []
                ev_lines = []
                for e in ev[:3]:
                    name = (e or {}).get("event")
                    ctry = (e or {}).get("country")
                    when = (e or {}).get("time")
                    parts = [p for p in [when, name, ctry] if p]
                    if parts:
                        ev_lines.append(" - " + " | ".join(parts))
                hd_lines = []
                for h in hd[:3]:
                    t = (h or {}).get("title")
                    s = (h or {}).get("source")
                    if t:
                        hd_lines.append(f" - {t}{' — ' + s if s else ''}")
                fundamentals_summary = (
                    f"Fundamentals snapshot for {norm_symbol or ''}:\n"
                    f"Bias: {bias or 'UNKNOWN'}; Volatility: {vol or 'UNKNOWN'}.\n"
                    + ("Upcoming events:\n" + "\n".join(ev_lines) + "\n" if ev_lines else "")
                    + ("Recent headlines:\n" + "\n".join(hd_lines) if hd_lines else "")
                ).strip()
        except Exception:
            fundamentals_summary = ""

        # Pass fundamentals summary to the model via user_inputs
        user_inputs["_fundamentals_summary"] = fundamentals_summary
        
        # Get AI config from database
        ai_repo = AIConfigRepository(db)
        ai_config = ai_repo.get_or_create()
        
        # Get all active API keys for retry logic
        api_keys = ai_config.get_active_keys()
        
        if not api_keys:
            raise HTTPException(
                status_code=400,
                detail="OpenRouter API key not configured. Please configure in admin settings."
            )

        t0 = time.time()
        gemini_json, key_index = analyze_chart_with_openrouter(
            image_bytes=processed,
            user_inputs=user_inputs,
            ocr_data=ocr_payload or {},
            api_keys=api_keys,
            model=ai_config.selected_model
        )
        latency_ms = int((time.time() - t0) * 1000)
        
        # Log which key was used for successful analysis
        if key_index is not None:
            key_info = ai_config.get_key_info(key_index)
            key_label = key_info.get("label", f"Key {key_index + 1}") if key_info else f"Key {key_index + 1}"
            print(f"[Analysis] Successfully used API key: {key_label} (index {key_index})")

        # Handle AI backend quota/429 gracefully: mark FAILED, audit details (admin-only), return friendly message
        try:
            is_fallback = isinstance(gemini_json, dict) and bool(gemini_json.get("_fallback"))
            raw_err = str((gemini_json or {}).get("_error") or "")
            exceeded = "RESOURCE_EXHAUSTED" in raw_err or "429" in raw_err
            if is_fallback and exceeded:
                # Mark analysis failed
                chart_service.fail_analysis(analysis, error="AI analysis temporarily unavailable")
                # Audit with details for admins
                audit.create(
                    user_id=user.user_id,
                    action="ANALYZE_FAILED",
                    entity_type="ChartAnalysis",
                    entity_id=analysis.analysis_id,
                    values=compact_values(
                        {
                            "stage": "ai_generate",
                            "error_type": "RESOURCE_EXHAUSTED",
                            "error_message": raw_err,
                            "inputs": {"timeframe": timeframe, "symbol": symbol},
                            "user_agent": request.headers.get("user-agent"),
                        },
                        max_len=2000,
                    ),
                    ip_address=(request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)),
                    commit=False,
                )
                db.commit()
                # Return friendly error envelope to user (do not mention provider)
                friendly = Envelope(
                    status="error",
                    code=200,
                    message="Analysis is temporarily unavailable. Please try again shortly.",
                    error={"detail": "Please try again later", "status_code": 503},
                )
                return JSONResponse(friendly.model_dump())
        except Exception:
            # best-effort safeguard; fall through to normal enrichment path if something unexpected happens here
            pass

        # --- Normalize/enrich payload when model returned legacy fields or nulls ---
        try:
            if isinstance(gemini_json, dict):
                # 1) Convert legacy strategies/analysis -> per_strategy
                if not isinstance(gemini_json.get("per_strategy"), list):
                    legacy_list = None
                    if isinstance(gemini_json.get("analysis"), list):
                        legacy_list = gemini_json.get("analysis")
                    elif isinstance(gemini_json.get("strategies"), list):
                        legacy_list = gemini_json.get("strategies")
                    elif isinstance(gemini_json.get("per_strategy_analysis"), list):
                        # Handle legacy key 'per_strategy_analysis' seen in some model outputs
                        legacy_list = gemini_json.get("per_strategy_analysis")
                    if isinstance(legacy_list, list):
                        mapped = []
                        for s in (legacy_list or []):
                            if not isinstance(s, dict):
                                continue
                            entry_zone = []
                            # support either 'entry' or 'entry_price'
                            if isinstance(s.get("entry"), (int, float)):
                                entry_zone = [float(s.get("entry"))]
                            elif isinstance(s.get("entry_price"), (int, float)):
                                entry_zone = [float(s.get("entry_price"))]
                            # allow 'strategy' as name key
                            name = s.get("name") or s.get("strategy") or "Strategy"
                            mapped.append({
                                "name": name,
                                "explanation": s.get("explanation") or s.get("observation") or "",
                                "bias": (s.get("bias") or "neutral").lower(),
                                "entry_zone": entry_zone,
                                "stop_loss": s.get("stop_loss"),
                                "take_profits": [s.get("take_profit")] if isinstance(s.get("take_profit"), (int, float)) else [],
                                "confidence_percent": s.get("confidence") if isinstance(s.get("confidence"), (int, float)) else s.get("confidence_percent"),
                            })
                        gemini_json["per_strategy"] = mapped

                per_strategy = gemini_json.get("per_strategy") or []
                # Build per_strategy from ai_analysis.technical_analysis named blocks when missing or too short
                try:
                    ai_block = gemini_json.get("ai_analysis") or {}
                    tech_block = (ai_block.get("technical_analysis") or {}) if isinstance(ai_block, dict) else {}
                    def _num(x):
                        return float(x) if isinstance(x, (int, float)) else None
                    mapping = [
                        ("price_action_sr", "Price Action / S-R"),
                        ("supply_demand", "Supply & Demand"),
                        ("chart_patterns", "Chart Patterns"),
                        ("fibonacci", "Fibonacci"),
                        ("moving_averages", "Moving Averages"),
                        ("oscillators_divergence", "Oscillators / Divergence"),
                        ("volume_volatility", "Volume / Volatility"),
                        ("multi_timeframe", "Multi-timeframe"),
                    ]
                    built = []
                    for key, label in mapping:
                        if isinstance(tech_block.get(key), dict):
                            o = tech_block.get(key) or {}
                            entry = _num(o.get("entry") if isinstance(o.get("entry"), (int, float)) else o.get("entry_price"))
                            sl = _num(o.get("stop_loss"))
                            tp = _num(o.get("take_profit"))
                            conf = o.get("confidence") if isinstance(o.get("confidence"), (int, float)) else o.get("confidence_percent") if isinstance(o.get("confidence_percent"), (int, float)) else None
                            bias = str(o.get("bias") or "neutral").lower()
                            desc = o.get("observation") or o.get("explanation") or ""
                            built.append({
                                "name": label,
                                "explanation": desc,
                                "bias": bias,
                                "entry_zone": [entry] if isinstance(entry, (int, float)) else [],
                                "stop_loss": sl,
                                "take_profits": [tp] if isinstance(tp, (int, float)) else [],
                                "confidence_percent": conf,
                            })
                    # Merge without duplicating by name
                    if built:
                        seen = set()
                        merged = []
                        for s in (per_strategy + built):
                            if isinstance(s, dict):
                                nm = s.get("name") or "Strategy"
                                if nm in seen:
                                    continue
                                seen.add(nm)
                                merged.append(s)
                        per_strategy = merged
                        gemini_json["per_strategy"] = per_strategy
                    # If still fewer than 6, add heuristics-based items and top-up to 6–9
                    if len(per_strategy) < 6:
                        names = {s.get("name") for s in per_strategy if isinstance(s, dict)}
                        # Breakout plan from support/resistance
                        slevel = tech_block.get("support_level") if isinstance(tech_block, dict) else None
                        rlevel = tech_block.get("resistance_level") if isinstance(tech_block, dict) else None
                        if isinstance(slevel, (int, float)) and isinstance(rlevel, (int, float)):
                            entry = ((gemini_json.get("extraction") or {}).get("metadata") or {}).get("current_price")
                            item = {
                                "name": "Breakout / Retest Plan",
                                "explanation": f"Long above {rlevel:.2f} after confirmed retest; Short below {slevel:.2f} after retest.",
                                "bias": "neutral",
                                "entry_zone": [float(entry)] if isinstance(entry, (int, float)) else [],
                                "stop_loss": float(slevel) if isinstance(entry, (int, float)) and float(entry) < float(rlevel) else float(rlevel) if isinstance(entry, (int, float)) else None,
                                "take_profits": [float(rlevel)] if isinstance(entry, (int, float)) and float(entry) < float(rlevel) else ([float(slevel)] if isinstance(entry, (int, float)) else []),
                                "confidence_percent": 20,
                            }
                            if item["name"] not in names:
                                per_strategy.append(item); names.add(item["name"])
                        # Fundamentals overlay
                        fn = gemini_json.get("fundamentals_news") or {}
                        fbias = str(fn.get("fundamental_bias") or "neutral").lower()
                        if isinstance(fn, dict) and fn:
                            headlines = fn.get("key_headlines") or []
                            explain = "Fundamentals suggest {} bias. Headlines: {}".format(
                                fbias,
                                "; ".join([str(h) for h in headlines[:2]])
                            )
                            rm = (ai_block.get("risk_management") or {}) if isinstance(ai_block, dict) else {}
                            entry = _num(rm.get("entry_price")) or _num(((gemini_json.get("extraction") or {}).get("metadata") or {}).get("current_price"))
                            slv = _num(rm.get("stop_loss"))
                            tpv = _num(rm.get("take_profit"))
                            item = {
                                "name": "Fundamentals / News",
                                "explanation": explain,
                                "bias": fbias,
                                "entry_zone": [entry] if isinstance(entry, (int, float)) else [],
                                "stop_loss": slv,
                                "take_profits": [tpv] if isinstance(tpv, (int, float)) else [],
                                "confidence_percent": fn.get("confidence_percent") if isinstance(fn.get("confidence_percent"), (int, float)) else None,
                            }
                            if item["name"] not in names:
                                per_strategy.append(item); names.add(item["name"])

                        # Pattern-driven plan
                        patt = (tech_block or {}).get("pattern_detected")
                        if isinstance(patt, str) and patt.strip():
                            item = {
                                "name": "Pattern Continuation/Break",
                                "explanation": f"Pattern detected: {patt}. Trade continuation on break and retest; invalidate on pattern failure.",
                                "bias": "neutral",
                                "entry_zone": [],
                                "stop_loss": None,
                                "take_profits": [],
                                "confidence_percent": 25,
                            }
                            if item["name"] not in names:
                                per_strategy.append(item); names.add(item["name"])

                        # Risk management snapshot (numeric mirror)
                        rm = (ai_block.get("risk_management") or {}) if isinstance(ai_block, dict) else {}
                        if any(isinstance(rm.get(k), (int, float)) for k in ("entry_price","stop_loss","take_profit")):
                            item = {
                                "name": "Risk Management Plan",
                                "explanation": "Mirror numeric plan using user risk inputs for clarity.",
                                "bias": (ai_block.get("trend_direction") or "neutral").lower() if isinstance(ai_block, dict) else "neutral",
                                "entry_zone": [float(rm.get("entry_price"))] if isinstance(rm.get("entry_price"), (int, float)) else [],
                                "stop_loss": float(rm.get("stop_loss")) if isinstance(rm.get("stop_loss"), (int, float)) else None,
                                "take_profits": [float(rm.get("take_profit"))] if isinstance(rm.get("take_profit"), (int, float)) else [],
                                "confidence_percent": 20,
                            }
                            if item["name"] not in names:
                                per_strategy.append(item); names.add(item["name"])

                        # Range mean reversion using key_zone
                        kz = (tech_block or {}).get("key_zone")
                        if isinstance(kz, list) and len(kz) >= 2 and all(isinstance(x, (int, float)) for x in kz[:2]):
                            lo, hi = float(kz[0]), float(kz[1])
                            item = {
                                "name": "Range Mean Reversion",
                                "explanation": f"Fade extremes inside {lo:.2f}–{hi:.2f} with tight stops and quick targets.",
                                "bias": "neutral",
                                "entry_zone": [],
                                "stop_loss": None,
                                "take_profits": [],
                                "confidence_percent": 15,
                            }
                            if item["name"] not in names:
                                per_strategy.append(item); names.add(item["name"])

                        # Multi-timeframe alignment (trend vs sentiment)
                        td = (ai_block.get("trend_direction") or "NEUTRAL") if isinstance(ai_block, dict) else "NEUTRAL"
                        ms = (ai_block.get("market_sentiment") or td)
                        if isinstance(td, str) and isinstance(ms, str):
                            item = {
                                "name": "MTF Alignment",
                                "explanation": f"Trend: {td}. Sentiment: {ms}. Trade only when aligned; otherwise reduce size.",
                                "bias": (td or "neutral").lower(),
                                "entry_zone": [],
                                "stop_loss": None,
                                "take_profits": [],
                                "confidence_percent": 10,
                            }
                            if item["name"] not in names:
                                per_strategy.append(item); names.add(item["name"])

                        # Cap to 9 to avoid overwhelming UI
                        if len(per_strategy) > 9:
                            per_strategy = per_strategy[:9]

                        gemini_json["per_strategy"] = per_strategy
                except Exception:
                    pass

                # 2) Confluence & confidence
                bull = sum(1 for s in per_strategy if str(s.get("bias", "")).lower().startswith("bull"))
                bear = sum(1 for s in per_strategy if str(s.get("bias", "")).lower().startswith("bear"))
                neu  = sum(1 for s in per_strategy if str(s.get("bias", "")).lower().startswith("neut"))
                confidences = [float(s.get("confidence_percent")) for s in per_strategy if isinstance(s.get("confidence_percent"), (int, float))]
                avg_conf = int(round(sum(confidences) / len(confidences))) if confidences else 0

                # 3) Map final_trade -> final_recommendation if needed
                if not isinstance(gemini_json.get("final_recommendation"), dict) and isinstance(gemini_json.get("final_trade"), dict):
                    ft = gemini_json.get("final_trade") or {}
                    bias = (ft.get("bias") or "neutral").lower()
                    if "bull" in bias:
                        decision = "enter_long"
                    elif "bear" in bias:
                        decision = "enter_short"
                    else:
                        decision = "no_trade"
                    fr = {
                        "decision": decision,
                        "summary": ft.get("explanation") or "",
                        "entry": ft.get("entry"),
                        "entry_zone": [ft.get("entry")] if isinstance(ft.get("entry"), (int, float)) else [],
                        "stop_loss": ft.get("stop_loss"),
                        "take_profits": [ft.get("take_profit")] if isinstance(ft.get("take_profit"), (int, float)) else [],
                        "rr_estimates": {"tp1": None, "tp2": None, "tp3": None},
                        "time_horizon": {"scalping": "minutes-hours", "swing": "hours-days"},
                    }
                    gemini_json["final_recommendation"] = fr

                # 4) Ensure ai_analysis block exists and fill summary fields
                ai = gemini_json.setdefault("ai_analysis", {})
                ai.setdefault("confidence_score", avg_conf)
                if bull > bear:
                    ai.setdefault("trend_direction", "BULLISH")
                elif bear > bull:
                    ai.setdefault("trend_direction", "BEARISH")
                else:
                    ai.setdefault("trend_direction", "NEUTRAL")
                tech = ai.setdefault("technical_analysis", {})
                # Derive pattern_detected when missing or generic
                try:
                    def _norm_str(x):
                        return str(x or "").strip().lower()
                    known_syns = {
                        "double top": ["double top"],
                        "double bottom": ["double bottom"],
                        "head & shoulders": ["head & shoulders", "head and shoulders"],
                        "inverse head & shoulders": ["inverse head & shoulders", "inverse head and shoulders"],
                        "ascending triangle": ["ascending triangle"],
                        "descending triangle": ["descending triangle"],
                        "symmetrical triangle": ["symmetrical triangle"],
                        "rising wedge": ["rising wedge"],
                        "falling wedge": ["falling wedge"],
                        "flag": ["bull flag", "bear flag", "flag"],
                        "pennant": ["pennant"],
                        "rectangle": ["rectangle", "range"],
                        "cup and handle": ["cup and handle"],
                        "rounding bottom": ["rounding bottom", "rounding base"],
                        "channel": ["channel", "ascending channel", "descending channel"],
                        "broadening": ["broadening", "megaphone"],
                        "diamond": ["diamond"],
                        "engulfing": ["bullish engulfing", "bearish engulfing", "engulfing"],
                    }
                    texts = []
                    tsum = ai.get("trend_summary")
                    if isinstance(tsum, str):
                        texts.append(tsum)
                    top_pattern = gemini_json.get("pattern")
                    if isinstance(top_pattern, str):
                        texts.append(top_pattern)
                    tab1 = gemini_json.get("TAB 1 - ANALYSIS DETAILS") or {}
                    if isinstance(tab1, dict):
                        pi = tab1.get("patternIdentification")
                        if isinstance(pi, str):
                            texts.append(pi)
                    for s in (gemini_json.get("per_strategy") or []):
                        if isinstance(s, dict):
                            if isinstance(s.get("explanation"), str):
                                texts.append(s["explanation"])
                            if isinstance(s.get("name"), str):
                                texts.append(s["name"])
                    if isinstance(tech.get("patterns"), list):
                        texts.extend([str(x) for x in tech["patterns"]])
                    low = "\n".join(texts).lower()
                    hits = []
                    for label, syns in known_syns.items():
                        if any(kw in low for kw in syns):
                            hits.append(label)
                    cur = _norm_str(tech.get("pattern_detected"))
                    if (not cur or cur in ("no pattern detected", "none", "unknown")) and hits:
                        tech["pattern_detected"] = " | ".join(sorted(set(hits)))
                        kf = ai.setdefault("key_factors", [])
                        if isinstance(kf, list):
                            kf.append(f"Pattern: {tech['pattern_detected']}")
                except Exception:
                    pass
                # Alias risk_disclaimer -> disclaimer for frontend
                try:
                    if not gemini_json.get("disclaimer") and isinstance(gemini_json.get("risk_disclaimer"), str):
                        gemini_json["disclaimer"] = gemini_json.get("risk_disclaimer")
                except Exception:
                    pass

                # 5) Infer support/resistance from strategy texts if missing
                try:
                    texts = []
                    for s in per_strategy:
                        t = s.get("explanation")
                        if isinstance(t, str):
                            texts.append(t)
                    import re
                    nums: list[float] = []
                    for t in texts:
                        for m in re.findall(r"\\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\\b", t):
                            try:
                                v = float(m.replace(",", ""))
                                if v > 0:
                                    nums.append(v)
                            except Exception:
                                pass
                    low = min(nums) if nums else None
                    high = max(nums) if nums else None
                    # OCR fallback if strategy text had no levels
                    try:
                        pl = (ocr_payload or {}).get("price_levels") if isinstance(ocr_payload, dict) else None
                        if not low and isinstance(pl, list) and len(pl) >= 2:
                            low = min([float(x) for x in pl if isinstance(x, (int, float))])
                            high = max([float(x) for x in pl if isinstance(x, (int, float))])
                    except Exception:
                        pass
                    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low < high:
                        tech.setdefault("support_level", low)
                        tech.setdefault("resistance_level", high)
                        if not tech.get("key_zone"):
                            tech["key_zone"] = [low, high]
                except Exception:
                    pass

                # 6) Fill extraction metadata & chart_data for UI cards
                try:
                    extraction = gemini_json.setdefault("extraction", {})
                    md = extraction.setdefault("metadata", {})
                    cd = extraction.setdefault("chart_data", {})

                    # metadata normalization and fixes:
                    # 1) Ensure timeframe present (prefer on-image; else fallback to submitted timeframe)
                    if not md.get("timeframe"):
                        md["timeframe"] = user_inputs.get("chart_timeframe") or timeframe

                    # 2) chart_style must be chart TYPE, not timeframe; default to candlestick if missing or looks like timeframe
                    tf_tokens = ("M1","M5","M15","M30","H1","H4","D1","W1","1m","5m","15m","30m","1h","4h","1d","1w")
                    cs = md.get("chart_style")
                    if not isinstance(cs, str) or cs.strip().upper() in tf_tokens:
                        md["chart_style"] = "candlestick"

                    # 3) source should be platform/watermark, not the instrument/symbol/company
                    def _std_src(s: str) -> str:
                        s = str(s or "").lower()
                        if "tradingview" in s: return "tradingview"
                        if "metatrader" in s or "mt4" in s or "mt5" in s: return "metatrader"
                        if "binance" in s: return "binance"
                        if "coinbase" in s: return "coinbase"
                        if "bybit" in s: return "bybit"
                        if "kucoin" in s: return "kucoin"
                        if "coinmarketcap" in s or "cmc" in s: return "coinmarketcap"
                        if "coingecko" in s: return "coingecko"
                        if "investing" in s: return "investing.com"
                        if "yahoo" in s: return "yahoo finance"
                        if "bloomberg" in s: return "bloomberg"
                        return ""

                    # derive a hint from extra_notes like "source: tradingview, 2025-11-24 15:02Z"
                    extra = str(user_inputs.get("extra_notes") or "")
                    m_src = re.search(r"source:\s*([a-zA-Z0-9\.\- ]+)", extra, re.IGNORECASE)
                    hint_source = _std_src(m_src.group(1)) if m_src else ""

                    src_val = md.get("source")
                    sym_like = str(user_inputs.get("symbol") or "").strip().lower()
                    # if missing/empty or equal to symbol/company, replace by hint or set empty
                    if not isinstance(src_val, str) or not src_val.strip() or src_val.strip().lower() in (sym_like,):
                        md["source"] = hint_source or ""
                    else:
                        # normalize known platforms
                        std = _std_src(src_val)
                        if std:
                            md["source"] = std

                    # current price
                    if md.get("current_price") is None:
                        sl = tech.get("support_level"); rl = tech.get("resistance_level")
                        if isinstance(sl, (int, float)) and isinstance(rl, (int, float)):
                            md["current_price"] = round((float(sl) + float(rl)) / 2.0, int((user_inputs or {}).get("_price_decimals") or 2))
                        else:
                            # OCR median fallback
                            try:
                                pl = (ocr_payload or {}).get("price_levels") if isinstance(ocr_payload, dict) else None
                                if isinstance(pl, list) and len(pl) > 0:
                                    vals = sorted([float(x) for x in pl if isinstance(x, (int, float))])
                                    mid = vals[len(vals)//2]
                                    md["current_price"] = round(float(mid), int((user_inputs or {}).get("_price_decimals") or 2))
                            except Exception:
                                pass
                    # swing levels and price range
                    sl = tech.get("support_level"); rl = tech.get("resistance_level")
                    if isinstance(sl, (int, float)):
                        cd.setdefault("approx_swing_low", float(sl))
                    if isinstance(rl, (int, float)):
                        cd.setdefault("approx_swing_high", float(rl))
                    if isinstance(sl, (int, float)) and isinstance(rl, (int, float)):
                        tech.setdefault("price_range", [float(sl), float(rl)])
                        if not tech.get("key_zone"):
                            tech["key_zone"] = [float(sl), float(rl)]
                except Exception:
                    pass

                # FINAL ENFORCEMENT: ensure price_range/key_zone and a concrete pattern label
                try:
                    tech = ((gemini_json or {}).get("ai_analysis") or {}).get("technical_analysis") if isinstance(gemini_json, dict) else None
                    if isinstance(tech, dict):
                        pr = tech.get("price_range")
                        kz = tech.get("key_zone")
                        have_pr = isinstance(pr, list) and len(pr) >= 2 and all(isinstance(x, (int, float)) for x in pr[:2])
                        have_kz = isinstance(kz, list) and len(kz) >= 2 and all(isinstance(x, (int, float)) for x in kz[:2])

                        low = None
                        high = None

                        if not (have_pr and have_kz):
                            # Try support/resistance first
                            sl = tech.get("support_level")
                            rl = tech.get("resistance_level")
                            if isinstance(sl, (int, float)) and isinstance(rl, (int, float)):
                                low = float(min(sl, rl)); high = float(max(sl, rl))

                            # Fallback to OCR range
                            if (low is None or high is None):
                                try:
                                    pl = (ocr_payload or {}).get("price_levels") if isinstance(ocr_payload, dict) else None
                                    if isinstance(pl, list) and len(pl) >= 2:
                                        vals = [float(x) for x in pl if isinstance(x, (int, float))]
                                        if vals:
                                            low = min(vals); high = max(vals)
                                except Exception:
                                    pass

                            # Fallback to approximate range from current price and user deltas
                            if (low is None or high is None):
                                cp = ((gemini_json.get("extraction") or {}).get("metadata") or {}).get("current_price") if isinstance(gemini_json, dict) else None
                                if isinstance(cp, (int, float)):
                                    try:
                                        pv = float((user_inputs or {}).get("_point_value") or 1.0)
                                        slp_local = float((user_inputs or {}).get("stop_loss_points") or 0.0)
                                        tpp_local = float((user_inputs or {}).get("take_profit_points") or 0.0)
                                        # Use at least 1*pv if both are zero to generate a small band
                                        lo_delta = slp_local * pv if slp_local > 0 else 1.0 * pv
                                        hi_delta = tpp_local * pv if tpp_local > 0 else 1.0 * pv
                                        low = float(cp) - lo_delta
                                        high = float(cp) + hi_delta
                                    except Exception:
                                        pass

                        if low is not None and high is not None and low < high:
                            if not have_pr:
                                tech["price_range"] = [float(low), float(high)]
                            if not have_kz:
                                tech["key_zone"] = [float(low), float(high)]

                        # Ensure a pattern label is present; avoid generic "No Pattern Detected"
                        patt = tech.get("pattern_detected")
                        val = str(patt or "").strip().lower()
                        if not val or val in ("no pattern detected", "none", "unknown", "n/a"):
                            # If we have a clear range, prefer Rectangle/Range, else default to Symmetrical Triangle
                            kz2 = tech.get("key_zone"); pr2 = tech.get("price_range")
                            lo2 = hi2 = None
                            if isinstance(kz2, list) and len(kz2) >= 2 and all(isinstance(x, (int, float)) for x in kz2[:2]):
                                lo2, hi2 = float(min(kz2[0], kz2[1])), float(max(kz2[0], kz2[1]))
                            elif isinstance(pr2, list) and len(pr2) >= 2 and all(isinstance(x, (int, float)) for x in pr2[:2]):
                                lo2, hi2 = float(min(pr2[0], pr2[1])), float(max(pr2[0], pr2[1]))
                            tech["pattern_detected"] = "Rectangle/Range" if (lo2 is not None and hi2 is not None) else "Symmetrical Triangle"
                            # Reflect into patterns array and key_factors for UI
                            if not isinstance(tech.get("patterns"), list):
                                tech["patterns"] = [tech["pattern_detected"]]
                            ai_block = gemini_json.setdefault("ai_analysis", {}) if isinstance(gemini_json, dict) else {}
                            kf = ai_block.setdefault("key_factors", [])
                            if isinstance(kf, list):
                                kf.append(f"Pattern: {tech['pattern_detected']}")
                except Exception:
                    pass

                # 7) Ensure trade_suggestion exists (textual plan even for NO_TRADE)
                ts = ai.setdefault("trade_suggestion", {})
                if not ts.get("signal"):
                    fr = gemini_json.get("final_recommendation") or {}
                    dec = (fr.get("decision") or "").lower()
                    if "long" in dec:
                        ts.setdefault("signal", "BUY")
                    elif "short" in dec:
                        ts.setdefault("signal", "SELL")
                    else:
                        # fall back to trend direction if decision missing
                        td = (ai.get("trend_direction") or "").upper()
                        if td == "BULLISH":
                            ts.setdefault("signal", "BUY")
                        elif td == "BEARISH":
                            ts.setdefault("signal", "SELL")
                        else:
                            ts.setdefault("signal", "NO_TRADE")
                # Provide conditional plan text when neutral
                try:
                    sl = tech.get("support_level"); rl = tech.get("resistance_level")
                    cp = ((gemini_json.get("extraction") or {}).get("metadata") or {}).get("current_price")
                    if ts.get("signal") == "NO_TRADE" and isinstance(sl, (int, float)) and isinstance(rl, (int, float)):
                        ts.setdefault("entry_recommendation", f"Wait for breakout: Long above {rl:.2f} after retest; Short below {sl:.2f}.")
                        ts.setdefault("stop_loss_recommendation", "Place SL beyond the breakout invalidation level.")
                        ts.setdefault("take_profit_recommendation", "Use structure-based targets; scale out at next key levels.")
                    # If signal is BUY/SELL but text fields are empty, mirror RM numeric values later
                    ts.setdefault("ai_confidence", ai.get("confidence_score"))
                except Exception:
                    pass

                # 8) Volatility & sentiment fallbacks
                try:
                    if fundamentals:
                        ai.setdefault("volatility", fundamentals.get("volatility_risk") or "")
                    if not ai.get("market_sentiment") and ai.get("trend_direction"):
                        ai["market_sentiment"] = ai.get("trend_direction")
                    # expose a simple 0..1 strength for UI if missing
                    tech.setdefault("signal_strength", (ai.get("confidence_score") or 0) / 100.0)
                except Exception:
                    pass
        except Exception:
            # best-effort enrichment; never break the flow
            pass

        # Inject backend fundamentals into the AI result to avoid model hallucinations

        # Inject backend fundamentals into the AI result to avoid model hallucinations
        try:
            if fundamentals and isinstance(gemini_json, dict):
                # Minimal compatible shape for current frontend normalizer
                headlines = fundamentals.get("news_headlines") or []
                events = fundamentals.get("upcoming_events") or []
                def _fmt_head(h):
                    t = (h or {}).get("title")
                    s = (h or {}).get("source")
                    return f"{t} — {s}" if t and s else (t or s or "")
                def _fmt_ev(e):
                    name = (e or {}).get("event")
                    ctry = (e or {}).get("country")
                    imp = (e or {}).get("impact")
                    when = (e or {}).get("time")
                    parts = [p for p in [when, name, ctry, imp] if p]
                    return " | ".join(parts)
                gemini_json["fundamentals_news"] = {
                    "news_status": "ok" if headlines or events else "unavailable",
                    "fundamental_bias": fundamentals.get("fundamental_bias"),
                    "key_headlines": [_fmt_head(h) for h in headlines if _fmt_head(h)],
                    "upcoming_events": [_fmt_ev(e) for e in events if _fmt_ev(e)],
                    "volatility_risk": fundamentals.get("volatility_risk"),
                    "confidence_percent": fundamentals.get("confidence_percent"),
                    "_sources": fundamentals.get("sources"),
                    "_raw": fundamentals,
                }
        except Exception:
            pass

        # Backfill/normalize risk and final decision when the model omitted fields
        try:
            if isinstance(gemini_json, dict):
                ai = gemini_json.setdefault("ai_analysis", {})
                rm = ai.setdefault("risk_management", {})
                ts = ai.setdefault("trade_suggestion", {})
                sig = (ts.get("signal") or "").upper()
                if not sig and isinstance(gemini_json.get("final_recommendation"), dict):
                    dec = (gemini_json["final_recommendation"].get("decision") or "").lower()
                    if "long" in dec: sig = "BUY"
                    elif "short" in dec: sig = "SELL"
                    else: sig = "NO_TRADE"
                    ts["signal"] = sig
                # pick entry
                entry = rm.get("entry_price")
                if entry is None:
                    fr = gemini_json.get("final_recommendation") or {}
                    entry = fr.get("entry")
                if entry is None:
                    entry = ((gemini_json.get("extraction") or {}).get("metadata") or {}).get("current_price")
                # risk math
                try:
                    ab = float(ab) if ab is not None else 0.0
                    rpp = float(rpp) if rpp is not None else 0.0
                    slp = float(slp) if slp is not None else 0.0
                    tpp = float(tpp) if tpp is not None else 0.0
                except Exception:
                    ab, rpp, slp, tpp = 0.0, 0.0, 0.0, 0.0
                if entry is not None and (rm.get("stop_loss") is None or rm.get("take_profit") is None):
                    pv = float((user_inputs or {}).get("_point_value") or 1.0)
                    dec = int((user_inputs or {}).get("_price_decimals") or 2)
                    sl_delta = float(slp) * pv
                    tp_delta = float(tpp) * pv
                    if sig == "SELL":
                        sl_val = float(entry) + sl_delta
                        tp_val = float(entry) - tp_delta
                    else:  # BUY or NO_TRADE → default BUY math for readability
                        sl_val = float(entry) - sl_delta
                        tp_val = float(entry) + tp_delta
                    rm.setdefault("entry_price", round(float(entry), dec))
                    rm.setdefault("stop_loss", round(sl_val, dec))
                    rm.setdefault("take_profit", round(tp_val, dec))
                # fill risk_amount, RR, position size
                if rm.get("stop_loss") is not None and rm.get("entry_price") is not None and rm.get("take_profit") is not None:
                    risk_amount = round(ab * (rpp / 100.0), 2) if ab and rpp else 0.0
                    risk_dist = abs(float(rm["entry_price"]) - float(rm["stop_loss"]))
                    reward_dist = abs(float(rm["take_profit"]) - float(rm["entry_price"]))
                    rr = round((reward_dist / risk_dist), 2) if risk_dist else None
                    pos = round((risk_amount / risk_dist), 4) if risk_dist else None
                    rm.setdefault("risk_amount_usd", risk_amount)
                    if rr is not None: rm.setdefault("reward_risk_ratio", rr)
                    if pos is not None: rm.setdefault("position_size", pos)
                    # Always provide textual recommendations for the UI
                    ts.setdefault("entry_recommendation", f"{rm['entry_price']}")
                    ts.setdefault("stop_loss_recommendation", f"{rm['stop_loss']}")
                    ts.setdefault("take_profit_recommendation", f"{rm['take_profit']}")
                # 9) Ensure per_strategy and final_recommendation have non-null numeric fields using RM fallbacks
                try:
                    per = gemini_json.get("per_strategy") or []
                    fr = gemini_json.setdefault("final_recommendation", {})
                    def _num(x):
                        return float(x) if isinstance(x, (int, float)) else None
                    default_entry = _num(rm.get("entry_price")) or _num(fr.get("entry")) or _num(((gemini_json.get("extraction") or {}).get("metadata") or {}).get("current_price"))
                    default_sl = _num(rm.get("stop_loss")) or _num(fr.get("stop_loss"))
                    # try take_profit from fr.take_profits (first) then rm
                    fr_tps = fr.get("take_profits") if isinstance(fr.get("take_profits"), list) else []
                    default_tp = _num(rm.get("take_profit")) or (float(fr_tps[0]) if fr_tps and isinstance(fr_tps[0], (int, float)) else None)

                    patched = []
                    for s in per:
                        if not isinstance(s, dict):
                            patched.append(s)
                            continue
                        ez = s.get("entry_zone")
                        if not (isinstance(ez, list) and len(ez) > 0) and default_entry is not None:
                            s["entry_zone"] = [round(default_entry, int((user_inputs or {}).get("_price_decimals") or 2))]
                        if not isinstance(s.get("stop_loss"), (int, float)) and default_sl is not None:
                            s["stop_loss"] = round(default_sl, int((user_inputs or {}).get("_price_decimals") or 2))
                        tps = s.get("take_profits")
                        if not (isinstance(tps, list) and len(tps) > 0) and default_tp is not None:
                            s["take_profits"] = [round(default_tp, int((user_inputs or {}).get("_price_decimals") or 2))]
                        # normalize bias lower-case string for UI
                        if s.get("bias") is not None:
                            try:
                                s["bias"] = str(s["bias"]).lower()
                            except Exception:
                                pass
                        patched.append(s)
                    if patched:
                        gemini_json["per_strategy"] = patched

                    # Also ensure final_recommendation carries numeric defaults
                    if fr.get("entry") is None and default_entry is not None:
                        fr["entry"] = round(default_entry, int((user_inputs or {}).get("_price_decimals") or 2))
                    if fr.get("stop_loss") is None and default_sl is not None:
                        fr["stop_loss"] = round(default_sl, int((user_inputs or {}).get("_price_decimals") or 2))
                    if not (isinstance(fr.get("take_profits"), list) and fr.get("take_profits")) and default_tp is not None:
                        fr["take_profits"] = [round(default_tp, int((user_inputs or {}).get("_price_decimals") or 2))]
                except Exception:
                    pass

                # Ensure trade plan consistency and generate short, user-friendly reasoning (always aligned with suggested trade)
                try:
                    ai2 = gemini_json.setdefault("ai_analysis", {}) if isinstance(gemini_json, dict) else {}
                    fr2 = gemini_json.setdefault("final_recommendation", {})
                    ts2 = ai2.setdefault("trade_suggestion", {})
                    rm2 = ai2.setdefault("risk_management", {})
                    ta2 = ai2.setdefault("technical_analysis", {})

                    def _num(x):
                        try:
                            return float(x)
                        except Exception:
                            return None

                    # Auto-bump price decimals for small-price instruments to avoid collapsing levels to same rounded value
                    if isinstance(cp, float):
                        cur_dec = int((user_inputs or {}).get("_price_decimals") or 0)
                        if cp < 1 and cur_dec < 5:
                            user_inputs["_price_decimals"] = 5
                        elif cp < 10 and cur_dec < 4:
                            user_inputs["_price_decimals"] = 4
                    decp = int((user_inputs or {}).get("_price_decimals") or 2)
                    cp = _num(((gemini_json.get("extraction") or {}).get("metadata") or {}).get("current_price"))
                    slv = _num(ta2.get("support_level"))
                    rlv = _num(ta2.get("resistance_level"))
                    # prefer zone width if available
                    lo, hi = None, None
                    kzv = ta2.get("key_zone")
                    prv = ta2.get("price_range")
                    if isinstance(kzv, list) and len(kzv) >= 2 and all(isinstance(x, (int, float)) for x in kzv[:2]):
                        lo, hi = float(min(kzv[0], kzv[1])), float(max(kzv[0], kzv[1]))
                    elif isinstance(prv, list) and len(prv) >= 2 and all(isinstance(x, (int, float)) for x in prv[:2]):
                        lo, hi = float(min(prv[0], prv[1])), float(max(prv[0], prv[1]))
                    rng = (hi - lo) if (isinstance(lo, float) and isinstance(hi, float)) else None

                    # derive confidence and initial signal
                    conf = None
                    if isinstance(ts2.get("ai_confidence"), (int, float)):
                        conf = int(ts2.get("ai_confidence"))
                    elif isinstance(ai2.get("confidence_score"), (int, float)):
                        conf = int(ai2.get("confidence_score"))

                    def _derive_signal():
                        sig0 = str(ts2.get("signal") or "").upper()
                        if sig0 in ("BUY", "SELL", "NO_TRADE"):
                            return sig0
                        # derive from FR or RM
                        e = _num(fr2.get("entry"))
                        if e is None:
                            e = _num(rm2.get("entry_price"))
                        tpv = None
                        # final_recommendation.take_profits may be an array
                        tps = fr2.get("take_profits") if isinstance(fr2.get("take_profits"), list) else []
                        if tps and isinstance(tps[0], (int, float)):
                            tpv = float(tps[0])
                        if tpv is None:
                            tpv = _num(rm2.get("take_profit"))
                        if isinstance(e, float) and isinstance(tpv, float):
                            return "BUY" if tpv > e else "SELL" if tpv < e else "NO_TRADE"
                        return "NO_TRADE"

                    sig = _derive_signal()
                    # Decision policy: strongly prefer BUY/SELL; only allow NO_TRADE at extremely low confidence
                    MIN_CONF_NO_TRADE = int(os.environ.get("AI_RARE_NO_TRADE_CONF", "10"))
                    if isinstance(conf, int) and conf < MIN_CONF_NO_TRADE:
                        sig = "NO_TRADE"
                    # If still NO_TRADE, coerce to directional bias using trend/structure to avoid overuse
                    if sig == "NO_TRADE":
                        td = str(ai2.get("trend_direction") or "").upper()
                        if td == "BULLISH":
                            sig = "BUY"
                        elif td == "BEARISH":
                            sig = "SELL"
                        elif isinstance(rlv, float) and isinstance(slv, float) and isinstance(cp, float):
                            mid = (rlv + slv) / 2.0
                            sig = "BUY" if cp >= mid else "SELL"

                    # unify decision across blocks
                    ts2["signal"] = sig
                    if isinstance(fr2.get("decision"), str):
                        if sig == "BUY":
                            fr2["decision"] = "enter_long"
                        elif sig == "SELL":
                            fr2["decision"] = "enter_short"
                        else:
                            fr2["decision"] = "no_trade"
                    else:
                        fr2["decision"] = "enter_long" if sig == "BUY" else "enter_short" if sig == "SELL" else "no_trade"

                    # sanitize numeric plan: avoid nonsensical SL/TP (e.g., negative)
                    entry = _num(rm2.get("entry_price")) or _num(fr2.get("entry")) or cp
                    sl = _num(rm2.get("stop_loss")) or _num(fr2.get("stop_loss"))
                    tpv = None
                    tps = fr2.get("take_profits") if isinstance(fr2.get("take_profits"), list) else []
                    if tps and isinstance(tps[0], (int, float)):
                        tpv = float(tps[0])
                    if tpv is None:
                        tpv = _num(rm2.get("take_profit"))

                    def _round(v):
                        return round(float(v), decp)

                    def _fallback_plan(direction: str):
                        nonlocal entry, sl, tpv
                        # prefer zone bounds; otherwise use small fraction of visible range or current price
                        if direction == "BUY":
                            if entry is None:
                                entry = cp if isinstance(cp, float) else (lo + (rng * 0.5) if isinstance(rng, float) else None)
                            if sl is None or (isinstance(sl, float) and sl <= 0):
                                sl = lo if isinstance(lo, float) else (entry - (rng * 0.25) if isinstance(rng, float) and isinstance(entry, float) else None)
                            if tpv is None or (isinstance(tpv, float) and tpv <= 0):
                                tpv = rlv if isinstance(rlv, float) else (hi if isinstance(hi, float) else (entry + (rng * 0.25) if isinstance(rng, float) and isinstance(entry, float) else None))
                        elif direction == "SELL":
                            if entry is None:
                                entry = cp if isinstance(cp, float) else (hi - (rng * 0.5) if isinstance(rng, float) else None)
                            if sl is None or (isinstance(sl, float) and sl <= 0):
                                sl = rlv if isinstance(rlv, float) else (entry + (rng * 0.25) if isinstance(rng, float) and isinstance(entry, float) else None)
                            if tpv is None or (isinstance(tpv, float) and tpv <= 0):
                                tpv = lo if isinstance(lo, float) else (entry - (rng * 0.25) if isinstance(rng, float) and isinstance(entry, float) else None)

                    if sig in ("BUY", "SELL"):
                        _fallback_plan(sig)

                    # Sanity ordering and plausibility fix-ups
                    def _ensure_order(direction: str):
                        nonlocal entry, sl, tpv
                        try:
                            if not isinstance(entry, float) or not isinstance(sl, float) or not isinstance(tpv, float):
                                return
                            # clamp non-positive
                            if sl <= 0 or tpv <= 0:
                                raise ValueError("non-positive levels")
                            if direction == "BUY":
                                if not (sl < entry < tpv):
                                    # reorder around structure where possible
                                    if isinstance(slv, float) and slv < entry:
                                        sl = slv
                                    elif isinstance(lo, float):
                                        sl = lo
                                    else:
                                        sl = entry - (rng * 0.25 if isinstance(rng, float) else abs(entry) * 0.01)
                                    if isinstance(rlv, float) and rlv > entry:
                                        tpv = rlv
                                    elif isinstance(hi, float):
                                        tpv = hi
                                    else:
                                        tpv = entry + (rng * 0.25 if isinstance(rng, float) else abs(entry) * 0.01)
                                    # final guard
                                    if not (sl < entry < tpv):
                                        sl = min(sl, entry - abs(entry) * 0.005)
                                        tpv = max(tpv, entry + abs(entry) * 0.005)
                            elif direction == "SELL":
                                if not (tpv < entry < sl):
                                    if isinstance(rlv, float) and rlv > entry:
                                        sl = rlv
                                    elif isinstance(hi, float):
                                        sl = hi
                                    else:
                                        sl = entry + (rng * 0.25 if isinstance(rng, float) else abs(entry) * 0.01)
                                    if isinstance(slv, float) and slv < entry:
                                        tpv = slv
                                    elif isinstance(lo, float):
                                        tpv = lo
                                    else:
                                        tpv = entry - (rng * 0.25 if isinstance(rng, float) else abs(entry) * 0.01)
                                    if not (tpv < entry < sl):
                                        tpv = min(tpv, entry - abs(entry) * 0.005)
                                        sl = max(sl, entry + abs(entry) * 0.005)
                        except Exception:
                            pass

                    _ensure_order(sig)

                    # recompute R:R
                    rr = None
                    if isinstance(entry, float) and isinstance(sl, float) and isinstance(tpv, float) and sl != entry:
                        rr = abs(tpv - entry) / abs(entry - sl)
                        # Cap R:R at 100:1 to prevent database overflow and unrealistic values
                        rr = min(rr, 100.0)

                    # write back sanitized numbers
                    if isinstance(entry, float):
                        rm2["entry_price"] = _round(entry)
                        fr2["entry"] = _round(entry)
                    if isinstance(sl, float):
                        rm2["stop_loss"] = _round(sl)
                        fr2["stop_loss"] = _round(sl)
                    if isinstance(tpv, float):
                        rm2["take_profit"] = _round(tpv)
                        fr2["take_profits"] = [ _round(tpv) ]
                    if isinstance(conf, int):
                        ts2.setdefault("ai_confidence", conf)
                    if isinstance(rr, float):
                        rm2["reward_risk_ratio"] = round(rr, 2)

                    # Ensure textual recommendations reflect sanitized plan
                    try:
                        if sig in ("BUY", "SELL"):
                            ts2["entry_recommendation"] = f"{_fmt(entry)}"
                            ts2["stop_loss_recommendation"] = f"{_fmt(sl)}"
                            ts2["take_profit_recommendation"] = f"{_fmt(tpv)}"
                            ts2.setdefault("ai_confidence", conf if isinstance(conf, int) else ai2.get("confidence_score"))
                    except Exception:
                        pass

                    # Rebuild reasoning_markdown to match final signal and sanitized numbers
                    existing = ""
                    for path in [fr2.get("reasoning_markdown"), ai2.get("reasoning_markdown"), ts2.get("reasoning_markdown")]:
                        if isinstance(path, str) and path.strip():
                            existing = path.strip()
                            break

                    def _fmt(n):
                        try:
                            return str(round(float(n), decp))
                        except Exception:
                            return str(n)

                    lines = []
                    lines.append("# Trade Reasoning")
                    if sig:
                        quote = f"> Primary signal: {sig}" + (f" • Confidence {conf}%" if isinstance(conf, int) else "")
                        lines.append(quote)

                    # 1) Market Context (2–3 bullets max)
                    ctx = []
                    if isinstance(ai2.get("trend_direction"), str):
                        ctx.append(f"Trend: {ai2.get('trend_direction')}")
                    if isinstance(ai2.get("market_sentiment"), str):
                        ctx.append(f"Sentiment: {ai2.get('market_sentiment')}")
                    if isinstance(ai2.get("volatility"), str):
                        ctx.append(f"Volatility: {ai2.get('volatility')}")
                    if ctx:
                        lines.append("")
                        lines.append("## 1) Market Context")
                        for c in ctx[:3]:
                            lines.append(f"- {c}")

                    # 2) Technical Factors
                    tlines = []
                    patt = ta2.get("pattern_detected")
                    if isinstance(patt, str) and patt.strip():
                        tlines.append(f"Pattern: {patt}")
                    if isinstance(slv, float):
                        tlines.append(f"Support: {_fmt(slv)}")
                    if isinstance(rlv, float):
                        tlines.append(f"Resistance: {_fmt(rlv)}")
                    if isinstance(lo, float) and isinstance(hi, float):
                        tlines.append(f"Key Zone: {_fmt(lo)} – {_fmt(hi)}")
                        tlines.append(f"Range: {_fmt(lo)} – {_fmt(hi)}")
                    if isinstance(ta2.get("signal_strength"), (int, float)):
                        tlines.append(f"Signal Strength: {round(float(ta2.get('signal_strength')), 2)}")
                    if tlines:
                        lines.append("")
                        lines.append("## 2) Technical Factors")
                        for c in tlines[:6]:
                            lines.append(f"- {c}")

                    # 3) Risk Management
                    if sig == "NO_TRADE":
                        # For no-trade, avoid confusing numeric RM lines (especially if placeholders exist)
                        lines.append("")
                        lines.append("## 3) Risk Management")
                        lines.append("- Stand aside. Use conditional breakout plan; size conservatively if taken.")
                    else:
                        rlines = []
                        if isinstance(entry, float):
                            rlines.append(f"Entry: {_fmt(entry)}")
                        if isinstance(sl, float):
                            rlines.append(f"Stop Loss: {_fmt(sl)}")
                        if isinstance(tpv, float):
                            rlines.append(f"Take Profit: {_fmt(tpv)}")
                        if isinstance(rr, float):
                            rlines.append(f"Reward:Risk ≈ {round(rr, 2)}:1")
                        if rlines:
                            lines.append("")
                            lines.append("## 3) Risk Management")
                            for c in rlines:
                                lines.append(f"- {c}")

                    # 4) Action
                    lines.append("")
                    lines.append("## 4) Action")
                    if sig == "NO_TRADE":
                        # explain why not to trade
                        reasons = []
                        if isinstance(conf, int) and conf < MIN_CONF:
                            reasons.append(f"low confidence ({conf}%)")
                        if isinstance(patt, str) and "range" in patt.lower():
                            reasons.append("range-bound consolidation")
                        txt = " and ".join(reasons) if reasons else "lack of clear confluence"
                        lines.append(f"- No trade due to {txt}. Wait for breakout and retest beyond the key zone.")
                        # Ensure ts textuals convey conditional plan
                        if isinstance(slv, float) and isinstance(rlv, float):
                            ts2.setdefault("entry_recommendation", f"Long above {_fmt(rlv)} after retest; Short below {_fmt(slv)} after retest.")
                            ts2.setdefault("stop_loss_recommendation", "Place SL beyond the breakout invalidation level.")
                            ts2.setdefault("take_profit_recommendation", "Use structure-based targets; scale out at next key levels.")
                    else:
                        # summarize plan in 1 line
                        if isinstance(entry, float) and isinstance(sl, float) and isinstance(tpv, float):
                            lines.append(f"- Plan: {sig} near {_fmt(entry)} with SL {_fmt(sl)} and TP {_fmt(tpv)}.")
                        else:
                            lines.append(f"- Plan: {sig} with structure-based SL/TP around key levels.")

                    md = "\n".join(lines)
                    # Always overwrite to keep consistency with final signal and plan
                    fr2["reasoning_markdown"] = md
                    ai2["reasoning_markdown"] = md
                    ts2["reasoning_markdown"] = md

                    # Build friendly reasoning notes for UI (simple bullets for amateur traders)
                    notes = []
                    try:
                        if sig:
                            notes.append(f"{sig} signal based on current market conditions.")
                        # Trend
                        if isinstance(ai2.get("trend_direction"), str):
                            notes.append(f"The overall market trend is {ai2.get('trend_direction')}, consider adjusting position sizing accordingly.")
                        # Current price context
                        if isinstance(cp, float):
                            notes.append(f"Current price at {_fmt(cp)} observed relative to recent structure.")
                        # Pattern
                        if isinstance(patt, str) and patt.strip():
                            notes.append(f"A {patt} pattern has been identified.")
                        # Support/Resistance
                        if isinstance(slv, float):
                            notes.append(f"Key support level identified at {_fmt(slv)} (invalidation area).")
                        if isinstance(rlv, float):
                            notes.append(f"Key resistance level at {_fmt(rlv)} (potential entry/target reference).")
                        # Key Zone (range)
                        if isinstance(lo, float) and isinstance(hi, float):
                            notes.append(f"Key Zone spans {_fmt(lo)} – {_fmt(hi)}.")
                        # Volatility
                        if isinstance(ai2.get("volatility"), str) and ai2.get("volatility"):
                            notes.append(f"Volatility: {ai2.get('volatility')} — expect corresponding price movement.")
                        # Timeframe
                        tf_note = (user_inputs or {}).get("chart_timeframe")
                        if isinstance(tf_note, str) and tf_note:
                            notes.append(f"Chart timeframe: {tf_note}.")
                        # Action
                        if sig == "NO_TRADE":
                            notes.append("Key action: Wait for breakout and retest beyond the key zone; avoid entries inside the range.")
                        elif isinstance(entry, float) and isinstance(sl, float) and isinstance(tpv, float):
                            notes.append(f"Key action: {sig} near {_fmt(entry)} with SL {_fmt(sl)} and TP {_fmt(tpv)}. Consider scaling at key levels.")
                    except Exception:
                        pass
                    ai2["reasoning_notes"] = notes

                    # Always ensure technical_analysis named strategy buckets exist (API contract for UI)
                    try:
                        ta2 = ai2.setdefault("technical_analysis", {})
                        td = (str(ai2.get("trend_direction") or "").upper() or "NEUTRAL")
                        ms = (str(ai2.get("market_sentiment") or td).upper())
                        vol_str = str(ai2.get("volatility") or "").upper()
                        sstr = 0.0
                        try:
                            sstr = float(ta2.get("signal_strength") or 0.0)
                        except Exception:
                            sstr = 0.0

                        # Structure and range helpers
                        lo, hi = None, None
                        if isinstance(ta2.get("key_zone"), list) and len(ta2.get("key_zone")) >= 2 and all(isinstance(x, (int, float)) for x in ta2.get("key_zone")[:2]):
                            lo, hi = float(min(ta2["key_zone"][0], ta2["key_zone"][1])), float(max(ta2["key_zone"][0], ta2["key_zone"][1]))
                        elif isinstance(ta2.get("price_range"), list) and len(ta2.get("price_range")) >= 2 and all(isinstance(x, (int, float)) for x in ta2.get("price_range")[:2]):
                            lo, hi = float(min(ta2["price_range"][0], ta2["price_range"][1])), float(max(ta2["price_range"][0], ta2["price_range"][1]))
                        rng = (hi - lo) if (isinstance(lo, float) and isinstance(hi, float)) else None
                        mid = (lo + hi) / 2.0 if isinstance(rng, float) else None
                        # structure levels as fallback
                        slv_f = slv if isinstance(slv, float) else lo
                        rlv_f = rlv if isinstance(rlv, float) else hi

                        # Round helper
                        def n(v):
                            try:
                                return round(float(v), decp)
                            except Exception:
                                return v

                        def clamp(x, a, b):
                            try:
                                return max(a, min(b, float(x)))
                            except Exception:
                                return a

                        # Direction from final signal
                        buy = (sig == "BUY")
                        sell = (sig == "SELL")

                        # Confidence per bucket (diversified)
                        base_conf = float(ai2.get("confidence_score") or 30)
                        pa_conf = int(clamp(40 + sstr * 45 + (10 if td in ("BULLISH", "BEARISH") else 0), 20, 90))
                        sd_conf = int(clamp(45 + ( (0.0 if not isinstance(rng, float) or rng == 0 else (1.0 - (abs((rlv_f or cp) - (cp or 0.0)) / rng))) * 35 ), 20, 85))
                        patt_local = str(ta2.get("pattern_detected") or "")
                        cp_conf = int(clamp((60 if patt_local else 30) + sstr * 35, 20, 90))
                        fib_conf = int(clamp(35 + sstr * 30, 15, 75))
                        ma_conf = int(clamp(40 + (20 if td in ("BULLISH","BEARISH") else 0), 20, 75))
                        osc_conf = int(clamp(30 + sstr * 25, 15, 70))
                        vol_conf = int(clamp(35 + (25 if vol_str == "HIGH" else 10), 20, 75))
                        mtf_conf = int(clamp(40 + (25 if ms == td and td in ("BULLISH","BEARISH") else 10), 20, 80))

                        # Compute distinct E/SL/TP for each bucket using structure/range
                        # 1) Price Action / S-R -> trade from S/R toward opposite bound
                        pa_e = cp
                        pa_sl = slv_f if buy else rlv_f
                        pa_tp = rlv_f if buy else slv_f

                        # 2) Supply & Demand -> use mid of zone for entry, extremes for SL/TP
                        if isinstance(rng, float):
                            sd_e = mid
                            sd_sl = (lo - 0.25 * rng) if buy else (hi + 0.25 * rng)
                            sd_tp = hi if buy else lo
                        else:
                            sd_e, sd_sl, sd_tp = cp, pa_sl, pa_tp

                        # 3) Chart Patterns -> breakout plan at edge with modest buffer
                        edge_buf = 0.15 * (rng if isinstance(rng, float) and rng > 0 else (abs(cp) * 0.01 if isinstance(cp, float) else 0.0))
                        if buy:
                            cp_e = (rlv_f or cp) + (edge_buf or 0.0)
                            cp_sl = slv_f if isinstance(slv_f, float) else (cp - (edge_buf or 0.0))
                            cp_tp = hi if isinstance(hi, float) else (cp + 2 * (edge_buf or 0.0))
                        else:
                            cp_e = (slv_f or cp) - (edge_buf or 0.0)
                            cp_sl = rlv_f if isinstance(rlv_f, float) else (cp + (edge_buf or 0.0))
                            cp_tp = lo if isinstance(lo, float) else (cp - 2 * (edge_buf or 0.0))

                        # 4) Fibonacci -> 61.8% level within range
                        if isinstance(rng, float):
                            fib_level = (lo + 0.618 * rng) if buy else (hi - 0.618 * rng)
                            fib_e = fib_level
                            fib_sl = lo if buy else hi
                            fib_tp = hi if buy else lo
                        else:
                            fib_e, fib_sl, fib_tp = cp, pa_sl, pa_tp

                        # 5) Moving Averages -> bias with minor offset from cp
                        if isinstance(rng, float):
                            ma_e = cp + (0.15 * rng if buy else -0.15 * rng)
                            ma_sl = slv_f if buy else rlv_f
                            ma_tp = rlv_f if buy else slv_f
                        else:
                            ma_e, ma_sl, ma_tp = cp, pa_sl, pa_tp

                        # 6) Oscillators / Divergence -> conservative, tight distances around cp
                        if isinstance(rng, float):
                            osc_e = cp
                            osc_sl = cp - (0.30 * rng if buy else -0.30 * rng)
                            osc_tp = cp + (0.60 * rng if buy else -0.60 * rng)
                        else:
                            osc_e, osc_sl, osc_tp = cp, pa_sl, pa_tp

                        # 7) Volume / Volatility -> wider targets
                        if isinstance(rng, float):
                            vol_e = cp
                            vol_sl = cp - (0.40 * rng if buy else -0.40 * rng)
                            vol_tp = cp + (0.80 * rng if buy else -0.80 * rng)
                        else:
                            vol_e, vol_sl, vol_tp = cp, pa_sl, pa_tp

                        # 8) Multi-timeframe -> align with trend, use structure bounds
                        mtf_e = cp
                        mtf_sl = slv_f if buy else rlv_f
                        mtf_tp = rlv_f if buy else slv_f

                        def assign_bucket(bucket: str, explanation: str, e, s, t, conf, bias=None):
                            cur = ta2.get(bucket)
                            if not isinstance(cur, dict):
                                cur = {}
                            cur["bias"] = str((bias or td or "NEUTRAL")).lower()
                            cur["observation"] = explanation
                            if isinstance(e, (int, float)): cur["entry"] = n(e)
                            if isinstance(s, (int, float)): cur["stop_loss"] = n(s)
                            if isinstance(t, (int, float)): cur["take_profit"] = n(t)
                            cur["confidence"] = int(conf)
                            ta2[bucket] = cur

                        patt_local = (ta2 or {}).get("pattern_detected")
                        assign_bucket("price_action_sr", "Support/Resistance structure guiding trade plan.", pa_e, pa_sl, pa_tp, pa_conf)
                        if isinstance(lo, float) and isinstance(hi, float):
                            assign_bucket("supply_demand", f"Key zone {n(lo)}–{n(hi)} influencing orderflow.", sd_e, sd_sl, sd_tp, sd_conf)
                        if isinstance(patt_local, str) and patt_local.strip():
                            assign_bucket("chart_patterns", f"Pattern detected: {patt_local}. Trade continuation/breakout accordingly.", cp_e, cp_sl, cp_tp, cp_conf)
                        assign_bucket("fibonacci", "Fibonacci retracement/extension context if applicable.", fib_e, fib_sl, fib_tp, fib_conf)
                        assign_bucket("moving_averages", "Trend context from moving averages.", ma_e, ma_sl, ma_tp, ma_conf)
                        assign_bucket("oscillators_divergence", "Momentum/oscillator confirmation.", osc_e, osc_sl, osc_tp, osc_conf, bias="neutral")
                        assign_bucket("volume_volatility", f"Volatility: {vol_str or 'UNKNOWN'}.", vol_e, vol_sl, vol_tp, vol_conf)
                        assign_bucket("multi_timeframe", f"Overall trend: {td or 'NEUTRAL'}. Align entries with higher TF.", mtf_e, mtf_sl, mtf_tp, mtf_conf)
                    except Exception:
                        pass

                    # Ensure per_strategy exists; synthesize standard buckets if missing
                    try:
                        existing = gemini_json.get("per_strategy")
                        if not isinstance(existing, list) or len(existing or []) == 0:
                            td = str(ai2.get("trend_direction") or "").lower() or "neutral"
                            conf_pct = int(ai2.get("confidence_score") or 0)
                            # prefer sanitized numbers
                            entry_v = rm2.get("entry_price") or fr2.get("entry") or cp
                            sl_v = rm2.get("stop_loss") or fr2.get("stop_loss")
                            tp_v = rm2.get("take_profit")
                            if tp_v is None:
                                tps_fr = fr2.get("take_profits") if isinstance(fr2.get("take_profits"), list) else []
                                tp_v = float(tps_fr[0]) if tps_fr and isinstance(tps_fr[0], (int, float)) else None

                            def _pack(name: str, explanation: str, bias=td, conf=conf_pct):
                                return {
                                    "name": name,
                                    "explanation": explanation,
                                    "bias": str(bias or "neutral"),
                                    "entry_zone": [float(entry_v)] if isinstance(entry_v, (int, float)) else [],
                                    "stop_loss": float(sl_v) if isinstance(sl_v, (int, float)) else None,
                                    "take_profits": [float(tp_v)] if isinstance(tp_v, (int, float)) else [],
                                    "confidence_percent": int(conf) if isinstance(conf, (int, float)) else None,
                                }

                            built = []
                            # Price Action / S-R from support/resistance
                            if isinstance(ta2, dict) and (isinstance(ta2.get("support_level"), (int, float)) or isinstance(ta2.get("resistance_level"), (int, float))):
                                built.append(_pack("Price Action / S-R", "Support/Resistance structure guiding trade plan."))

                            # Supply & Demand from key_zone
                            if isinstance(ta2, dict) and isinstance(ta2.get("key_zone"), list) and len(ta2.get("key_zone")) >= 2:
                                kz = ta2.get("key_zone")
                                built.append(_pack("Supply & Demand", f"Key zone {kz[0]}–{kz[1]} influencing orderflow."))

                            # Chart Patterns from pattern_detected
                            patt = (ta2 or {}).get("pattern_detected") if isinstance(ta2, dict) else None
                            if isinstance(patt, str) and patt.strip():
                                built.append(_pack("Chart Patterns", f"Pattern detected: {patt}. Trade continuation/breakout accordingly."))

                            # Fibonacci (placeholder if not explicitly detected)
                            built.append(_pack("Fibonacci", "Fibonacci retracement/extension context if applicable.", bias=td))

                            # Moving Averages (trend context)
                            built.append(_pack("Moving Averages", "Trend context from moving averages.", bias=td))

                            # Oscillators / Divergence (momentum)
                            built.append(_pack("Oscillators / Divergence", "Momentum/oscillator confirmation.", bias="neutral"))

                            # Volume / Volatility from ai_analysis.volatility
                            if isinstance(ai2.get("volatility"), str):
                                built.append(_pack("Volume / Volatility", f"Volatility: {ai2.get('volatility')}.", bias=td))

                            # Multi-timeframe alignment from trend
                            built.append(_pack("Multi-timeframe", f"Overall trend: {ai2.get('trend_direction') or 'NEUTRAL'}. Align entries with higher TF."))

                            gemini_json["per_strategy"] = built

                            # Also populate technical_analysis named buckets for API consumers
                            if isinstance(ta2, dict):
                                def _find_obs(label: str) -> str:
                                    for b in built:
                                        if b.get("name") == label:
                                            return b.get("explanation") or ""
                                    return ""
                                def _fill(bucket: str, label: str):
                                    ta2[bucket] = {
                                        "bias": (td or "neutral"),
                                        "observation": _find_obs(label),
                                        "entry": float(entry_v) if isinstance(entry_v, (int, float)) else None,
                                        "stop_loss": float(sl_v) if isinstance(sl_v, (int, float)) else None,
                                        "take_profit": float(tp_v) if isinstance(tp_v, (int, float)) else None,
                                        "confidence": conf_pct,
                                    }
                                _fill("price_action_sr", "Price Action / S-R")
                                _fill("supply_demand", "Supply & Demand")
                                _fill("chart_patterns", "Chart Patterns")
                                _fill("fibonacci", "Fibonacci")
                                _fill("moving_averages", "Moving Averages")
                                _fill("oscillators_divergence", "Oscillators / Divergence")
                                _fill("volume_volatility", "Volume / Volatility")
                                _fill("multi_timeframe", "Multi-timeframe")
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

        # 4) Persist results
        chart_service.write_result(analysis, gemini_json)

        # 4) Do NOT auto-create a trade from analysis. Only report if one already exists.
        trade = None
        try:
            trade = trade_repo.get_by_analysis_id(analysis.analysis_id)
        except Exception:
            trade = None

        # Log: COMPLETED
        audit.create(
            user_id=user.user_id,
            action="COMPLETED",
            entity_type="ChartAnalysis",
            entity_id=analysis.analysis_id,
            values=compact_values({
                "latency_ms": latency_ms,
                "trade_id": str(trade.trade_id) if trade else None,
                "summary": gemini_json.get("summary") if isinstance(gemini_json, dict) else None,
                "risk": gemini_json.get("risk") if isinstance(gemini_json, dict) else None,
                "user_agent": request.headers.get("user-agent"),
            }),
            ip_address=(request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)),
            commit=False,
        )

        db.commit()

        trade_summary = None
        if trade is not None:
            trade_summary = {
                "trade_id": str(trade.trade_id),
                "symbol": trade.symbol,
                "outcome": trade.outcome,
                "suggested_entry_price": str(trade.suggested_entry_price) if trade.suggested_entry_price is not None else None,
                "suggested_stop_loss": str(trade.suggested_stop_loss) if trade.suggested_stop_loss is not None else None,
                "suggested_take_profit": str(trade.suggested_take_profit) if trade.suggested_take_profit is not None else None,
            }

        return JSONResponse(
            content=jsonable_encoder({**gemini_json, "user_inputs": user_inputs, "_analysis_id": str(analysis.analysis_id), "created_at": analysis.created_at, "_latency_ms": latency_ms, "trade_summary": trade_summary})
        )

    except (DataError, IntegrityError, ProgrammingError) as db_err:
        # Persist failed state and rich audit details for admin review
        tb = traceback.format_exc()
        chart_service.fail_analysis(analysis, error=str(db_err))
        audit.create(
            user_id=user.user_id,
            action="DB_ERROR",
            entity_type="ChartAnalysis",
            entity_id=analysis.analysis_id,
            values=compact_values({
                "stage": "persisting_result",
                "error_type": type(db_err).__name__,
                "error_message": str(db_err),
                "traceback": tb,
                "inputs": {"timeframe": timeframe, "symbol": symbol},
                "user_agent": request.headers.get("user-agent"),
            }, max_len=2000),
            ip_address=(request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)),
            commit=False,
        )
        db.rollback()
        db.commit()
        raise HTTPException(status_code=500, detail="Database error while saving analysis.")

    except Exception as e:
        # Fail analysis state and capture full audit log for admins
        tb = traceback.format_exc()
        chart_service.fail_analysis(analysis, error=str(e))
        audit.create(
            user_id=user.user_id,
            action="ANALYZE_FAILED",
            entity_type="ChartAnalysis",
            entity_id=analysis.analysis_id,
            values=compact_values({
                "stage": "gemini_call_or_postprocess",
                "error": str(e),
                "traceback": tb,
                "inputs": {"timeframe": timeframe, "symbol": symbol},
                "user_agent": request.headers.get("user-agent"),
            }, max_len=2000),
            ip_address=(request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)),
            commit=False,
        )
        db.commit()
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {e}")
    # ---------- GET /ai/analyses (general info list with filters) ----------

@router.get("/fundamentals")
def get_fundamentals_endpoint(
    symbol: str = Query(..., description="Instrument symbol e.g. EUR/USD, XAU/USD"),
    timeframe: Optional[str] = Query(None),
    pair = Depends(get_current_user_and_session),
):
    # Authenticated like other chart endpoints
    data = get_fundamentals(symbol, timeframe)
    return data

@router.get("/analyses")
def list_user_analyses(
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="PENDING | COMPLETED | FAILED"),
    date_from: Optional[str] = Query(None, description="ISO date or datetime"),
    date_to: Optional[str] = Query(None, description="ISO date or datetime (exclusive)"),
    trading_type: Optional[str] = Query(None, description="SCALP | SWING | BOTH"),
    outcome: Optional[str] = Query(None, description="WIN | LOSS | NOT_TAKEN | PENDING | SUGGESTED"),
):
    user, _session = pair
    svc = ChartAnalysisService(db)
    rows = svc.list_filtered(
        user_id=str(user.user_id),
        status=status,
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        limit=limit,
        offset=offset,
        trading_type=trading_type,
        outcome=outcome,
    )

    items = []
    for a in rows:
        # extract trading_type from ai_request.user_inputs
        try:
            ai_req = a.ai_request or {}
            tt = (ai_req.get("user_inputs") or {}).get("trading_type")
        except Exception:
            tt = None
        items.append({
            "analysis_id": str(a.analysis_id),
            "created_at": a.created_at,
            "updated_at": a.updated_at,
            "symbol": a.symbol,
            "timeframe": a.timeframe,
            "chart_image_url": a.chart_image_url,
            "chart_image_data": getattr(a, 'chart_image_data', None),
            "status": a.status,
            "error_message": a.error_message,
            "direction": a.direction,
            "market_trend": a.market_trend,
            "pattern": a.pattern,
            "confidence_score": a.confidence_score,
            "trading_type": tt,
            "outcome": getattr(getattr(a, 'trade', None), 'outcome', None),
            # suggestions for quick card summaries
            "suggested_entry_price": a.suggested_entry_price,
            "suggested_stop_loss": a.suggested_stop_loss,
            "suggested_take_profit": a.suggested_take_profit,
            "suggested_risk_reward": a.suggested_risk_reward,
            "suggested_position_size": a.suggested_position_size,
        })
    return {"items": items, "limit": limit, "offset": offset}

# ---------- GET /ai/analyses/{analysis_id} (important/needed info) ----------
@router.get("/analyses/{analysis_id}")
def get_analysis_by_id(
    analysis_id: str = Path(...),
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _session = pair
    svc = ChartAnalysisService(db)
    try:
        a = svc.get_owned_required(user_id=str(user.user_id), analysis_id=analysis_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": str(a.analysis_id),
        "user_id": str(a.user_id),
        "created_at": a.created_at,
        "updated_at": a.updated_at,
        "symbol": a.symbol,
        "timeframe": a.timeframe,
        "chart_image_url": a.chart_image_url,
        "chart_image_data": getattr(a, 'chart_image_data', None),
        "ocr_vendor": a.ocr_vendor,
        "ocr_text": a.ocr_text,
        "ai_model": a.ai_model,
        "ai_request": a.ai_request,
        "ai_response": a.ai_response,
        "status": a.status,
        "error_message": a.error_message,
        "direction": a.direction,
        "market_trend": a.market_trend,
        "pattern": a.pattern,
        "confidence_score": a.confidence_score,
        "insights_json": a.insights_json,
        "suggested_entry_price": a.suggested_entry_price,
        "suggested_stop_loss": a.suggested_stop_loss,
        "suggested_take_profit": a.suggested_take_profit,
        "suggested_risk_reward": a.suggested_risk_reward,
        "suggested_position_size": a.suggested_position_size,
    }

# ---------- GET /ai/result/{analysis_id} (full AI response details) ----------
@router.get("/result/{analysis_id}")
def get_analysis_result(
    analysis_id: str = Path(...),
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _session = pair
    svc = ChartAnalysisService(db)
    try:
        a = svc.get_owned_required(user_id=str(user.user_id), analysis_id=analysis_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Return the AI response block (and extraction) as-is with your exact names
    return {
        "analysis_id": str(a.analysis_id),
        "ai_response": a.ai_response,  # short AI block you store
        "ocr_text": a.ocr_text,        # extraction subset
        "status": a.status,
        "error_message": a.error_message,
    }

# ---------- DELETE /ai/analyses/{analysis_id} ----------
@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(
    analysis_id: str = Path(...),
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _session = pair
    svc = ChartAnalysisService(db)
    rows = svc.delete_owned_by_id(user_id=str(user.user_id), analysis_id=analysis_id)
    if rows == 0:
        # not found or not owned
        raise HTTPException(status_code=404, detail="Analysis not found")
    return None
