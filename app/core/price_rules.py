from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


# Basic symbol/asset classification + price precision and "point" size (how many price units per point).
# This is intentionally conservative and covers common retail platforms.
#
# Definitions:
# - decimals: number of decimal places used to round prices for the instrument.
# - point_value: how many price units correspond to 1 "point" of SL/TP distance provided by the user.
#   Examples:
#     * EURUSD: user enters points in pips - 1 point == 0.0001
#     * USDJPY: 1 point == 0.01
#     * Indices (US100, US30, SPX, DAX): 1 point == 1.0
#     * Commodities (XAUUSD, WTI): 1 point == 1.0 (display with 2 decimals)
#     * Crypto (BTCUSD, ETHUSD): 1 point == 1.0 (display with 0-2 decimals typically)
#
# Also provides a loose price range hint for sanity checks in AI prompt context.

_INDEX_TICKERS = {
    "US100", "NAS100", "NASDAQ", "US30", "DJI", "DOW", "SPX", "SP500", "SP-500", "DAX", "DE40", "UK100", "FTSE100",
    "CAC40", "FR40", "JP225", "NIKKEI", "HK50"
}

def _looks_like_forex(sym: str) -> bool:
    s = sym.upper()
    if "/" in s:
        parts = s.split("/")
        return len(parts) == 2 and all(len(p) in (3, 4, 5) for p in parts)
    return bool(re.fullmatch(r"[A-Z]{6,7}", s)) and any(ccy in s for ccy in ("USD","EUR","GBP","JPY","AUD","NZD","CAD","CHF","CNH","CNY"))

def _is_jpy_pair(sym: str) -> bool:
    s = sym.upper().replace("/", "")
    return "JPY" in s and len(s) >= 6

def _looks_like_index(sym: str) -> bool:
    s = sym.upper().replace("/", "")
    if s in _INDEX_TICKERS:
        return True
    # Common aliases embedded
    return any(k in s for k in ("US100","NAS100","NAS","US30","DJI","SPX","SP500","DAX","DE40","FTSE","UK100","NIKKEI","JP225","HK50"))

def _looks_like_crypto(sym: str) -> bool:
    s = sym.upper().replace("/", "")
    return any(s.endswith(q) for q in ("USD","USDT","USDC","EUR")) and any(
        s.startswith(p) for p in ("BTC","ETH","SOL","BNB","XRP","ADA","DOGE","TON","AVAX","DOT","MATIC","LINK","SHIB")
    )

def _looks_like_gold(sym: str) -> bool:
    s = sym.upper().replace("/", "")
    return "XAU" in s or "GOLD" in s

def _looks_like_oil(sym: str) -> bool:
    s = sym.upper().replace("/", "")
    return "WTI" in s or "OIL" in s or "BRENT" in s

def classify_symbol(symbol: Optional[str], asset_type_hint: Optional[str] = None) -> Dict:
    """
    Return a dict with: asset_type, decimals, point_value, price_range (tuple or None)
    Asset types: forex | index | crypto | commodity | stock | other
    """
    sym = (symbol or "").strip().upper()
    at = (asset_type_hint or "").strip().lower()

    # Resolve asset type with heuristics
    resolved = at if at in {"forex","index","crypto","commodity","stock"} else None
    if not resolved:
        if sym:
            if _looks_like_index(sym):
                resolved = "index"
            elif _looks_like_gold(sym) or _looks_like_oil(sym):
                resolved = "commodity"
            elif _looks_like_forex(sym):
                resolved = "forex"
            elif _looks_like_crypto(sym):
                resolved = "crypto"
            elif re.fullmatch(r"[A-Z\.]{1,6}", sym):
                resolved = "stock"
    if not resolved:
        resolved = "other"

    # Defaults by resolved type
    decimals = 2
    point_value = 1.0
    price_range: Optional[Tuple[float,float]] = None

    if resolved == "forex":
        if _is_jpy_pair(sym):
            decimals = 2     # e.g., USDJPY ~150.00 (display)
            point_value = 0.01  # 1 point == 1 pip == 0.01
            price_range = (60.0, 250.0)
        else:
            decimals = 5         # e.g., EURUSD 1.08650 (display)
            point_value = 0.0001 # 1 point == 1 pip == 0.0001
            price_range = (0.5, 2.0)
    elif resolved == "index":
        decimals = 0
        point_value = 1.0
        # Loose combined range for major indices
        price_range = (1000.0, 50000.0)
        if any(k in sym for k in ("US100","NAS","NAS100")):
            price_range = (10000.0, 45000.0)
    elif resolved == "crypto":
        decimals = 2
        point_value = 1.0
        price_range = None  # very volatile; skip strict hint
        if sym.startswith("BTC"):
            price_range = (5000.0, 200000.0)
        elif sym.startswith("ETH"):
            price_range = (100.0, 20000.0)
    elif resolved == "commodity":
        decimals = 2
        point_value = 1.0
        if _looks_like_gold(sym):
            price_range = (1000.0, 3500.0)
        elif _looks_like_oil(sym):
            price_range = (20.0, 200.0)
        else:
            price_range = None
    elif resolved == "stock":
        decimals = 2
        point_value = 1.0
        price_range = (1.0, 5000.0)
    else:
        decimals = 2
        point_value = 1.0
        price_range = None

    return {
        "asset_type": resolved,
        "decimals": int(decimals),
        "point_value": float(point_value),
        "price_range": price_range,
        "symbol_upper": sym,
    }


def points_to_price_delta(points: Optional[float], point_value: float) -> float:
    try:
        if points is None:
            return 0.0
        return float(points) * float(point_value)
    except Exception:
        return 0.0


def round_price(value: Optional[float], decimals: int) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value), int(max(0, decimals)))
    except Exception:
        return value
