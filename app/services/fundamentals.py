import os
import time
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple, Optional

# Simple in-memory cache (process-local)
_CACHE: Dict[Tuple[str, Optional[str]], Tuple[float, Dict[str, Any]]] = {}


def _now_ts() -> float:
    return time.time()


def _ttl_seconds() -> int:
    try:
        return int(float(os.environ.get("FUNDAMENTALS_TTL_MIN", "30")) * 60)
    except Exception:
        return 1800


# --- Instrument context mapping (FX-first, extensible) ---
_CCY_MAP = {
    "USD": {"country": "United States", "central_bank": "Federal Reserve", "aliases": ["US", "Fed"]},
    "EUR": {"country": "Euro Area", "central_bank": "European Central Bank", "aliases": ["EU", "ECB", "Eurozone"]},
    "GBP": {"country": "United Kingdom", "central_bank": "Bank of England", "aliases": ["UK", "BoE"]},
    "JPY": {"country": "Japan", "central_bank": "Bank of Japan", "aliases": ["BoJ"]},
    "AUD": {"country": "Australia", "central_bank": "Reserve Bank of Australia", "aliases": ["RBA"]},
    "CAD": {"country": "Canada", "central_bank": "Bank of Canada", "aliases": ["BoC"]},
    "CHF": {"country": "Switzerland", "central_bank": "Swiss National Bank", "aliases": ["SNB"]},
    "NZD": {"country": "New Zealand", "central_bank": "Reserve Bank of New Zealand", "aliases": ["RBNZ"]},
    "XAU": {"country": "Global", "central_bank": None, "aliases": ["Gold", "bullion"]},
}


def resolve_instrument_context(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").upper().replace("-", "/").strip()
    parts = sym.split("/") if "/" in sym else [sym]
    countries: List[str] = []
    banks: List[str] = []
    keywords: List[str] = [sym]

    # Global or category-based scopes (e.g., GLOBAL, GLOBAL:CRYPTO, GLOBAL:FOREX, GLOBAL:STOCKS)
    if sym in ("", "GLOBAL") or sym.startswith("GLOBAL"):
        scope = sym.split(":", 1)[1] if ":" in sym else ""
        base_kw = ["markets", "finance", "economy", "trading"]
        if scope == "CRYPTO":
            base_kw += ["crypto", "bitcoin", "ethereum", "binance", "coinbase", "blockchain"]
        elif scope == "FOREX":
            base_kw += ["forex", "FX", "currencies", "ECB", "Fed", "BoJ", "BoE"]
        elif scope == "STOCKS":
            base_kw += ["stocks", "equities", "S&P 500", "NASDAQ", "earnings"]
        keywords = base_kw

    if len(parts) == 2 and parts[0] in _CCY_MAP and parts[1] in _CCY_MAP:
        a = _CCY_MAP[parts[0]]
        b = _CCY_MAP[parts[1]]
        countries.extend([a["country"], b["country"]])
        if a.get("central_bank"): banks.append(a["central_bank"]) 
        if b.get("central_bank"): banks.append(b["central_bank"]) 
        keywords.extend([parts[0], parts[1]] + a.get("aliases", []) + b.get("aliases", []))
    else:
        # Fallback: single asset (index/equity/crypto)
        keywords.extend(parts)

    # Deduplicate, keep order
    def _dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if not s: 
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return {
        "symbol": sym,
        "countries": _dedup(countries),
        "central_banks": _dedup(banks),
        "keywords": _dedup(keywords),
    }


# --- Providers ---

def _http_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = resp.read()
        try:
            return json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return None


def fetch_news_marketaux(context: Dict[str, Any], *, lookback_hours: int = 72, limit: int = 8) -> List[Dict[str, Any]]:
    key = os.environ.get("MARKETAUX_API_KEY") or os.environ.get("MARKETEAUX_API_KEY")
    if not key:
        return []
    # Marketaux query
    base = "https://api.marketaux.com/v1/news/all"
    # Fallback to general market query when symbol/keywords are absent
    q = " ".join(context.get("keywords", [])[:8]) or context.get("symbol", "") or "markets finance"
    params = {
        "api_token": key,
        "limit": str(limit),
        "language": "en",
        # Marketaux accepts relative time like 72h, keep as-is
        "published_after": f"{lookback_hours}h",
        "search": q,
    }
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        js = _http_get_json(url)
        items = js.get("data", []) if isinstance(js, dict) else []
        out: List[Dict[str, Any]] = []
        for it in items:
            out.append({
                "title": it.get("title"),
                "source": (it.get("source") or {}).get("name") if isinstance(it.get("source"), dict) else it.get("source"),
                "url": it.get("url"),
                "published_at": it.get("published_at"),
                "summary": it.get("description") or it.get("snippet"),
            })
        return out
    except Exception:
        return []


def fetch_news_google_rss(context: Dict[str, Any], *, limit: int = 8) -> List[Dict[str, Any]]:
    """Free fallback: Google News RSS search on keywords.
    Note: RSS may not always include publisher or full links; we keep best-effort."""
    q = " ".join(context.get("keywords", [])[:6]) or context.get("symbol", "")
    if not q:
        q = "markets finance"
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(q)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        items = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            source_el = item.find("{*}source")  # namespaced sometimes
            items.append({
                "title": title_el.text if title_el is not None else None,
                "source": source_el.text if source_el is not None else None,
                "url": link_el.text if link_el is not None else None,
                "published_at": (item.find("pubDate").text if item.find("pubDate") is not None else None),
                "summary": None,
            })
            if len(items) >= limit:
                break
        return items
    except Exception:
        return []


def fetch_news_gemini(context: Dict[str, Any], *, limit: int = 8) -> List[Dict[str, Any]]:
    """LLM fallback: ask Gemini to summarize latest market headlines.
    This does not browse the web; treat as heuristic summaries."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        sym = context.get("symbol", "markets")
        prompt = (
            f"Provide up to {limit} concise current market headlines relevant to {sym}. "
            "Return as JSON array of objects with keys: title, source. No prose."
        )
        res = client.models.generate(model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), contents=prompt)
        text = res.text if hasattr(res, "text") else (res.candidates[0].content.parts[0].text if res.candidates else "")
        # Try to parse JSON array
        try:
            arr = json.loads(text)
            out: List[Dict[str, Any]] = []
            for it in arr[:limit]:
                out.append({
                    "title": it.get("title"),
                    "source": it.get("source"),
                    "url": it.get("url"),
                    "published_at": None,
                    "summary": None,
                })
            return out
        except Exception:
            # Fallback: split lines
            lines = [l.strip("- • ") for l in text.splitlines() if l.strip()]
            out: List[Dict[str, Any]] = []
            for l in lines[:limit]:
                out.append({"title": l, "source": "gemini", "url": None, "published_at": None, "summary": None})
            return out
    except Exception:
        return []


def fetch_calendar_trading_economics(context: Dict[str, Any], *, horizon_hours: int = 72, limit: int = 20) -> List[Dict[str, Any]]:
    token = os.environ.get("TRADING_ECONOMICS_API_KEY")
    if not token:
        return []
    countries = context.get("countries", [])
    # TE expects comma-separated country names; if none, fetch global high importance
    country_param = ",".join([urllib.parse.quote_plus(c) for c in countries]) if countries else ""
    base = "https://api.tradingeconomics.com/calendar"
    # importance=2 for high; we can fetch all and filter
    params = {
        "importance": "2,3",  # medium/high
        "limit": str(limit),
    }
    if country_param:
        params["country"] = country_param
    # token can be "key:secret" or just key
    params["c"] = token
    url = base + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
    try:
        js = _http_get_json(url)
        events = js if isinstance(js, list) else []
        out: List[Dict[str, Any]] = []
        now_ts = _now_ts()
        horizon_sec = horizon_hours * 3600
        for ev in events:
            # Common TE fields
            dt = ev.get("DateUTC") or ev.get("Date") or ev.get("DateISO")
            country = ev.get("Country")
            event = ev.get("Event") or ev.get("Category")
            importance = ev.get("Importance") or ev.get("Impact")
            forecast = ev.get("Forecast")
            previous = ev.get("Previous")
            link = ev.get("URL") or ev.get("Link")
            # TE may return lots of past events; we rely on API but keep simple time filter via published date if any
            out.append({
                "time": dt,
                "event": event,
                "country": country,
                "impact": importance,
                "forecast": forecast,
                "previous": previous,
                "url": link,
            })
        return out[:limit]
    except Exception:
        return []


def _volatility_from_events(events: List[Dict[str, Any]]) -> str:
    txt = ",".join([str(e.get("impact", "")) for e in events]).lower()
    if any(x in txt for x in ["high", "3", "very important"]):
        return "HIGH"
    if any(x in txt for x in ["medium", "2"]):
        return "MEDIUM"
    return "LOW" if events else "LOW"


def summarize_fundamentals(symbol: str, news: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Heuristic summary (LLM-free for now). We can upgrade to LLM later.
    bias = "NEUTRAL"
    rationale: List[str] = []
    if events:
        rationale.append(f"{len(events)} medium/high-impact events within 72h")
    if news:
        rationale.append(f"{len(news)} recent headlines relevant to {symbol}")
    vol = _volatility_from_events(events)
    conf = 60 if events or news else 40

    return {
        "fundamental_bias": bias,
        "confidence_percent": conf,
        "rationale": rationale[:4],
        "volatility_risk": vol,
        "upcoming_events": events[:5],
        "news_headlines": news[:5],
    }


def get_fundamentals(symbol: str, timeframe: Optional[str] = None, *, lookback_hours: int = 72, horizon_hours: int = 72) -> Dict[str, Any]:
    sym = (symbol or "").upper()
    key = (sym, (timeframe or None))
    ttl = _ttl_seconds()
    now = _now_ts()
    cached = _CACHE.get(key)
    if cached and (now - cached[0] < ttl):
        return cached[1]

    context = resolve_instrument_context(symbol)
    # Collect news using layered fallbacks (no calendar requirement)
    news = fetch_news_marketaux(context, lookback_hours=lookback_hours)
    news_source = "marketaux" if news else None
    if not news:
        rss = fetch_news_google_rss(context)
        if rss:
            news = rss
            news_source = "rss"
    if not news:
        gem = fetch_news_gemini(context)
        if gem:
            news = gem
            news_source = "gemini"

    # Calendar events (if API key configured)
    events: List[Dict[str, Any]] = []
    calendar_source = None
    try:
        ev = fetch_calendar_trading_economics(context, horizon_hours=horizon_hours)
        if ev:
            events = ev
            calendar_source = "tradingeconomics"
    except Exception:
        events = []

    summary = summarize_fundamentals(sym, news, events)
    result = {
        "symbol": sym,
        "as_of": int(now),
        "sources": {
            "news": news_source,
            "calendar": calendar_source,
        },
        **summary,
    }

    _CACHE[key] = (now, result)
    return result
