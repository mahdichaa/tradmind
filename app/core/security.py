from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from typing import Optional
from fastapi import Depends, HTTPException, Request , status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
import hashlib
import secrets
import os
from pwdlib import PasswordHash

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME")          # override via env in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", "7"))

_pwd = PasswordHash.recommended()  # Argon2id; no 72-byte issue

def hash_password(raw: str) -> str:
    return _pwd.hash(raw)

def verify_password(raw: str, hashed: str) -> bool:
    return _pwd.verify(raw, hashed)

def create_access_token(sub: str, minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> tuple[str, datetime]:
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    token = jwt.encode({"sub": sub, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
    return token, expire



bearer_scheme = HTTPBearer(auto_error=False)

def get_bearer_token(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return creds.credentials

# === Refresh token (opaque) ===
def new_refresh_token() -> str:
    return secrets.token_urlsafe(64)

def hash_refresh_token(raw: str) -> str:
    # store only the hash in DB
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def refresh_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=REFRESH_TTL_DAYS)

def _utcnow() -> datetime:
    # Always give an aware UTC datetime
    return datetime.now(timezone.utc)

def _as_aware_utc(dt: datetime | None) -> datetime | None:
    # If DB gave naive datetime, treat it as UTC by adding tzinfo
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

def _first_ip(xff: str) -> Optional[str]:
    for part in xff.split(","):
        s = part.strip()
        if s:
            return s
    return None

def _norm(ip: str) -> Optional[str]:
    try:
        a = ip_address(ip)
        return str(a.ipv4_mapped) if getattr(a, "ipv4_mapped", None) else str(a)
    except ValueError:
        return None

def _client_ip(req: Request) -> Optional[str]:
    for h in ("cf-connecting-ip", "true-client-ip", "x-real-ip", "x-forwarded-for"):
        v = req.headers.get(h)
        if not v:
            continue
        ip = _first_ip(v) if h == "x-forwarded-for" else v.strip()
        norm = _norm(ip)
        if norm:
            return norm
    return _norm(req.client.host) if req.client else None
def _user_agent(req: Request) -> str | None:
    return req.headers.get("user-agent")
