import os
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from pwdlib import PasswordHash

from app.repositories.password_reset import PasswordResetRepository
from app.repositories.user import UserRepository
from app.repositories.session import SessionRepository
from app.core.security import hash_password  
from app.core.mailer_ses import GmailMailer

FRONTEND_RESET_URL = os.getenv("FRONTEND_RESET_URL", "http://localhost:5173/reset-password")
RESET_TOKEN_TTL_MINUTES = int(os.getenv("RESET_TOKEN_TTL_MINUTES", "30"))

password_hash = PasswordHash.recommended()

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _as_aware_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

class PasswordResetService:
    def __init__(self, db):
        self.db = db
        self.resets = PasswordResetRepository(db)
        self.users = UserRepository(db)
        self.sessions = SessionRepository(db)
        self.mailer = GmailMailer()

    # --- Forgot password ---
    def request_reset(self, email: str, ip_address: str | None, user_agent: str | None) -> dict:
        user = self.users.find_one(where={"email": email})
        # Always respond OK (avoid email enumeration)
        if not user:
            return {"status": "ok", "code": 200, "message": "If the email exists, a reset link has been sent."}

        selector = secrets.token_hex(8)          # short id
        verifier = secrets.token_urlsafe(32)     # long secret
        verifier_hash = password_hash.hash(verifier)
        expires_at = _utcnow() + timedelta(minutes=RESET_TOKEN_TTL_MINUTES)

        self.resets.create({
            "user_id": user.user_id,
            "selector": selector,
            "verifier_hash": verifier_hash,
            "expires_at": expires_at,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "reason": "forgot",
        }, commit=True)

        reset_link = f"{FRONTEND_RESET_URL}?selector={selector}&token={verifier}"

        # Send email; swallow exceptions to avoid leaking internals
        try:
            self.mailer.send_password_reset(user.email, reset_link)
        except Exception:
            pass

        return {"status": "ok", "code": 200, "message": "If the email exists, a reset link has been sent."}

    # --- Optional pre-check for UX ---
    def verify(self, selector: str, token: str) -> bool:
        row = self.resets.find_one(where={"selector": selector})
        if not row or row.is_used:
            return False
        exp = _as_aware_utc(row.expires_at)
        if not exp or exp <= _utcnow():
            return False
        try:
            return password_hash.verify(token, row.verifier_hash)
        except Exception:
            return False

    # --- Reset password (commit) ---
    def reset_password(self, *, selector: str, token: str, new_password: str, revoke_all_sessions: bool = True) -> bool:
        row = self.resets.find_one(where={"selector": selector})
        invalid = HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset link")

        if not row or row.is_used:
            raise invalid

        exp = _as_aware_utc(row.expires_at)
        if not exp or exp <= _utcnow():
            raise invalid

        try:
            valid = password_hash.verify(token, row.verifier_hash)
        except Exception:
            valid = False
        if not valid:
            raise invalid

        user = self.users.find_one(where={"user_id": row.user_id})
        if not user:
            raise invalid

        # Update user password
        user.password_hash = hash_password(new_password)
        user.updated_at = _utcnow()

        # Mark token as used
        self.resets.update(row, {"is_used": True, "used_at": _utcnow()}, commit=False)

        # Session strategy (secure default): revoke all active sessions
        if revoke_all_sessions:
            active = self.sessions.find(where={"and": [{"user_id": user.user_id}, {"is_active": True}]})
            for s in active:
                s.is_active = False

        self.db.commit()
        return True
