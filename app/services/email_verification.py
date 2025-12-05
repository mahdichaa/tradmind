import os, secrets
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from pwdlib import PasswordHash

from app.repositories.email_verification import EmailVerificationRepository
from app.repositories.user import UserRepository
from app.core.mailer_ses import GmailMailer

FRONTEND_VERIFY_URL = os.getenv("FRONTEND_VERIFY_URL", "http://localhost:5173/verify-email")
VERIFY_TOKEN_TTL_MINUTES = int(os.getenv("VERIFY_TOKEN_TTL_MINUTES", "60"))
password_hash = PasswordHash.recommended()

def _utcnow(): return datetime.now(timezone.utc)
def _aware(dt): return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

class EmailVerificationService:
    def __init__(self, db):
        self.db = db
        self.repo = EmailVerificationRepository(db)
        self.users = UserRepository(db)
        self.mailer = GmailMailer()

    def request_verification(self, *, user_id, email: str, ip_address=None, user_agent=None) -> None:
        # Generate selector + verifier
        selector = secrets.token_hex(8)
        verifier = secrets.token_urlsafe(32)
        verifier_hash = password_hash.hash(verifier)
        expires_at = _utcnow() + timedelta(minutes=VERIFY_TOKEN_TTL_MINUTES)

        self.repo.create({
            "user_id": user_id,
            "selector": selector,
            "verifier_hash": verifier_hash,
            "sent_to_email": email,
            "expires_at": expires_at,
            "ip_address": ip_address,
            "user_agent": user_agent,
        }, commit=True)

        verify_link = f"{FRONTEND_VERIFY_URL}?selector={selector}&token={verifier}"
        try:
            self.mailer.send_email_verification(email, verify_link)
        except Exception:
            pass  # avoid leaking infra errors

    def confirm(self, *, selector: str, token: str) -> bool:
        row = self.repo.find_one(where={"selector": selector})
        invalid = HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired verification link")
        if not row or row.is_used:
            raise invalid
        if _aware(row.expires_at) <= _utcnow():
            raise invalid

        # verify token
        try:
            ok = password_hash.verify(token, row.verifier_hash)
        except Exception:
            ok = False
        if not ok:
            raise invalid

        # mark used and set user.email_verified = True
        user = self.users.find_one(where={"user_id": row.user_id})
        if not user:
            raise invalid

        user.email_verified = True
        user.updated_at = _utcnow()
        self.repo.update(row, {"is_used": True, "used_at": _utcnow()}, commit=False)
        self.db.commit()
        return user
