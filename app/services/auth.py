from datetime import datetime
from typing import Optional, Tuple
from fastapi import HTTPException, Request, status 

from psycopg import IntegrityError
from sqlalchemy.orm import Session

from app.repositories.session import SessionRepository
from app.repositories.user import UserRepository
from app.core.security import _as_aware_utc, _client_ip, _user_agent, _utcnow, create_access_token, hash_password, hash_refresh_token, new_refresh_token, refresh_expiry, verify_password
from app.models.user import User
from app.models.enums import UserRole, UserStatus
from app.schemas.auth import UserCreate
from app.models.session import Session as SessionModel
from app.services.email_verification import EmailVerificationService 

def _norm(s: str | None) -> str | None:
    return s.strip() if isinstance(s, str) else s

MAX_ACTIVE_SESSIONS_PER_USER = 10  # safety cap
class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.users = UserRepository(db)
        self.sessions = SessionRepository(db)

    def register(self, payload: UserCreate ,request :Request) -> User:
        email = _norm(payload.email.lower())
        username = _norm(payload.username)

        # Uniqueness
        if self.users.get_by_email(email):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
        if self.users.get_by_username(username):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")

        user = User(
            email=email,
            password_hash=hash_password(payload.password),
            username=username,
            first_name=_norm(payload.first_name),
            last_name=_norm(payload.last_name),
            avatar_url=str(payload.avatar_url) if payload.avatar_url else None,
            timezone=payload.timezone or "UTC",
            role="user",
            status="active",
            email_verified=False,
        )
        self.users.create(user, commit=True)
        EmailVerificationService(self.db).request_verification(
            user_id=user.user_id,
            email=user.email,
            ip_address=_client_ip(request),
            user_agent=_user_agent(request),          # pass from endpoint request if you have it
        )
        
        # persist via your BaseRepository
       
        return user
    
     # ---------- LOGIN with single-session-per-device ----------
    
    def login_with_username_password(
        self,
        *,
        identifier: str,
        password: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
        device_type: Optional[str] = "web",
    ) -> Tuple[str, datetime, str, datetime]:
        ident = (identifier or "").strip()
        user = self.users.get_by_username(ident) or self.users.get_by_email(ident.lower())
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
        if user.status != "active":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is not active")
        if not verify_password(password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

        # issue tokens
        access_token, access_exp = create_access_token(sub=user.username)
        raw_refresh = new_refresh_token()
        refresh_hash = hash_refresh_token(raw_refresh)
        refresh_exp = refresh_expiry()

        # Reuse single session per device (same device_type + user_agent)
        existing = self.sessions.find_active_for_user_and_device(user.user_id, device_type or "web", user_agent or "")
        if existing:
            self.sessions.update(existing, {
                "token": access_token,
                "expires_at": _as_aware_utc(access_exp) or _utcnow(),
                "refresh_token_hash": refresh_hash,
                "refresh_expires_at": _as_aware_utc(refresh_exp) or _utcnow(),
                "rotated_at": _utcnow(),
                "ip_address": ip_address,
                "last_activity": _utcnow(),
                "is_active":True,
            }, commit=True)

            sess = existing
        else:
            # enforce max active sessions
            active = self.sessions.find_active_for_user(user.user_id)
            if len(active) >= MAX_ACTIVE_SESSIONS_PER_USER:
                # revoke oldest extra sessions
                # simple approach: mark all inactive except the newest (keep last MAX-1)
                to_revoke = sorted(active, key=lambda s: s.created_at)[:max(0, len(active) - (MAX_ACTIVE_SESSIONS_PER_USER - 1))]
                if to_revoke:
                    for s in to_revoke:
                        s.is_active = False
                    self.db.commit()

            sess = SessionModel(
                user_id=user.user_id,
                token=access_token,
                refresh_token_hash=refresh_hash,
                refresh_expires_at=refresh_exp,
                ip_address=ip_address,
                user_agent=user_agent,
                device_type=device_type or "web",
                is_active=True,
                expires_at=access_exp,
            )
            self.sessions.create(sess, commit=True)

        user.last_login = _utcnow()
        self.db.commit()

        return access_token, access_exp, raw_refresh, refresh_exp

    # ---------- LOGIN FOR VERIFIED EMAIL (no password) ----------
    def login_for_user(
        self,
        *,
        user: User,
        ip_address: Optional[str],
        user_agent: Optional[str],
        device_type: Optional[str] = "web",
    ) -> Tuple[str, datetime, str, datetime]:
        if user.status != "active":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is not active")

        # issue tokens
        access_token, access_exp = create_access_token(sub=user.username)
        raw_refresh = new_refresh_token()
        refresh_hash = hash_refresh_token(raw_refresh)
        refresh_exp = refresh_expiry()

        # Reuse single session per device (same device_type + user_agent)
        existing = self.sessions.find_active_for_user_and_device(user.user_id, device_type or "web", user_agent or "")
        if existing:
            self.sessions.update(existing, {
                "token": access_token,
                "expires_at": _as_aware_utc(access_exp) or _utcnow(),
                "refresh_token_hash": refresh_hash,
                "refresh_expires_at": _as_aware_utc(refresh_exp) or _utcnow(),
                "rotated_at": _utcnow(),
                "ip_address": ip_address,
                "last_activity": _utcnow(),
                "is_active": True,
            }, commit=True)
        else:
            # enforce max active sessions
            active = self.sessions.find_active_for_user(user.user_id)
            if len(active) >= MAX_ACTIVE_SESSIONS_PER_USER:
                to_revoke = sorted(active, key=lambda s: s.created_at)[:max(0, len(active) - (MAX_ACTIVE_SESSIONS_PER_USER - 1))]
                if to_revoke:
                    for s in to_revoke:
                        s.is_active = False
                    self.db.commit()

            sess = SessionModel(
                user_id=user.user_id,
                token=access_token,
                refresh_token_hash=refresh_hash,
                refresh_expires_at=refresh_exp,
                ip_address=ip_address,
                user_agent=user_agent,
                device_type=device_type or "web",
                is_active=True,
                expires_at=access_exp,
            )
            self.sessions.create(sess, commit=True)

        user.last_login = _utcnow()
        self.db.commit()

        return access_token, access_exp, raw_refresh, refresh_exp

    # ---------- REFRESH (rotate) ----------
    def refresh(self, raw_refresh_token: str) -> Tuple[str, datetime, str, datetime]:
        if not raw_refresh_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")

        refresh_hash = hash_refresh_token(raw_refresh_token)
        sess = self.sessions.get_active_by_refresh_hash(refresh_hash)
        if not sess:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        sess_exp = _as_aware_utc(sess.refresh_expires_at)
        if not sess_exp or sess_exp <= _utcnow():
            sess.is_active = False
            self.db.commit()
            raise HTTPException(status_code=401, detail="Refresh token expired")

        # load user
        from app.repositories.user import UserRepository
        user = UserRepository(self.db).find_one(where={"user_id": sess.user_id})
        if not user or user.status != "active":
            sess.is_active = False
            self.db.commit()
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User invalid")

        # rotate refresh + issue new access
        new_access, new_access_exp = create_access_token(sub=user.username)
        new_raw_refresh = new_refresh_token()
        new_refresh_hash = hash_refresh_token(new_raw_refresh)
        new_refresh_exp = refresh_expiry()

        self.sessions.update(sess, {
            "token": new_access,
            "expires_at": _as_aware_utc(new_access_exp) or _utcnow(),
            "refresh_token_hash": new_refresh_hash,
            "refresh_expires_at": _as_aware_utc(new_refresh_exp) or _utcnow(),
            "rotated_at": _utcnow(),
            "last_activity": _utcnow(),
        }, commit=True)

        return new_access, new_access_exp, new_raw_refresh, new_refresh_exp

    # ---------- LOGOUT ----------
    def logout_by_token(self, token: str) -> SessionModel:
        sess = self.sessions.get_active_by_token(token)
        if not sess:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        sess.is_active = False
        sess.last_activity = _utcnow()
        self.db.commit()
        self.db.refresh(sess)
        return sess
    
    def get_me(self, token: str):
        """
        Resolve the active session by Bearer token and return (user, session).
        """
      
        sess = self.sessions.get_active_by_token(token)
        if not sess or not sess.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

        # optional: also ensure access token not past its (stored) expires_at
        exp = _as_aware_utc(sess.expires_at)
        if exp and exp <= _utcnow():
            # still “active” in DB but access is expired → ask client to refresh
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Access token expired")

        user = self.users.find_one(where={"user_id": sess.user_id})
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        # touch last_activity
        sess.last_activity = _utcnow()
        self.db.commit()

        return user, sess
    
    def change_password(
            self,
            *,
            user_id,
            current_password: str,
            new_password: str,
            current_session_id: Optional[str] = None,
        ) -> Tuple[str, datetime, str, datetime]:
            """
            Change the user's password and (optionally) rotate tokens for the CURRENT session only.
            - Does NOT revoke other sessions/devices.
            - Returns: (new_access, new_access_exp, new_refresh, new_refresh_exp)
            If current_session_id is not provided/found, password is changed but no tokens are returned.

            Raises:
            404  -> user not found
            400  -> current password incorrect OR new password equals old
            400  -> current session not found/inactive (when current_session_id provided)
            """
            users = UserRepository(self.db)
            sessions = SessionRepository(self.db)

            # 1) Load and validate user
            user = users.find_one(where={"user_id": user_id})
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            if not verify_password(current_password, user.password_hash):
                raise HTTPException(status_code=400, detail="Current password is incorrect")

            if verify_password(new_password, user.password_hash):
                raise HTTPException(status_code=400, detail="New password must be different")

            # 2) Update password
            user.password_hash = hash_password(new_password)
            user.updated_at = _utcnow()

            # 3) If no session context provided, finish here (password changed only)
            if not current_session_id:
                self.db.commit()
                # Return “empty” tokens (caller can ignore)
                now = _utcnow()
                return "", now, "", now

            # 4) Rotate ONLY the current session (keep others as-is)
            sess = sessions.find_one(where={
                "and": [
                    {"session_id": current_session_id},
                    {"user_id": user_id},
                    {"is_active": True},
                ]
            })
            if not sess:
                self.db.commit()
                raise HTTPException(status_code=400, detail="Current session not found or inactive")

            # Issue fresh tokens
            new_access, new_access_exp = create_access_token(sub=user.username)
            new_refresh = new_refresh_token()
            new_refresh_hash = hash_refresh_token(new_refresh)
            new_refresh_exp = refresh_expiry()

            # Persist rotation
            sessions.update(sess, {
                "token": new_access,
                "expires_at": _as_aware_utc(new_access_exp) or _utcnow(),
                "refresh_token_hash": new_refresh_hash,
                "refresh_expires_at": _as_aware_utc(new_refresh_exp) or _utcnow(),
                "rotated_at": _utcnow(),
                "last_activity": _utcnow(),
                "is_active": True,
            }, commit=True)

            return new_access, new_access_exp, new_refresh, new_refresh_exp

    def update_user_profile( self, *, user_id, data: dict):
            users = UserRepository(self.db)

            # Whitelist fields (defense-in-depth)
            allowed = {"first_name", "last_name", "username", "avatar_url", "timezone"}
            payload = {k: v for k, v in data.items() if k in allowed and v is not None}

            if not payload:
                return users.find_one(where={"user_id": user_id})

            try:
                updated = users.update_one(where={"user_id": user_id}, data={**payload, "updated_at": _utcnow()})
                if not updated:
                    raise HTTPException(status_code=404, detail="User not found")
                return updated
            except IntegrityError as e:
                # Handle unique violations (e.g., username)
                self.db.rollback()
                raise HTTPException(status_code=400, detail="Username or data already in use")
