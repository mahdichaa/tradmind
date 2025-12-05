
from datetime import datetime
from datetime import datetime, timezone
from typing import Annotated, Optional
import os
import secrets

from fastapi import APIRouter, Depends, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.deps import get_current_user_and_session
from app.core.security import _client_ip, _user_agent, hash_password
from app.database.session import get_db
from app.services.auth import AuthService
from app.services.audit_logs import AuditLogService
from app.schemas.common import Envelope
from app.schemas.auth import ChangePasswordIn, ForgotPasswordIn, MeOut, ResetPasswordIn, SessionOut, Token, LogoutOut, UpdateUserIn, UserCreate, UserOut
from app.services.email_verification import EmailVerificationService
from app.services.password_reset import PasswordResetService
from app.services.google_oauth import verify_google_id_token
from app.repositories.user import UserRepository


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
router = APIRouter(prefix="", tags=["auth"])

# For logout / protected routes we use HTTP Bearer so Swagger shows a token box
bearer_scheme = HTTPBearer(auto_error=False)



COOKIE_NAME = "refresh_token"
COOKIE_PATH = "/api/auth"   # limit cookie to auth paths

# Configurable cookie flags (to support cross-origin dev)
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")

# Sign UP
@router.post("/register", response_model=Envelope[UserOut], status_code=200)
def register(payload: UserCreate,request : Request, db: Session = Depends(get_db)):
    svc = AuthService(db)
    audit = AuditLogService(db)
    try:
        user = svc.register(payload,request)
        # audit: registration success
        audit.create(
            user_id=getattr(user, "user_id", None),
            action="REGISTER_OK",
            entity_type="Auth",
            entity_id=getattr(user, "user_id", None),
            values={"email": (payload.email or "").lower(), "user_agent": _user_agent(request)},
            ip_address=_client_ip(request),
            commit=False,
        )
        db.commit()
        return Envelope(status="ok", code=200, message="User created", data=user)
    except HTTPException as e:
        # audit: registration failed
        try:
            audit.create(
                user_id=None,
                action="REGISTER_FAILED",
                entity_type="Auth",
                entity_id=None,
                values={"email": (payload.email or "").lower(), "reason": e.detail, "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return Envelope(
            status="error",
            code=200,  # you said keep 200 and signal error inside the envelope
            message="Registration failed",
            error={"detail": e.detail, "status_code": e.status_code},
        )


# --------- LOGIN (/token) - OAuth2PasswordRequestForm ----------

@router.post("/google", response_model=Envelope[Token], status_code=200)
def google_login(request: Request, body: dict, db: Session = Depends(get_db)):
    audit = AuditLogService(db)
    """Login/register using a Google ID token (from Google Identity Services)."""
    id_token_raw = (body or {}).get("id_token")
    svc = AuthService(db)
    try:
        claims = verify_google_id_token(id_token_raw)
        sub = claims.get("sub")
        email = (claims.get("email") or "").lower()
        email_verified = bool(claims.get("email_verified"))
        name = claims.get("name") or ""
        picture = claims.get("picture")

        repo = UserRepository(db)
        user = None
        if sub:
            user = repo.get_by_criteria({"google_sub": sub})
        if not user and email:
            user = repo.get_by_email(email)

        if user:
            if getattr(user, "status", None) != "active":
                raise HTTPException(status_code=403, detail="User is not active")
            # mark verified if Google says so
            if email_verified and not getattr(user, "email_verified", False):
                user.email_verified = True
                db.commit()
        else:
            # create a new user from claims
            first_name = None
            last_name = None
            if name:
                parts = name.split(" ")
                first_name = parts[0]
                last_name = " ".join(parts[1:]) or None
            # generate username
            base_username = (email.split("@")[0] if email else f"g_{sub[:8]}") if (email or sub) else f"g_{secrets.token_hex(4)}"
            username = base_username
            i = 1
            while repo.get_by_username(username):
                username = f"{base_username}{i}"
                i += 1
            # generate a random password to satisfy not-null constraint
            random_pw = secrets.token_urlsafe(32)
            from app.models.user import User
            user = User(
                email=email or f"_{sub}@google.local",
                password_hash=hash_password(random_pw),
                username=username,
                first_name=first_name,
                last_name=last_name,
                avatar_url=picture,
                timezone="UTC",
                role="user",
                status="active",
                email_verified=True,  # trust Google
                google_sub=sub,
            )
            repo.create(user, commit=True)

        # Issue tokens and set refresh cookie
        access, access_exp, refresh, refresh_exp = svc.login_for_user(
            user=user,
            ip_address=_client_ip(request),
            user_agent=_user_agent(request),
            device_type="web",
        )
        payload = Envelope(status="ok", code=200, message="Logged in", data=Token(access_token=access, token_type="bearer"))
        response = JSONResponse(payload.model_dump())
        response.set_cookie(
            key=COOKIE_NAME,
            value=refresh,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            path=COOKIE_PATH,
            expires=int(refresh_exp.timestamp()),
        )
        # audit: google login ok
        try:
            audit.create(
                user_id=getattr(user, "user_id", None),
                action="GOOGLE_LOGIN_OK",
                entity_type="Auth",
                entity_id=getattr(user, "user_id", None),
                values={"email": email, "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return response
    except HTTPException as e:
        try:
            audit.create(
                user_id=None,
                action="GOOGLE_LOGIN_FAILED",
                entity_type="Auth",
                entity_id=None,
                values={"reason": e.detail, "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return JSONResponse(Envelope(status="error", code=200, message="Google login failed", error={"detail": e.detail, "status_code": e.status_code}).model_dump())
    except Exception as e:
        try:
            audit.create(
                user_id=None,
                action="GOOGLE_LOGIN_FAILED",
                entity_type="Auth",
                entity_id=None,
                values={"reason": str(e), "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return JSONResponse(Envelope(status="error", code=200, message="Google login failed", error={"detail": str(e), "status_code": 500}).model_dump())


@router.get("/verify-email", response_model=Envelope[Token], status_code=200)
def confirm_email(selector: str, token: str, request: Request, db: Session = Depends(get_db)):
    svc = EmailVerificationService(db)
    try:
        user = svc.confirm(selector=selector, token=token)

        # Issue tokens and set refresh cookie so user is logged in immediately
        auth = AuthService(db)
        access, access_exp, refresh, refresh_exp = auth.login_for_user(
            user=user,
            ip_address=_client_ip(request),
            user_agent=_user_agent(request),
            device_type="web",
        )

        payload = Envelope(
            status="ok",
            code=200,
            message="Email verified",
            data=Token(access_token=access, token_type="bearer"),
        )
        response = JSONResponse(payload.model_dump())
        response.set_cookie(
            key=COOKIE_NAME,
            value=refresh,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            path=COOKIE_PATH,
            expires=int(refresh_exp.timestamp()),
        )
        return response
    except Exception as e:
        from fastapi import HTTPException
        if isinstance(e, HTTPException):
            return Envelope(status="error", code=200, message="Verification failed",
                            error={"detail": e.detail, "status_code": e.status_code})
        return Envelope(status="error", code=200, message="Verification failed",
                        error={"detail": str(e), "status_code": 500})

@router.post("/Login", response_model=Envelope[Token], status_code=200)
def login_for_access_token(
    request: Request,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db),
):
    svc = AuthService(db)
    audit = AuditLogService(db)
    try:
        access, access_exp, refresh, refresh_exp = svc.login_with_username_password(
            identifier=form.username,
            password=form.password,
            ip_address=_client_ip(request),
            user_agent=_user_agent(request),
            device_type="web",
        )
        payload = Envelope(status="ok", code=200, message="Logged in",
                           data=Token(access_token=access, token_type="bearer"))
        response = JSONResponse(payload.model_dump())

        # set HttpOnly refresh cookie
        response.set_cookie(
            key=COOKIE_NAME,
            value=refresh,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            path=COOKIE_PATH,
            expires=int(refresh_exp.timestamp()),
        )
        # audit: login ok
        try:
            audit.create(
                user_id=getattr(getattr(svc, "user_repo", None), "user", None) or None,
                action="LOGIN_OK",
                entity_type="Auth",
                entity_id=None,
                values={"identifier": form.username, "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return response
    except HTTPException as e:
        try:
            audit.create(
                user_id=None,
                action="LOGIN_FAILED",
                entity_type="Auth",
                entity_id=None,
                values={"identifier": form.username, "reason": e.detail, "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return JSONResponse(Envelope(
            status="error", code=200, message="Login failed",
            error={"detail": e.detail, "status_code": e.status_code}
        ).model_dump())

# --------- REFRESH (rotate) ----------
@router.post("/refresh", response_model=Envelope[Token], status_code=200)
def refresh_token(request: Request, db: Session = Depends(get_db)):
    raw_refresh = request.cookies.get(COOKIE_NAME)
    svc = AuthService(db)
    audit = AuditLogService(db)
    try:
        new_access, new_access_exp, new_refresh, new_refresh_exp = svc.refresh(raw_refresh)
        payload = Envelope(status="ok", code=200, message="Refreshed",
                           data=Token(access_token=new_access, token_type="bearer"))
        response = JSONResponse(payload.model_dump())
        response.set_cookie(
            key=COOKIE_NAME,
            value=new_refresh,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            path=COOKIE_PATH,
            expires=int(new_refresh_exp.timestamp()),
        )
        # audit: refresh ok
        try:
            audit.create(
                user_id=None,
                action="REFRESH_OK",
                entity_type="Auth",
                entity_id=None,
                values={"user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return response
    except HTTPException as e:
        try:
            audit.create(
                user_id=None,
                action="REFRESH_FAILED",
                entity_type="Auth",
                entity_id=None,
                values={"reason": e.detail, "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return JSONResponse(Envelope(
            status="error", code=200, message="Refresh failed",
            error={"detail": e.detail, "status_code": e.status_code}
        ).model_dump())

# --------- LOGOUT (Bearer required) ----------
@router.post("/logout", response_model=Envelope[LogoutOut], status_code=200)
def logout(
    response: Response,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    if not credentials or credentials.scheme.lower() != "bearer":
        return Envelope(
            status="error", code=200, message="Logout failed",
            error={"detail": "Not authenticated", "status_code": 401},
        )

    token = credentials.credentials
    svc = AuthService(db)
    sess = svc.logout_by_token(token)

    # Clear refresh cookie
    response.delete_cookie(
        key=COOKIE_NAME,
        path=COOKIE_PATH,
    )

    # Return a Pydantic model; FastAPI handles datetime serialization
    return Envelope(
        status="ok",
        code=200,
        message="Logged out",
        data=LogoutOut(revoked=True, revoked_at=sess.last_activity or datetime.now(timezone.utc)),
    )
# --------------- get me ------------------

@router.get("/me", response_model=Envelope[MeOut], status_code=200)
def get_me(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
):
    if not credentials or credentials.scheme.lower() != "bearer":
        return Envelope(
            status="error", code=200, message="Not authenticated",
            error={"detail": "Missing bearer token", "status_code": 401},
        )

    token = credentials.credentials
    svc = AuthService(db)
    try:
        user, sess = svc.get_me(token)
        print(user.username)
        return Envelope(
            status="ok",
            code=200,
            message="Active session",
            data=MeOut(
                user=UserOut.model_validate(user, from_attributes=True),
                session=SessionOut.model_validate(sess, from_attributes=True),
            ),
        )
    except Exception as e:
        # Let FastAPI’s handler format HTTPException; otherwise wrap as error
        from fastapi import HTTPException
        if isinstance(e, HTTPException):
            return Envelope(
                status="error", code=200, message="Not authenticated",
                error={"detail": e.detail, "status_code": e.status_code},
            )
        # unexpected
        return Envelope(
            status="error", code=200, message="Failed to resolve session",
            error={"detail": str(e), "status_code": 500},
        )

@router.post("/change-password", response_model=Envelope[dict], status_code=200)
def change_password(
    body: ChangePasswordIn,
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, sess = pair
    svc = AuthService(db)
    try:
        svc.change_password(
            user_id=user.user_id,
            current_password=body.current_password,
            new_password=body.new_password,
            current_session_id=str(sess.session_id),
        )
        return Envelope(status="ok", code=200, message="Password changed", data={"revoked_other_sessions": True})
    except Exception as e:
        from fastapi import HTTPException
        if isinstance(e, HTTPException):
            return Envelope(status="error", code=200, message="Password change failed",
                            error={"detail": e.detail, "status_code": e.status_code})
        return Envelope(status="error", code=200, message="Password change failed",
                        error={"detail": str(e), "status_code": 500})

@router.patch("/change_me", response_model=Envelope[UserOut], status_code=200)
def update_me(
    body: UpdateUserIn,
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    user, _ = pair
    svc = AuthService(db)
    try:
        updated = svc.update_user_profile(user_id=user.user_id, data=body.model_dump(exclude_none=True))
        return Envelope(status="ok", code=200, message="Profile updated",
                        data=UserOut.model_validate(updated, from_attributes=True))
    except Exception as e:
        from fastapi import HTTPException
        if isinstance(e, HTTPException):
            return Envelope(status="error", code=200, message="Update failed",
                            error={"detail": e.detail, "status_code": e.status_code})
        return Envelope(status="error", code=200, message="Update failed",
                        error={"detail": str(e), "status_code": 500})
    
@router.post("/forgot-password", response_model=Envelope[dict], status_code=200)
def forgot_password(payload: ForgotPasswordIn, request: Request, db: Session = Depends(get_db)):
    svc = PasswordResetService(db)
    res = svc.request_reset(
        email=payload.email,
        ip_address=getattr(request.client, "host", None),
        user_agent=request.headers.get("user-agent"),
    )
    try:
        AuditLogService(db).create(
            user_id=None,
            action="FORGOT_REQUEST",
            entity_type="Auth",
            entity_id=None,
            values={"email": (payload.email or "").lower(), "status": getattr(res, "status", None), "user_agent": _user_agent(request)},
            ip_address=_client_ip(request),
            commit=False,
        )
        db.commit()
    except Exception:
        pass
    # 'res' is already envelope-shaped
    return res

@router.get("/reset-password/verify", response_model=Envelope[dict], status_code=200)
def verify_reset_token(selector: str, token: str, db: Session = Depends(get_db)):
    svc = PasswordResetService(db)
    ok = svc.verify(selector=selector, token=token)
    try:
        AuditLogService(db).create(
            user_id=None,
            action="RESET_VERIFY_OK" if ok else "RESET_VERIFY_FAILED",
            entity_type="Auth",
            entity_id=None,
            values={"selector": selector},
            ip_address=None,
            commit=False,
        )
        db.commit()
    except Exception:
        pass
    if ok:
        return Envelope(status="ok", code=200, message="Token valid", data={"valid": True})
    return Envelope(status="error", code=200, message="Invalid or expired reset link",
                    error={"detail": "Invalid or expired reset link", "status_code": 400})

@router.post("/reset-password", response_model=Envelope[dict], status_code=200)
def reset_password(body: ResetPasswordIn, db: Session = Depends(get_db)):
    svc = PasswordResetService(db)
    try:
        svc.reset_password(selector=body.selector, token=body.token, new_password=body.new_password, revoke_all_sessions=True)
        try:
            AuditLogService(db).create(
                user_id=None,
                action="RESET_OK",
                entity_type="Auth",
                entity_id=None,
                values={"selector": body.selector},
                ip_address=None,
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return Envelope(status="ok", code=200, message="Password has been reset", data={})
    except Exception as e:
        from fastapi import HTTPException
        try:
            AuditLogService(db).create(
                user_id=None,
                action="RESET_FAILED",
                entity_type="Auth",
                entity_id=None,
                values={"selector": body.selector, "error": str(e)},
                ip_address=None,
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        if isinstance(e, HTTPException):
            return Envelope(status="error", code=200, message="Reset failed",
                            error={"detail": e.detail, "status_code": e.status_code})
        return Envelope(status="error", code=200, message="Reset failed",
                        error={"detail": str(e), "status_code": 500})

@router.post("/resend-verification", response_model=Envelope[dict], status_code=200)
def resend_verification(
    request: Request,
    pair = Depends(get_current_user_and_session),
    db: Session = Depends(get_db),
):
    """Send a fresh email verification link to the authenticated user."""
    user, _sess = pair
    svc = EmailVerificationService(db)
    try:
        svc.request_verification(
            user_id=user.user_id,
            email=user.email,
            ip_address=_client_ip(request),
            user_agent=_user_agent(request),
        )
        try:
            AuditLogService(db).create(
                user_id=user.user_id,
                action="RESEND_VERIFICATION_OK",
                entity_type="Auth",
                entity_id=user.user_id,
                values={"email": user.email, "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        return Envelope(status="ok", code=200, message="Verification email sent", data={"sent": True})
    except Exception as e:
        from fastapi import HTTPException
        try:
            AuditLogService(db).create(
                user_id=user.user_id,
                action="RESEND_VERIFICATION_FAILED",
                entity_type="Auth",
                entity_id=user.user_id,
                values={"email": user.email, "error": str(e), "user_agent": _user_agent(request)},
                ip_address=_client_ip(request),
                commit=False,
            )
            db.commit()
        except Exception:
            pass
        if isinstance(e, HTTPException):
            return Envelope(status="error", code=200, message="Resend failed",
                            error={"detail": e.detail, "status_code": e.status_code})
        return Envelope(status="error", code=200, message="Resend failed",
                        error={"detail": str(e), "status_code": 500})
