# app/core/deps.py
from typing import Optional, Tuple
from datetime import datetime, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.auth import AuthService

bearer_scheme = HTTPBearer(auto_error=False)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

async def get_current_user_and_session(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Tuple[object, object]:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    token = credentials.credentials
    svc = AuthService(db)
    user, sess = svc.get_me(token)  # we wrote this earlier
    return user, sess

async def get_current_user(
    pair: Tuple[object, object] = Depends(get_current_user_and_session),
):
    user, _ = pair
    return user

def require_roles(*allowed_roles: str):
    async def _dep(pair: Tuple[object, object] = Depends(get_current_user_and_session)):
        user, _ = pair
        role = getattr(user, "role", None)
        if role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user
    return _dep
