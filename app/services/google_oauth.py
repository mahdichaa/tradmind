import os
from fastapi import HTTPException, status
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID")


def verify_google_id_token(raw_id_token: str) -> dict:
    """
    Verify a Google ID token and return its claims.
    Raises HTTPException 400/401 on failure.
    """
    if not raw_id_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing id_token")
    if not GOOGLE_OAUTH_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Server not configured for Google OAuth (missing GOOGLE_OAUTH_CLIENT_ID)")
    try:
        claims = id_token.verify_oauth2_token(raw_id_token, grequests.Request(), GOOGLE_OAUTH_CLIENT_ID)
        # Basic sanity checks
        if claims.get("aud") != GOOGLE_OAUTH_CLIENT_ID:
            raise HTTPException(status_code=401, detail="Invalid audience for Google token")
        if claims.get("iss") not in ("https://accounts.google.com", "accounts.google.com"):
            raise HTTPException(status_code=401, detail="Invalid issuer for Google token")
        return claims
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")
