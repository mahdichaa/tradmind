# app/middleware/ip_blocklist.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Optional
from app.core.ip_blocklist import IPBlocklistStore

def get_client_ip(headers, client) -> Optional[str]:
    xff = headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return client.host if client else None

class IPBlocklistMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, store: IPBlocklistStore):
        super().__init__(app)
        self.store = store

    async def dispatch(self, request, call_next):
        ip = get_client_ip(request.headers, request.client)
        if ip and self.store.is_blocked(ip):
            return JSONResponse(
                status_code=403,
                content={"detail": "Your IP is blocked."},
            )
        return await call_next(request)
