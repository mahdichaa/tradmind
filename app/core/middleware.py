"""
Custom middleware for the Golden Recipe API.

This module contains middleware components that handle cross-cutting concerns
across all requests.
"""

import logging
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingCleanupMiddleware(BaseHTTPMiddleware):
    """
    Middleware to clean up duplicate handlers dynamically added by SQLAlchemy.

    Background:
    SQLAlchemy adds its own StreamHandler when creating new database connections.
    This happens dynamically during request processing, bypassing our monkey-patch.
    The result is duplicate log lines (one from SQLAlchemy, one from root logger).

    Solution:
    Check before and after each request if SQLAlchemy has added handlers.
    If found, call setup_logging() to remove them.
    This ensures single, colored log output in development.
    """

    async def dispatch(self, request, call_next):
        """Process request and clean up SQLAlchemy handlers before and after."""
        # Clean up before request (in case handlers were added by previous request)
        self._cleanup_sqlalchemy_handlers()

        response = await call_next(request)

        # Clean up after request (in case new DB connection added handlers during this request)
        self._cleanup_sqlalchemy_handlers()

        return response

    def _cleanup_sqlalchemy_handlers(self):
        """Remove SQLAlchemy handlers if they exist."""
        sql_logger = logging.getLogger("sqlalchemy.engine.Engine")
        if sql_logger.handlers:
            from utils.logger import setup_logging

            setup_logging()
