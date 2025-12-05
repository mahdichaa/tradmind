"""
Global exception handlers for the API layer.

These handlers transform domain exceptions (from the service layer)
into appropriate HTTP responses. This follows the clean architecture
principle where each layer transforms exceptions as they bubble up:

Repository (DB exceptions) -> Service (Domain exceptions) -> API (HTTP exceptions)

By using global handlers, we avoid repetitive try-except blocks in every endpoint.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    NotFoundError,
    AlreadyExistsError,
    ValidationError,
    AuthorizationError,
    RepositoryError,
    AppException,
)


async def not_found_exception_handler(
        request: Request, exc: NotFoundError
) -> JSONResponse:
    """
    Handle NotFoundError and its subclasses (like YeastNotFoundError).
    Maps to HTTP 404 Not Found.
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.message},
    )


async def already_exists_exception_handler(
        request: Request, exc: AlreadyExistsError
) -> JSONResponse:
    """
    Handle AlreadyExistsError and its subclasses (like YeastAlreadyExistsError).
    Maps to HTTP 409 Conflict.
    """
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": exc.message},
    )


async def validation_exception_handler(
        request: Request, exc: ValidationError
) -> JSONResponse:
    """
    Handle ValidationError (business validation failures).
    Maps to HTTP 400 Bad Request.

    Note: This is different from Pydantic validation errors,
    which are handled by FastAPI automatically.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message},
    )


async def authorization_exception_handler(
        request: Request, exc: AuthorizationError
) -> JSONResponse:
    """
    Handle AuthorizationError.
    Maps to HTTP 403 Forbidden.
    """
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": exc.message},
    )


async def repository_exception_handler(
        request: Request, exc: RepositoryError
) -> JSONResponse:
    """
    Handle RepositoryError (database/technical errors that weren't caught by service layer).
    Maps to HTTP 500 Internal Server Error.

    Note: These should ideally be caught and transformed in the service layer.
    If we're seeing these at the API layer, it means we need better exception
    handling in the service layer.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal database error occurred"},
    )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """
    Fallback handler for any AppException that wasn't caught by more specific handlers.
    Maps to HTTP 500 Internal Server Error.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": exc.message},
    )


# Dictionary mapping exception types to their handlers
# This can be used to register all handlers at once in main.py
EXCEPTION_HANDLERS = {
    NotFoundError: not_found_exception_handler,
    AlreadyExistsError: already_exists_exception_handler,
    ValidationError: validation_exception_handler,
    AuthorizationError: authorization_exception_handler,
    RepositoryError: repository_exception_handler,
    AppException: app_exception_handler,
}
