"""
Base exception classes for the application.

Exception hierarchy should follow the layer structure:
- Repository layer raises technical exceptions
- Service layer transforms them to domain exceptions
- API layer transforms them to HTTP exceptions

This file contains base exception classes that can be extended
for specific use cases in your application.
"""


class AppException(Exception):
    """
    Base exception for all application-specific exceptions.

    All custom exceptions should inherit from this class.
    This allows for easy catching of all app exceptions if needed.
    """

    def __init__(self, message: str = "An application error occurred"):
        self.message = message
        super().__init__(self.message)


# ============================================================================
# DOMAIN EXCEPTIONS (raised by service layer)
# ============================================================================


class ValidationError(AppException):
    """
    Raised when business validation rules are violated.

    Examples:
    - User age below minimum requirement
    - Invalid date range
    - Required field missing in business context

    Note: This is different from Pydantic validation (which happens at API layer)
    """

    pass


class NotFoundError(AppException):
    """
    Raised when a requested resource doesn't exist.

    Examples:
    - User not found by ID
    - Order not found
    - Resource deleted or never existed

    Typically maps to HTTP 404
    """

    pass


class AlreadyExistsError(AppException):
    """
    Raised when trying to create a resource that already exists.

    Examples:
    - User with email already exists
    - Duplicate order ID
    - Unique constraint would be violated

    Typically maps to HTTP 409 (Conflict)
    """

    pass


class AuthorizationError(AppException):
    """
    Raised when user doesn't have permission to perform an action.

    Examples:
    - User trying to access another user's data
    - Non-admin trying to perform admin action

    Typically maps to HTTP 403 (Forbidden)
    """

    pass


# ============================================================================
# SPECIFIC DOMAIN EXCEPTIONS
# ============================================================================
    """
    To be used if need specific exception
    """

# ============================================================================
# REPOSITORY/TECHNICAL EXCEPTIONS
# ============================================================================


class RepositoryError(AppException):
    """
    Base exception for repository layer errors.

    Used when database operations fail in ways that aren't
    covered by specific SQLAlchemy exceptions.

    Examples:
    - Generic database errors
    - Transaction failures
    - Connection issues
    """

    pass


class DatabaseConstraintError(RepositoryError):
    """
    Raised when a database constraint is violated.

    This is typically caught in the service layer and transformed
    to a more specific domain exception (like AlreadyExistsError).
    """

    pass


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Repository → Service transformation

# In repository
def create_user(self, email: str):
    try:
        # ... SQLAlchemy code ...
    except IntegrityError as e:
        if "unique constraint" in str(e).lower():
            raise DatabaseConstraintError(f"Constraint violation: {e}")
        raise RepositoryError("Database error") from e

# In service
def create_user(self, data: UserCreate):
    try:
        return self.repo.create_user(data.email)
    except DatabaseConstraintError:
        raise YeastAlreadyExistsError(f"Yest {data.email} already exists")


Example 2: Service → API transformation

# In API endpoint
@router.post("/users")
def create_user(data: UserCreate, service: Yestervice = Depends()):
    try:
        return service.create_user(data)
    except YeastAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


Example 3: Global exception handler (cleaner alternative)

# In api/exception_handlers.py
@app.exception_handler(YeastAlreadyExistsError)
async def already_exists_exception_handler(request: Request, exc: YeastAlreadyExistsError):
    return JSONResponse(status_code=409, content={"detail": exc.message})

# Then endpoints don't need try/except
@router.post("/yest")
def create_user(data: YeastCreate, service: YeastService = Depends()):
    return service.create_yest(data)  # Exceptions handled globally
"""
