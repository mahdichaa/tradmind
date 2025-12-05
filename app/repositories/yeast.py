from typing import Optional, Any
from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.core.exceptions import DatabaseConstraintError, RepositoryError
from app.database.session import get_db
from app.models.user import User
from app.repositories.base import BaseRepository, WhereExpr


class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session = Depends(get_db)):
        super().__init__(session, User)

    def create(self, data: dict[str, Any] | User, commit: bool = True) -> User:
        """
        Create a new User record with proper exception handling.

        Raises:
            DatabaseConstraintError: When unique/foreign key constraints are violated
            RepositoryError: For other database errors
        """
        try:
            return super().create(data, commit)
        except IntegrityError as e:
            if "unique constraint" in str(e).lower() or "duplicate" in str(e).lower():
                raise DatabaseConstraintError(
                    f"User with these attributes already exists"
                ) from e
            raise RepositoryError(f"Database constraint violation: {str(e)}") from e
        except SQLAlchemyError as e:
            raise RepositoryError(f"Database error during create: {str(e)}") from e

    def update(
        self, where: WhereExpr | User, data: dict[str, Any], commit: bool = True
    ) -> int | User:
        """
        Update User record(s) with proper exception handling.

        Raises:
            DatabaseConstraintError: When unique/foreign key constraints are violated
            RepositoryError: For other database errors
        """
        try:
            return super().update(where, data, commit)
        except IntegrityError as e:
            if "unique constraint" in str(e).lower() or "duplicate" in str(e).lower():
                raise DatabaseConstraintError(
                    f"Update would violate unique constraint"
                ) from e
            raise RepositoryError(f"Database constraint violation: {str(e)}") from e
        except SQLAlchemyError as e:
            raise RepositoryError(f"Database error during update: {str(e)}") from e

    def delete(self, where: WhereExpr | User | list[User], commit: bool = True) -> int:
        """
        Delete User record(s) with proper exception handling.

        Raises:
            RepositoryError: For database errors
        """
        try:
            return super().delete(where, commit)
        except SQLAlchemyError as e:
            raise RepositoryError(f"Database error during delete: {str(e)}") from e
