from typing import Optional, Literal

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.exceptions import (
    DatabaseConstraintError,
    RepositoryError, NotFoundError, AlreadyExistsError,
)
from app.core.pagination import PaginationParams, paginate, PaginatedResponse
from app.database.session import get_db
from app.repositories import YeastRepository
from app.repositories.base import WhereExpr
from app.schemas.yeast import YeastCreate, YeastUpdate, YeastPatch, YeastSchema


class YeastService:
    def __init__(self, db: Session = Depends(get_db)):
        self.yeast_repository = YeastRepository(db)

    def find_one(self, where: WhereExpr) -> YeastSchema:
        """
        Find a single yeast by filter.

        Raises:
            YeastNotFoundError: When yeast is not found
        """
        yeast = self.yeast_repository.find_one(where=where)
        if yeast is None:
            raise NotFoundError("Yeast not found")
        return YeastSchema.model_validate(yeast)

    def find_paginated(
            self,
            where: WhereExpr,
            order_by: Optional[list[tuple[str, Literal["asc", "desc"]]]] = None,
            pagination: PaginationParams = None,
    ) -> Optional[PaginatedResponse[YeastSchema]]:
        """
        Find paginated yeasts.

        Raises:
            YeastNotFoundError: When page is out of range
        """
        yeast_page, total = self.yeast_repository.find_paginated(
            where=where, order_by=order_by, pagination=pagination
        )

        if not yeast_page and pagination.page > 1:
            raise AlreadyExistsError(f"No results found on page {pagination.page}")

        return paginate(
            [YeastSchema.model_validate(yeast) for yeast in yeast_page],
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )

    def delete_one(self, yeast_id: str) -> bool:
        """
        Delete a yeast by ID.

        Raises:
            YeastNotFoundError: When yeast is not found
            RepositoryError: For database errors
        """
        yeast = self.find_one(where={"id": yeast_id})
        if not yeast:
            raise NotFoundError("Yeast not found")
        try:
            count = self.yeast_repository.delete({**yeast.model_dump()})
            return count == 1
        except RepositoryError:
            raise

    def create(self, payload: YeastCreate) -> YeastSchema:
        """
        Create a new yeast.

        Raises:
            YeastAlreadyExistsError: When yeast already exists
            RepositoryError: For other database errors
        """
        try:
            yeast = self.yeast_repository.create(data={**payload.model_dump()})
            return YeastSchema.model_validate(yeast)
        except DatabaseConstraintError as e:
            raise AlreadyExistsError(
                "A yeast with these attributes already exists"
            ) from e
        except RepositoryError:
            raise

    def update_one(self, yeast_id: str, payload: YeastUpdate) -> YeastSchema:
        """
        Update a yeast completely (PUT).

        Raises:
            YeastNotFoundError: When yeast is not found
            YeastAlreadyExistsError: When update would violate unique constraint
            RepositoryError: For other database errors
        """
        yeast = self.find_one(where={"id": yeast_id})
        data = payload.model_dump()

        try:
            self.yeast_repository.update(where=yeast.model_dump(), data=data)
            return yeast.model_copy(update=data)
        except DatabaseConstraintError as e:
            raise AlreadyExistsError(
                "Update would violate unique constraint"
            ) from e
        except RepositoryError:
            raise

    def patch_one(self, yeast_id: str, payload: YeastPatch) -> YeastSchema:
        """
        Partially update a yeast (PATCH).

        Raises:
            YeastNotFoundError: When yeast is not found
            YeastAlreadyExistsError: When update would violate unique constraint
            RepositoryError: For other database errors
        """
        yeast = self.find_one(where={"id": yeast_id})
        update_data = {
            field_name: field_value
            for field_name, field_value in payload.model_dump().items()
            if field_value is not None
        }
        try:
            self.yeast_repository.update(where=yeast.model_dump(), data=update_data)
            return yeast.model_copy(update=update_data)
        except DatabaseConstraintError as e:
            raise AlreadyExistsError(
                "Update would violate unique constraint"
            ) from e
        except RepositoryError:
            raise
