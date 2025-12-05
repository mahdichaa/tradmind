from math import ceil
from typing import TypeVar, Generic, List

from fastapi import Query
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Query parameters for pagination"""

    page: int = Field(1, ge=1, description="Page number (starts at 1)")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")

    @property
    def skip(self) -> int:
        """Calculate offset for database query"""
        return (self.page - 1) * self.page_size

    @property
    def take(self) -> int:
        """Alias for page_size"""
        return self.page_size


class PageMeta(BaseModel):
    """Pagination metadata"""

    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper"""

    data: List[T] = Field(..., description="List of items for current page")
    meta: PageMeta = Field(..., description="Pagination metadata")


# -------------------------
# Pagination Dependency
# -------------------------


def get_pagination_params(
        page: int = Query(1, ge=1, description="Page number (starts at 1)"),
        page_size: int = Query(10, ge=1, le=100, description="Items per page"),
) -> PaginationParams:
    """FastAPI dependency for pagination parameters"""
    return PaginationParams(page=page, page_size=page_size)


# -------------------------
# Helper Function
# -------------------------


def paginate(
        items: List[T],
        total: int,
        page: int,
        page_size: int,
) -> PaginatedResponse[T]:
    """
    Create a paginated response from items and total count.

    Args:
        items: List of items for current page
        total: Total number of items across all pages
        page: Current page number
        page_size: Number of items per page

    Returns:
        PaginatedResponse with data and metadata
    """
    total_pages = ceil(total / page_size) if page_size > 0 else 0

    meta = PageMeta(
        page=page,
        page_size=page_size,
        total_items=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )

    return PaginatedResponse(data=items, meta=meta)
