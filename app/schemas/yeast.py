from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import Query
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
)

from app.core.query_engine import QueryMixin, OrderByMixin
from app.schemas.base import PartialModel


class YeastBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, example="SafAle BE-256")
    species: str = Field(default="Saccharomyces cerevisiae", max_length=255)
    attenuation: float = Field(..., ge=0, le=100, example=75.5)
    origin: str = Field(..., max_length=100, example="Belgique")


class YeastCreate(YeastBase):
    pass


class YeastUpdate(YeastBase):
    pass


YeastPatch = PartialModel.from_model(YeastBase)


class YeastInDB(YeastBase):
    id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("id", mode="before")
    def convert_uuid(cls, value):
        if isinstance(value, UUID):
            return str(value)
        return value

    model_config = ConfigDict(
        from_attributes=True, json_encoders={datetime: lambda v: v.isoformat()}
    )


class YeastSchema(YeastInDB):
    class Meta:
        description = "Représentation complète d'une levure avec toutes ses propriétés"


class YeastQuery(BaseModel, QueryMixin):
    name_eq: Optional[str] = Field(
        None, min_length=1, max_length=255, example="SafAle BE-256"
    )
    name_like: Optional[str] = Field(
        None, min_length=1, max_length=255, example="SafAle"
    )
    name_in: Optional[List[str]] = Field(
        None, example=["SafAle BE-256", "SafAle US-05"]
    )

    id_eq: Optional[str] = None
    id_in: Optional[List[str]] = None
    id_not_in: Optional[List[str]] = None

    species_eq: Optional[str] = Field(
        None, max_length=255, example="Saccharomyces cerevisiae"
    )
    species_like: Optional[str] = Field(None, max_length=255, example="Saccharomyces")
    species_in: Optional[List[str]] = None

    origin_eq: Optional[str] = Field(None, max_length=100, example="Belgique")
    origin_like: Optional[str] = Field(None, max_length=100, example="Belgique")
    origin_in: Optional[List[str]] = Field(None, example=["Belgique", "France"])

    updated_at_eq: Optional[datetime] = None
    updated_at_lt: Optional[datetime] = None
    updated_at_lte: Optional[datetime] = None
    updated_at_gt: Optional[datetime] = None
    updated_at_gte: Optional[datetime] = None

    created_at_eq: Optional[datetime] = None
    created_at_lt: Optional[datetime] = None
    created_at_lte: Optional[datetime] = None
    created_at_gt: Optional[datetime] = None
    created_at_gte: Optional[datetime] = None


def yeast_query_params(
        name_eq: Optional[str] = Query(
            None, min_length=1, max_length=255
        ),
        name_like: Optional[str] = Query(
            None, min_length=1, max_length=255
        ),
        name_in: Optional[List[str]] = Query(
            None, example=[]
        ),
        id_eq: Optional[str] = Query(None),
        id_in: Optional[List[str]] = Query(None),
        id_not_in: Optional[List[str]] = Query(None),
        species_eq: Optional[str] = Query(
            None, max_length=255
        ),
        species_like: Optional[str] = Query(None, max_length=255),
        species_in: Optional[List[str]] = Query(None),
        origin_eq: Optional[str] = Query(None, max_length=100),
        origin_like: Optional[str] = Query(None, max_length=100),
        origin_in: Optional[List[str]] = Query(None, example=[]),
        updated_at_eq: Optional[datetime] = Query(None),
        updated_at_lt: Optional[datetime] = Query(None),
        updated_at_lte: Optional[datetime] = Query(None),
        updated_at_gt: Optional[datetime] = Query(None),
        updated_at_gte: Optional[datetime] = Query(None),
        created_at_eq: Optional[datetime] = Query(None),
        created_at_lt: Optional[datetime] = Query(None),
        created_at_lte: Optional[datetime] = Query(None),
        created_at_gt: Optional[datetime] = Query(None),
        created_at_gte: Optional[datetime] = Query(None),
) -> YeastQuery:
    """
    Converts query parameters into a YeastQuery model instance.
    This function can be used as a FastAPI dependency.
    """
    return YeastQuery(
        name_eq=name_eq,
        name_like=name_like,
        name_in=name_in,
        id_eq=id_eq,
        id_in=id_in,
        id_not_in=id_not_in,
        species_eq=species_eq,
        species_like=species_like,
        species_in=species_in,
        origin_eq=origin_eq,
        origin_like=origin_like,
        origin_in=origin_in,
        updated_at_eq=updated_at_eq,
        updated_at_lt=updated_at_lt,
        updated_at_lte=updated_at_lte,
        updated_at_gt=updated_at_gt,
        updated_at_gte=updated_at_gte,
        created_at_eq=created_at_eq,
        created_at_lt=created_at_lt,
        created_at_lte=created_at_lte,
        created_at_gt=created_at_gt,
        created_at_gte=created_at_gte,
    )


class YeastOrderBy(BaseModel, OrderByMixin):
    _sortable_fields = {"created_at", "updated_at", "name", "origin", "species"}
    order_by: Optional[str]


def yeast_order_by(
        order_by: Optional[str] = Query(
            None,
            examples=["name:asc,created_at:desc", "name,-created_at"],
        )
) -> YeastOrderBy:
    return YeastOrderBy(order_by=order_by)
