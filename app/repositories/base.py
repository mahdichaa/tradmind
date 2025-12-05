from typing import (
    TypeVar,
    Generic,
    Sequence,
    Optional,
    Any,
    Literal,
    Union,
    overload,
)

import pandas as pd
from sqlalchemy import func, inspect
from sqlalchemy import select, asc, desc, and_, or_, not_, delete
from sqlalchemy.orm import Session, DeclarativeBase
from sqlalchemy.sql import Select

from app.core.pagination import PaginationParams


# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass


T = TypeVar("T", bound=Base)

# Operator keys
ComparisonKey = Literal["=", ">", "<", ">=", "<=", "!=", "like", "in", "not_in"]

# Comparison operator filter
ComparisonFilter = dict[ComparisonKey, Any]

# Recursive where expression
WhereExpr = Union[
    dict[str, Any | ComparisonFilter],  # column filters
    dict[Literal["and", "or"], list["WhereExpr"]],  # logical nesting
    dict[Literal["not"], "WhereExpr"],
]


class BaseRepository(Generic[T]):
    model: type[T]

    def __init__(self, session: Session, model: type[T]):
        self.session = session
        self.model = model

    def _build_where(self, where: WhereExpr):
        """Recursively converts a WhereExpr into a SQLAlchemy expression."""
        if not where:
            return None

        expressions = []

        for key, value in where.items():
            key_lower = key.lower()

            # Logical operators
            if key_lower == "and":
                sub_exprs = [self._build_where(cond) for cond in value]
                expressions.append(and_(*[e for e in sub_exprs if e is not None]))
            elif key_lower == "or":
                sub_exprs = [self._build_where(cond) for cond in value]
                expressions.append(or_(*[e for e in sub_exprs if e is not None]))
            elif key_lower == "not":
                expr = self._build_where(value)
                if expr is not None:
                    expressions.append(not_(expr))
            else:
                # Field condition
                column = getattr(self.model, key, None)
                if column is None:
                    raise ValueError(f"{self.model} has no column '{key}'")

                if isinstance(value, dict):  # operator-based
                    for op, v in value.items():
                        match op:
                            case "=":
                                expressions.append(column == v)
                            case "!=":
                                expressions.append(column != v)
                            case ">":
                                expressions.append(column > v)
                            case "<":
                                expressions.append(column < v)
                            case ">=":
                                expressions.append(column >= v)
                            case "<=":
                                expressions.append(column <= v)
                            case "like":
                                expressions.append(column.like(v))
                            case "in":
                                expressions.append(column.in_(v))
                            case "not_in":
                                expressions.append(column.not_in(v))
                            case "eq":
                                expressions.append(column.__eq__(v))
                            case _:
                                raise ValueError(f"Unsupported operator '{op}'")
                elif isinstance(value, (list, tuple, set)):
                    expressions.append(column.in_(value))
                else:
                    expressions.append(column == value)

        return and_(*expressions) if len(expressions) > 1 else expressions[0]

    def find(
            self,
            where: Optional[WhereExpr] = None,
            order_by: Optional[list[tuple[str, Literal["asc", "desc"]]]] = None,
            skip: Optional[int] = None,
            take: Optional[int] = None,
            select_cols: Optional[list[str]] = None,
            as_df: bool = False,
            raw: bool = False,
    ) -> Sequence[T] | pd.DataFrame | Select:
        """
        TypeORM-style find() with typed nested filters and operators.
        """
        stmt = (
            select(self.model)
            if not select_cols
            else select(*[getattr(self.model, c) for c in select_cols])
        )

        where_expr = self._build_where(where) if where else None
        if where_expr is not None:
            stmt = stmt.where(where_expr)

        if order_by is not None:
            for col, direction in order_by:
                column = getattr(self.model, col, None)
                if not column:
                    raise ValueError(f"{self.model} has no column '{col}'")
                stmt = stmt.order_by(
                    asc(column) if direction == "asc" else desc(column)
                )

        if skip is not None:
            stmt = stmt.offset(skip)
        if take is not None:
            stmt = stmt.limit(take)

        if raw:
            return stmt
        if as_df:
            return pd.read_sql(stmt, self.session.bind)
        return self.session.execute(stmt).scalars().all()

    def find_one(self, where: Optional[WhereExpr] = None) -> Optional[T]:
        stmt = self.find(where=where, raw=True, take=1)
        return self.session.execute(stmt).scalars().first()

    def count(self, where: Optional[WhereExpr] = None) -> int:
        """Count total items matching the filter"""

        stmt = select(func.count()).select_from(self.model)

        if where:
            where_expr = self._build_where(where)
            if where_expr is not None:
                stmt = stmt.where(where_expr)

        return self.session.execute(stmt).scalar_one()

    def find_paginated(
            self,
            where: Optional[WhereExpr] = None,
            order_by: Optional[list[tuple[str, Literal["asc", "desc"]]]] = None,
            pagination: PaginationParams = None,
    ):
        """
        Find items with pagination support.

        Returns tuple of (items, total_count)
        """
        # Get total count
        total = self.count(where=where)

        # Get paginated items
        items = self.find(
            where=where,
            order_by=order_by,
            skip=pagination.skip if pagination else None,
            take=pagination.take if pagination else None,
        )

        return items, total

    @overload
    def create(self, data: T, commit: bool = True) -> T:
        ...

    @overload
    def create(self, data: dict[str, Any], commit: bool = True) -> T:
        ...

    def create(self, data: dict[str, Any] | T, commit: bool = True) -> T:
        """
        Create a new record.

        Args:
            data: Either a dictionary of column values or a model instance
            commit: Whether to commit the transaction immediately

        Returns:
            The created model instance

        Example:
            # Using dict (validates columns at runtime)
            user = repo.create({"name": "John", "email": "john@example.com"})

            # Using model instance (type-safe at compile time)
            user = repo.create(User(name="John", email="john@example.com"))
        """
        if isinstance(data, dict):
            # Validate that all keys are valid columns
            mapper = inspect(self.model)
            valid_columns = {col.key for col in mapper.columns}
            invalid_keys = set(data.keys()) - valid_columns

            if invalid_keys:
                raise ValueError(
                    f"{self.model.__name__} has no columns: {', '.join(invalid_keys)}"
                )

            instance = self.model(**data)
        else:
            instance = data

        self.session.add(instance)

        if commit:
            self.session.commit()
            self.session.refresh(instance)
        else:
            self.session.flush()

        return instance

    @overload
    def create_many(self, data: list[T], commit: bool = True) -> list[T]:
        ...

    @overload
    def create_many(
            self, data: list[dict[str, Any]], commit: bool = True
    ) -> list[T]:
        ...

    def create_many(
            self, data: list[dict[str, Any]] | list[T], commit: bool = True
    ) -> list[T]:
        """
        Create multiple records in a single transaction.

        Args:
            data: List of either dictionaries or model instances
            commit: Whether to commit the transaction immediately

        Returns:
            List of created model instances

        Example:
            users = repo.create_many([
                {"name": "John", "email": "john@example.com"},
                {"name": "Jane", "email": "jane@example.com"},
            ])
        """
        instances = []

        if data and isinstance(data[0], dict):
            # Validate columns once
            mapper = inspect(self.model)
            valid_columns = {col.key for col in mapper.columns}

            for item in data:
                invalid_keys = set(item.keys()) - valid_columns
                if invalid_keys:
                    raise ValueError(
                        f"{self.model.__name__} has no columns: {', '.join(invalid_keys)}"
                    )
                instances.append(self.model(**item))
        else:
            instances = data

        self.session.add_all(instances)

        if commit:
            self.session.commit()
            for instance in instances:
                self.session.refresh(instance)
        else:
            self.session.flush()

        return instances

    @overload
    def delete(self, where: WhereExpr, commit: bool = True) -> int:
        ...

    @overload
    def delete(self, where: T, commit: bool = True) -> int:
        ...

    @overload
    def delete(self, where: list[T], commit: bool = True) -> int:
        ...

    def delete(self, where: WhereExpr | T | list[T], commit: bool = True) -> int:
        """
        Delete records matching the filter, delete a specific instance, or delete multiple instances.

        Args:
            where: Either a filter expression, a model instance, or a list of model instances to delete
            commit: Whether to commit the transaction immediately

        Returns:
            Number of deleted records

        Example:
            # Delete by filter
            count = repo.delete(where={"status": "inactive"})

            # Delete a specific instance
            user = repo.find_one(where={"id": 123})
            repo.delete(where=user)

            # Delete multiple instances
            users = repo.find(where={"status": "inactive"})
            count = repo.delete(where=users)
        """
        # If it's a list of model instances, delete them all
        if isinstance(where, list):
            if not where:
                return 0

            if not isinstance(where[0], self.model):
                raise ValueError(f"Expected list of {self.model.__name__} instances")

            for instance in where:
                self.session.delete(instance)

            if commit:
                self.session.commit()
            else:
                self.session.flush()

            return len(where)

        # If it's a single model instance, delete it directly
        if isinstance(where, self.model):
            self.session.delete(where)

            if commit:
                self.session.commit()
            else:
                self.session.flush()

            return 1

        # Otherwise treat as a filter expression
        if where is None:
            raise ValueError("where parameter is required for delete operation")

        stmt = delete(self.model)

        where_expr = self._build_where(where)
        if where_expr is not None:
            stmt = stmt.where(where_expr)

        result = self.session.execute(stmt)

        if commit:
            self.session.commit()
        else:
            self.session.flush()

        return result.rowcount

    @overload
    def update(
            self, where: WhereExpr, data: dict[str, Any], commit: bool = True
    ) -> int:
        ...

    @overload
    def update(self, where: T, data: dict[str, Any], commit: bool = True) -> T:
        ...

    def update(
            self, where: WhereExpr | T, data: dict[str, Any], commit: bool = True
    ) -> int | T:
        """
        Update records matching the filter or update a specific instance.

        Args:
            where: Either a filter expression or a model instance to update
            data: Dictionary of column values to update
            commit: Whether to commit the transaction immediately

        Returns:
            Number of updated records, or the updated instance if passing an object

        Example:
            # Update by filter
            count = repo.update(
                where={"status": "inactive"},
                data={"status": "active"}
            )

            # Update a specific instance
            user = repo.find_one(where={"id": 123})
            updated_user = repo.update(
                where=user,
                data={"name": "New Name"}
            )
        """
        # Validate that all keys are valid columns
        mapper = inspect(self.model)
        valid_columns = {col.key for col in mapper.columns}
        invalid_keys = set(data.keys()) - valid_columns

        if invalid_keys:
            raise ValueError(
                f"{self.model.__name__} has no columns: {', '.join(invalid_keys)}"
            )

        # If it's a model instance, update it directly
        if isinstance(where, self.model):
            for key, value in data.items():
                setattr(where, key, value)

            if commit:
                self.session.commit()
                self.session.refresh(where)
            else:
                self.session.flush()

            return where

        # Otherwise find and update records matching the filter
        stmt = self.find(where=where, raw=True)
        items = self.session.execute(stmt).scalars().all()

        for item in items:
            for key, value in data.items():
                setattr(item, key, value)

        if commit:
            self.session.commit()
        else:
            self.session.flush()

        return len(items)

    def update_one(
            self, where: WhereExpr, data: dict[str, Any], commit: bool = True
    ) -> Optional[T]:
        """
        Update a single record matching the filter.

        Args:
            where: Filter expression to match the record to update
            data: Dictionary of column values to update
            commit: Whether to commit the transaction immediately

        Returns:
            The updated model instance, or None if not found

        Example:
            user = repo.update_one(
                where={"id": 123},
                data={"name": "John Updated"}
            )
        """
        # Validate columns
        mapper = inspect(self.model)
        valid_columns = {col.key for col in mapper.columns}
        invalid_keys = set(data.keys()) - valid_columns

        if invalid_keys:
            raise ValueError(
                f"{self.model.__name__} has no columns: {', '.join(invalid_keys)}"
            )

        # Find the record
        item = self.find_one(where=where)

        if item is None:
            return None

        # Update fields
        for key, value in data.items():
            setattr(item, key, value)

        if commit:
            self.session.commit()
            self.session.refresh(item)
        else:
            self.session.flush()

        return item

    def exists(self, where: Optional[WhereExpr] = None) -> bool:
        """
        Check if any records match the filter.

        Args:
            where: Filter expression to check

        Returns:
            True if at least one matching record exists, False otherwise
        """
        stmt = select(func.count()).select_from(self.model)

        if where:
            where_expr = self._build_where(where)
            if where_expr is not None:
                stmt = stmt.where(where_expr)

        count = self.session.execute(stmt).scalar_one()
        return count > 0
