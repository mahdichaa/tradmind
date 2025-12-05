from pydantic import BaseModel
from typing import Dict, Any, List, get_origin, get_args, Union, Literal, Optional
from datetime import date, datetime
from typing import Dict, Any, get_origin, get_args, Union, Literal, Optional

from pydantic import BaseModel


def unwrap_optional(annotation):
    """Extract the actual type from Optional[T]"""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if args:
            return args[0]
    return annotation


def to_where(query_model: BaseModel) -> Dict[str, Dict[str, Any]]:
    """
    Convert a Pydantic query model to nested where conditions.

    Convention: field names must end with _<operator>
    Examples: name_eq, age_gte, status_in, created_at_lte

    Supported operators:
    - Comparison: eq, neq, gt, gte, lt, lte
    - Pattern: like, ilike
    - List: in, not_in (use List[T] type for these)

    Args:
        query_model: Any Pydantic model following the naming convention

    Returns:
        Nested dict: {"field_name": {"operator": value}}

    Example:
        class UserQuery(BaseModel):
            name_eq: Optional[str] = None
            age_gte: Optional[int] = None
            status_in: Optional[List[str]] = Query(None)  # Multiple values

        query = UserQuery(name_eq="john", age_gte=18, status_in=["active", "pending"])
        where = to_where(query)
        # Returns: {
        #     "name": {"eq": "john"},
        #     "age": {"gte": 18},
        #     "status": {"in": ["active", "pending"]}
        # }

        # URL: ?name_eq=john&age_gte=18&status_in=active&status_in=pending
    """
    nested = {}

    # Known operators
    operators = {"eq", "neq", "gt", "gte", "lt", "lte", "like", "ilike", "in", "not_in"}

    for field_name, field_info in query_model.model_fields.items():
        value = getattr(query_model, field_name, None)

        # Skip None values and empty lists
        if value is None or (isinstance(value, list) and len(value) == 0):
            continue

        # Find operator suffix
        operator = None
        base_field = None

        for op in operators:
            if field_name.endswith(f"_{op}"):
                operator = op
                base_field = field_name[: -len(op) - 1]  # Remove _operator
                break

        if not operator or not base_field:
            # No valid operator found, skip this field
            continue

        # Get the actual field type for type conversion
        field_type = unwrap_optional(field_info.annotation)

        # Handle date/datetime string conversion
        if field_type is date and isinstance(value, str):
            try:
                value = date.fromisoformat(value)
            except ValueError:
                continue
        elif field_type is datetime and isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                continue

        # Build nested structure
        if base_field not in nested:
            nested[base_field] = {}
        nested[base_field][operator] = value

    return nested


# -------------------------
# Optional: Mixin class
# -------------------------
class QueryMixin:
    """
    Mixin to add query helper methods to any Pydantic model.

    Usage:
        class UserQuery(BaseModel, QueryMixin):
            name_eq: Optional[str] = None
            age_gte: Optional[int] = None
            status_in: Optional[List[str]] = Query(None)

        query = UserQuery(name_eq="john", status_in=["active", "pending"])
        where = query.to_where()
    """

    def to_where(self) -> Dict[str, Dict[str, Any]]:
        """Convert query parameters to nested where conditions"""
        return to_where(self)

    def has_filters(self) -> bool:
        """Check if any filters are applied"""
        return bool(self.to_where())

    def to_dict(self) -> Dict[str, Any]:
        """Get all non-None query parameters"""
        return {k: v for k, v in self.model_dump().items() if v is not None}


# -------------------------
# Optional: Decorator
# -------------------------
def add_to_where_method(model_class):
    """
    Decorator to add to_where() method to a Pydantic model.

    Usage:
        @add_to_where_method
        class UserQuery(BaseModel):
            name_eq: Optional[str] = None
            age_gte: Optional[int] = None

        query = UserQuery(name_eq="john")
        where = query.to_where()
    """

    def _to_where(self) -> Dict[str, Dict[str, Any]]:
        return to_where(self)

    def _has_filters(self) -> bool:
        return bool(to_where(self))

    def _to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    model_class.to_where = _to_where
    model_class.has_filters = _has_filters
    model_class.to_dict = _to_dict

    return model_class


def to_order_by(
        order_by_str: Optional[str],
        valid_fields: set[str],
        default_order: Optional[list[tuple[str, Literal["asc", "desc"]]]] = None
) -> Optional[list[tuple[str, Literal["asc", "desc"]]]]:
    """
    Parse order_by string into list of tuples.

    Args:
        order_by_str: "name:asc,created_at:desc" or "name,-created_at" (- prefix for desc)
        valid_fields: {"name", "created_at", "id"}
        default_order: Default ordering if order_by_str is None

    Returns:
        [("name", "asc"), ("created_at", "desc")] or None

    Raises:
        ValueError: If field or direction is invalid
    """
    if not order_by_str:
        return default_order

    result = []
    for item in order_by_str.split(","):
        item = item.strip()
        if not item:
            continue

        # Support both "field:desc" and "-field" syntax
        if ":" in item:
            field, direction = item.split(":", 1)
            field = field.strip()
            direction = direction.lower().strip()
        elif item.startswith("-"):
            field = item[1:].strip()
            direction = "desc"
        else:
            field = item.strip()
            direction = "asc"

        if field not in valid_fields:
            raise ValueError(f"Invalid field: {field}. Valid fields: {', '.join(sorted(valid_fields))}")

        if direction not in ("asc", "desc"):
            raise ValueError(f"Invalid direction: {direction}. Must be 'asc' or 'desc'")

        result.append((field, direction))

    return result if result else default_order


class OrderByMixin:
    """
    Mixin to add order_by parsing to any Pydantic model.

    Usage:
        class UserQuery(BaseModel, OrderByMixin):
            order_by: Optional[str] = None
            
            # Define valid sortable fields
            _sortable_fields = {"name", "age", "created_at"}

        query = UserQuery(order_by="-created_at,name")
        order = query.parse_order_by()
    """

    def parse_order_by(
            self,
            default_order: Optional[list[tuple[str, Literal["asc", "desc"]]]] = None
    ) -> Optional[list[tuple[str, Literal["asc", "desc"]]]]:
        """
        Parse the order_by field using the model's _sortable_fields.
        
        The model must define a class variable _sortable_fields.
        The model should have an order_by field (typically Optional[str]).
        
        Args:
            default_order: Default ordering if order_by is None
            
        Returns:
            Parsed order_by list or None
            
        Raises:
            AttributeError: If _sortable_fields is not defined
            ValueError: If field or direction is invalid
        """
        if not hasattr(self.__class__, '_sortable_fields'):
            raise AttributeError(
                f"{self.__class__.__name__} must define _sortable_fields class variable "
                "to use parse_order_by()"
            )

        order_by_value = getattr(self, 'order_by', None)
        return to_order_by(order_by_value, self.__class__._sortable_fields, default_order)
