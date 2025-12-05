from typing import TypeVar, Type, Optional, Generic
# app/common/enums.py
from enum import Enum
from pydantic import BaseModel, create_model

T = TypeVar("T", bound=BaseModel)

class PartialModel(Generic[T]):
    """
    A mixin to generate a 'partial' version of a Pydantic model
    where all fields become optional (like PATCH schemas).
    """

    @classmethod
    def from_model(cls, model: Type[T]) -> Type[T]:
        """
        Dynamically creates a partial version of the given model.
        Example:
            PartialUser = PartialModel.from_model(User)
        """
        return create_model(
            f"Partial{model.__name__}",
            **{
                name: (Optional[field.annotation], None)
                for name, field in model.model_fields.items()
            },
            __base__=(BaseModel,)
        )


