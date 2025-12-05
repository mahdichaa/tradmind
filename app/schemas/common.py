from typing import Generic, Optional, TypeVar, Literal, Any
from pydantic import BaseModel

T = TypeVar("T")

class Envelope(BaseModel, Generic[T]):
    status: Literal["ok", "error"]
    code: int
    message: Optional[str] = None
    data: Optional[T] = None
    error: Optional[dict[str, Any]] = None
