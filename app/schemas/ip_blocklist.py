from pydantic import BaseModel, Field
from typing import List

class BlockItemIn(BaseModel):
    item: str = Field(..., description="IP or CIDR, e.g. '203.0.113.10' or '203.0.113.0/24'")

class BlockedIPListOut(BaseModel):
    ips: List[str]

class CheckIPResponse(BaseModel):
    ip: str
    blocked: bool
