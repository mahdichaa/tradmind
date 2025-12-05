# app/schemas/ai_settings.py
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class OpenRouterModel(BaseModel):
    """OpenRouter model information"""
    id: str
    name: str
    context_length: Optional[int] = None
    pricing: Optional[Dict[str, Any]] = None
    architecture: Optional[Dict[str, Any]] = None


class AIConfigOut(BaseModel):
    """AI configuration output schema"""
    id: str
    openrouter_api_key_masked: Optional[str] = Field(default=None, description="Masked API key (last 4 chars)")
    api_keys: Optional[List[Dict[str, Any]]] = Field(default=None, description="Array of masked API keys with metadata")
    selected_model: Optional[str] = None
    risk_defaults: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True


class AIConfigUpdateIn(BaseModel):
    """AI configuration update input schema"""
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key (plaintext, will be stored)")
    selected_model: Optional[str] = None
    risk_defaults: Optional[Dict[str, Any]] = None
