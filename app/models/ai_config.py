# app/models/ai_config.py
from sqlalchemy import Column, String, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm.attributes import flag_modified
from app.models.base import Base
import uuid
from typing import List, Optional, Dict, Any


class AIConfig(Base):
    """
    AI Configuration model for storing OpenRouter settings.
    This is a single-row table that stores the active AI configuration.
    
    API keys are stored as JSONB array of objects:
    [
        {"key": "sk-or-...", "label": "Primary Key", "is_active": true},
        {"key": "sk-or-...", "label": "Backup Key", "is_active": true}
    ]
    """
    __tablename__ = "ai_config"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # JSONB array of API key objects: [{"key": "...", "label": "...", "is_active": true}, ...]
    openrouter_api_keys = Column(
        JSONB, 
        nullable=True, 
        comment="Array of OpenRouter API keys with labels and status"
    )
    
    # Legacy single key field - kept for backward compatibility during transition
    openrouter_api_key = Column(
        Text, 
        nullable=True, 
        comment="[DEPRECATED] Legacy single API key - use openrouter_api_keys instead"
    )
    
    selected_model = Column(String(255), nullable=True, comment="Selected OpenRouter model ID")
    risk_defaults = Column(JSONB, nullable=False, default=dict, comment="Risk management defaults for SWING/SCALP modes")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def get_active_keys(self) -> List[str]:
        """
        Returns list of active API keys in priority order.
        Handles both new multi-key format and legacy single key.
        """
        keys = []
        
        # New multi-key format
        if self.openrouter_api_keys and isinstance(self.openrouter_api_keys, list):
            for key_obj in self.openrouter_api_keys:
                if isinstance(key_obj, dict) and key_obj.get("is_active", True):
                    key = key_obj.get("key")
                    if key:
                        keys.append(key)
        
        # Fallback to legacy single key if no multi-keys configured
        if not keys and self.openrouter_api_key:
            keys.append(self.openrouter_api_key)
        
        return keys

    def get_key_info(self, index: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific key by index."""
        if not self.openrouter_api_keys or not isinstance(self.openrouter_api_keys, list):
            return None
        
        if 0 <= index < len(self.openrouter_api_keys):
            return self.openrouter_api_keys[index]
        
        return None

    def get_next_key(self, failed_index: Optional[int] = None) -> Optional[tuple[str, int]]:
        """
        Get the next available API key after a failure.
        Returns tuple of (key, index) or None if no keys available.
        
        Args:
            failed_index: Index of the key that failed (to skip it)
        """
        active_keys = self.get_active_keys()
        
        if not active_keys:
            return None
        
        # If no failed index, return first key
        if failed_index is None:
            return (active_keys[0], 0)
        
        # Return next key after failed index
        next_index = failed_index + 1
        if next_index < len(active_keys):
            return (active_keys[next_index], next_index)
        
        return None

    def add_key(self, api_key: str, label: Optional[str] = None):
        """Add a new API key to the configuration."""
        if not self.openrouter_api_keys:
            self.openrouter_api_keys = []
        
        if not isinstance(self.openrouter_api_keys, list):
            self.openrouter_api_keys = []
        
        key_obj = {
            "key": api_key,
            "label": label or f"API Key {len(self.openrouter_api_keys) + 1}",
            "is_active": True
        }
        
        self.openrouter_api_keys.append(key_obj)
        
        # CRITICAL: Tell SQLAlchemy that the JSONB field has been modified
        flag_modified(self, "openrouter_api_keys")

    def remove_key(self, index: int) -> bool:
        """Remove API key at specified index. Returns True if successful."""
        if not self.openrouter_api_keys or not isinstance(self.openrouter_api_keys, list):
            return False
        
        if 0 <= index < len(self.openrouter_api_keys):
            self.openrouter_api_keys.pop(index)
            # Tell SQLAlchemy that the JSONB field has been modified
            flag_modified(self, "openrouter_api_keys")
            return True
        
        return False

    def reorder_keys(self, new_order: List[int]) -> bool:
        """
        Reorder API keys based on new index order.
        
        Args:
            new_order: List of indices in desired order, e.g., [2, 0, 1]
        
        Returns:
            True if successful, False otherwise
        """
        if not self.openrouter_api_keys or not isinstance(self.openrouter_api_keys, list):
            return False
        
        if len(new_order) != len(self.openrouter_api_keys):
            return False
        
        try:
            reordered = [self.openrouter_api_keys[i] for i in new_order]
            self.openrouter_api_keys = reordered
            # Tell SQLAlchemy that the JSONB field has been modified
            flag_modified(self, "openrouter_api_keys")
            return True
        except (IndexError, TypeError):
            return False

    def mark_key_inactive(self, index: int) -> bool:
        """Mark a key as inactive (won't be used for API calls)."""
        if not self.openrouter_api_keys or not isinstance(self.openrouter_api_keys, list):
            return False
        
        if 0 <= index < len(self.openrouter_api_keys):
            self.openrouter_api_keys[index]["is_active"] = False
            # Tell SQLAlchemy that the JSONB field has been modified
            flag_modified(self, "openrouter_api_keys")
            return True
        
        return False

    def mark_key_active(self, index: int) -> bool:
        """Mark a key as active (will be used for API calls)."""
        if not self.openrouter_api_keys or not isinstance(self.openrouter_api_keys, list):
            return False
        
        if 0 <= index < len(self.openrouter_api_keys):
            self.openrouter_api_keys[index]["is_active"] = True
            # Tell SQLAlchemy that the JSONB field has been modified
            flag_modified(self, "openrouter_api_keys")
            return True
        
        return False

    def __repr__(self):
        key_count = len(self.openrouter_api_keys) if self.openrouter_api_keys else 0
        return f"<AIConfig(id={self.id}, model={self.selected_model}, keys={key_count})>"
