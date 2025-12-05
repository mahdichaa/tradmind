# app/repositories/ai_config.py
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from app.models.ai_config import AIConfig
import uuid


class AIConfigRepository:
    """Repository for AI configuration management."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_config(self) -> Optional[AIConfig]:
        """
        Get the current AI configuration (single row).
        Returns None if no configuration exists.
        """
        return self.db.query(AIConfig).first()
    
    def create_default(self) -> AIConfig:
        """
        Create a default AI configuration with empty settings.
        """
        default_risk_defaults = {
            "SWING": {
                "H1": {
                    "account_balance": 1000.0,
                    "risk_per_trade_percent": 1.0,
                    "stop_loss_points": 13.5,
                    "take_profit_points": 24.3
                },
                "H4": {
                    "account_balance": 1000.0,
                    "risk_per_trade_percent": 1.0,
                    "stop_loss_points": 27.0,
                    "take_profit_points": 48.6
                },
                "D1": {
                    "account_balance": 1000.0,
                    "risk_per_trade_percent": 1.0,
                    "stop_loss_points": 40.5,
                    "take_profit_points": 72.9
                },
                "W1": {
                    "account_balance": 1000.0,
                    "risk_per_trade_percent": 1.0,
                    "stop_loss_points": 67.5,
                    "take_profit_points": 121.5
                }
            },
            "SCALP": {
                "M1": {
                    "account_balance": 1000.0,
                    "risk_per_trade_percent": 0.25,
                    "stop_loss_points": 6.0,
                    "take_profit_points": 7.8
                },
                "M5": {
                    "account_balance": 1000.0,
                    "risk_per_trade_percent": 0.25,
                    "stop_loss_points": 12.0,
                    "take_profit_points": 15.6
                },
                "M15": {
                    "account_balance": 1000.0,
                    "risk_per_trade_percent": 0.25,
                    "stop_loss_points": 18.0,
                    "take_profit_points": 23.4
                }
            }
        }
        
        config = AIConfig(
            id=str(uuid.uuid4()),
            openrouter_api_key=None,
            openrouter_api_keys=None,  # New multi-key field
            selected_model=None,
            risk_defaults=default_risk_defaults
        )
        self.db.add(config)
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def update_config(
        self,
        openrouter_api_key: Optional[str] = None,
        selected_model: Optional[str] = None,
        risk_defaults: Optional[Dict[str, Any]] = None
    ) -> AIConfig:
        """
        Update the AI configuration. Creates default if none exists.
        Only updates fields that are provided (not None).
        
        Note: For single API key updates, this will convert to multi-key format.
        """
        config = self.get_config()
        
        if not config:
            config = self.create_default()
        
        # Update only provided fields
        if openrouter_api_key is not None:
            # Convert single key to multi-key format
            if openrouter_api_key:
                # If there are no existing multi-keys, create first one
                if not config.openrouter_api_keys:
                    config.openrouter_api_keys = [{
                        "key": openrouter_api_key,
                        "label": "Primary Key",
                        "is_active": True
                    }]
                else:
                    # Update first key if it exists
                    config.openrouter_api_keys[0]["key"] = openrouter_api_key
            
            # Also update legacy field for backward compatibility
            config.openrouter_api_key = openrouter_api_key
        
        if selected_model is not None:
            config.selected_model = selected_model
        
        if risk_defaults is not None:
            config.risk_defaults = risk_defaults
        
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def add_api_key(self, api_key: str, label: Optional[str] = None) -> AIConfig:
        """
        Add a new API key to the configuration.
        
        Args:
            api_key: The API key to add
            label: Optional label for the key
        
        Returns:
            Updated AIConfig
        """
        config = self.get_or_create()
        config.add_key(api_key, label)
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def remove_api_key(self, index: int) -> AIConfig:
        """
        Remove API key at specified index.
        
        Args:
            index: Index of the key to remove
        
        Returns:
            Updated AIConfig
        
        Raises:
            ValueError: If index is invalid
        """
        config = self.get_or_create()
        
        if not config.remove_key(index):
            raise ValueError(f"Invalid key index: {index}")
        
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def reorder_api_keys(self, new_order: List[int]) -> AIConfig:
        """
        Reorder API keys based on new index order.
        
        Args:
            new_order: List of indices in desired order
        
        Returns:
            Updated AIConfig
        
        Raises:
            ValueError: If new_order is invalid
        """
        config = self.get_or_create()
        
        if not config.reorder_keys(new_order):
            raise ValueError("Invalid reorder operation")
        
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def get_api_keys(self) -> List[str]:
        """
        Get all active API keys in priority order.
        
        Returns:
            List of API keys
        """
        config = self.get_or_create()
        return config.get_active_keys()
    
    def get_or_create(self) -> AIConfig:
        """
        Get existing config or create default if none exists.
        """
        config = self.get_config()
        if not config:
            config = self.create_default()
        return config
