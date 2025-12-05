import os, secrets
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from pwdlib import PasswordHash
from app.models.enums  import ProviderType, SubscriptionStatus
from app.repositories.user_subscription import subscriptionRepository
from sqlalchemy import and_


FRONTEND_VERIFY_URL = os.getenv("FRONTEND_VERIFY_URL", "http://localhost:3000/verify-email")
VERIFY_TOKEN_TTL_MINUTES = int(os.getenv("VERIFY_TOKEN_TTL_MINUTES", "60"))
password_hash = PasswordHash.recommended()

def _utcnow(): return datetime.now(timezone.utc)
def _aware(dt): return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

class UserSubscriptionService:
    def __init__(self, db):
        self.db = db
        self.repo = subscriptionRepository(db)

    def create(self,payload):
        self.repo.create(data=payload,commit=True)
    
    def update_user_subscription(self, subscription_id: str, data: dict):
        if not data or not isinstance(data, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Update data must be a non-empty dictionary.",
            )

        # Fetch the existing subscription
        subscription = self.repo.get_by_id(subscription_id)
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User subscription not found.",
            )

        # Dynamically update provided fields
        for key, value in data.items():
            if hasattr(subscription, key):
                setattr(subscription, key, value)

        # Commit changes to the database
        self.db.commit()
        self.db.refresh(subscription)

        return subscription

    def get_by_criteria(self,criteria):
        return self.repo.find_one(where=criteria)

    def update_by_id(self,id:str,payload:dict):
        return self.repo.update_one(where={'subscription_id':id},data=payload)

    def update_by_criteria(self,criteria:dict,payload:dict):
        return self.repo.update_one(where=criteria,data=payload)
        
    def create_or_update(self,criteria:dict,payload:dict):
        existing=self.get_by_criteria(criteria)
        if(existing):
            self.update_by_id(existing.subscription_id,payload)
        else:
            self.create(payload)