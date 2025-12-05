from typing import Optional
from sqlalchemy.orm import Session
from app.repositories.base import BaseRepository
from app.models.subscription_plan import SubscriptionPlan

class SubscriptionPlanRepository(BaseRepository[SubscriptionPlan]):
    def __init__(self, session: Session):
        super().__init__(session, SubscriptionPlan)

    def get_by_id(self,id:str):
        return self.find_one(where={"plan_id":id})
        
    def get_by_criteria(self,criteria):
        return self.find_one(where=criteria)
