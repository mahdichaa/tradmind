from sqlalchemy.orm import Session
from app.models.subscription_plan import SubscriptionPlan
from app.repositories.subscription_plan import SubscriptionPlanRepository
from fastapi import HTTPException

class SubscriptionPlanService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = SubscriptionPlanRepository(db)
    
    def create(self, payload):
        try:
            # Create plan directly with provided paddle_price_id and paypal_plan_id
            plan = self.repo.create({
                "name": payload.name,
                "description": payload.description,
                "currency": payload.currency,
                "price": payload.price,
                "billing_interval": payload.billing_interval,
                "features": payload.features,
                "paddle_price_id": getattr(payload, 'paddle_price_id', None),
                "paypal_plan_id": getattr(payload, 'paypal_plan_id', None),
                "is_active": payload.is_active
            })
            plan_dict = {col.name: getattr(plan, col.name) for col in SubscriptionPlan.__table__.columns}
            return plan_dict
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def get_by_id(self,id:str)->SubscriptionPlan:
        return self.repo.get_by_id(id)
    
    def get_by_criteria(self,criteria):
        return self.repo.get_by_criteria(criteria)
    
    def disactivate(self,plan_id):
        try:
            subscription_plan=self.get_by_criteria({"plan_id":plan_id})
            if(not subscription_plan):
                raise HTTPException(status_code=404,detail="not found")
            return self.repo.update({"plan_id":plan_id},{"is_active":False})
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def activate(self,plan_id):
        try:
            subscription_plan=self.get_by_criteria({"plan_id":plan_id})
            if(not subscription_plan):
                raise HTTPException(status_code=404,detail="not found")
            return self.repo.update({"plan_id":plan_id},{"is_active":True})
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def get_all(self):
        try:
            plans=self.repo.find()
            response = []
            for plan in plans:
                # Convert SQLAlchemy model → dict (fast)
                plan_dict = { col.name: getattr(plan, col.name) for col in SubscriptionPlan.__table__.columns }
                response.append(plan_dict)

            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
