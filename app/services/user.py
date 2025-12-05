# app/services/audit_log_service.py
from typing import Any, Dict, Optional, Union
from uuid import UUID
from sqlalchemy.orm import Session

from app.repositories.user import UserRepository
from app.models.user import User


class UserService:
    def __init__(self, db: Session):
        self.session = db
        self.repo = UserRepository(db)

    def update_by_id(self,user_id,payload):
        return self.repo.update_by_id(user_id,payload)

    def get_by_criteria(
        self,
        criteria
    ) -> User:
        return self.repo.get_by_criteria(criteria)
