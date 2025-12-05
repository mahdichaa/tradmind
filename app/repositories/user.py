from typing import Optional
from sqlalchemy.orm import Session

from app.repositories.base import BaseRepository   # <-- your generic BaseRepository
from app.models.user import User                   # <-- your SQLAlchemy User model

class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session):
        super().__init__(session, User)

    def get_by_id(self, user_id) -> Optional[User]:
        return self.find_one(where={"user_id": user_id})

    def get_by_email(self, email: str) -> Optional[User]:
        # keep it simple; if you want case-insensitive later we can add it
        return self.find_one(where={"email": email})

    def get_by_username(self, username: str) -> Optional[User]:
        return self.find_one(where={"username": username})
    
    def get_by_criteria(self,criteria)->Optional[User]:
        return self.find_one(where=criteria)
    def update_by_id(self,user_id,payload):
        return self.update(where={"user_id":user_id},data=payload,commit=True)

