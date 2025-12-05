from typing import Optional
from sqlalchemy.orm import Session
from app.repositories.base import BaseRepository
from app.models.session import Session as SessionModel

class SessionRepository(BaseRepository[SessionModel]):
    def __init__(self, session: Session):
        super().__init__(session, SessionModel)

    def get_active_by_token(self, token: str) -> Optional[SessionModel]:
        print("token is " , token )
        return self.find_one(where={"and": [{"token": token}, {"is_active": True}]})

    def get_active_by_refresh_hash(self, refresh_hash: str) -> Optional[SessionModel]:
        return self.find_one(where={"and": [{"refresh_token_hash": refresh_hash}, {"is_active": True}]})

    def find_active_for_user(self, user_id) -> list[SessionModel]:
        return list(self.find(where={"and": [{"user_id": user_id}, {"is_active": True}]}))

    def find_active_for_user_and_device(self, user_id, device_type: str, user_agent: str) -> Optional[SessionModel]:
        return self.find_one(where={"and": [
            {"user_id": user_id},
            {"device_type": device_type},
            {"user_agent": user_agent},
        ]})
