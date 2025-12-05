from sqlalchemy.orm import Session
from app.repositories.base import BaseRepository
from app.models.email_verification import EmailVerificationToken

class EmailVerificationRepository(BaseRepository[EmailVerificationToken]):
    def __init__(self, session: Session):
        super().__init__(session, EmailVerificationToken)
