from app.repositories.base import BaseRepository
from app.models.password_reset import PasswordResetToken


class PasswordResetRepository(BaseRepository[PasswordResetToken]):
    def __init__(self, session):
        super().__init__(session, PasswordResetToken)
