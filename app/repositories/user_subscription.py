from app.models.session import Session
from app.models.user_subscription import UserSubscription
from app.repositories.base import BaseRepository
from app.models.enums import ProviderType,SubscriptionStatus

class subscriptionRepository(BaseRepository[UserSubscription]):
    def __init__(self, session: Session):
        super().__init__(session, UserSubscription)
        
    def get_by_id(self, id_):
        return self.session.query(self.model).get(id_)



