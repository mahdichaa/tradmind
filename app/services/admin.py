
from ast import Subscript
from datetime import datetime
from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy import func, select
from app.core.ip_blocklist import IPBlocklistStore
from app.core.email_blocklist import EmailBlocklistStore
from app.core.paths import config_path
from app.core.pagination import PaginationParams
from app.models.chart_analysis import ChartAnalysis
from app.models.trade import Trade
from app.models.user import User
from app.models.user_subscription import UserSubscription
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.user import UserRepository
from app.repositories.user_subscription import subscriptionRepository
from app.schemas.user_subscription import SubscriptionItem, SubscriptionsResponse
from app.core.security import hash_password


USER_ROLE_VALUES = ["user", "admin"]
USER_STATUS_VALUES = ["active", "suspended", "deleted"]

def validate_enum(value: str, allowed: list[str], field_name: str):
    if value not in allowed:
        raise ValueError(
            f"Invalid {field_name}: '{value}'. Allowed values: {allowed}"
        )
class AdminService:
    def __init__(self, db):
        self.repo = UserRepository(db)  
        self.subscription_repo = subscriptionRepository(db)
        self.audit_logs_repo= AuditLogRepository(db)
        self.ip_repo = IPBlocklistStore(config_path("blocked_ips.json"))
        self.email_repo = EmailBlocklistStore(config_path("blocked_emails.json"))
        self.db=db 

    def get_all_users_with_stats(self):
        
        stmt = (
            select(
                User,  # selects ALL columns of User
                func.count(Trade.trade_id).label("trade_count"),
                func.count(ChartAnalysis.analysis_id).label("analyse_count"),
            )
            .outerjoin(Trade, Trade.user_id == User.user_id)
            .outerjoin(ChartAnalysis, ChartAnalysis.user_id == User.user_id)
            .group_by(User.user_id)
        )

        rows = self.db.execute(stmt).all()

        # Build the JSON response
        response = []
        for row in rows:
            user_obj = row.User  # this is the full User model instance

            # Convert SQLAlchemy model → dict (fast)
            user_dict = {col.name: getattr(user_obj, col.name) for col in User.__table__.columns}
            # Remove sensitive fields
            if "password_hash" in user_dict:
                del user_dict["password_hash"]

            # Add counts
            user_dict["trade_count"] = row.trade_count
            user_dict["analyse_count"] = row.analyse_count

            response.append(user_dict)

        return response

    # Get specific user
    def get_user(self, user_id: str):
        user = self.repo.find_one(where={"user_id": user_id})
        if not user:
            raise HTTPException(404, "User not found")
        return user

    def create_user(self, payload):
        # Uniqueness checks
        if self.repo.get_by_email(payload.email):
            raise HTTPException(status_code=400, detail="Email already registered")
        if self.repo.get_by_username(payload.username):
            raise HTTPException(status_code=400, detail="Username already taken")

        user = User(
            email=payload.email.lower(),
            username=(payload.username or "").strip(),
            password_hash=hash_password(payload.password),
            first_name=(payload.first_name or None),
            last_name=(payload.last_name or None),
            timezone=payload.timezone or "UTC",
            role=payload.role or "user",
            status="active",
            email_verified=True,
        )
        self.repo.create(user, commit=True)
        return user

    def update_user(self, user_id: str, payload):
        user = self.repo.find_one(where={"user_id": user_id})
        if not user:
            raise HTTPException(404, "User not found")

        updates: dict = {}

        # Change email if provided and unique
        new_email = getattr(payload, "email", None)
        if new_email and new_email != user.email:
            if self.repo.get_by_email(new_email):
                raise HTTPException(status_code=400, detail="Email already registered")
            updates["email"] = new_email.lower()

        # Change username if provided and unique
        new_username = getattr(payload, "username", None)
        if new_username and new_username != user.username:
            if self.repo.get_by_username(new_username):
                raise HTTPException(status_code=400, detail="Username already taken")
            updates["username"] = new_username.strip()

        # Optional simple fields
        for k in ["first_name", "last_name", "timezone", "role", "status"]:
            v = getattr(payload, k, None)
            if v is not None:
                updates[k] = v

        # Optional password reset by admin
        new_password = getattr(payload, "password", None)
        if new_password:
            updates["password_hash"] = hash_password(new_password)

        if not updates:
            return user

        updated = self.repo.update_one(where={"user_id": user_id}, data=updates, commit=True)
        return updated

  # Change user role
    def change_role(self, user_id: str, new_role: str):
        validate_enum(new_role, USER_ROLE_VALUES, "role")

        return self.repo.update(
            where={"user_id": user_id},
            data={"role": new_role},
            commit=True
        )

    def change_status(self, user_id: str, new_status: str):
        validate_enum(new_status, USER_STATUS_VALUES, "status")

        return self.repo.update(
            where={"user_id": user_id},
            data={"status": new_status},
            commit=True
        )
    # Delete user
    def delete_user(self, user_id: str):
        self.repo.delete(where={"user_id":user_id}, commit=True)
        return {"status": "success", "message": "User deleted"}
    
    def get_all_subscriptions(self) -> list[UserSubscription]:
        """
        Fetch all subscriptions from the repository, optionally with pagination.
        """
        return self.subscription_repo.find()


    def get_subscriptions_by_user(self, user_id: int, skip: Optional[int] = None, take: Optional[int] = None) -> SubscriptionsResponse:
        items = self.subscription_repo.find(where={"user_id": user_id}, skip=skip, take=take)
        total = self.subscription_repo.count(where={"user_id": user_id})
        
        total_active = self.subscription_repo.count(where={"user_id": user_id, "status": "active"})
        total_profit = sum(sub.profit for sub in self.subscription_repo.find(where={"user_id": user_id}))
        average_profit = total_profit / total if total else 0

        subscription_items = [
            SubscriptionItem(
                id=sub.id,
                user_id=sub.user_id,
                plan_name=sub.plan_name,
                status=sub.status,
                start_date=sub.start_date,
                end_date=sub.end_date,
                price=sub.price,
                profit=sub.profit
            )
            for sub in items
        ]

        return SubscriptionsResponse(
            total=total,
            total_active=total_active,
            total_profit=total_profit,
            average_profit=average_profit,
            items=subscription_items
        )
    def list_logs(
        self,
        *,
        entity_type: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_desc: bool = True,
    ):
        """
        Return audit logs filtered by entity_type/action/date with pagination.
        - limit/offset: page controls
        - order_desc: latest-first when True (default)
        """
        where: dict = {}

        if entity_type:
            where["entity_type"] = entity_type
        if action:
            where["action"] = action
        if start_date or end_date:
            date_filter: dict = {}
            if start_date:
                date_filter[">="] = start_date
            if end_date:
                date_filter["<="] = end_date
            where["created_at"] = date_filter

        return self.audit_logs_repo.list_with_user(
            where=where if where else None,
            limit=limit,
            offset=offset,
            order_desc=order_desc,
        )
    
      # ---------------- IP Blocklist (delegates to repo) ----------------
    def list_blocked_ips(self) -> List[str]:
        return self.ip_repo.list()

    def add_blocked_ip(self, item: str) -> None:
        self.ip_repo.add(item)

    def remove_blocked_ip(self, item: str) -> None:
        self.ip_repo.remove(item)

    def is_ip_blocked(self, ip: str) -> bool:
        return self.ip_repo.is_blocked(ip)

    # ---------------- Email Blocklist ----------------
    def list_blocked_emails(self) -> List[str]:
        return self.email_repo.list()

    def add_blocked_email(self, email: str) -> None:
        self.email_repo.add(email)
        # suspend user if exists
        user = self.repo.get_by_email(email.lower())
        if user and user.status != "suspended":
            self.repo.update_by_id(user.user_id, {"status": "suspended"})

    def remove_blocked_email(self, email: str) -> None:
        self.email_repo.remove(email)
        # optionally reactivate if previously suspended
        user = self.repo.get_by_email(email.lower())
        if user and user.status == "suspended":
            self.repo.update_by_id(user.user_id, {"status": "active"})

    def is_email_blocked(self, email: str) -> bool:
        return self.email_repo.is_blocked(email)
