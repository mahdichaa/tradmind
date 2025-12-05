# app/main.py
from __future__ import annotations
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.api.router import router
from sqlalchemy.orm import configure_mappers
from sqlalchemy import text
import os
from app.database.session import engine, SessionLocal
from app.models.base import Base
import app.models  # ensure models are imported so Base.metadata is populated
from app.core.ip_blocklist import IPBlocklistStore
from app.middleware.ip_blocklist import IPBlocklistMiddleware
from app.core.paths import config_path


from fastapi.middleware.cors import CORSMiddleware


configure_mappers()



class Settings(BaseSettings):
    app_name: str = "Tradvio-API"
    app_version: str = "0.1.0"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

# CORS origins (CSV in env CORS_ORIGINS), defaults to localhost:5173
_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:5173")
origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

settings = Settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"persistAuthorization": True},
)
ip_store = IPBlocklistStore(config_path("blocked_ips.json"))
app.add_middleware(IPBlocklistMiddleware, store=ip_store) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of allowed origins
    allow_credentials=True,      # Allow cookies
    allow_methods=["*"],         # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],         # Allow all headers
)

# Initialize DB schema and seed defaults at startup (no Alembic)
@app.on_event("startup")
def _init_db_and_seed():
    from app.repositories.user import UserRepository
    from app.core.security import hash_password
    from app.models.user import User
    from app.repositories.ai_config import AIConfigRepository

    # Ensure required Postgres extensions used by models (e.g., gen_random_uuid())
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    except Exception as e:
        print(f"Warning: could not ensure pgcrypto extension: {e}")

    # Create all tables if they don't exist
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Error creating tables: {e}")

    # Seed default/config rows idempotently
    db = SessionLocal()
    try:
        # Ensure AI config exists
        AIConfigRepository(db).get_or_create()

        # Ensure default admin exists / stays active
        repo = UserRepository(db)
        admin = repo.get_by_username("admin_9KXr7wQ2")
        if not admin:
            user = User(
                email="admin_9KXr7wQ2@example.com",
                username="admin_9KXr7wQ2",
                password_hash=hash_password("S3cureAdm1nP@ssw0rd_9KXr"),
                first_name="Admin",
                last_name="User",
                timezone="UTC",
                email_verified=True,
                status="active",
                role="admin",
            )
            repo.create(user, commit=True)
        else:
            # enforce active admin on every restart
            repo.update_one(
                where={"user_id": admin.user_id},
                data={
                    "status": "active",
                    "role": "admin",
                    "email_verified": True,
                    "password_hash": hash_password("S3cureAdm1nP@ssw0rd_9KXr"),
                },
                commit=True,
            )
    finally:
        db.close()

@app.get("/")
def root():
    return {"status": "ok"}

# Mount all routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
