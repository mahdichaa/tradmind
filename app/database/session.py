from __future__ import annotations
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL

# Prefer DATABASE_URL if provided; otherwise build from components
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # drops dead connections
        future=True,
    )
else:
    # Read component env vars (fallback)
    POSTGRES_USERNAME = os.getenv("POSTGRES_CONN_USERNAME")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_CONN_PASSWORD")
    POSTGRES_HOST = os.getenv("POSTGRES_CONN_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_CONN_PORT", "5432"))
    POSTGRES_DBNAME = os.getenv("POSTGRES_CONN_DBNAME")

    # Safe URL creation (handles special characters)
    url = URL.create(
        "postgresql+psycopg",
        username=POSTGRES_USERNAME,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DBNAME,
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,  # drops dead connections
        future=True,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
