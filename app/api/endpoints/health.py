from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.database.session import get_db

router = APIRouter()

@router.get("/db")
def health_db(db: Session = Depends(get_db)):
    version = db.execute(text("SELECT version()")).scalar()
    return {"ok": True, "postgres": version}