from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database.session import get_db
from app.models.user import User
from app.core.security import hash_password
import os

router = APIRouter()

def verify_token(x_grant_access_token: str = Header(...)):
    expected_token = os.getenv("GRANT_ACCESS_TOKEN")
    if not expected_token:
        # If env var is not set, fail safe by denying access
        raise HTTPException(status_code=500, detail="Server misconfiguration: GRANT_ACCESS_TOKEN not set")
    
    if x_grant_access_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid access token")
    return x_grant_access_token

@router.delete("/reset-database", dependencies=[Depends(verify_token)])
def reset_database(db: Session = Depends(get_db)):
    """
    DANGER: This endpoint deletes ALL data from the database.
    It truncates all tables except 'alembic_version'.
    """
    try:
        # Disable foreign key checks to allow truncation of tables with relationships
        db.execute(text("SET session_replication_role = 'replica';"))
        
        # Get all table names
        # Note: This query is specific to PostgreSQL
        result = db.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename != 'alembic_version'
        """))
        tables = [row[0] for row in result]
        
        if tables:
            # Truncate all tables
            # We use CASCADE just in case, though replication_role='replica' usually handles it
            tables_str = ", ".join([f'"{t}"' for t in tables])
            db.execute(text(f"TRUNCATE TABLE {tables_str} CASCADE;"))
            
        # Re-enable foreign key checks
        db.execute(text("SET session_replication_role = 'origin';"))
        
        # Create default admin user
        # Username and password are the GRANT_ACCESS_TOKEN
        token_value = verify_token(x_grant_access_token=os.getenv("GRANT_ACCESS_TOKEN"))
        
        admin_user = User(
            email="admin@tradvio.com",
            username=token_value,
            password_hash=hash_password(token_value),
            first_name="System",
            last_name="Admin",
            role="admin",
            status="active",
            email_verified=True
        )
        db.add(admin_user)
        db.commit()
        
        return {"status": "ok", "message": "All database tables cleared successfully. Default admin created."}
        
    except Exception as e:
        db.rollback()
        # Ensure we try to reset the replication role even on error
        try:
            db.execute(text("SET session_replication_role = 'origin';"))
            db.commit()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Database reset failed: {str(e)}")
