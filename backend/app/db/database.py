"""
SQLAlchemy engine and session factory.
Each request gets a scoped session via the get_db() dependency.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

# check_same_thread=False is required for SQLite when FastAPI runs background tasks
# on threads other than the main one.
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables. Called once at application startup."""
    from app.db import models  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that provides a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
