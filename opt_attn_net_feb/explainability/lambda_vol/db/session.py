from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def build_engine(uri: str) -> Engine:
    """Create SQLAlchemy engine for Lambda-Vol store."""
    return create_engine(str(uri), future=True)


def initialize_database(engine: Engine) -> None:
    """Create Lambda-Vol tables if absent."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create typed session factory."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


__all__ = ["build_engine", "initialize_database", "make_session_factory"]
