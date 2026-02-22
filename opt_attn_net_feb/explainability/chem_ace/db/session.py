from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def build_engine(uri: str) -> Engine:
    """Create SQLAlchemy engine for Chem-ACE database."""
    return create_engine(uri, future=True)


def initialize_database(engine: Engine) -> None:
    """Create all Chem-ACE tables when they do not exist."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build typed SQLAlchemy session factory."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


__all__ = ["build_engine", "initialize_database", "make_session_factory"]
