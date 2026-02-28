from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def build_engine(uri: str) -> Engine:
    """
    Builds and returns a SQLAlchemy Engine instance for database interactions.

    This function creates and configures an SQLAlchemy Engine object with the
    provided database URI. The engine is set to use future behavior as per
    SQLAlchemy standards.

    Args:
        uri (str): The database connection URI used to configure the engine.

    Returns:
        Engine: A SQLAlchemy Engine configured with the given URI.
    """
    return create_engine(uri, future=True)


def initialize_database(engine: Engine) -> None:
    """
    Initializes the database by creating all the tables defined in the metadata.

    This function ensures that all the tables declared using SQLAlchemy's
    Base object are created in the database connected through the provided
    engine. If the tables already exist, no changes will be made.

    Args:
        engine (Engine): The SQLAlchemy engine object used to connect to the
        database.

    Returns:
        None
    """
    Base.metadata.create_all(engine)
    _apply_compat_migrations(engine)


def _apply_compat_migrations(engine: Engine) -> None:
    """
    Apply lightweight additive migrations for existing SQLite databases.

    These migrations only add nullable columns and keep legacy DB files usable.
    """
    if str(engine.dialect.name).lower() != "sqlite":
        return
    planned = {
        "concepts": {"modality": "TEXT"},
        "concept_memberships": {"modality": "TEXT"},
        "concept_tags": {"modality": "TEXT"},
        "cavs": {"concept_modality": "TEXT"},
        "tcav_epoch": {"concept_modality": "TEXT"},
    }
    with engine.begin() as conn:
        for table_name, cols in planned.items():
            existing = {
                str(row[1])
                for row in conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
            }
            for col_name, col_type in cols.items():
                if str(col_name) in existing:
                    continue
                conn.exec_driver_sql(
                    f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """
    Creates a session factory for interacting with the database using the provided
    SQLAlchemy engine.

    This function returns a SQLAlchemy sessionmaker instance configured with the
    given engine. The returned sessionmaker can be used to create database sessions
    for performing queries and transactions.

    Arguments:
        engine: The SQLAlchemy engine to be used for binding the sessionmaker.

    Returns:
        sessionmaker[Session]: A sessionmaker instance configured with the provided
        engine.
    """
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


__all__ = ["build_engine", "initialize_database", "make_session_factory"]
