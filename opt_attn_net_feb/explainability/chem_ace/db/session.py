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
