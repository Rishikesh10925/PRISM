import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL
from app.database import get_db
from app.main import app
from app.models import Base


@pytest.fixture(scope="session")
def test_engine():
    """Runs against the real local Postgres+PostGIS (docker-compose), in a schema
    dedicated to tests so it never touches dev data -- there's no in-memory PostGIS
    substitute, so this is the real database, just isolated by schema."""
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS test"))
        conn.execute(text("SET search_path TO test, public"))
    Base.metadata.schema = "test"
    Base.metadata.create_all(engine)
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA test CASCADE"))
    Base.metadata.schema = None


@pytest.fixture
def db_session(test_engine):
    connection = test_engine.connect()
    transaction = connection.begin()
    connection.execute(text("SET search_path TO test, public"))
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    from fastapi.testclient import TestClient

    def _get_test_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_test_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
