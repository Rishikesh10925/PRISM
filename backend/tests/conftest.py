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


@pytest.fixture(autouse=True)
def fake_overpass(monkeypatch):
    """Never hit the live Overpass API from backend tests: by default every lookup finds
    a residential road. Tests can set .result to a list of OSM elements, or to an
    exception instance to simulate the API being down."""
    import road_type  # importable once app.services.inference has extended sys.path

    class FakeOverpass:
        result: object = [{"tags": {"highway": "residential"}}]

        def __call__(self, lat, lon, radius_m=50, timeout=15.0):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    fake = FakeOverpass()
    road_type._source_cache.clear()
    monkeypatch.setattr(road_type, "fetch_highway_elements", fake)
    yield fake
    road_type._source_cache.clear()


@pytest.fixture(autouse=True)
def upload_dir(tmp_path, monkeypatch):
    """Store uploaded photos in a per-test temp dir, not the real data/reports."""
    from app import config

    path = tmp_path / "uploads"
    monkeypatch.setattr(config, "UPLOAD_DIR", path)
    return path


@pytest.fixture(scope="session")
def test_engine():
    """Runs against the real local Postgres+PostGIS (docker-compose), in a schema
    dedicated to tests so it never touches dev data -- there's no in-memory PostGIS
    substitute, so this is the real database, just isolated by schema."""
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS test CASCADE"))
        conn.execute(text("CREATE SCHEMA test"))
        conn.execute(text("SET LOCAL search_path TO test, public"))
        # checkfirst=False: with search_path "test, public" the existence check would
        # find the dev tables in public and silently skip creating the test ones (which
        # previously made every test run against dev data in public).
        Base.metadata.create_all(conn, checkfirst=False)
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA test CASCADE"))


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
