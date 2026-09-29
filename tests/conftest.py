import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from sqlalchemy import text

from prism.api.app import create_app
from prism.config import Settings

ROOT = Path(__file__).resolve().parents[1]

# Integration tests expect a throwaway Postgres database and Redis db index.
TEST_DB = os.environ.get(
    "PRISM_TEST_DATABASE_URL", "postgresql+asyncpg://prism:prism-test@localhost/prism_test"
)
TEST_REDIS = os.environ.get("PRISM_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture(scope="session")
def migrated_db() -> Iterator[str]:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["database_url"] = TEST_DB
    cfg.attributes["configure_logger"] = False
    try:
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
    except OSError as exc:
        pytest.skip(f"test database unavailable: {exc}")
    yield TEST_DB


@pytest.fixture
def settings(tmp_path: Path, migrated_db: str) -> Settings:
    return Settings(
        database_url=migrated_db,
        redis_url=TEST_REDIS,
        secret_key="test-secret-" + "x" * 32,
        upload_dir=tmp_path / "uploads",
        cookie_secure=False,
        max_upload_mb=1,
        _env_file=None,
    )


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with app.state.sessionmaker() as session:
            await session.execute(text("TRUNCATE workspaces CASCADE"))
            await session.commit()
        await app.state.queue.flushdb()
        yield app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    r: Redis = Redis.from_url(TEST_REDIS)
    yield r
    await r.aclose()


@pytest.fixture
def worker_ctx(app: FastAPI, settings: Settings) -> dict[str, Any]:
    from tests.fakes import FakeDetector

    return {
        "settings": settings,
        "sessionmaker": app.state.sessionmaker,
        "redis": app.state.queue,
        "detector": FakeDetector(),
    }
