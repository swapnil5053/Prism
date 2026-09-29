import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi import FastAPI

from prism.db.models import Base

pytestmark = pytest.mark.integration


async def test_models_match_migrations(app: FastAPI) -> None:
    def diff(conn):  # type: ignore[no-untyped-def]
        return compare_metadata(MigrationContext.configure(conn), Base.metadata)

    async with app.state.sessionmaker() as session:
        conn = await session.connection()
        assert await conn.run_sync(diff) == []
