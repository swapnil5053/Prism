import pytest
from fastapi import FastAPI
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_liveness(client: AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_readiness_checks_dependencies(client: AsyncClient) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"database": "ok", "redis": "ok"}


async def test_unhandled_error_hides_details(app: FastAPI, client: AsyncClient) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("db password is hunter2")

    r = await client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert "hunter2" not in r.text
    assert body["request_id"] == r.headers["x-request-id"]


async def test_forged_request_id_is_replaced(client: AsyncClient) -> None:
    r = await client.get("/healthz", headers={"x-request-id": "evil\nlog line"})
    assert r.headers["x-request-id"] != "evil\nlog line"
    assert r.headers["x-request-id"].isalnum()
