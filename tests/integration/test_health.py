from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from prism.api.app import create_app
from prism.config import Settings

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


async def test_security_headers(client: AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


async def test_serves_built_frontend(settings: Settings, tmp_path: Path) -> None:
    web = tmp_path / "dist"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>Prism</title>")
    app = create_app(settings.model_copy(update={"web_dir": web}))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c,
    ):
        assert "<title>Prism</title>" in (await c.get("/")).text
        assert (await c.get("/healthz")).json() == {"status": "ok"}
