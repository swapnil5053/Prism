import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from httpx_ws import WebSocketDisconnect, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from prism import events
from prism.api.workspace import COOKIE_NAME
from prism.worker.tasks import analyze
from tests.integration.test_analyses import upload

pytestmark = pytest.mark.integration


def ws_client(app: FastAPI, cookie: str | None) -> AsyncClient:
    client = AsyncClient(transport=ASGIWebSocketTransport(app), base_url="http://test")
    if cookie:
        client.cookies[COOKIE_NAME] = cookie
    return client


async def collect(app: FastAPI, cookie: str, aid: str, after: str = "0") -> list[dict[str, Any]]:
    received = []
    async with (
        ws_client(app, cookie) as c,
        aconnect_ws(f"/api/v1/analyses/{aid}/events?after={after}", c) as ws,
    ):
        try:
            while True:
                received.append(await ws.receive_json(timeout=5))
        except WebSocketDisconnect:
            pass
    return received


async def test_streams_until_done(
    app: FastAPI, client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    aid = (await upload(client)).json()["id"]
    cookie = client.cookies[COOKIE_NAME]

    listener = asyncio.create_task(collect(app, cookie, aid))
    await asyncio.sleep(0.1)  # connect before the job starts
    await analyze(worker_ctx, aid)
    got = await asyncio.wait_for(listener, 10)
    assert [e["status"] for e in got] == ["running", "completed"]
    assert got[-1]["data"]["elements"] == 1


async def test_late_client_replays_missed_events(
    app: FastAPI, client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    aid = (await upload(client)).json()["id"]
    await analyze(worker_ctx, aid)  # finishes before anyone listens
    cookie = client.cookies[COOKIE_NAME]

    got = await collect(app, cookie, aid)
    assert [e["status"] for e in got] == ["running", "completed"]

    resumed = await collect(app, cookie, aid, after=got[0]["id"])
    assert [e["status"] for e in resumed] == ["completed"]


async def test_snapshot_when_stream_expired(
    app: FastAPI, client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    aid = (await upload(client)).json()["id"]
    await analyze(worker_ctx, aid)
    await worker_ctx["redis"].delete(events.stream_key(aid))

    got = await collect(app, client.cookies[COOKIE_NAME], aid)
    assert len(got) == 1
    assert got[0]["snapshot"]["status"] == "completed"


async def test_other_workspace_is_refused(app: FastAPI, client: AsyncClient) -> None:
    aid = (await upload(client)).json()["id"]
    async with ws_client(app, None) as c:
        with pytest.raises(WebSocketDisconnect) as err:
            async with aconnect_ws(f"/api/v1/analyses/{aid}/events", c) as ws:
                await ws.receive_json(timeout=2)
    assert err.value.code == 1008


async def test_cross_site_origin_is_refused(app: FastAPI, client: AsyncClient) -> None:
    aid = (await upload(client)).json()["id"]
    async with ws_client(app, client.cookies[COOKIE_NAME]) as c:
        with pytest.raises(WebSocketDisconnect):
            async with aconnect_ws(
                f"/api/v1/analyses/{aid}/events", c, headers={"origin": "https://evil.example"}
            ) as ws:
                await ws.receive_json(timeout=2)
