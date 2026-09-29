import asyncio
from typing import Any

import pytest
from httpx import AsyncClient

from prism import events
from prism.worker.tasks import FAILED_MESSAGE, analyze
from tests.fakes import FakeDetector
from tests.integration.test_analyses import upload

pytestmark = pytest.mark.integration


async def stages(ctx: dict[str, Any], aid: str) -> list[str]:
    return [e.status for _, e in await events.read(ctx["redis"], aid, "0", block_ms=10)]


async def test_completes_and_stores_result(client: AsyncClient, worker_ctx: dict[str, Any]) -> None:
    aid = (await upload(client)).json()["id"]
    assert await analyze(worker_ctx, aid) == "completed"

    body = (await client.get(f"/api/v1/analyses/{aid}")).json()
    assert body["status"] == "completed"
    assert body["model_version"] == "fake-detector"
    assert body["elapsed_ms"] >= 0
    assert body["result"]["elements"][0]["kind"] == "button"
    assert await stages(worker_ctx, aid) == ["running", "completed"]


async def test_detector_error_is_not_leaked(
    client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    worker_ctx["detector"] = FakeDetector(error=RuntimeError("CUDA out of memory at 0x7f"))
    aid = (await upload(client)).json()["id"]
    assert await analyze(worker_ctx, aid) == "failed"

    body = (await client.get(f"/api/v1/analyses/{aid}")).json()
    assert body["status"] == "failed"
    assert body["error"] == FAILED_MESSAGE


async def test_canceled_while_queued_is_skipped(
    client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    aid = (await upload(client)).json()["id"]
    r = await client.post(f"/api/v1/analyses/{aid}/cancel")
    assert r.status_code == 202
    assert r.json()["status"] == "canceled"

    assert await analyze(worker_ctx, aid) == "skipped"
    assert worker_ctx["detector"].calls == 0


async def test_cancel_stops_running_detection(
    client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    worker_ctx["detector"] = FakeDetector(delay_s=10)
    aid = (await upload(client)).json()["id"]
    job = asyncio.create_task(analyze(worker_ctx, aid))

    for _ in range(100):  # wait until the worker has claimed it
        if (await client.get(f"/api/v1/analyses/{aid}")).json()["status"] == "running":
            break
        await asyncio.sleep(0.02)
    r = await client.post(f"/api/v1/analyses/{aid}/cancel")
    assert r.status_code == 202

    assert await asyncio.wait_for(job, timeout=5) == "canceled"
    assert (await client.get(f"/api/v1/analyses/{aid}")).json()["status"] == "canceled"
    assert not await worker_ctx["redis"].exists(events.cancel_key(aid))


async def test_cancel_finished_is_conflict(client: AsyncClient, worker_ctx: dict[str, Any]) -> None:
    aid = (await upload(client)).json()["id"]
    await analyze(worker_ctx, aid)
    assert (await client.post(f"/api/v1/analyses/{aid}/cancel")).status_code == 409


async def test_runs_once_even_if_enqueued_twice(
    client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    aid = (await upload(client)).json()["id"]
    results = await asyncio.gather(analyze(worker_ctx, aid), analyze(worker_ctx, aid))
    assert sorted(results) == ["completed", "skipped"]
    assert worker_ctx["detector"].calls == 1
