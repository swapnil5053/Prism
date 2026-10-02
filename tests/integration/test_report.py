from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from prism.domain import Box, Element, ElementKind
from prism.worker.tasks import analyze
from tests.fakes import FakeDetector
from tests.integration.test_analyses import other_client, upload

pytestmark = pytest.mark.integration


async def test_reports_after_completion(client: AsyncClient, worker_ctx: dict[str, Any]) -> None:
    worker_ctx["detector"] = FakeDetector(
        [
            Element(
                id=0,
                kind=ElementKind.ICON,
                box=Box(x1=0.1, y1=0.1, x2=0.15, y2=0.15),
                text="<script>alert(1)</script>",
            )
        ]
    )
    aid = (await upload(client, name="<b>home</b>.png")).json()["id"]
    assert (await client.get(f"/api/v1/analyses/{aid}/report")).status_code == 409

    await analyze(worker_ctx, aid)

    html = await client.get(f"/api/v1/analyses/{aid}/report")
    assert html.status_code == 200
    assert "attachment" in html.headers["content-disposition"]
    assert "default-src 'none'" in html.headers["content-security-policy"]
    assert "<script>alert(1)</script>" not in html.text
    assert "&lt;script&gt;" in html.text
    assert "<b>home</b>" not in html.text
    assert "2.5.8" in html.text

    data = (await client.get(f"/api/v1/analyses/{aid}/report", params={"format": "json"})).json()
    assert data["result"]["findings"][0]["rule"] == "target-size"

    # The history list carries the two numbers its table shows, not the whole result.
    [item] = (await client.get("/api/v1/analyses")).json()["items"]
    assert item["finding_count"] == 1
    assert item["score"] == data["result"]["score"]
    assert "result" not in item


async def test_report_is_private(
    app: FastAPI, client: AsyncClient, worker_ctx: dict[str, Any]
) -> None:
    aid = (await upload(client)).json()["id"]
    await analyze(worker_ctx, aid)
    async with other_client(app) as stranger:
        assert (await stranger.get(f"/api/v1/analyses/{aid}/report")).status_code == 404
