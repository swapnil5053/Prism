import uuid
from io import BytesIO

import pytest
from arq.jobs import Job, JobStatus
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from PIL import Image
from redis.exceptions import ConnectionError as RedisConnectionError

from prism.api.workspace import COOKIE_NAME, sign

pytestmark = pytest.mark.integration


def png(size: tuple[int, int] = (120, 80), color: str = "white") -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


async def upload(
    client: AsyncClient, data: bytes | None = None, name: str = "shot.png"
) -> Response:
    return await client.post("/api/v1/analyses", files={"file": (name, data or png(), "image/png")})


def other_client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_upload_creates_queued_analysis(app: FastAPI, client: AsyncClient) -> None:
    r = await upload(client)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "queued"
    assert (body["image_width"], body["image_height"]) == (120, 80)
    assert body["original_filename"] == "shot.png"
    assert COOKIE_NAME in r.cookies

    job = Job(f"analysis:{body['id']}", app.state.queue)
    assert await job.status() == JobStatus.queued


async def test_cookie_is_http_only(client: AsyncClient) -> None:
    r = await upload(client)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie


async def test_second_upload_reuses_workspace(client: AsyncClient) -> None:
    await upload(client)
    r = await upload(client)
    assert "set-cookie" not in r.headers
    page = (await client.get("/api/v1/analyses")).json()
    assert len(page["items"]) == 2


async def test_client_filename_cannot_pick_the_path(app: FastAPI, client: AsyncClient) -> None:
    r = await upload(client, name="../../../../tmp/evil.php")
    assert r.status_code == 201
    assert r.json()["original_filename"] == "evil.php"
    stored = list(app.state.settings.upload_dir.iterdir())
    assert len(stored) == 1
    assert stored[0].suffix == ".png"


async def test_rejects_oversized_upload(client: AsyncClient) -> None:
    noisy = Image.effect_noise((1200, 1200), 100).convert("RGB")
    buf = BytesIO()
    noisy.save(buf, format="PNG")
    assert len(buf.getvalue()) > 1024 * 1024  # test settings cap uploads at 1 MB
    r = await upload(client, buf.getvalue())
    assert r.status_code == 413


async def test_rejects_non_image(client: AsyncClient) -> None:
    r = await upload(client, b"GIF89a but actually text", name="x.png")
    assert r.status_code == 422
    assert "set-cookie" not in r.headers


async def test_other_workspace_gets_404(app: FastAPI, client: AsyncClient) -> None:
    analysis_id = (await upload(client)).json()["id"]
    assert (await client.get(f"/api/v1/analyses/{analysis_id}")).status_code == 200

    async with other_client(app) as stranger:
        await upload(stranger)  # has its own workspace now
        assert (await stranger.get(f"/api/v1/analyses/{analysis_id}")).status_code == 404
        assert (await stranger.get(f"/api/v1/analyses/{analysis_id}/image")).status_code == 404
        assert len((await stranger.get("/api/v1/analyses")).json()["items"]) == 1


async def test_forged_cookie_is_ignored(app: FastAPI, client: AsyncClient) -> None:
    analysis_id = (await upload(client)).json()["id"]
    real = client.cookies[COOKIE_NAME]
    victim_id = uuid.UUID(hex=real.split(".")[0])

    async with other_client(app) as attacker:
        attacker.cookies[COOKIE_NAME] = sign(victim_id, "wrong-secret-" + "x" * 32)
        assert (await attacker.get(f"/api/v1/analyses/{analysis_id}")).status_code == 404


async def test_no_cookie_means_empty_history(app: FastAPI) -> None:
    async with other_client(app) as fresh:
        r = await fresh.get("/api/v1/analyses")
    assert r.json() == {"items": [], "next_before": None}
    assert "set-cookie" not in r.headers


async def test_malformed_id_is_422(client: AsyncClient) -> None:
    r = await client.get("/api/v1/analyses/1' OR '1'='1")
    assert r.status_code == 422


async def test_pagination(client: AsyncClient) -> None:
    ids = [(await upload(client)).json()["id"] for _ in range(5)]
    first = (await client.get("/api/v1/analyses", params={"limit": 2})).json()
    assert [a["id"] for a in first["items"]] == ids[::-1][:2]

    second = (
        await client.get("/api/v1/analyses", params={"limit": 2, "before": first["next_before"]})
    ).json()
    assert [a["id"] for a in second["items"]] == ids[::-1][2:4]


async def test_image_download(client: AsyncClient) -> None:
    body = (await upload(client, png(color="red"))).json()
    r = await client.get(body["image_url"])
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    with Image.open(BytesIO(r.content)) as img:
        assert img.getpixel((0, 0)) == (255, 0, 0)


async def test_queue_down_marks_failed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*args: object, **kwargs: object) -> None:
        raise RedisConnectionError("redis is down")

    monkeypatch.setattr(app.state.queue, "enqueue_job", broken)
    r = await upload(client)
    assert r.status_code == 503
    assert r.json()["status"] == "failed"
    assert COOKIE_NAME in r.cookies  # a first-time visitor can still see the failure

    items = (await client.get("/api/v1/analyses")).json()["items"]
    detail = (await client.get(f"/api/v1/analyses/{items[0]['id']}")).json()
    assert detail["status"] == "failed"
    assert detail["error"]


async def test_device_pixel_ratio(client: AsyncClient) -> None:
    r = await client.post(
        "/api/v1/analyses",
        files={"file": ("s.png", png(), "image/png")},
        data={"device_pixel_ratio": "2"},
    )
    assert r.json()["device_pixel_ratio"] == 2
    bad = await client.post(
        "/api/v1/analyses",
        files={"file": ("s.png", png(), "image/png")},
        data={"device_pixel_ratio": "9"},
    )
    assert bad.status_code == 422
