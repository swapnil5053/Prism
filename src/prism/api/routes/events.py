import asyncio
import contextlib
import re
import time
import uuid
from collections import Counter
from urllib.parse import urlsplit

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from prism import events
from prism.config import Settings
from prism.db.models import Analysis
from prism.schemas import AnalysisOut

from ..workspace import COOKIE_NAME, verify

router = APIRouter()

READ_BLOCK_MS = 10_000
# Each open socket holds a Redis connection in a blocking read, so one visitor
# can't be allowed an unbounded number. Per API process.
MAX_SOCKETS_PER_WORKSPACE = 8
_open_sockets: Counter[uuid.UUID] = Counter()
_EVENT_ID = re.compile(r"^\d{1,20}-\d{1,20}$")


def _origin_allowed(ws: WebSocket, settings: Settings) -> bool:
    # Browsers don't apply CORS to WebSockets, so check Origin ourselves.
    origin = ws.headers.get("origin")
    if origin is None:  # non-browser client
        return True
    return origin in settings.cors_origins or urlsplit(origin).netloc == ws.headers.get("host")


@router.websocket("/api/v1/analyses/{analysis_id}/events")
async def analysis_events(ws: WebSocket, analysis_id: uuid.UUID, after: str = "0") -> None:
    """Stream progress events. Reconnect with ?after=<last id> to resume."""
    settings: Settings = ws.app.state.settings
    sessions = ws.app.state.sessionmaker

    workspace_id = verify(ws.cookies.get(COOKIE_NAME), settings.secret_key.get_secret_value())
    async with sessions() as session:
        analysis = await session.get(Analysis, analysis_id)
    if (
        analysis is None
        or workspace_id is None
        or analysis.workspace_id != workspace_id
        or not _origin_allowed(ws, settings)
    ):
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    if _open_sockets[workspace_id] >= MAX_SOCKETS_PER_WORKSPACE:
        await ws.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    _open_sockets[workspace_id] += 1
    try:
        await _stream(ws, analysis, after)
    finally:
        _open_sockets[workspace_id] -= 1
        if not _open_sockets[workspace_id]:
            del _open_sockets[workspace_id]


async def _stream(ws: WebSocket, analysis: Analysis, after: str) -> None:
    settings: Settings = ws.app.state.settings
    sessions = ws.app.state.sessionmaker
    redis = ws.app.state.queue
    analysis_id = analysis.id

    await ws.accept()
    if analysis.status.is_terminal and not await redis.exists(events.stream_key(analysis_id)):
        await _send_snapshot(ws, analysis)
        await ws.close()
        return

    cursor = after if _EVENT_ID.match(after) else "0"
    disconnected = asyncio.create_task(_wait_for_disconnect(ws))
    deadline = time.monotonic() + settings.job_timeout_s + 120
    try:
        while not disconnected.done() and time.monotonic() < deadline:
            batch = await events.read(redis, analysis_id, cursor, READ_BLOCK_MS)
            for event_id, event in batch:
                cursor = event_id
                await ws.send_json(
                    {"id": event_id, **event.model_dump(mode="json", exclude_none=True)}
                )
                if event.status.is_terminal:
                    return
            if not batch:
                # Nothing new for a while. If the job ended and its stream has
                # already expired, send the final state from the database.
                async with sessions() as session:
                    current = await session.get(Analysis, analysis_id)
                if current is not None and current.status.is_terminal:
                    await _send_snapshot(ws, current)
                    return
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away mid-send
    finally:
        disconnected.cancel()
        with contextlib.suppress(RuntimeError):
            await ws.close()


async def _send_snapshot(ws: WebSocket, analysis: Analysis) -> None:
    snapshot = AnalysisOut.model_validate(analysis).model_dump(mode="json")
    await ws.send_json({"id": None, "status": analysis.status, "snapshot": snapshot})


async def _wait_for_disconnect(ws: WebSocket) -> None:
    # Clients don't send us anything; this just notices when they leave.
    # Leaving does not cancel the job (a page refresh shouldn't throw away work).
    with contextlib.suppress(WebSocketDisconnect):
        while (await ws.receive())["type"] != "websocket.disconnect":
            pass
