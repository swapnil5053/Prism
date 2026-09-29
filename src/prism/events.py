"""Progress events for an analysis, stored in a Redis stream per analysis.

A stream (not pub/sub) so that a browser which connects late, or reconnects
after a network blip, can replay everything it missed from its last event id.
"""

import json
import uuid
from typing import Any

from pydantic import BaseModel
from redis.asyncio import Redis

from prism.db.models import AnalysisStatus

STREAM_MAXLEN = 200
# Keep finished streams around long enough for a slow client to catch up.
FINISHED_TTL_S = 3600


class Event(BaseModel):
    status: AnalysisStatus
    stage: str | None = None
    message: str | None = None
    data: dict[str, Any] | None = None


def stream_key(analysis_id: uuid.UUID | str) -> str:
    return f"prism:events:{analysis_id}"


def cancel_key(analysis_id: uuid.UUID | str) -> str:
    return f"prism:cancel:{analysis_id}"


async def publish(redis: Redis, analysis_id: uuid.UUID | str, event: Event) -> str:
    key = stream_key(analysis_id)
    payload = event.model_dump_json(exclude_none=True)
    event_id: bytes | str = await redis.xadd(
        key, {"e": payload}, maxlen=STREAM_MAXLEN, approximate=True
    )
    if event.status.is_terminal:
        await redis.expire(key, FINISHED_TTL_S)
    return event_id.decode() if isinstance(event_id, bytes) else event_id


async def read(
    redis: Redis, analysis_id: uuid.UUID | str, after: str, block_ms: int
) -> list[tuple[str, Event]]:
    """Events newer than `after` ("0" = from the start), waiting up to block_ms."""
    reply = await redis.xread({stream_key(analysis_id): after}, block=block_ms, count=50)
    out: list[tuple[str, Event]] = []
    for _key, entries in reply or []:
        for raw_id, fields in entries:
            event_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            raw = fields.get(b"e", fields.get("e"))
            out.append((event_id, Event.model_validate(json.loads(raw))))
    return out
