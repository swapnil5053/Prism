import uuid

import pytest
from redis.asyncio import Redis

from prism import events
from prism.db.models import AnalysisStatus
from prism.events import Event

pytestmark = pytest.mark.integration


async def test_replay_from_start_and_from_cursor(redis: Redis) -> None:
    aid = uuid.uuid4()
    first = await events.publish(redis, aid, Event(status=AnalysisStatus.RUNNING, stage="a"))
    await events.publish(redis, aid, Event(status=AnalysisStatus.RUNNING, stage="b"))

    everything = await events.read(redis, aid, "0", block_ms=10)
    assert [e.stage for _, e in everything] == ["a", "b"]

    rest = await events.read(redis, aid, first, block_ms=10)
    assert [e.stage for _, e in rest] == ["b"]


async def test_read_times_out_empty(redis: Redis) -> None:
    assert await events.read(redis, uuid.uuid4(), "0", block_ms=10) == []


async def test_terminal_event_sets_expiry(redis: Redis) -> None:
    aid = uuid.uuid4()
    await events.publish(redis, aid, Event(status=AnalysisStatus.RUNNING))
    assert await redis.ttl(events.stream_key(aid)) == -1
    await events.publish(redis, aid, Event(status=AnalysisStatus.COMPLETED))
    assert 0 < await redis.ttl(events.stream_key(aid)) <= events.FINISHED_TTL_S
