import asyncio
import logging
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from prism import events
from prism.a11y import audit
from prism.api.uploads import resolve_key
from prism.config import Settings
from prism.db.models import Analysis, AnalysisStatus
from prism.domain import AnalysisResult
from prism.events import Event
from prism.vision.base import DetectionCanceled, Detector

log = logging.getLogger(__name__)

CANCEL_POLL_S = 0.5
FAILED_MESSAGE = "Analysis failed. The error has been logged."
STOPPED_MESSAGE = "Analysis didn't finish: it took too long or the worker stopped."


async def analyze(ctx: dict[str, Any], analysis_id: str) -> str:
    settings: Settings = ctx["settings"]
    sessions: async_sessionmaker[AsyncSession] = ctx["sessionmaker"]
    redis: Redis = ctx["redis"]
    detector: Detector = ctx["detector"]
    aid = uuid.UUID(analysis_id)

    claimed = await _claim(sessions, aid)
    if claimed is None:
        log.info("analysis %s is no longer queued, skipping", aid)
        return "skipped"

    await events.publish(redis, aid, Event(status=AnalysisStatus.RUNNING, stage="detecting"))
    stop = threading.Event()
    watcher = asyncio.create_task(_watch_cancel(redis, aid, stop))
    started = time.perf_counter()
    try:
        image_key, dpr = claimed
        image = await asyncio.to_thread(_load_image, settings.upload_dir, image_key)
        elements = await asyncio.to_thread(detector.detect, image, stop.is_set)
        await events.publish(redis, aid, Event(status=AnalysisStatus.RUNNING, stage="auditing"))
        result = await asyncio.to_thread(audit, image, elements, dpr)
    except DetectionCanceled:
        await _finish(sessions, redis, aid, AnalysisStatus.CANCELED)
        return "canceled"
    except asyncio.CancelledError:
        # arq's job timeout and worker shutdown cancel this task. Stop the
        # generation thread too, and record why, or the row stays "running".
        stop.set()
        await asyncio.shield(
            _finish(sessions, redis, aid, AnalysisStatus.FAILED, error=STOPPED_MESSAGE)
        )
        raise
    except Exception:
        log.exception("analysis %s failed", aid)
        await _finish(sessions, redis, aid, AnalysisStatus.FAILED, error=FAILED_MESSAGE)
        return "failed"
    finally:
        stop.set()
        watcher.cancel()

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    await _finish(
        sessions,
        redis,
        aid,
        AnalysisStatus.COMPLETED,
        result=result,
        model_version=detector.version,
        elapsed_ms=elapsed_ms,
    )
    return "completed"


async def _claim(
    sessions: async_sessionmaker[AsyncSession], aid: uuid.UUID
) -> tuple[str, float] | None:
    """Move queued -> running atomically. None if it was canceled or already taken."""
    async with sessions() as session:
        row = await session.execute(
            update(Analysis)
            .where(Analysis.id == aid, Analysis.status == AnalysisStatus.QUEUED)
            .values(status=AnalysisStatus.RUNNING, started_at=datetime.now(UTC))
            .returning(Analysis.image_key, Analysis.device_pixel_ratio)
        )
        claimed = row.one_or_none()
        await session.commit()
        return (claimed[0], claimed[1]) if claimed else None


async def _finish(
    sessions: async_sessionmaker[AsyncSession],
    redis: Redis,
    aid: uuid.UUID,
    status: AnalysisStatus,
    *,
    result: AnalysisResult | None = None,
    error: str | None = None,
    model_version: str | None = None,
    elapsed_ms: int | None = None,
) -> None:
    async with sessions() as session:
        await session.execute(
            update(Analysis)
            .where(Analysis.id == aid, Analysis.status == AnalysisStatus.RUNNING)
            .values(
                status=status,
                result=result.model_dump(mode="json") if result else None,
                error=error,
                model_version=model_version,
                elapsed_ms=elapsed_ms,
                finished_at=datetime.now(UTC),
            )
        )
        await session.commit()

    data = None
    if result is not None:
        data = {
            "elements": len(result.elements),
            "findings": len(result.findings),
            "score": result.score,
        }
    await events.publish(redis, aid, Event(status=status, message=error, data=data))
    await redis.delete(events.cancel_key(aid))


async def _watch_cancel(redis: Redis, aid: uuid.UUID, stop: threading.Event) -> None:
    # The detector runs in a thread and only checks a threading.Event, so the
    # generation loop never waits on Redis.
    key = events.cancel_key(aid)
    while not stop.is_set():
        try:
            if await redis.exists(key):
                stop.set()
                return
        except RedisError:
            log.warning("cancel check for %s failed; retrying", aid)
        await asyncio.sleep(CANCEL_POLL_S)


async def fail_stale(sessions: async_sessionmaker[AsyncSession], older_than_s: int) -> int:
    """Mark analyses that have been "running" longer than the job timeout as failed.

    Covers a worker that crashed or was killed mid-job: nothing else would ever
    finish those rows. Called when a worker starts.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=older_than_s)
    async with sessions() as session:
        rows = await session.execute(
            update(Analysis)
            .where(Analysis.status == AnalysisStatus.RUNNING, Analysis.started_at < cutoff)
            .values(
                status=AnalysisStatus.FAILED, error=STOPPED_MESSAGE, finished_at=datetime.now(UTC)
            )
            .returning(Analysis.id)
        )
        stale = len(rows.all())
        await session.commit()
    return stale


def _load_image(upload_dir: Path, key: str) -> Image.Image:
    with Image.open(resolve_key(upload_dir, key)) as img:
        return img.convert("RGB")
