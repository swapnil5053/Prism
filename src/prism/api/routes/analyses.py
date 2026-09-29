import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from redis.exceptions import RedisError
from sqlalchemy import select, update

from prism import events
from prism.db.models import Analysis, AnalysisStatus
from prism.domain import AnalysisResult
from prism.events import Event
from prism.report import render_html
from prism.schemas import AnalysisOut, AnalysisPage, AnalysisSummary

from ..deps import SessionDep, SettingsDep
from ..uploads import UploadRejected, display_name, read_limited, resolve_key, store_image
from ..workspace import CurrentWorkspace, EnsuredWorkspace

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analyses", tags=["analyses"])

ANALYZE_TASK = "analyze"
REPORT_CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'"
_MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


async def _owned(session: SessionDep, analysis_id: uuid.UUID, ws: uuid.UUID | None) -> Analysis:
    analysis = await session.get(Analysis, analysis_id)
    # Same 404 whether it doesn't exist or belongs to someone else.
    if analysis is None or ws is None or analysis.workspace_id != ws:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found")
    return analysis


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_analysis(
    request: Request,
    response: Response,
    file: UploadFile,
    session: SessionDep,
    settings: SettingsDep,
    workspace_id: EnsuredWorkspace,
    device_pixel_ratio: Annotated[float, Form(ge=1, le=4)] = 1.0,
) -> AnalysisOut:
    try:
        data = await read_limited(file, settings.max_upload_bytes)
        stored = await asyncio.to_thread(
            store_image, data, settings.upload_dir, settings.max_image_pixels
        )
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc

    analysis = Analysis(
        workspace_id=workspace_id,
        image_key=stored.key,
        image_width=stored.width,
        image_height=stored.height,
        original_filename=display_name(file.filename),
        device_pixel_ratio=device_pixel_ratio,
        status=AnalysisStatus.QUEUED,
    )
    session.add(analysis)
    try:
        await session.commit()
    except Exception:
        resolve_key(settings.upload_dir, stored.key).unlink(missing_ok=True)
        raise
    await session.refresh(analysis)

    try:
        # A fixed job id makes a retried enqueue a no-op instead of a duplicate run.
        await request.app.state.queue.enqueue_job(
            ANALYZE_TASK, str(analysis.id), _job_id=f"analysis:{analysis.id}"
        )
    except (RedisError, OSError):
        log.exception("could not enqueue analysis %s", analysis.id)
        analysis.status = AnalysisStatus.FAILED
        analysis.error = "Could not queue the analysis. Try again shortly."
        analysis.finished_at = datetime.now(UTC)
        await session.commit()
        # Return the failed record rather than raising, so a first-time visitor still
        # gets their workspace cookie and can see what happened.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return AnalysisOut.model_validate(analysis)


@router.get("")
async def list_analyses(
    session: SessionDep,
    workspace_id: CurrentWorkspace,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    before: datetime | None = None,
) -> AnalysisPage:
    if workspace_id is None:
        return AnalysisPage(items=[], next_before=None)

    query = (
        select(Analysis)
        .where(Analysis.workspace_id == workspace_id)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit)
    )
    if before is not None:
        query = query.where(Analysis.created_at < before)
    rows = (await session.scalars(query)).all()

    items = [AnalysisSummary.model_validate(a) for a in rows]
    next_before = rows[-1].created_at if len(rows) == limit else None
    return AnalysisPage(items=items, next_before=next_before)


@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: uuid.UUID, session: SessionDep, workspace_id: CurrentWorkspace
) -> AnalysisOut:
    return AnalysisOut.model_validate(await _owned(session, analysis_id, workspace_id))


@router.get("/{analysis_id}/image")
async def get_image(
    analysis_id: uuid.UUID,
    session: SessionDep,
    settings: SettingsDep,
    workspace_id: CurrentWorkspace,
) -> FileResponse:
    analysis = await _owned(session, analysis_id, workspace_id)
    path = resolve_key(settings.upload_dir, analysis.image_key)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES[path.suffix.lstrip(".")],
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/{analysis_id}/report")
async def get_report(
    analysis_id: uuid.UUID,
    session: SessionDep,
    settings: SettingsDep,
    workspace_id: CurrentWorkspace,
    format: Literal["html", "json"] = "html",
) -> Response:
    analysis = await _owned(session, analysis_id, workspace_id)
    if analysis.status != AnalysisStatus.COMPLETED or analysis.result is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Analysis has not completed")
    result = AnalysisResult.model_validate(analysis.result)
    stem = f"prism-{analysis.id.hex[:8]}"

    if format == "json":
        return Response(
            AnalysisOut.model_validate(analysis).model_dump_json(indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{stem}.json"'},
        )

    def build() -> str:
        with Image.open(resolve_key(settings.upload_dir, analysis.image_key)) as img:
            return render_html(
                analysis.original_filename or "Screenshot",
                analysis.created_at.strftime("%Y-%m-%d %H:%M UTC"),
                result,
                img,
                analysis.model_version,
            )

    return HTMLResponse(
        await asyncio.to_thread(build),
        headers={
            "Content-Disposition": f'attachment; filename="{stem}.html"',
            # The report is a static document; nothing in it should run.
            "Content-Security-Policy": REPORT_CSP,
        },
    )


@router.post("/{analysis_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_analysis(
    analysis_id: uuid.UUID,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    workspace_id: CurrentWorkspace,
) -> AnalysisOut:
    analysis = await _owned(session, analysis_id, workspace_id)
    if analysis.status.is_terminal:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Analysis is already {analysis.status}")

    redis = request.app.state.queue
    # Still in the queue: cancel it here. The worker skips anything not queued.
    canceled = await session.scalar(
        update(Analysis)
        .where(Analysis.id == analysis_id, Analysis.status == AnalysisStatus.QUEUED)
        .values(status=AnalysisStatus.CANCELED, finished_at=datetime.now(UTC))
        .returning(Analysis.id)
    )
    await session.commit()
    if canceled is not None:
        await events.publish(redis, analysis_id, Event(status=AnalysisStatus.CANCELED))
    else:
        # Already running: flag it and let the worker stop generation.
        await redis.set(events.cancel_key(analysis_id), "1", ex=settings.job_timeout_s + 60)

    await session.refresh(analysis)
    return AnalysisOut.model_validate(analysis)
