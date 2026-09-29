import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from redis.exceptions import RedisError
from sqlalchemy import select

from prism.db.models import Analysis, AnalysisStatus
from prism.schemas import AnalysisOut, AnalysisPage, AnalysisSummary

from ..deps import SessionDep, SettingsDep
from ..uploads import UploadRejected, display_name, read_limited, resolve_key, store_image
from ..workspace import CurrentWorkspace, EnsuredWorkspace

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analyses", tags=["analyses"])

ANALYZE_TASK = "analyze"
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
