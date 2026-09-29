from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readiness(request: Request, response: Response) -> dict[str, str]:
    checks: dict[str, str] = {}
    try:
        async with request.app.state.sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"
    try:
        await request.app.state.queue.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"

    if any(v != "ok" for v in checks.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return checks
