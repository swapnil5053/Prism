import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prism import __version__
from prism.config import Settings, get_settings
from prism.db.session import make_engine, make_sessionmaker

from .routes import analyses, events, health

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        engine = make_engine(settings)
        app.state.sessionmaker = make_sessionmaker(engine)
        app.state.queue = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        try:
            yield
        finally:
            await app.state.queue.aclose()
            await engine.dispose()

    app = FastAPI(title="Prism", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        # Only accept ids that look like ours, so log lines can't be forged.
        if len(rid) > 64 or not rid.isalnum():
            rid = uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", "-")
        log.exception("unhandled error request_id=%s path=%s", rid, request.url.path)
        # Details stay in the logs; the client only gets an id to report.
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": rid},
            headers={"x-request-id": rid},
        )

    app.include_router(health.router)
    app.include_router(analyses.router)
    app.include_router(events.router)
    return app
