import logging
from typing import Any, ClassVar

from arq import func
from arq.connections import RedisSettings

from prism.config import Settings, get_settings
from prism.db.session import make_engine, make_sessionmaker
from prism.vision.base import Detector

from .tasks import analyze

log = logging.getLogger(__name__)


def load_detector(settings: Settings) -> Detector:
    # Imported here so the API and tests never pull in torch.
    from prism.vision.qwen import QwenDetector

    detector = QwenDetector(
        settings.detector_model,
        quant=settings.detector_quant,
        max_side=settings.detector_max_side,
        prompt=settings.detector_prompt,
    )
    if settings.detector_text == "model":
        return detector

    from prism.vision.hybrid import HybridDetector
    from prism.vision.ocr import OcrReader

    return HybridDetector(detector, OcrReader())


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    engine = make_engine(settings)
    ctx.update(settings=settings, engine=engine, sessionmaker=make_sessionmaker(engine))
    # Load once, before taking jobs: the first request shouldn't pay for it.
    ctx["detector"] = detector = load_detector(settings)
    log.info("detector ready: %s", detector.version)


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["engine"].dispose()


class WorkerSettings:
    functions: ClassVar = [func(analyze, name="analyze", max_tries=1)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    job_timeout = get_settings().job_timeout_s
    # One GPU, one model in memory: run jobs one at a time.
    max_jobs = 1
