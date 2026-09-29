"""Per-client upload limit: a fixed-window counter in Redis.

Uploads are the expensive call (each one is a GPU job), so they get a budget
per client IP per hour. Behind a proxy, run uvicorn with --forwarded-allow-ips
so request.client is the real client.
"""

import time

from fastapi import HTTPException, Request, status

WINDOW_S = 3600


async def limit_uploads(request: Request) -> None:
    per_hour: int = request.app.state.settings.uploads_per_hour
    if per_hour <= 0:
        return
    client = request.client.host if request.client else "unknown"
    window = int(time.time() // WINDOW_S)
    key = f"prism:rl:upload:{client}:{window}"

    redis = request.app.state.queue
    async with redis.pipeline(transaction=True) as pipe:
        count, _ = await pipe.incr(key).expire(key, WINDOW_S).execute()
    if count > per_hour:
        retry_after = WINDOW_S - int(time.time()) % WINDOW_S
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Upload limit reached. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )
