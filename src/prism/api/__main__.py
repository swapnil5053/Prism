import logging

import uvicorn

from prism.config import get_settings


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "prism.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
