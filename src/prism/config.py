from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from PRISM_* environment variables or .env."""

    model_config = SettingsConfigDict(env_prefix="PRISM_", env_file=".env", extra="ignore")

    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    secret_key: SecretStr

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)
    cookie_secure: bool = True
    sql_echo: bool = False
    log_level: str = "INFO"

    upload_dir: Path = Path("data/uploads")
    # Built frontend (web/dist). When set, the API serves it at /.
    web_dir: Path | None = None
    uploads_per_hour: int = Field(default=30, ge=0)  # per client IP; 0 disables
    max_upload_mb: int = Field(default=10, gt=0, le=50)
    detector_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    detector_quant: Literal["nf4", "int8", "none"] = "nf4"
    # Longest image side fed to the model. Bigger finds small icons but costs VRAM and time.
    detector_max_side: int = Field(default=896, ge=448, le=2048)
    detector_prompt: Literal["v1", "v2", "v3", "v4"] = "v2"
    # "ocr": text lines come from OCR and the VLM's text items are dropped
    # (vision/hybrid.py). "model": the VLM finds text too.
    detector_text: Literal["ocr", "model"] = "ocr"
    job_timeout_s: int = Field(default=300, gt=0)

    # Pillow's own bomb guard trips at ~89M pixels; screenshots never need that many.
    max_image_pixels: int = Field(default=40_000_000, gt=0)

    @field_validator("secret_key")
    @classmethod
    def _secret_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("PRISM_SECRET_KEY must be at least 32 characters")
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # values come from the environment
