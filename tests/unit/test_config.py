import pytest
from pydantic import ValidationError

from prism.config import Settings

DB = "postgresql+asyncpg://u:p@localhost/db"


def test_rejects_short_secret() -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(database_url=DB, secret_key="too-short", _env_file=None)


def test_upload_limit_in_bytes() -> None:
    s = Settings(database_url=DB, secret_key="k" * 32, max_upload_mb=2, _env_file=None)
    assert s.max_upload_bytes == 2 * 1024 * 1024


def test_reads_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_DATABASE_URL", DB)
    monkeypatch.setenv("PRISM_SECRET_KEY", "k" * 40)
    monkeypatch.setenv("PRISM_CORS_ORIGINS", '["http://localhost:5173"]')
    s = Settings(_env_file=None)
    assert s.cors_origins == ["http://localhost:5173"]


def test_detector_defaults_use_ocr_for_text() -> None:
    s = Settings(database_url=DB, secret_key="k" * 32, _env_file=None)
    assert (s.detector_text, s.detector_prompt) == ("ocr", "v2")
    with pytest.raises(ValidationError):
        Settings(database_url=DB, secret_key="k" * 32, detector_prompt="v9", _env_file=None)
