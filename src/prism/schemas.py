"""Request/response models shared by the API and the worker."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field

from prism.db.models import AnalysisStatus


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: AnalysisStatus
    original_filename: str | None
    image_width: int
    image_height: int
    device_pixel_ratio: float
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def image_url(self) -> str:
        return f"/api/v1/analyses/{self.id}/image"


class AnalysisOut(AnalysisSummary):
    model_version: str | None
    elapsed_ms: int | None
    error: str | None
    result: dict[str, Any] | None
    started_at: datetime | None
    finished_at: datetime | None


class AnalysisPage(BaseModel):
    items: list[AnalysisSummary]
    # Pass as ?before= to get the next (older) page. None when there are no more.
    next_before: datetime | None
