import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AnalysisStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def is_terminal(self) -> bool:
        return self in {AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELED}


class Workspace(Base):
    """An anonymous owner of analyses. Identified by a signed cookie, no login."""

    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan", passive_deletes=True
    )


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_workspace_created", "workspace_id", "created_at"),
        # jsonb_path_ops is smaller than the default opclass and still serves @> queries,
        # e.g. "every analysis that found a contrast problem".
        Index(
            "ix_analyses_result_gin",
            "result",
            postgresql_using="gin",
            postgresql_ops={"result": "jsonb_path_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(
            AnalysisStatus,
            name="analysis_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        default=AnalysisStatus.QUEUED,
    )

    # Server-generated file name inside settings.upload_dir. Never a user-supplied path.
    image_key: Mapped[str] = mapped_column(String(64))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    # Screenshot pixels per CSS pixel (2 for most phone and retina screenshots).
    device_pixel_ratio: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0")

    model_version: Mapped[str | None] = mapped_column(String(128))
    elapsed_ms: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    workspace: Mapped[Workspace] = relationship(back_populates="analyses")

    @property
    def score(self) -> float | None:
        return self.result.get("score") if self.result else None

    @property
    def finding_count(self) -> int | None:
        return len(self.result.get("findings", [])) if self.result else None
