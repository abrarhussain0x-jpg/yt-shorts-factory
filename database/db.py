"""
database/db.py — SQLAlchemy 2.0 ORM with SQLite backend.

Complete database layer for yt-shorts-factory providing:
  - Six ORM models: Job, Video, ExportRecord, TranscriptionCache,
    PipelineMetric, AnalyticsSnapshot
  - Thread-safe engine with WAL mode for concurrent read/write access
  - Session context manager for safe, auto-rolling-back session handling
  - Migration-safe schema initialisation (create tables, add missing columns)
  - Comprehensive indexing strategy for fast queries
  - Full CRUD and analytics API with proper type annotations

All public functions use explicit Session management with try/except/finally
blocks ensuring sessions are always closed even on unexpected errors.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    desc,
    event,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)
from sqlalchemy.engine import Engine

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("db")


# ── SQLAlchemy Base ────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""

    pass


# ── Enum Constants ─────────────────────────────────────────────────────────
JOB_STATUS_VALUES: tuple[str, ...] = (
    "pending", "running", "done", "failed", "retrying", "cancelled",
)
PRIORITY_VALUES: tuple[str, ...] = ("high", "medium", "low")
PLATFORM_VALUES: tuple[str, ...] = (
    "youtube", "tiktok", "reels", "twitter", "facebook", "snapchat",
)
CLIP_QUALITY_GRADES: tuple[str, ...] = ("A", "B", "C", "D")


# ── Models ─────────────────────────────────────────────────────────────────
class Job(Base):
    """Pipeline job tracking record.

    Represents a single processing job from URL submission through
    completion. Supports priority queuing, dependency chains, batch
    tracking, heartbeat monitoring, and delayed scheduling.

    Attributes:
        id: UUID-based primary key.
        url: Source YouTube URL (max 500 chars).
        status: Current job state (pending/running/done/failed/retrying/cancelled).
        priority: Queue priority (high/medium/low).
        created_at: Timestamp when job was created.
        started_at: Timestamp when processing began.
        finished_at: Timestamp when processing ended.
        duration_seconds: Total processing time in seconds.
        output_path: Filesystem path to the primary output.
        error_message: Error details if the job failed.
        retry_count: Number of retry attempts made.
        settings_json: JSON-encoded job-specific settings override.
        source: Source identifier for rate limiting.
        batch_id: UUID grouping related jobs into a batch.
        depends_on: Foreign key to a predecessor Job.
        clip_index: Zero-based index for multi-clip jobs.
        heartbeat_at: Last heartbeat timestamp from the worker.
        scheduled_at: Timestamp for delayed processing.
        parent_job_id: Foreign key to parent Job for multi-clip relationships.
    """

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    url = Column(String(500), nullable=False)
    status = Column(
        Enum(*JOB_STATUS_VALUES, name="job_status"),
        default="pending",
        nullable=False,
    )
    priority = Column(String(10), default="medium", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, default=0.0)
    output_path = Column(Text, default="")
    error_message = Column(Text, default="")
    retry_count = Column(Integer, default=0)
    settings_json = Column(Text, default="{}")
    source = Column(String(200), default="")
    batch_id = Column(String(36), default="")
    depends_on = Column(String(36), ForeignKey("jobs.id"), nullable=True)
    clip_index = Column(Integer, default=0)
    heartbeat_at = Column(DateTime, nullable=True)
    scheduled_at = Column(DateTime, nullable=True)
    parent_job_id = Column(String(36), ForeignKey("jobs.id"), nullable=True)

    # Relationships
    video = relationship("Video", back_populates="job", uselist=False, cascade="all, delete-orphan")
    exports = relationship("ExportRecord", back_populates="job", cascade="all, delete-orphan")
    metrics = relationship("PipelineMetric", back_populates="job", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_priority", "priority"),
        Index("ix_jobs_created_at", "created_at"),
        Index("ix_jobs_batch_id", "batch_id"),
        Index("ix_jobs_source", "source"),
        Index("ix_jobs_scheduled_at", "scheduled_at"),
    )

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, status={self.status}, priority={self.priority}, url={self.url[:50]})>"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the Job to a plain dictionary.

        Returns:
            Dictionary of all Job column values with datetimes as ISO strings.
        """
        return {
            "id": self.id,
            "url": self.url,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "settings_json": self.settings_json,
            "source": self.source,
            "batch_id": self.batch_id,
            "depends_on": self.depends_on,
            "clip_index": self.clip_index,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "parent_job_id": self.parent_job_id,
        }


class Video(Base):
    """Per-clip metadata record.

    Stores comprehensive metadata about a processed video clip including
    source information, quality metrics, content analysis results,
    subtitle data, and multi-platform export paths.

    Attributes:
        id: Auto-incrementing primary key.
        job_id: Foreign key to the parent Job.
        youtube_id: YouTube video identifier.
        title: Video title (max 500 chars).
        channel: Channel name (max 200 chars).
        duration: Source video duration in seconds.
        view_count: Source video view count.
        clip_start: Clip start time in seconds.
        clip_end: Clip end time in seconds.
        energy_score: Audio energy score for the clip.
        whisper_model_used: Name of the Whisper model used.
        word_count: Number of words in the transcript.
        language: Detected language code (max 10 chars).
        fps: Frames per second.
        has_subtitles: Whether subtitles were detected/extracted.
        thumbnail_path: Path to the thumbnail image.
        srt_path: Path to the SRT subtitle file.
        vtt_path: Path to the VTT subtitle file.
        audio_quality_score: Audio quality metric.
        content_rating: Content safety rating.
        moderation_flags: JSON array of moderation flags.
        clip_quality_grade: Quality grade (A/B/C/D).
        speech_rate_wpm: Speech rate in words per minute.
        music_likelihood: Probability of music presence.
        face_detected: Whether a face was detected in the clip.
        motion_score: Motion intensity score.
        output_youtube: Path to YouTube-formatted export.
        output_tiktok: Path to TikTok-formatted export.
        output_reels: Path to Reels-formatted export.
        created_at: Timestamp when record was created.
    """

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    youtube_id = Column(String(20), default="")
    title = Column(String(500), default="")
    channel = Column(String(200), default="")
    duration = Column(Float, default=0.0)
    view_count = Column(Integer, default=0)
    clip_start = Column(Float, default=0.0)
    clip_end = Column(Float, default=0.0)
    energy_score = Column(Float, default=0.0)
    whisper_model_used = Column(String(50), default="")
    word_count = Column(Integer, default=0)
    language = Column(String(10), default="")
    fps = Column(Float, default=0.0)
    has_subtitles = Column(Boolean, default=False)
    thumbnail_path = Column(Text, default="")
    srt_path = Column(Text, default="")
    vtt_path = Column(Text, default="")
    audio_quality_score = Column(Float, default=0.0)
    content_rating = Column(String(10), default="")
    moderation_flags = Column(Text, default="[]")
    clip_quality_grade = Column(String(2), default="")
    speech_rate_wpm = Column(Float, default=0.0)
    music_likelihood = Column(Float, default=0.0)
    face_detected = Column(Boolean, default=False)
    motion_score = Column(Float, default=0.0)
    output_youtube = Column(Text, default="")
    output_tiktok = Column(Text, default="")
    output_reels = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    job = relationship("Job", back_populates="video")

    __table_args__ = (
        Index("ix_videos_job_id", "job_id"),
        Index("ix_videos_youtube_id", "youtube_id"),
        Index("ix_videos_channel", "channel"),
    )

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, title={self.title[:50]})>"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the Video to a plain dictionary.

        Returns:
            Dictionary of all Video column values with datetimes as ISO strings.
        """
        return {
            "id": self.id,
            "job_id": self.job_id,
            "youtube_id": self.youtube_id,
            "title": self.title,
            "channel": self.channel,
            "duration": self.duration,
            "view_count": self.view_count,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "energy_score": self.energy_score,
            "whisper_model_used": self.whisper_model_used,
            "word_count": self.word_count,
            "language": self.language,
            "fps": self.fps,
            "has_subtitles": self.has_subtitles,
            "thumbnail_path": self.thumbnail_path,
            "srt_path": self.srt_path,
            "vtt_path": self.vtt_path,
            "audio_quality_score": self.audio_quality_score,
            "content_rating": self.content_rating,
            "moderation_flags": self.moderation_flags,
            "clip_quality_grade": self.clip_quality_grade,
            "speech_rate_wpm": self.speech_rate_wpm,
            "music_likelihood": self.music_likelihood,
            "face_detected": self.face_detected,
            "motion_score": self.motion_score,
            "output_youtube": self.output_youtube,
            "output_tiktok": self.output_tiktok,
            "output_reels": self.output_reels,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExportRecord(Base):
    """Export platform record.

    Tracks individual platform-specific exports, including file details,
    encoding parameters, and validation status.

    Attributes:
        id: Auto-incrementing primary key.
        job_id: Foreign key to the parent Job.
        platform: Target platform name (youtube/tiktok/reels/twitter/facebook/snapchat).
        file_path: Filesystem path to the exported file.
        file_size: File size in bytes.
        duration: Export duration in seconds.
        resolution: Resolution string (e.g. '1080x1920').
        codec: Video codec used (e.g. 'libx264').
        crf: Constant Rate Factor value used during encoding.
        validated: Whether the export passed validation checks.
        validation_errors: JSON or text describing any validation errors.
        created_at: Timestamp when the record was created.
    """

    __tablename__ = "export_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    platform = Column(String(20), default="")
    file_path = Column(Text, default="")
    file_size = Column(Integer, default=0)
    duration = Column(Float, default=0.0)
    resolution = Column(String(20), default="")
    codec = Column(String(20), default="")
    crf = Column(Integer, default=23)
    validated = Column(Boolean, default=False)
    validation_errors = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    job = relationship("Job", back_populates="exports")

    __table_args__ = (
        Index("ix_export_records_job_id", "job_id"),
        Index("ix_export_records_platform", "platform"),
    )

    def __repr__(self) -> str:
        return f"<ExportRecord(id={self.id}, job_id={self.job_id}, platform={self.platform})>"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the ExportRecord to a plain dictionary.

        Returns:
            Dictionary of all ExportRecord column values.
        """
        return {
            "id": self.id,
            "job_id": self.job_id,
            "platform": self.platform,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "duration": self.duration,
            "resolution": self.resolution,
            "codec": self.codec,
            "crf": self.crf,
            "validated": self.validated,
            "validation_errors": self.validation_errors,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TranscriptionCache(Base):
    """Whisper transcription result cache.

    Caches transcription results keyed by audio hash and model name
    to avoid redundant Whisper calls for the same audio content.

    Attributes:
        id: Auto-incrementing primary key.
        audio_hash: SHA-256 hex digest of the audio file (unique, indexed).
        model_name: Name of the Whisper model used.
        language: Detected or specified language code.
        word_count: Number of words in the transcription.
        result_json: JSON-encoded full transcription result.
        created_at: Timestamp when the cache entry was created.
        accessed_at: Timestamp of the most recent cache hit.
    """

    __tablename__ = "transcription_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audio_hash = Column(String(64), unique=True, nullable=False)
    model_name = Column(String(50), default="")
    language = Column(String(10), default="")
    word_count = Column(Integer, default=0)
    result_json = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    accessed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_transcription_cache_audio_hash", "audio_hash", unique=True),
        Index("ix_transcription_cache_model_name", "model_name"),
    )

    def __repr__(self) -> str:
        return f"<TranscriptionCache(id={self.id}, hash={self.audio_hash[:16]}, model={self.model_name})>"


class PipelineMetric(Base):
    """Individual pipeline step timing and status record.

    Captures per-step performance data for each job, enabling
    fine-grained performance analysis and bottleneck detection.

    Attributes:
        id: Auto-incrementing primary key.
        job_id: Foreign key to the parent Job.
        step_name: Name of the pipeline step (e.g. 'download', 'transcribe').
        duration_seconds: Wall-clock time for this step in seconds.
        status: Step completion status (e.g. 'success', 'error', 'skipped').
        error_message: Error details if the step failed.
        timestamp: When this metric was recorded.
    """

    __tablename__ = "pipeline_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    step_name = Column(String(50), default="")
    duration_seconds = Column(Float, default=0.0)
    status = Column(String(20), default="")
    error_message = Column(Text, default="")
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    job = relationship("Job", back_populates="metrics")

    __table_args__ = (
        Index("ix_pipeline_metrics_job_id", "job_id"),
        Index("ix_pipeline_metrics_step_name", "step_name"),
    )

    def __repr__(self) -> str:
        return f"<PipelineMetric(id={self.id}, job_id={self.job_id}, step={self.step_name})>"


class AnalyticsSnapshot(Base):
    """Daily analytics snapshot record.

    Stores pre-computed aggregate metrics for each day, enabling
    fast dashboard queries without scanning the entire jobs table.

    Attributes:
        id: Auto-incrementing primary key.
        snapshot_date: The calendar date this snapshot covers (unique).
        total_jobs: Cumulative total of all jobs up to this date.
        completed_jobs: Cumulative total of completed jobs.
        failed_jobs: Cumulative total of failed jobs.
        avg_processing_time: Average processing time in seconds for completed jobs.
        total_videos_created: Cumulative total of video records.
        total_disk_usage_mb: Estimated total disk usage in megabytes.
        most_used_whisper_model: Most frequently used Whisper model name.
        top_source_channel: Channel with the most processed videos.
        created_at: Timestamp when this snapshot was computed.
    """

    __tablename__ = "analytics_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date, unique=True, nullable=False)
    total_jobs = Column(Integer, default=0)
    completed_jobs = Column(Integer, default=0)
    failed_jobs = Column(Integer, default=0)
    avg_processing_time = Column(Float, default=0.0)
    total_videos_created = Column(Integer, default=0)
    total_disk_usage_mb = Column(Float, default=0.0)
    most_used_whisper_model = Column(String(50), default="")
    top_source_channel = Column(String(200), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_analytics_snapshots_date", "snapshot_date", unique=True),
    )

    def __repr__(self) -> str:
        return f"<AnalyticsSnapshot(id={self.id}, date={self.snapshot_date})>"


# ── Engine & Session Factory ───────────────────────────────────────────────
_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def _get_engine() -> Engine:
    """Get or create the SQLAlchemy engine (lazy singleton).

    Creates a thread-safe SQLite engine with ``check_same_thread=False``
    and WAL journal mode for improved concurrent read/write performance.

    Returns:
        The global SQLAlchemy Engine instance.

    Raises:
        RuntimeError: If the database path cannot be resolved.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        db_path: Path = settings.DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )

        # Enable WAL mode for better concurrent access
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        logger.debug("Database engine created at %s", db_path)
    return _engine


def _get_session_factory() -> sessionmaker:
    """Get or create the session factory (lazy singleton).

    Returns:
        The global sessionmaker instance bound to the engine.
    """
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionFactory


def _new_session() -> Session:
    """Create a new database session.

    Returns:
        A fresh SQLAlchemy Session instance.
    """
    return _get_session_factory()()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations.

    The session is automatically committed on success and rolled back
    on any exception. The session is always closed when the block exits.

    Yields:
        A SQLAlchemy Session instance.

    Example::

        with session_scope() as session:
            job = session.get(Job, job_id)
            job.status = "running"
    """
    session = _new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Migration Helpers ──────────────────────────────────────────────────────

# Column definitions for migration: (table_name, column_name, column_type_sql)
_MIGRATION_COLUMNS: list[tuple[str, str, str]] = [
    # Job enhancements (v1 -> v2)
    ("jobs", "priority", "VARCHAR(10) DEFAULT 'medium'"),
    ("jobs", "source", "VARCHAR(200) DEFAULT ''"),
    ("jobs", "batch_id", "VARCHAR(36) DEFAULT ''"),
    ("jobs", "depends_on", "VARCHAR(36)"),
    ("jobs", "clip_index", "INTEGER DEFAULT 0"),
    ("jobs", "heartbeat_at", "DATETIME"),
    ("jobs", "scheduled_at", "DATETIME"),
    ("jobs", "parent_job_id", "VARCHAR(36)"),
    # Video enhancements (v1 -> v2)
    ("videos", "language", "VARCHAR(10) DEFAULT ''"),
    ("videos", "fps", "FLOAT DEFAULT 0.0"),
    ("videos", "has_subtitles", "BOOLEAN DEFAULT 0"),
    ("videos", "thumbnail_path", "TEXT DEFAULT ''"),
    ("videos", "srt_path", "TEXT DEFAULT ''"),
    ("videos", "vtt_path", "TEXT DEFAULT ''"),
    ("videos", "audio_quality_score", "FLOAT DEFAULT 0.0"),
    ("videos", "content_rating", "VARCHAR(10) DEFAULT ''"),
    ("videos", "moderation_flags", "TEXT DEFAULT '[]'"),
    ("videos", "clip_quality_grade", "VARCHAR(2) DEFAULT ''"),
    ("videos", "speech_rate_wpm", "FLOAT DEFAULT 0.0"),
    ("videos", "music_likelihood", "FLOAT DEFAULT 0.0"),
    ("videos", "face_detected", "BOOLEAN DEFAULT 0"),
    ("videos", "motion_score", "FLOAT DEFAULT 0.0"),
]


def _run_migrations(engine: Engine) -> None:
    """Apply schema migrations by adding missing columns to existing tables.

    Uses SQLAlchemy inspection to detect existing columns and only adds
    those that are not yet present, making this safe for repeat calls.

    Args:
        engine: The SQLAlchemy engine to inspect and migrate.
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    for table_name, column_name, column_type_sql in _MIGRATION_COLUMNS:
        if table_name not in existing_tables:
            # Table doesn't exist yet; it will be created by create_all()
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        if column_name not in existing_columns:
            try:
                alter_sql = (
                    f"ALTER TABLE {table_name} "
                    f"ADD COLUMN {column_name} {column_type_sql}"
                )
                with engine.connect() as conn:
                    conn.execute(text(alter_sql))
                    conn.commit()
                logger.info("Migration: added column %s.%s", table_name, column_name)
            except Exception as exc:
                logger.warning(
                    "Migration failed for %s.%s: %s", table_name, column_name, exc
                )


# ── Public API ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """Initialise the database: create tables, run migrations, set WAL mode, create indexes.

    This function is idempotent — safe to call on every application startup.
    It performs the following steps in order:
      1. Creates all tables defined in the ORM metadata that don't yet exist.
      2. Runs column-level migrations to add missing columns to existing tables.
      3. Verifies WAL journal mode is active for concurrent read/write safety.

    Raises:
        OperationalError: If the database cannot be opened or created.
    """
    engine = _get_engine()

    # Step 1: Create all tables (no-op if they already exist)
    Base.metadata.create_all(engine)
    logger.debug("Base metadata create_all completed")

    # Step 2: Run column migrations for existing tables
    _run_migrations(engine)
    logger.debug("Migrations completed")

    # Step 3: Verify WAL mode
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode"))
        mode = result.scalar()
        if mode and mode.lower() != "wal":
            logger.warning(
                "Journal mode is '%s', expected 'WAL'. Attempting to set WAL mode.",
                mode,
            )
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
        else:
            logger.debug("WAL journal mode confirmed")

    logger.info("Database initialised successfully at %s", engine.url)


def create_job(
    url: str,
    priority: str = "medium",
    settings_json: dict[str, Any] | None = None,
    source: str = "",
    batch_id: str = "",
    depends_on: str = "",
    clip_index: int = 0,
    scheduled_at: datetime | None = None,
    parent_job_id: str = "",
) -> Job:
    """Create a new pending pipeline job.

    Generates a UUID-based job ID, validates the priority level, and
    inserts the job record into the database.

    Args:
        url: YouTube video URL to process.
        priority: Queue priority — one of 'high', 'medium', 'low'. Defaults to 'medium'.
        settings_json: Optional dict of per-job settings overrides. Stored as JSON text.
        source: Source identifier for rate limiting (e.g. 'api', 'cli', channel name).
        batch_id: UUID string grouping related jobs into a batch.
        depends_on: UUID of a predecessor Job that must complete before this one starts.
        clip_index: Zero-based index for multi-clip extraction jobs.
        scheduled_at: Optional datetime for delayed processing.
        parent_job_id: UUID of the parent Job for multi-clip relationships.

    Returns:
        The newly created Job instance with all fields populated.

    Raises:
        ValueError: If priority is not one of the allowed values.
        RuntimeError: If the database insert fails.
    """
    if priority not in PRIORITY_VALUES:
        raise ValueError(
            f"Priority must be one of {PRIORITY_VALUES}, got '{priority}'"
        )

    job_id = str(uuid.uuid4())

    job = Job(
        id=job_id,
        url=url,
        status="pending",
        priority=priority,
        created_at=datetime.now(timezone.utc),
        settings_json=json.dumps(settings_json or {}),
        source=source,
        batch_id=batch_id,
        depends_on=depends_on if depends_on else None,
        clip_index=clip_index,
        scheduled_at=scheduled_at,
        parent_job_id=parent_job_id if parent_job_id else None,
    )

    session = _new_session()
    try:
        session.add(job)
        session.commit()
        session.refresh(job)
        logger.info(
            "Created job %s for %s (priority=%s, source=%s)",
            job_id, url[:60], priority, source,
        )
        return job
    except Exception as exc:
        session.rollback()
        logger.error("Failed to create job: %s", exc)
        raise
    finally:
        session.close()


def update_job_status(
    job_id: str,
    status: str,
    error_message: str = "",
    output_path: str = "",
) -> None:
    """Update a job's status and related timestamps.

    Automatically sets ``started_at`` when status becomes ``'running'``,
    and ``finished_at`` + ``duration_seconds`` when status becomes
    ``'done'``, ``'failed'``, or ``'cancelled'``. Also increments
    ``retry_count`` when status is ``'retrying'``.

    Args:
        job_id: UUID of the job to update.
        status: New status value — must be one of :data:`JOB_STATUS_VALUES`.
        error_message: Optional error message for failed jobs. Truncated to 2000 chars.
        output_path: Optional output file path for completed jobs.

    Raises:
        ValueError: If status is not a recognised job status value.
    """
    if status not in JOB_STATUS_VALUES:
        raise ValueError(
            f"Status must be one of {JOB_STATUS_VALUES}, got '{status}'"
        )

    session = _new_session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            logger.warning("Job %s not found for status update", job_id)
            return

        job.status = status

        now = datetime.now(timezone.utc)

        if status == "running":
            job.started_at = now
            job.heartbeat_at = now
        elif status == "retrying":
            job.retry_count = (job.retry_count or 0) + 1
            job.heartbeat_at = now
        elif status in ("done", "failed", "cancelled"):
            job.finished_at = now
            if job.started_at:
                # Handle both timezone-aware and naive datetimes
                started = job.started_at
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                elapsed = (now - started).total_seconds()
                job.duration_seconds = round(elapsed, 3)

        if error_message:
            job.error_message = error_message[:2000]
        if output_path:
            job.output_path = output_path

        session.commit()
        logger.debug("Job %s status updated to %s", job_id, status)
    except Exception as exc:
        session.rollback()
        logger.error("Failed to update job %s: %s", job_id, exc)
    finally:
        session.close()


def get_job(job_id: str) -> Job | None:
    """Retrieve a job by its UUID.

    Args:
        job_id: UUID of the job to retrieve.

    Returns:
        The Job instance if found, otherwise ``None``.
    """
    session = _new_session()
    try:
        return session.get(Job, job_id)
    finally:
        session.close()


def list_jobs(
    limit: int = 20,
    status: str | None = None,
    priority: str | None = None,
) -> list[Job]:
    """List recent jobs with optional filtering.

    Results are ordered by creation date (newest first).

    Args:
        limit: Maximum number of jobs to return. Defaults to 20.
        status: Optional status filter (e.g. ``'pending'``, ``'done'``).
        priority: Optional priority filter (e.g. ``'high'``, ``'medium'``, ``'low'``).

    Returns:
        List of Job instances matching the filters, up to ``limit``.
    """
    session = _new_session()
    try:
        stmt = select(Job).order_by(desc(Job.created_at))
        if status:
            stmt = stmt.where(Job.status == status)
        if priority:
            stmt = stmt.where(Job.priority == priority)
        stmt = stmt.limit(limit)
        return list(session.scalars(stmt).all())
    finally:
        session.close()


def cancel_job(job_id: str) -> bool:
    """Cancel a pending or retrying job.

    Only jobs in ``'pending'`` or ``'retrying'`` status can be cancelled.
    Jobs that are already ``'running'``, ``'done'``, ``'failed'``, or
    ``'cancelled'`` will not be modified.

    Args:
        job_id: UUID of the job to cancel.

    Returns:
        ``True`` if the job was successfully cancelled, ``False`` if the
        job was not found or was not in a cancellable state.
    """
    session = _new_session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            logger.warning("Job %s not found for cancellation", job_id)
            return False

        if job.status not in ("pending", "retrying"):
            logger.info(
                "Job %s cannot be cancelled (current status: %s)",
                job_id, job.status,
            )
            return False

        job.status = "cancelled"
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
        logger.info("Job %s cancelled", job_id)
        return True
    except Exception as exc:
        session.rollback()
        logger.error("Failed to cancel job %s: %s", job_id, exc)
        return False
    finally:
        session.close()


def save_video_record(job_id: str, data: dict[str, Any]) -> Video:
    """Create a Video record for a completed pipeline run.

    Extracts known fields from the data dictionary and creates a new
    Video row linked to the specified job. Unknown keys in ``data``
    are silently ignored.

    Args:
        job_id: UUID of the parent Job.
        data: Dictionary of video metadata. Supported keys match
              the Video model column names.

    Returns:
        The newly created Video instance.

    Raises:
        RuntimeError: If the database insert fails.
    """
    video = Video(
        job_id=job_id,
        youtube_id=data.get("youtube_id", ""),
        title=data.get("title", ""),
        channel=data.get("channel", ""),
        duration=data.get("duration", 0.0),
        view_count=data.get("view_count", 0),
        clip_start=data.get("clip_start", 0.0),
        clip_end=data.get("clip_end", 0.0),
        energy_score=data.get("energy_score", 0.0),
        whisper_model_used=data.get("whisper_model_used", ""),
        word_count=data.get("word_count", 0),
        language=data.get("language", ""),
        fps=data.get("fps", 0.0),
        has_subtitles=data.get("has_subtitles", False),
        thumbnail_path=data.get("thumbnail_path", ""),
        srt_path=data.get("srt_path", ""),
        vtt_path=data.get("vtt_path", ""),
        audio_quality_score=data.get("audio_quality_score", 0.0),
        content_rating=data.get("content_rating", ""),
        moderation_flags=json.dumps(data.get("moderation_flags", [])),
        clip_quality_grade=data.get("clip_quality_grade", ""),
        speech_rate_wpm=data.get("speech_rate_wpm", 0.0),
        music_likelihood=data.get("music_likelihood", 0.0),
        face_detected=data.get("face_detected", False),
        motion_score=data.get("motion_score", 0.0),
        output_youtube=data.get("output_youtube", ""),
        output_tiktok=data.get("output_tiktok", ""),
        output_reels=data.get("output_reels", ""),
    )

    session = _new_session()
    try:
        session.add(video)
        session.commit()
        session.refresh(video)
        logger.info(
            "Saved video record for job %s: %s", job_id, data.get("title", "untitled")
        )
        return video
    except Exception as exc:
        session.rollback()
        logger.error("Failed to save video record: %s", exc)
        raise
    finally:
        session.close()


def save_export_record(
    job_id: str,
    platform: str,
    file_path: str,
    file_size: int,
    duration: float,
    resolution: str,
    codec: str,
    crf: int,
    validated: bool,
    validation_errors: str,
) -> ExportRecord:
    """Create an export record for a platform-specific output.

    Args:
        job_id: UUID of the parent Job.
        platform: Target platform name (youtube/tiktok/reels/etc.).
        file_path: Filesystem path to the exported file.
        file_size: File size in bytes.
        duration: Export duration in seconds.
        resolution: Resolution string (e.g. ``'1080x1920'``).
        codec: Video codec used (e.g. ``'libx264'``).
        crf: Constant Rate Factor value used during encoding.
        validated: Whether the export passed validation checks.
        validation_errors: Description of any validation errors.

    Returns:
        The newly created ExportRecord instance.

    Raises:
        ValueError: If platform is not a recognised value.
        RuntimeError: If the database insert fails.
    """
    if platform not in PLATFORM_VALUES:
        raise ValueError(
            f"Platform must be one of {PLATFORM_VALUES}, got '{platform}'"
        )

    record = ExportRecord(
        job_id=job_id,
        platform=platform,
        file_path=file_path,
        file_size=file_size,
        duration=duration,
        resolution=resolution,
        codec=codec,
        crf=crf,
        validated=validated,
        validation_errors=validation_errors,
    )

    session = _new_session()
    try:
        session.add(record)
        session.commit()
        session.refresh(record)
        logger.info(
            "Saved export record for job %s: platform=%s, size=%d",
            job_id, platform, file_size,
        )
        return record
    except Exception as exc:
        session.rollback()
        logger.error("Failed to save export record: %s", exc)
        raise
    finally:
        session.close()


def get_transcription_cache(
    audio_hash: str,
    model_name: str,
) -> dict[str, Any] | None:
    """Look up a cached transcription result by audio hash and model.

    Updates the ``accessed_at`` timestamp on cache hit to support
    LRU-style eviction.

    Args:
        audio_hash: SHA-256 hex digest of the audio file.
        model_name: Name of the Whisper model used.

    Returns:
        Parsed JSON dict of the cached transcription result, or ``None``
        if no matching cache entry exists.
    """
    session = _new_session()
    try:
        stmt = select(TranscriptionCache).where(
            TranscriptionCache.audio_hash == audio_hash,
            TranscriptionCache.model_name == model_name,
        )
        cache_entry = session.scalar(stmt)
        if cache_entry is None:
            return None

        # Update accessed_at for LRU tracking
        cache_entry.accessed_at = datetime.now(timezone.utc)
        session.commit()

        try:
            return json.loads(cache_entry.result_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Corrupt cache entry for hash=%s, model=%s",
                audio_hash[:16], model_name,
            )
            return None
    finally:
        session.close()


def save_transcription_cache(
    audio_hash: str,
    model_name: str,
    language: str,
    word_count: int,
    result_data: dict[str, Any] | list[Any],
) -> None:
    """Store a transcription result in the cache.

    If a cache entry with the same ``audio_hash`` already exists, it is
    updated in place. Otherwise a new entry is created.

    Args:
        audio_hash: SHA-256 hex digest of the audio file.
        model_name: Name of the Whisper model used.
        language: Detected or specified language code.
        word_count: Number of words in the transcription.
        result_data: Full transcription result (dict or list) to cache as JSON.

    Raises:
        RuntimeError: If the database upsert fails.
    """
    session = _new_session()
    try:
        existing = session.scalar(
            select(TranscriptionCache).where(
                TranscriptionCache.audio_hash == audio_hash,
            )
        )

        if existing:
            existing.model_name = model_name
            existing.language = language
            existing.word_count = word_count
            existing.result_json = json.dumps(result_data, ensure_ascii=False)
            existing.accessed_at = datetime.now(timezone.utc)
        else:
            cache_entry = TranscriptionCache(
                audio_hash=audio_hash,
                model_name=model_name,
                language=language,
                word_count=word_count,
                result_json=json.dumps(result_data, ensure_ascii=False),
            )
            session.add(cache_entry)

        session.commit()
        logger.debug(
            "Cached transcription for hash=%s, model=%s",
            audio_hash[:16], model_name,
        )
    except Exception as exc:
        session.rollback()
        logger.error("Failed to save transcription cache: %s", exc)
        raise
    finally:
        session.close()


def record_pipeline_metric(
    job_id: str,
    step_name: str,
    duration_seconds: float,
    status: str,
    error_message: str = "",
) -> None:
    """Record a pipeline step execution metric.

    Args:
        job_id: UUID of the parent Job.
        step_name: Name of the pipeline step (e.g. ``'download'``, ``'transcribe'``).
        duration_seconds: Wall-clock time for this step.
        status: Step completion status (e.g. ``'success'``, ``'error'``, ``'skipped'``).
        error_message: Error details if the step failed.
    """
    metric = PipelineMetric(
        job_id=job_id,
        step_name=step_name,
        duration_seconds=round(duration_seconds, 3),
        status=status,
        error_message=error_message[:2000] if error_message else "",
    )

    session = _new_session()
    try:
        session.add(metric)
        session.commit()
        logger.debug(
            "Recorded metric for job %s: step=%s, duration=%.2fs, status=%s",
            job_id, step_name, duration_seconds, status,
        )
    except Exception as exc:
        session.rollback()
        logger.error("Failed to record pipeline metric: %s", exc)
    finally:
        session.close()


def get_pipeline_metrics(job_id: str) -> list[PipelineMetric]:
    """Retrieve all pipeline step metrics for a given job.

    Args:
        job_id: UUID of the job.

    Returns:
        List of PipelineMetric instances ordered by timestamp.
    """
    session = _new_session()
    try:
        stmt = (
            select(PipelineMetric)
            .where(PipelineMetric.job_id == job_id)
            .order_by(PipelineMetric.timestamp)
        )
        return list(session.scalars(stmt).all())
    finally:
        session.close()


def save_analytics_snapshot() -> AnalyticsSnapshot:
    """Compute and save a daily analytics snapshot.

    Aggregates job, video, and metric data into a single snapshot
    record for the current date. If a snapshot already exists for
    today, it is updated in place.

    Returns:
        The created or updated AnalyticsSnapshot instance.

    Raises:
        RuntimeError: If the database upsert fails.
    """
    today = date.today()
    now = datetime.now(timezone.utc)

    session = _new_session()
    try:
        # Check for existing snapshot
        existing = session.scalar(
            select(AnalyticsSnapshot).where(
                AnalyticsSnapshot.snapshot_date == today,
            )
        )

        # Compute aggregate metrics
        total_jobs = session.scalar(select(func.count(Job.id))) or 0
        completed_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "done")
        ) or 0
        failed_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "failed")
        ) or 0
        avg_processing_time = session.scalar(
            select(func.avg(Job.duration_seconds)).where(
                Job.status == "done",
                Job.duration_seconds > 0,
            )
        ) or 0.0
        total_videos = session.scalar(select(func.count(Video.id))) or 0

        # Most used whisper model
        most_used_model_row = session.scalar(
            select(Video.whisper_model_used)
            .where(Video.whisper_model_used != "")
            .group_by(Video.whisper_model_used)
            .order_by(func.count(Video.whisper_model_used).desc())
            .limit(1)
        )
        most_used_model = most_used_model_row or ""

        # Top source channel
        top_channel_row = session.scalar(
            select(Video.channel)
            .where(Video.channel != "")
            .group_by(Video.channel)
            .order_by(func.count(Video.channel).desc())
            .limit(1)
        )
        top_channel = top_channel_row or ""

        # Estimate disk usage from export records
        total_disk_bytes = session.scalar(
            select(func.coalesce(func.sum(ExportRecord.file_size), 0))
        ) or 0
        total_disk_mb = round(total_disk_bytes / (1024 * 1024), 2)

        if existing:
            existing.total_jobs = total_jobs
            existing.completed_jobs = completed_jobs
            existing.failed_jobs = failed_jobs
            existing.avg_processing_time = round(avg_processing_time, 2)
            existing.total_videos_created = total_videos
            existing.total_disk_usage_mb = total_disk_mb
            existing.most_used_whisper_model = most_used_model
            existing.top_source_channel = top_channel
            existing.created_at = now
            snapshot = existing
        else:
            snapshot = AnalyticsSnapshot(
                snapshot_date=today,
                total_jobs=total_jobs,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                avg_processing_time=round(avg_processing_time, 2),
                total_videos_created=total_videos,
                total_disk_usage_mb=total_disk_mb,
                most_used_whisper_model=most_used_model,
                top_source_channel=top_channel,
                created_at=now,
            )
            session.add(snapshot)

        session.commit()
        session.refresh(snapshot)
        logger.info("Saved analytics snapshot for %s", today)
        return snapshot
    except Exception as exc:
        session.rollback()
        logger.error("Failed to save analytics snapshot: %s", exc)
        raise
    finally:
        session.close()


def get_stats() -> dict[str, Any]:
    """Return aggregate pipeline statistics.

    Provides a comprehensive overview of job processing including
    counts by status, success rate, average duration, video counts,
    cache hit rate, and retry statistics.

    Returns:
        Dictionary with keys:
          - total_jobs, done_jobs, failed_jobs, running_jobs,
            pending_jobs, retrying_jobs, cancelled_jobs
          - avg_duration_seconds, median_duration_seconds
          - total_videos_created
          - success_rate (percentage)
          - total_export_records
          - total_cache_entries
          - avg_retry_count
          - jobs_last_24h, jobs_last_7d
    """
    session = _new_session()
    try:
        total_jobs = session.scalar(select(func.count(Job.id))) or 0
        done_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "done")
        ) or 0
        failed_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "failed")
        ) or 0
        running_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "running")
        ) or 0
        pending_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "pending")
        ) or 0
        retrying_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "retrying")
        ) or 0
        cancelled_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.status == "cancelled")
        ) or 0

        avg_duration = session.scalar(
            select(func.avg(Job.duration_seconds)).where(Job.status == "done")
        ) or 0.0

        # Median duration using subquery
        median_duration = 0.0
        try:
            done_count = session.scalar(
                select(func.count(Job.id)).where(
                    Job.status == "done", Job.duration_seconds > 0
                )
            ) or 0
            if done_count > 0:
                mid_offset = done_count // 2
                median_row = session.scalar(
                    select(Job.duration_seconds)
                    .where(Job.status == "done", Job.duration_seconds > 0)
                    .order_by(Job.duration_seconds)
                    .offset(mid_offset)
                    .limit(1)
                )
                median_duration = median_row or 0.0
        except Exception:
            pass

        total_videos = session.scalar(select(func.count(Video.id))) or 0
        total_exports = session.scalar(select(func.count(ExportRecord.id))) or 0
        total_cache = session.scalar(select(func.count(TranscriptionCache.id))) or 0
        avg_retry = session.scalar(
            select(func.avg(Job.retry_count)).where(Job.retry_count > 0)
        ) or 0.0

        # Jobs in last 24h / 7d
        now = datetime.now(timezone.utc)
        jobs_24h = session.scalar(
            select(func.count(Job.id)).where(
                Job.created_at >= now - timedelta(hours=24)
            )
        ) or 0
        jobs_7d = session.scalar(
            select(func.count(Job.id)).where(
                Job.created_at >= now - timedelta(days=7)
            )
        ) or 0

        return {
            "total_jobs": total_jobs,
            "done_jobs": done_jobs,
            "failed_jobs": failed_jobs,
            "running_jobs": running_jobs,
            "pending_jobs": pending_jobs,
            "retrying_jobs": retrying_jobs,
            "cancelled_jobs": cancelled_jobs,
            "avg_duration_seconds": round(avg_duration, 1),
            "median_duration_seconds": round(median_duration, 1),
            "total_videos_created": total_videos,
            "total_export_records": total_exports,
            "total_cache_entries": total_cache,
            "success_rate": round(done_jobs / total_jobs * 100, 1) if total_jobs > 0 else 0.0,
            "avg_retry_count": round(avg_retry, 2),
            "jobs_last_24h": jobs_24h,
            "jobs_last_7d": jobs_7d,
        }
    finally:
        session.close()


def get_daily_analytics(days: int = 30) -> list[dict[str, Any]]:
    """Retrieve daily analytics snapshots for the last N days.

    Args:
        days: Number of days to look back. Defaults to 30.

    Returns:
        List of dictionaries, one per day, ordered by snapshot_date ascending.
        Each dict contains all AnalyticsSnapshot column values.
    """
    cutoff_date = date.today() - timedelta(days=days)

    session = _new_session()
    try:
        stmt = (
            select(AnalyticsSnapshot)
            .where(AnalyticsSnapshot.snapshot_date >= cutoff_date)
            .order_by(AnalyticsSnapshot.snapshot_date)
        )
        snapshots = list(session.scalars(stmt).all())
        return [
            {
                "snapshot_date": s.snapshot_date.isoformat() if s.snapshot_date else None,
                "total_jobs": s.total_jobs,
                "completed_jobs": s.completed_jobs,
                "failed_jobs": s.failed_jobs,
                "avg_processing_time": s.avg_processing_time,
                "total_videos_created": s.total_videos_created,
                "total_disk_usage_mb": s.total_disk_usage_mb,
                "most_used_whisper_model": s.most_used_whisper_model,
                "top_source_channel": s.top_source_channel,
            }
            for s in snapshots
        ]
    finally:
        session.close()


def get_top_channels(limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve the top source channels by number of videos processed.

    Args:
        limit: Maximum number of channels to return. Defaults to 10.

    Returns:
        List of dicts with keys ``'channel'`` and ``'video_count'``,
        ordered by video count descending.
    """
    session = _new_session()
    try:
        stmt = (
            select(
                Video.channel,
                func.count(Video.id).label("video_count"),
            )
            .where(Video.channel != "")
            .group_by(Video.channel)
            .order_by(func.count(Video.id).desc())
            .limit(limit)
        )
        rows = session.execute(stmt).all()
        return [
            {"channel": row.channel, "video_count": row.video_count}
            for row in rows
        ]
    finally:
        session.close()


def get_processing_time_percentiles() -> dict[str, float]:
    """Compute processing time percentiles for completed jobs.

    Calculates p50, p75, p90, and p99 of the ``duration_seconds``
    field across all successfully completed jobs.

    Returns:
        Dictionary with keys ``'p50'``, ``'p75'``, ``'p90'``, ``'p99'``
        mapping to their respective percentile values in seconds.
        Returns all zeros if there are no completed jobs.
    """
    session = _new_session()
    try:
        # Get all completed job durations sorted
        stmt = (
            select(Job.duration_seconds)
            .where(Job.status == "done", Job.duration_seconds > 0)
            .order_by(Job.duration_seconds)
        )
        durations = [d for d in session.scalars(stmt).all() if d is not None]

        if not durations:
            return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0}

        def _percentile(data: list[float], pct: float) -> float:
            """Calculate the given percentile from a sorted list."""
            idx = int(len(data) * pct / 100.0)
            idx = min(idx, len(data) - 1)
            return round(data[idx], 2)

        return {
            "p50": _percentile(durations, 50),
            "p75": _percentile(durations, 75),
            "p90": _percentile(durations, 90),
            "p99": _percentile(durations, 99),
        }
    finally:
        session.close()


def cleanup_old_jobs(days: int = 30) -> int:
    """Remove jobs older than N days that are done, failed, or cancelled.

    Cascades deletion to related Video, ExportRecord, and PipelineMetric
    records via the ORM relationship configuration.

    Args:
        days: Age threshold in days. Jobs older than this are removed.
              Defaults to 30.

    Returns:
        Number of jobs removed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    session = _new_session()
    try:
        stmt = select(Job).where(
            Job.created_at < cutoff,
            Job.status.in_(["done", "failed", "cancelled"]),
        )
        old_jobs = list(session.scalars(stmt).all())
        count = len(old_jobs)
        for job in old_jobs:
            session.delete(job)
        session.commit()
        if count > 0:
            logger.info("Cleaned up %d old jobs (older than %d days)", count, days)
        return count
    except Exception as exc:
        session.rollback()
        logger.error("Failed to cleanup old jobs: %s", exc)
        return 0
    finally:
        session.close()


def cleanup_old_cache(days: int = 90) -> int:
    """Remove transcription cache entries older than N days.

    Uses the ``accessed_at`` timestamp so frequently-used cache entries
    are preserved even if they were created long ago.

    Args:
        days: Age threshold in days based on last access time.
              Defaults to 90.

    Returns:
        Number of cache entries removed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    session = _new_session()
    try:
        stmt = delete(TranscriptionCache).where(
            TranscriptionCache.accessed_at < cutoff,
        )
        result = session.execute(stmt)
        session.commit()
        count = result.rowcount  # type: ignore[union-attr]
        if count > 0:
            logger.info(
                "Cleaned up %d old cache entries (not accessed in %d days)",
                count, days,
            )
        return count
    except Exception as exc:
        session.rollback()
        logger.error("Failed to cleanup old cache: %s", exc)
        return 0
    finally:
        session.close()


def vacuum_database() -> None:
    """Run VACUUM on the SQLite database to reclaim disk space.

    This rebuilds the database file, removing fragmentation and
    deleted-row overhead. Should be called periodically or after
    large cleanup operations.

    Note:
        VACUUM requires an exclusive lock on the database and may
        take significant time for large databases.
    """
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("VACUUM"))
            conn.commit()
        logger.info("Database VACUUM completed")
    except Exception as exc:
        logger.error("Database VACUUM failed: %s", exc)


def get_job_with_details(job_id: str) -> dict[str, Any] | None:
    """Retrieve a job with all related records in a single dict.

    Performs a joined query to fetch the Job, its associated Video,
    ExportRecords, and PipelineMetrics, returning everything as a
    nested dictionary.

    Args:
        job_id: UUID of the job to retrieve.

    Returns:
        Dictionary with keys ``'job'``, ``'video'``, ``'exports'``,
        ``'metrics'``, or ``None`` if the job is not found.
    """
    session = _new_session()
    try:
        job = session.get(Job, job_id)
        if job is None:
            return None

        result: dict[str, Any] = {
            "job": job.to_dict(),
            "video": job.video.to_dict() if job.video else None,
            "exports": [e.to_dict() for e in job.exports] if job.exports else [],
            "metrics": [],
        }

        # Fetch metrics separately for reliable ordering
        metrics_stmt = (
            select(PipelineMetric)
            .where(PipelineMetric.job_id == job_id)
            .order_by(PipelineMetric.timestamp)
        )
        metrics = list(session.scalars(metrics_stmt).all())
        result["metrics"] = [
            {
                "id": m.id,
                "step_name": m.step_name,
                "duration_seconds": m.duration_seconds,
                "status": m.status,
                "error_message": m.error_message,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in metrics
        ]

        return result
    finally:
        session.close()


def search_videos(query: str, limit: int = 20) -> list[Video]:
    """Search videos by title or channel name.

    Performs a case-insensitive LIKE search on both the ``title``
    and ``channel`` columns.

    Args:
        query: Search string to match against title and channel.
        limit: Maximum number of results. Defaults to 20.

    Returns:
        List of Video instances matching the search query.
    """
    session = _new_session()
    try:
        pattern = f"%{query}%"
        stmt = (
            select(Video)
            .where(
                (Video.title.ilike(pattern)) | (Video.channel.ilike(pattern))
            )
            .order_by(desc(Video.created_at))
            .limit(limit)
        )
        return list(session.scalars(stmt).all())
    finally:
        session.close()


def get_disk_usage_summary() -> dict[str, Any]:
    """Calculate disk usage statistics from export records.

    Aggregates file sizes from ExportRecord entries, grouping by
    platform. Also computes total database file size on disk.

    Returns:
        Dictionary with keys:
          - ``'total_bytes'``: Sum of all export file sizes.
          - ``'total_mb'``: Total size in megabytes.
          - ``'by_platform'``: Dict mapping platform name to ``{'bytes', 'mb', 'count'}``.
          - ``'database_size_mb'``: Size of the SQLite database file on disk.
    """
    session = _new_session()
    try:
        # Aggregate by platform
        stmt = (
            select(
                ExportRecord.platform,
                func.sum(ExportRecord.file_size).label("total_bytes"),
                func.count(ExportRecord.id).label("count"),
            )
            .group_by(ExportRecord.platform)
        )
        rows = session.execute(stmt).all()

        by_platform: dict[str, dict[str, Any]] = {}
        grand_total = 0
        for row in rows:
            platform_total = row.total_bytes or 0
            grand_total += platform_total
            by_platform[row.platform] = {
                "bytes": platform_total,
                "mb": round(platform_total / (1024 * 1024), 2),
                "count": row.count,
            }

        # Database file size
        db_size_mb = 0.0
        try:
            settings = get_settings()
            db_path = settings.DB_PATH
            if db_path.exists():
                db_size_mb = round(db_path.stat().st_size / (1024 * 1024), 2)
        except Exception:
            pass

        return {
            "total_bytes": grand_total,
            "total_mb": round(grand_total / (1024 * 1024), 2),
            "by_platform": by_platform,
            "database_size_mb": db_size_mb,
        }
    finally:
        session.close()


def get_failure_analysis() -> dict[str, Any]:
    """Analyse job and pipeline step failures.

    Computes failure rates, common error messages, and failure
    rates broken down by pipeline step.

    Returns:
        Dictionary with keys:
          - ``'total_failures'``: Count of failed jobs.
          - ``'failure_rate'``: Percentage of jobs that failed.
          - ``'common_errors'``: List of ``{'message', 'count'}`` dicts.
          - ``'failure_rate_by_step'``: List of ``{'step', 'failure_rate', 'count'}`` dicts.
          - ``'retry_success_rate'``: Percentage of retried jobs that eventually succeeded.
    """
    session = _new_session()
    try:
        total_jobs = session.scalar(select(func.count(Job.id))) or 0
        total_failures = session.scalar(
            select(func.count(Job.id)).where(Job.status == "failed")
        ) or 0

        failure_rate = round(total_failures / total_jobs * 100, 1) if total_jobs > 0 else 0.0

        # Common error messages (top 10)
        error_stmt = (
            select(
                Job.error_message,
                func.count(Job.id).label("count"),
            )
            .where(
                Job.status == "failed",
                Job.error_message != "",
            )
            .group_by(Job.error_message)
            .order_by(func.count(Job.id).desc())
            .limit(10)
        )
        error_rows = session.execute(error_stmt).all()
        common_errors = [
            {"message": row.error_message[:200], "count": row.count}
            for row in error_rows
        ]

        # Failure rate by pipeline step
        step_stmt = (
            select(
                PipelineMetric.step_name,
                func.count(PipelineMetric.id).label("total"),
                func.sum(
                    func.cast(
                        PipelineMetric.status == "error", Integer
                    )
                ).label("failures"),
            )
            .group_by(PipelineMetric.step_name)
        )
        step_rows = session.execute(step_stmt).all()
        failure_rate_by_step: list[dict[str, Any]] = []
        for row in step_rows:
            total = row.total or 0
            failures = row.failures or 0
            failure_rate_by_step.append({
                "step": row.step_name,
                "failure_rate": round(failures / total * 100, 1) if total > 0 else 0.0,
                "count": failures,
            })
        failure_rate_by_step.sort(key=lambda x: x["failure_rate"], reverse=True)

        # Retry success rate
        retried_jobs = session.scalar(
            select(func.count(Job.id)).where(Job.retry_count > 0)
        ) or 0
        retried_success = session.scalar(
            select(func.count(Job.id)).where(
                Job.retry_count > 0, Job.status == "done",
            )
        ) or 0
        retry_success_rate = (
            round(retried_success / retried_jobs * 100, 1) if retried_jobs > 0 else 0.0
        )

        return {
            "total_failures": total_failures,
            "failure_rate": failure_rate,
            "common_errors": common_errors,
            "failure_rate_by_step": failure_rate_by_step,
            "retry_success_rate": retry_success_rate,
        }
    finally:
        session.close()
