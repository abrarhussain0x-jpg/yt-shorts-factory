"""
scheduler/job_queue.py — Priority-based persistent job queue backed by SQLite.

Provides enqueue, dequeue, completion, failure, retry, dead-letter,
batching, scheduling, rate-limiting, dependency tracking, health
monitoring, and time-series statistics with atomic state transitions
and thread-safe operations via SQLAlchemy sessions.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Index,
    select, func, desc, and_, or_, update,
)
from sqlalchemy.orm import DeclarativeBase, Session

from database.db import _new_session, _get_engine, init_db
from utils.logger import get_logger

logger = get_logger("job_queue")

# ── Priority constants ─────────────────────────────────────
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"
PRIORITY_ORDER = {PRIORITY_HIGH: 0, PRIORITY_MEDIUM: 1, PRIORITY_LOW: 2}
VALID_PRIORITIES = set(PRIORITY_ORDER.keys())

# ── Concurrency limits per priority ────────────────────────
DEFAULT_MAX_CONCURRENT_HIGH = 3
DEFAULT_MAX_CONCURRENT_MEDIUM = 2
DEFAULT_MAX_CONCURRENT_LOW = 1

# ── Rate limiting defaults ────────────────────────────────
DEFAULT_RATE_LIMIT_PER_HOUR = 30


# ══════════════════════════════════════════════════════════
#  Extra ORM models (supplementary to database.db.Job)
# ══════════════════════════════════════════════════════════

class _ExtraBase(DeclarativeBase):
    pass


class DeadLetterJob(_ExtraBase):
    """Permanently failed job stored for manual inspection / re-queue."""

    __tablename__ = "dead_letter_jobs"

    id = Column(String(36), primary_key=True)
    original_job_id = Column(String(36), nullable=False, index=True)
    url = Column(String(500), nullable=False)
    reason = Column(Text, default="")
    error_message = Column(Text, default="")
    priority = Column(String(10), default="medium")
    source = Column(String(200), default="")
    settings_json = Column(Text, default="{}")
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    failed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    moved_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class QueueStatsSnapshot(_ExtraBase):
    """Time-series snapshot of queue statistics for trend analysis."""

    __tablename__ = "queue_stats_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    pending_high = Column(Integer, default=0)
    pending_medium = Column(Integer, default=0)
    pending_low = Column(Integer, default=0)
    running_count = Column(Integer, default=0)
    done_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    dead_letter_count = Column(Integer, default=0)
    avg_wait_seconds = Column(Float, default=0.0)
    avg_process_seconds = Column(Float, default=0.0)


# ── Ensure supplementary tables exist ─────────────────────
_tables_initialized = False
_init_lock = threading.Lock()


def _ensure_tables() -> None:
    """Create supplementary tables and add columns to jobs if missing."""
    global _tables_initialized
    if _tables_initialized:
        return
    with _init_lock:
        if _tables_initialized:
            return
        engine = _get_engine()
        _ExtraBase.metadata.create_all(engine)

        # Add extended columns to the jobs table if they don't exist
        with engine.connect() as conn:
            # SQLite introspection for existing columns
            result = conn.execute(
                __import__("sqlalchemy").text("PRAGMA table_info(jobs)")
            )
            existing = {row[1] for row in result}

            new_columns = [
                ("priority", "VARCHAR(10) DEFAULT 'medium'"),
                ("scheduled_at", "DATETIME DEFAULT NULL"),
                ("depends_on", "VARCHAR(36) DEFAULT NULL"),
                ("source", "VARCHAR(200) DEFAULT ''"),
                ("batch_id", "VARCHAR(36) DEFAULT NULL"),
                ("worker_id", "VARCHAR(100) DEFAULT NULL"),
                ("heartbeat_at", "DATETIME DEFAULT NULL"),
                ("progress_pct", "FLOAT DEFAULT 0.0"),
                ("expires_at", "DATETIME DEFAULT NULL"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing:
                    try:
                        conn.execute(
                            __import__("sqlalchemy").text(
                                f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}"
                            )
                        )
                        conn.commit()
                        logger.debug("Added column %s to jobs table", col_name)
                    except Exception:
                        # Column may already exist from a concurrent process
                        conn.rollback()

        _tables_initialized = True


# ══════════════════════════════════════════════════════════
#  Main class
# ══════════════════════════════════════════════════════════

class JobQueue:
    """Priority-based persistent job queue with full lifecycle management.

    Features:
      - Priority queuing (high / medium / low)
      - Job scheduling (run at specific time or after delay)
      - Job dependencies (B depends on A)
      - Per-source rate limiting (max N jobs / hour)
      - Dead-letter queue for permanently failed jobs
      - Batch enqueue / grouping
      - Queue statistics with time-series snapshots
      - Job expiration for stale pending entries
      - Concurrency control per priority
      - Queue health monitoring
    """

    def __init__(
        self,
        max_concurrent_high: int = DEFAULT_MAX_CONCURRENT_HIGH,
        max_concurrent_medium: int = DEFAULT_MAX_CONCURRENT_MEDIUM,
        max_concurrent_low: int = DEFAULT_MAX_CONCURRENT_LOW,
        rate_limit_per_hour: int = DEFAULT_RATE_LIMIT_PER_HOUR,
    ) -> None:
        init_db()
        _ensure_tables()

        self._max_concurrent = {
            PRIORITY_HIGH: max_concurrent_high,
            PRIORITY_MEDIUM: max_concurrent_medium,
            PRIORITY_LOW: max_concurrent_low,
        }
        self._rate_limit_per_hour = rate_limit_per_hour
        self._op_lock = threading.Lock()

    # ── Enqueue ──────────────────────────────────────────

    def enqueue(
        self,
        url: str,
        priority: str = "medium",
        scheduled_at: datetime | None = None,
        depends_on: str | None = None,
        settings_override: dict[str, Any] | None = None,
        source: str = "",
        batch_id: str | None = None,
        delay_seconds: float | None = None,
    ) -> str:
        """Add a URL to the queue as a pending job.

        Args:
            url: YouTube video URL to process.
            priority: Job priority — 'high', 'medium', or 'low'.
            scheduled_at: Run the job no earlier than this UTC datetime.
            depends_on: Job ID that must complete before this one runs.
            settings_override: Dict of settings overrides stored as JSON.
            source: Identifier for rate-limit grouping (e.g. channel name).
            batch_id: Optional batch ID for grouping related jobs.
            delay_seconds: Enqueue the job this many seconds in the future.

        Returns:
            The job ID of the created job.
        """
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'; use {VALID_PRIORITIES}")

        if delay_seconds is not None and delay_seconds > 0:
            scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        settings_json = json.dumps(settings_override or {})
        expires_at = now + timedelta(hours=72)  # default expiration window

        session = _new_session()
        try:
            # Use raw INSERT via SQLAlchemy Core to include extended columns
            from database.db import Job
            job = Job(
                id=job_id,
                url=url,
                status="pending",
                created_at=now,
                settings_json=settings_json,
            )
            # Set extended attributes (may or may not be ORM-mapped)
            for attr, val in [
                ("priority", priority),
                ("scheduled_at", scheduled_at),
                ("depends_on", depends_on),
                ("source", source[:200] if source else ""),
                ("batch_id", batch_id),
                ("worker_id", None),
                ("heartbeat_at", None),
                ("progress_pct", 0.0),
                ("expires_at", expires_at),
            ]:
                try:
                    setattr(job, attr, val)
                except AttributeError:
                    pass  # Column not in ORM model; will be handled by ALTER TABLE

            session.add(job)
            session.commit()
            logger.info(
                "Enqueued job %s: %s (priority=%s, schedule=%s, depends=%s)",
                job_id, url[:60], priority,
                scheduled_at.isoformat() if scheduled_at else "now",
                depends_on or "none",
            )
            return job_id
        except Exception as exc:
            session.rollback()
            logger.error("Failed to enqueue job: %s", exc)
            raise
        finally:
            session.close()

    # ── Dequeue ──────────────────────────────────────────

    def dequeue(self, worker_id: str = "") -> str | None:
        """Dequeue the highest-priority eligible pending job.

        A job is eligible when:
          - status is 'pending'
          - scheduled_at is NULL or in the past
          - depends_on is NULL or the dependency job is 'done'
          - priority concurrency limit is not exceeded
          - source rate limit is not exceeded

        Returns:
            Job ID of the dequeued job, or None if no eligible job.
        """
        now = datetime.now(timezone.utc)
        session = _new_session()
        try:
            from database.db import Job

            # Build candidate query: pending, scheduled or due, dependency met
            stmt = (
                select(Job)
                .where(Job.status == "pending")
                .order_by(
                    # Priority order: high=0, medium=1, low=2
                    __import__("sqlalchemy").case(
                        PRIORITY_ORDER,
                        value=Job.priority if hasattr(Job, "priority") else "medium",
                    ),
                    Job.created_at,
                )
            )
            candidates = list(session.scalars(stmt).all())

            for job in candidates:
                # Check scheduled_at
                scheduled = getattr(job, "scheduled_at", None)
                if scheduled and scheduled > now:
                    continue

                # Check dependency
                dep_id = getattr(job, "depends_on", None)
                if dep_id:
                    dep_job = session.get(Job, dep_id)
                    if dep_job is None or dep_job.status != "done":
                        continue

                # Check expiration
                expires = getattr(job, "expires_at", None)
                if expires and expires < now:
                    continue

                # Check concurrency
                priority = getattr(job, "priority", "medium") or "medium"
                running_count = session.scalar(
                    select(func.count(Job.id)).where(
                        and_(
                            Job.status == "running",
                            Job.priority == priority if hasattr(Job, "priority") else True,
                        )
                    )
                ) or 0
                max_conc = self._max_concurrent.get(priority, 2)
                if running_count >= max_conc:
                    continue

                # Check rate limit
                source = getattr(job, "source", "") or ""
                if source:
                    hour_ago = now - timedelta(hours=1)
                    recent = session.scalar(
                        select(func.count(Job.id)).where(
                            and_(
                                Job.source == source if hasattr(Job, "source") else True,
                                Job.created_at >= hour_ago,
                            )
                        )
                    ) or 0
                    if recent >= self._rate_limit_per_hour:
                        continue

                # Mark running
                job.status = "running"
                job.started_at = now
                if hasattr(job, "worker_id"):
                    job.worker_id = worker_id
                if hasattr(job, "heartbeat_at"):
                    job.heartbeat_at = now
                session.commit()

                logger.info("Dequeued job %s (priority=%s)", job.id, priority)
                return job.id

            return None
        except Exception as exc:
            session.rollback()
            logger.error("Failed to dequeue job: %s", exc)
            return None
        finally:
            session.close()

    # ── Complete / Fail / Retry ──────────────────────────

    def complete(self, job_id: str, output_path: str = "") -> None:
        """Mark a job as successfully completed."""
        session = _new_session()
        try:
            from database.db import Job
            job = session.get(Job, job_id)
            if job is None:
                logger.warning("Job %s not found for completion", job_id)
                return
            job.status = "done"
            now = datetime.now(timezone.utc)
            job.finished_at = now
            if job.started_at:
                # Handle timezone-naive datetimes from SQLite
                started = job.started_at
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                job.duration_seconds = (now - started).total_seconds()
            if output_path:
                job.output_path = output_path
            session.commit()
            logger.info("Job %s completed", job_id)
        except Exception as exc:
            session.rollback()
            logger.error("Failed to complete job %s: %s", job_id, exc)
        finally:
            session.close()

    def fail(self, job_id: str, error: str = "") -> None:
        """Mark a job as failed."""
        session = _new_session()
        try:
            from database.db import Job
            job = session.get(Job, job_id)
            if job is None:
                logger.warning("Job %s not found for failure", job_id)
                return
            job.status = "failed"
            now = datetime.now(timezone.utc)
            job.finished_at = now
            if job.started_at:
                started = job.started_at
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                job.duration_seconds = (now - started).total_seconds()
            if error:
                job.error_message = error[:2000]
            session.commit()
            logger.warning("Job %s failed: %s", job_id, error[:100])
        except Exception as exc:
            session.rollback()
            logger.error("Failed to mark job %s as failed: %s", job_id, exc)
        finally:
            session.close()

    def retry(self, job_id: str) -> bool:
        """Attempt to retry a failed job.

        Returns:
            True if the job was re-queued, False if max retries exceeded.
        """
        from config.settings import get_settings
        max_retries = get_settings().JOB_RETRY_ATTEMPTS

        session = _new_session()
        try:
            from database.db import Job
            job = session.get(Job, job_id)
            if job is None:
                return False

            job.retry_count = (job.retry_count or 0) + 1
            if job.retry_count >= max_retries:
                job.status = "failed"
                session.commit()
                logger.warning(
                    "Job %s exceeded max retries (%d)", job_id, max_retries
                )
                return False

            job.status = "pending"
            job.finished_at = None
            job.started_at = None
            session.commit()
            logger.info(
                "Job %s re-queued (retry %d/%d)",
                job_id, job.retry_count, max_retries,
            )
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Failed to retry job %s: %s", job_id, exc)
            return False
        finally:
            session.close()

    # ── Dead Letter Queue ────────────────────────────────

    def move_to_dead_letter(self, job_id: str, reason: str = "") -> None:
        """Move a permanently failed job to the dead-letter queue."""
        session = _new_session()
        try:
            from database.db import Job
            job = session.get(Job, job_id)
            if job is None:
                logger.warning("Job %s not found for DLQ move", job_id)
                return

            dlq_entry = DeadLetterJob(
                id=str(uuid.uuid4()),
                original_job_id=job.id,
                url=job.url,
                reason=reason[:500],
                error_message=job.error_message or "",
                priority=getattr(job, "priority", "medium") or "medium",
                source=getattr(job, "source", "") or "",
                settings_json=job.settings_json or "{}",
                retry_count=job.retry_count or 0,
                created_at=job.created_at,
                failed_at=job.finished_at or datetime.now(timezone.utc),
                moved_at=datetime.now(timezone.utc),
            )
            session.add(dlq_entry)

            # Remove from active table
            session.delete(job)
            session.commit()
            logger.info("Job %s moved to dead letter queue: %s", job_id, reason[:80])
        except Exception as exc:
            session.rollback()
            logger.error("Failed to move job %s to DLQ: %s", job_id, exc)
        finally:
            session.close()

    def get_dead_letter_jobs(self, limit: int = 20) -> list[dict]:
        """Return dead-letter jobs as a list of dicts.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with keys: id, original_job_id, url, reason,
            error_message, priority, source, retry_count, created_at, failed_at.
        """
        session = _new_session()
        try:
            stmt = (
                select(DeadLetterJob)
                .order_by(desc(DeadLetterJob.moved_at))
                .limit(limit)
            )
            rows = list(session.scalars(stmt).all())
            return [
                {
                    "id": r.id,
                    "original_job_id": r.original_job_id,
                    "url": r.url,
                    "reason": r.reason,
                    "error_message": r.error_message,
                    "priority": r.priority,
                    "source": r.source,
                    "retry_count": r.retry_count,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                    "failed_at": r.failed_at.isoformat() if r.failed_at else "",
                    "moved_at": r.moved_at.isoformat() if r.moved_at else "",
                }
                for r in rows
            ]
        finally:
            session.close()

    def requeue_from_dead_letter(self, job_id: str) -> bool:
        """Move a dead-letter job back to the active queue.

        Args:
            job_id: The dead-letter entry ID.

        Returns:
            True if successfully re-queued.
        """
        session = _new_session()
        try:
            from database.db import Job
            dlq = session.get(DeadLetterJob, job_id)
            if dlq is None:
                logger.warning("DLQ entry %s not found", job_id)
                return False

            new_id = str(uuid.uuid4())
            new_job = Job(
                id=new_id,
                url=dlq.url,
                status="pending",
                created_at=datetime.now(timezone.utc),
                settings_json=dlq.settings_json,
                retry_count=0,
            )
            for attr, val in [
                ("priority", dlq.priority),
                ("source", dlq.source),
            ]:
                try:
                    setattr(new_job, attr, val)
                except AttributeError:
                    pass

            session.add(new_job)
            session.delete(dlq)
            session.commit()
            logger.info("Re-queued DLQ job %s as new job %s", job_id, new_id)
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Failed to re-queue DLQ job %s: %s", job_id, exc)
            return False
        finally:
            session.close()

    # ── Counts ───────────────────────────────────────────

    def pending_count(self) -> int:
        """Return total number of pending jobs."""
        session = _new_session()
        try:
            from database.db import Job
            return session.scalar(
                select(func.count(Job.id)).where(Job.status == "pending")
            ) or 0
        finally:
            session.close()

    def pending_count_by_priority(self) -> dict[str, int]:
        """Return pending job counts grouped by priority."""
        session = _new_session()
        try:
            from database.db import Job
            result = {}
            for p in VALID_PRIORITIES:
                if hasattr(Job, "priority"):
                    count = session.scalar(
                        select(func.count(Job.id)).where(
                            and_(Job.status == "pending", Job.priority == p)
                        )
                    ) or 0
                else:
                    count = self.pending_count() if p == "medium" else 0
                result[p] = count
            return result
        finally:
            session.close()

    # ── Statistics ───────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return comprehensive queue statistics."""
        session = _new_session()
        try:
            from database.db import Job
            now = datetime.now(timezone.utc)

            total = session.scalar(select(func.count(Job.id))) or 0
            done = session.scalar(
                select(func.count(Job.id)).where(Job.status == "done")
            ) or 0
            failed = session.scalar(
                select(func.count(Job.id)).where(Job.status == "failed")
            ) or 0
            running = session.scalar(
                select(func.count(Job.id)).where(Job.status == "running")
            ) or 0
            pending = session.scalar(
                select(func.count(Job.id)).where(Job.status == "pending")
            ) or 0

            avg_duration = session.scalar(
                select(func.avg(Job.duration_seconds)).where(Job.status == "done")
            ) or 0.0

            # Per-priority pending counts
            priority_counts = self.pending_count_by_priority()

            # Average wait time (started_at - created_at for running/done)
            avg_wait = session.scalar(
                select(
                    func.avg(
                        __import__("sqlalchemy").func.strftime(
                            "%s", Job.started_at
                        ) - __import__("sqlalchemy").func.strftime(
                            "%s", Job.created_at
                        )
                    )
                ).where(
                    and_(
                        Job.status.in_(["done", "running"]),
                        Job.started_at.isnot(None),
                    )
                )
            ) or 0.0

            # Dead letter count
            dlq_count = session.scalar(select(func.count(DeadLetterJob.id))) or 0

            # Jobs completed in last hour
            hour_ago = now - timedelta(hours=1)
            done_last_hour = session.scalar(
                select(func.count(Job.id)).where(
                    and_(Job.status == "done", Job.finished_at >= hour_ago)
                )
            ) or 0

            # Jobs completed in last 24 hours
            day_ago = now - timedelta(hours=24)
            done_last_day = session.scalar(
                select(func.count(Job.id)).where(
                    and_(Job.status == "done", Job.finished_at >= day_ago)
                )
            ) or 0

            return {
                "total_jobs": total,
                "pending_jobs": pending,
                "running_jobs": running,
                "done_jobs": done,
                "failed_jobs": failed,
                "dead_letter_jobs": dlq_count,
                "success_rate": round(done / total * 100, 1) if total > 0 else 0.0,
                "avg_duration_seconds": round(float(avg_duration), 1),
                "avg_wait_seconds": round(float(avg_wait), 1),
                "pending_by_priority": priority_counts,
                "done_last_hour": done_last_hour,
                "done_last_24h": done_last_day,
                "throughput_per_hour": done_last_hour,
            }
        finally:
            session.close()

    def record_stats_snapshot(self) -> None:
        """Persist a time-series snapshot of queue statistics."""
        s = self.stats()
        session = _new_session()
        try:
            snap = QueueStatsSnapshot(
                pending_high=s["pending_by_priority"].get("high", 0),
                pending_medium=s["pending_by_priority"].get("medium", 0),
                pending_low=s["pending_by_priority"].get("low", 0),
                running_count=s["running_jobs"],
                done_count=s["done_jobs"],
                failed_count=s["failed_jobs"],
                dead_letter_count=s["dead_letter_jobs"],
                avg_wait_seconds=s["avg_wait_seconds"],
                avg_process_seconds=s["avg_duration_seconds"],
            )
            session.add(snap)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Failed to record stats snapshot: %s", exc)
        finally:
            session.close()

    def get_stats_history(self, hours: int = 24) -> list[dict]:
        """Return stats snapshots from the last N hours."""
        session = _new_session()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            stmt = (
                select(QueueStatsSnapshot)
                .where(QueueStatsSnapshot.timestamp >= cutoff)
                .order_by(QueueStatsSnapshot.timestamp)
            )
            rows = list(session.scalars(stmt).all())
            return [
                {
                    "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                    "pending_high": r.pending_high,
                    "pending_medium": r.pending_medium,
                    "pending_low": r.pending_low,
                    "running": r.running_count,
                    "done": r.done_count,
                    "failed": r.failed_count,
                    "dead_letter": r.dead_letter_count,
                    "avg_wait": r.avg_wait_seconds,
                    "avg_process": r.avg_process_seconds,
                }
                for r in rows
            ]
        finally:
            session.close()

    # ── Health Check ─────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Return queue health status with diagnostic info."""
        now = datetime.now(timezone.utc)
        session = _new_session()
        try:
            from database.db import Job

            # Stale running jobs (no heartbeat for 30 min)
            stale_threshold = now - timedelta(minutes=30)
            stale_running = 0
            if hasattr(Job, "heartbeat_at"):
                stale_running = session.scalar(
                    select(func.count(Job.id)).where(
                        and_(
                            Job.status == "running",
                            or_(
                                Job.heartbeat_at < stale_threshold,
                                Job.heartbeat_at.is_(None),
                            ),
                        )
                    )
                ) or 0
            else:
                # Fallback: running for > 2 hours
                two_hours_ago = now - timedelta(hours=2)
                stale_running = session.scalar(
                    select(func.count(Job.id)).where(
                        and_(
                            Job.status == "running",
                            Job.started_at < two_hours_ago,
                        )
                    )
                ) or 0

            # Old pending jobs (waiting > 24h)
            day_ago = now - timedelta(hours=24)
            old_pending = session.scalar(
                select(func.count(Job.id)).where(
                    and_(Job.status == "pending", Job.created_at < day_ago)
                )
            ) or 0

            # DLQ size
            dlq_count = session.scalar(select(func.count(DeadLetterJob.id))) or 0

            # Recent failure rate (last hour)
            hour_ago = now - timedelta(hours=1)
            recent_done = session.scalar(
                select(func.count(Job.id)).where(
                    and_(Job.status == "done", Job.finished_at >= hour_ago)
                )
            ) or 0
            recent_failed = session.scalar(
                select(func.count(Job.id)).where(
                    and_(Job.status == "failed", Job.finished_at >= hour_ago)
                )
            ) or 0
            recent_total = recent_done + recent_failed
            failure_rate = (recent_failed / recent_total * 100) if recent_total > 0 else 0.0

            # Determine health status
            status = "healthy"
            warnings: list[str] = []
            if stale_running > 0:
                status = "degraded"
                warnings.append(f"{stale_running} stale running job(s)")
            if old_pending > 10:
                status = "degraded"
                warnings.append(f"{old_pending} pending job(s) older than 24h")
            if dlq_count > 20:
                status = "unhealthy"
                warnings.append(f"{dlq_count} dead-letter job(s)")
            if failure_rate > 50:
                status = "unhealthy"
                warnings.append(f"High failure rate: {failure_rate:.0f}% in last hour")

            return {
                "status": status,
                "stale_running_jobs": stale_running,
                "old_pending_jobs": old_pending,
                "dead_letter_count": dlq_count,
                "recent_failure_rate_pct": round(failure_rate, 1),
                "warnings": warnings,
                "checked_at": now.isoformat(),
            }
        finally:
            session.close()

    # ── Cleanup / Expiration ─────────────────────────────

    def cleanup_expired(self, max_age_hours: int = 72) -> int:
        """Remove expired pending jobs older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours for pending jobs.

        Returns:
            Number of expired jobs removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        session = _new_session()
        try:
            from database.db import Job
            stmt = select(Job).where(
                and_(
                    Job.status == "pending",
                    Job.created_at < cutoff,
                )
            )
            expired = list(session.scalars(stmt).all())
            count = len(expired)
            for job in expired:
                session.delete(job)
            session.commit()
            if count > 0:
                logger.info("Cleaned up %d expired pending jobs (older than %dh)", count, max_age_hours)
            return count
        except Exception as exc:
            session.rollback()
            logger.error("Failed to cleanup expired jobs: %s", exc)
            return 0
        finally:
            session.close()

    # ── Queue Position / Wait Time ───────────────────────

    def get_job_position(self, job_id: str) -> int:
        """Return the 0-based position of a pending job in the queue.

        Positions are ordered by priority then created_at.

        Returns:
            Position (0 = next to be dequeued), or -1 if not pending.
        """
        session = _new_session()
        try:
            from database.db import Job
            job = session.get(Job, job_id)
            if job is None or job.status != "pending":
                return -1

            job_priority = getattr(job, "priority", "medium") or "medium"
            job_created = job.created_at

            # Count jobs ahead in the queue
            ahead = 0
            for p_name, p_val in PRIORITY_ORDER.items():
                if p_val < PRIORITY_ORDER.get(job_priority, 1):
                    # Higher priority tier — all count
                    if hasattr(Job, "priority"):
                        ahead += session.scalar(
                            select(func.count(Job.id)).where(
                                and_(Job.status == "pending", Job.priority == p_name)
                            )
                        ) or 0
                elif p_val == PRIORITY_ORDER.get(job_priority, 1):
                    # Same tier — only earlier jobs count
                    if hasattr(Job, "priority"):
                        ahead += session.scalar(
                            select(func.count(Job.id)).where(
                                and_(
                                    Job.status == "pending",
                                    Job.priority == p_name,
                                    Job.created_at < job_created,
                                )
                            )
                        ) or 0
            return ahead
        finally:
            session.close()

    def estimate_wait_time(self, job_id: str) -> float:
        """Estimate wait time in seconds for a pending job.

        Uses average processing duration and queue position.

        Returns:
            Estimated wait in seconds, or 0 if job is not pending.
        """
        session = _new_session()
        try:
            from database.db import Job
            job = session.get(Job, job_id)
            if job is None or job.status != "pending":
                return 0.0

            position = self.get_job_position(job_id)
            avg_duration = session.scalar(
                select(func.avg(Job.duration_seconds)).where(Job.status == "done")
            ) or 120.0  # default 2 min estimate

            # Assume max concurrency across all priorities
            total_concurrency = sum(self._max_concurrent.values())
            if total_concurrency < 1:
                total_concurrency = 1

            estimated = (position / total_concurrency) * float(avg_duration)
            return round(estimated, 1)
        finally:
            session.close()

    # ── Batch Enqueue ────────────────────────────────────

    def batch_enqueue(
        self,
        urls: list[str],
        priority: str = "medium",
        source: str = "",
        settings_override: dict[str, Any] | None = None,
    ) -> list[str]:
        """Add multiple URLs to the queue under one batch ID.

        Args:
            urls: List of YouTube video URLs.
            priority: Priority for all jobs in the batch.
            source: Source identifier for rate limiting.
            settings_override: Settings overrides applied to all jobs.

        Returns:
            List of created job IDs.
        """
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'")

        batch_id = str(uuid.uuid4())
        job_ids: list[str] = []

        for url in urls:
            url = url.strip()
            if not url or url.startswith("#"):
                continue
            try:
                jid = self.enqueue(
                    url=url,
                    priority=priority,
                    source=source,
                    batch_id=batch_id,
                    settings_override=settings_override,
                )
                job_ids.append(jid)
            except Exception as exc:
                logger.error("Failed to enqueue %s: %s", url[:60], exc)

        logger.info(
            "Batch enqueued %d/%d jobs (batch=%s, priority=%s)",
            len(job_ids), len(urls), batch_id[:8], priority,
        )
        return job_ids
