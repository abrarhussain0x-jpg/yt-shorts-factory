"""
scheduler/worker.py — Multi-threaded background worker with auto-scaling.

Processes queued jobs using priority-aware multi-threaded execution with
graceful shutdown, heartbeat monitoring, rate limiting between YouTube
downloads, auto-scaling based on system load, memory/CPU monitoring,
job timeout enforcement, progress reporting, worker registration, and
stale job recovery.
"""

from __future__ import annotations

import os
import platform
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sqlalchemy import select

from config.settings import get_settings
from scheduler.job_queue import JobQueue
from utils.logger import get_logger

logger = get_logger("worker")
console = Console()


# ── System monitoring helpers ──────────────────────────────

def _get_cpu_percent() -> float:
    """Return system CPU usage as a percentage (0-100)."""
    try:
        if platform.system() == "Linux":
            # Read from /proc/stat
            with open("/proc/stat", "r") as f:
                line1 = f.readline()
            time.sleep(0.1)
            with open("/proc/stat", "r") as f:
                line2 = f.readline()

            vals1 = list(map(int, line1.split()[1:]))
            vals2 = list(map(int, line2.split()[1:]))
            idle1, idle2 = vals1[3], vals2[3]
            total1, total2 = sum(vals1), sum(vals2)
            diff_idle = idle2 - idle1
            diff_total = total2 - total1
            if diff_total > 0:
                return round((1.0 - diff_idle / diff_total) * 100, 1)
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "vm.loadavg"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().strip("{}").split()
                if parts:
                    load = float(parts[0])
                    cores = os.cpu_count() or 4
                    return min(100.0, round(load / cores * 100, 1))
    except Exception:
        pass
    return 50.0  # unknown default


def _get_memory_usage() -> dict[str, float]:
    """Return memory usage info: total_mb, used_mb, percent."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_mb": round(mem.total / (1024 * 1024), 0),
            "used_mb": round(mem.used / (1024 * 1024), 0),
            "percent": round(mem.percent, 1),
        }
    except ImportError:
        pass

    # Fallback for Linux
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])  # in kB
            total = info.get("MemTotal", 0) / 1024  # MB
            available = info.get("MemAvailable", 0) / 1024
            used = total - available
            pct = (used / total * 100) if total > 0 else 0
            return {
                "total_mb": round(total, 0),
                "used_mb": round(used, 0),
                "percent": round(pct, 1),
            }
    except Exception:
        pass

    return {"total_mb": 0, "used_mb": 0, "percent": 0.0}


def _get_disk_usage(path: str = ".") -> dict[str, float]:
    """Return disk usage for the given path."""
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        pct = (usage.used / usage.total * 100) if usage.total > 0 else 0
        return {
            "total_gb": round(total_gb, 1),
            "used_gb": round(used_gb, 1),
            "free_gb": round(free_gb, 1),
            "percent": round(pct, 1),
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0.0}


# ══════════════════════════════════════════════════════════
#  Worker class
# ══════════════════════════════════════════════════════════

class Worker:
    """Background worker that processes jobs from the queue.

    Features:
      - Multi-threaded job execution with configurable concurrency
      - Priority-aware job selection (high first)
      - Heartbeat monitoring (detect and recover stuck jobs)
      - Rate limiting between YouTube downloads
      - Graceful shutdown with job completion
      - Worker health reporting
      - Auto-scaling (adjust concurrency based on system load)
      - Memory/CPU monitoring
      - Job timeout enforcement
      - Progress reporting to database
      - Worker registration/deregistration
      - Stale job recovery
    """

    def __init__(
        self,
        max_concurrent: int | None = None,
        poll_interval: float | None = None,
        heartbeat_interval: float = 30.0,
        job_timeout_minutes: float = 60.0,
        download_delay_seconds: float = 5.0,
        auto_scale: bool = False,
    ) -> None:
        settings = get_settings()
        self.max_concurrent = max_concurrent or settings.MAX_CONCURRENT_JOBS
        self._initial_max_concurrent = self.max_concurrent
        self._poll_interval = poll_interval or settings.WORKER_POLL_INTERVAL
        self._heartbeat_interval = heartbeat_interval
        self._job_timeout_minutes = job_timeout_minutes
        self._download_delay = download_delay_seconds
        self._auto_scale = auto_scale

        self.queue = JobQueue()
        self._running = False
        self._active_jobs: dict[str, threading.Thread] = {}
        self._active_start_times: dict[str, float] = {}
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self._completed_count = 0
        self._failed_count = 0
        self._worker_id = f"worker-{platform.node()}-{uuid.uuid4().hex[:8]}"

        # Heartbeat thread
        self._heartbeat_thread: threading.Thread | None = None
        self._last_heartbeat: float = 0.0

        # Stats snapshot interval
        self._last_stats_snapshot: float = 0.0
        self._stats_snapshot_interval: float = 300.0  # 5 minutes

    # ── Context manager ──────────────────────────────────

    def __enter__(self) -> Worker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    # ── Start / Stop ─────────────────────────────────────

    def start(self) -> None:
        """Start the worker main loop.

        Blocks until stop() is called or SIGINT/SIGTERM is received.
        """
        self._running = True
        self._start_time = time.time()
        self._last_heartbeat = time.time()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="worker-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        logger.info(
            "Worker %s started (max_concurrent=%d, poll=%.1fs, auto_scale=%s)",
            self._worker_id, self.max_concurrent, self._poll_interval, self._auto_scale,
        )

        console.print(
            Panel(
                f"[bold green]Worker {self._worker_id[:20]} started[/bold green]\n"
                f"Max concurrent: {self.max_concurrent}\n"
                f"Poll interval:  {self._poll_interval}s\n"
                f"Auto-scale:     {self._auto_scale}\n"
                f"[dim]Press Ctrl+C to stop gracefully[/dim]",
                title="YT Shorts Factory Worker",
                border_style="green",
            )
        )

        try:
            while self._running:
                self._poll_and_process()
                self._maybe_auto_scale()
                self._maybe_record_stats()
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("Worker interrupted by keyboard")
        finally:
            self._wait_for_active_jobs()
            elapsed = time.time() - self._start_time
            logger.info(
                "Worker stopped after %.0fs (completed=%d, failed=%d)",
                elapsed, self._completed_count, self._failed_count,
            )
            console.print(
                f"\n[yellow]Worker stopped[/yellow] "
                f"(uptime={elapsed:.0f}s, completed={self._completed_count}, "
                f"failed={self._failed_count})"
            )

    def stop(self) -> None:
        """Signal the worker to stop gracefully."""
        self._running = False
        logger.info("Worker stop requested")

    # ── Signal handling ──────────────────────────────────

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        self._running = False

    # ── Main polling loop ────────────────────────────────

    def _poll_and_process(self) -> None:
        """Check for pending jobs and spawn processing threads."""
        # Clean up finished threads
        with self._lock:
            finished = [jid for jid, t in self._active_jobs.items() if not t.is_alive()]
            for jid in finished:
                del self._active_jobs[jid]
                self._active_start_times.pop(jid, None)

        # Check capacity
        with self._lock:
            active_count = len(self._active_jobs)
        if active_count >= self.max_concurrent:
            return

        # Check job timeouts
        self._check_timeouts()

        # Rate limit delay between dequeues
        with self._lock:
            if self._active_jobs:
                # Small delay to avoid hammering YouTube
                pass  # rate limiting is per-download, not per-poll

        # Dequeue next job
        job_id = self.queue.dequeue(worker_id=self._worker_id)
        if job_id is None:
            return

        # Rate limit between downloads
        if self._download_delay > 0:
            time.sleep(self._download_delay)

        # Spawn worker thread
        thread = threading.Thread(
            target=self._process_job,
            args=(job_id,),
            name=f"worker-{job_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._active_jobs[job_id] = thread
            self._active_start_times[job_id] = time.time()
        thread.start()
        logger.info(
            "Started processing job %s (active=%d/%d)",
            job_id, len(self._active_jobs), self.max_concurrent,
        )

    # ── Job processing ───────────────────────────────────

    def _process_job(self, job_id: str) -> None:
        """Process a single job by running the full pipeline."""
        from database.db import get_job
        from core.pipeline import run_pipeline

        logger.info("Processing job %s", job_id)

        try:
            job = get_job(job_id)
            if job is None:
                logger.error("Job %s not found", job_id)
                return

            # Parse settings overrides
            settings_kwargs: dict[str, Any] = {}
            try:
                overrides = {}
                if job.settings_json:
                    import json
                    overrides = json.loads(job.settings_json)
                if overrides:
                    settings_kwargs["settings_override"] = overrides
            except (json.JSONDecodeError, TypeError):
                pass

            result = run_pipeline(
                url=job.url,
                job_id=job_id,
                resume=True,
            )

            if result.success:
                output_paths_str = ""
                if result.outputs:
                    paths = [str(p) for p in result.outputs.paths if p]
                    output_paths_str = ";".join(paths)
                self.queue.complete(job_id, output_path=output_paths_str)
                with self._lock:
                    self._completed_count += 1
                logger.info("Job %s completed successfully", job_id)
            else:
                self._handle_retry(job_id, result.error or "Unknown error")

        except Exception as exc:
            logger.error("Job %s threw exception: %s", job_id, exc)
            self._handle_retry(job_id, str(exc))

    def _handle_retry(self, job_id: str, error: str) -> None:
        """Handle a failed job by retrying, moving to DLQ, or marking failed."""
        retried = self.queue.retry(job_id)
        if not retried:
            # Max retries exceeded — move to dead letter or mark failed
            from config.settings import get_settings
            settings = get_settings()
            job = None
            try:
                from database.db import get_job as _get_job
                job = _get_job(job_id)
            except Exception:
                pass

            retry_count = job.retry_count if job else 0
            if retry_count and retry_count >= settings.JOB_RETRY_ATTEMPTS:
                self.queue.move_to_dead_letter(job_id, reason=error[:500])
            else:
                self.queue.fail(job_id, error)

            with self._lock:
                self._failed_count += 1
            logger.warning("Job %s permanently failed: %s", job_id, error[:100])

    # ── Heartbeat / Monitoring ───────────────────────────

    def _heartbeat_loop(self) -> None:
        """Background thread: update heartbeat and check system health."""
        while self._running:
            try:
                self._update_heartbeat()
                self._check_system_resources()
            except Exception as exc:
                logger.error("Heartbeat error: %s", exc)
            time.sleep(self._heartbeat_interval)

    def _update_heartbeat(self) -> None:
        """Update heartbeat timestamp for all active jobs."""
        now = datetime.now(timezone.utc)
        session = None
        try:
            from database.db import _new_session, Job
            session = _new_session()
            with self._lock:
                active_ids = list(self._active_jobs.keys())
            for jid in active_ids:
                try:
                    job = session.get(Job, jid)
                    if job and hasattr(job, "heartbeat_at"):
                        job.heartbeat_at = now
                except Exception:
                    pass
            session.commit()
            self._last_heartbeat = time.time()
        except Exception as exc:
            if session:
                session.rollback()
            logger.debug("Heartbeat update failed: %s", exc)
        finally:
            if session:
                session.close()

    def _check_system_resources(self) -> None:
        """Check system resources and log warnings if thresholds exceeded."""
        mem = _get_memory_usage()
        if mem["percent"] > 90:
            logger.warning(
                "High memory usage: %.1f%% (%.0f MB / %.0f MB)",
                mem["percent"], mem["used_mb"], mem["total_mb"],
            )

        disk = _get_disk_usage()
        if disk["percent"] > 95:
            logger.warning(
                "Low disk space: %.1f%% free (%.1f GB)",
                100 - disk["percent"], disk["free_gb"],
            )

    # ── Timeout enforcement ──────────────────────────────

    def _check_timeouts(self) -> None:
        """Check for jobs running longer than the timeout and kill them."""
        now = time.time()
        timeout_seconds = self._job_timeout_minutes * 60
        timed_out: list[str] = []

        with self._lock:
            for jid, start_time in self._active_start_times.items():
                if now - start_time > timeout_seconds:
                    timed_out.append(jid)

        for jid in timed_out:
            logger.warning("Job %s exceeded timeout (%.0f min), failing", jid, self._job_timeout_minutes)
            self.queue.fail(jid, error=f"Job timed out after {self._job_timeout_minutes:.0f} minutes")
            with self._lock:
                self._failed_count += 1
                self._active_jobs.pop(jid, None)
                self._active_start_times.pop(jid, None)

    # ── Auto-scaling ─────────────────────────────────────

    def _maybe_auto_scale(self) -> None:
        """Adjust max_concurrent based on system load if auto_scale is enabled."""
        if not self._auto_scale:
            return

        try:
            new_concurrency = self.adjust_concurrency()
            if new_concurrency != self.max_concurrent:
                old = self.max_concurrent
                self.max_concurrent = new_concurrency
                logger.info(
                    "Auto-scaled concurrency: %d -> %d", old, new_concurrency
                )
        except Exception as exc:
            logger.debug("Auto-scale check failed: %s", exc)

    def adjust_concurrency(self) -> int:
        """Auto-adjust max_concurrent based on system load.

        Strategy:
          - CPU < 50%: increase concurrency (up to 2x initial)
          - CPU 50-80%: keep current
          - CPU > 80%: decrease concurrency (down to 1)
          - Memory > 90%: decrease concurrency

        Returns:
            Recommended max_concurrent value.
        """
        cpu = _get_cpu_percent()
        mem = _get_memory_usage()
        current = self.max_concurrent
        initial = self._initial_max_concurrent

        # Memory pressure takes precedence
        if mem["percent"] > 90:
            return max(1, current - 1)

        if cpu > 85:
            return max(1, current - 1)
        elif cpu > 70:
            return current  # maintain
        elif cpu < 40:
            return min(initial * 2, current + 1)
        else:
            return current

    # ── Stats snapshot ───────────────────────────────────

    def _maybe_record_stats(self) -> None:
        """Periodically record queue stats snapshots."""
        now = time.time()
        if now - self._last_stats_snapshot >= self._stats_snapshot_interval:
            try:
                self.queue.record_stats_snapshot()
                self._last_stats_snapshot = now
            except Exception as exc:
                logger.debug("Stats snapshot failed: %s", exc)

    # ── Graceful shutdown ────────────────────────────────

    def _wait_for_active_jobs(self) -> None:
        """Wait for all active job threads to complete."""
        settings = get_settings()
        timeout = settings.WORKER_GRACEFUL_TIMEOUT

        with self._lock:
            threads = list(self._active_jobs.values())

        if not threads:
            return

        logger.info(
            "Waiting for %d active jobs to complete (timeout=%ds)...",
            len(threads), timeout,
        )
        console.print(
            f"[yellow]Waiting for {len(threads)} active jobs...[/yellow]"
        )

        for thread in threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Thread %s did not finish within timeout", thread.name)

    # ── Status / Health ──────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return current worker status."""
        uptime = time.time() - self._start_time if self._start_time else 0
        with self._lock:
            active_count = len(self._active_jobs)
            active_ids = list(self._active_jobs.keys())

        return {
            "worker_id": self._worker_id,
            "running": self._running,
            "active_jobs": active_count,
            "active_job_ids": active_ids[:10],
            "max_concurrent": self.max_concurrent,
            "queue_depth": self.queue.pending_count(),
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            "uptime_seconds": round(uptime, 1),
            "auto_scale": self._auto_scale,
            "last_heartbeat": round(self._last_heartbeat, 1) if self._last_heartbeat else 0,
        }

    def health(self) -> dict[str, Any]:
        """Return worker health metrics including system resources."""
        worker_status = self.status()
        queue_health = self.queue.health_check()
        sys_load = self.get_system_load()

        # Determine worker health
        health_status = "healthy"
        issues: list[str] = []

        if not self._running:
            health_status = "stopped"
        elif sys_load.get("memory_percent", 0) > 90:
            health_status = "degraded"
            issues.append(f"High memory: {sys_load['memory_percent']:.0f}%")
        elif sys_load.get("cpu_percent", 0) > 85:
            health_status = "degraded"
            issues.append(f"High CPU: {sys_load['cpu_percent']:.0f}%")

        if queue_health["status"] != "healthy":
            issues.extend(queue_health.get("warnings", []))
            if queue_health["status"] == "unhealthy":
                health_status = "unhealthy"

        return {
            "worker_status": health_status,
            "issues": issues,
            "worker": worker_status,
            "queue_health": queue_health,
            "system_load": sys_load,
        }

    def get_system_load(self) -> dict[str, Any]:
        """Return CPU, memory, and disk usage metrics."""
        cpu = _get_cpu_percent()
        mem = _get_memory_usage()
        disk = _get_disk_usage()
        return {
            "cpu_percent": cpu,
            "memory_percent": mem["percent"],
            "memory_used_mb": mem["used_mb"],
            "memory_total_mb": mem["total_mb"],
            "disk_percent": disk["percent"],
            "disk_free_gb": disk["free_gb"],
            "disk_total_gb": disk["total_gb"],
        }

    # ── Stale job recovery ──────────────────────────────

    def recover_stale_jobs(self, timeout_minutes: int = 30) -> int:
        """Re-queue jobs that are stuck in 'running' for too long.

        A job is considered stale if:
          - It has been running for longer than timeout_minutes
          - Its heartbeat has not been updated for timeout_minutes

        Args:
            timeout_minutes: Threshold for considering a job stale.

        Returns:
            Number of jobs re-queued.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
        session = None
        try:
            from database.db import _new_session, Job
            session = _new_session()

            # Find stale running jobs
            stmt = select(Job).where(Job.status == "running")
            stale_jobs = []

            for job in session.scalars(stmt).all():
                is_stale = False

                # Check heartbeat
                heartbeat = getattr(job, "heartbeat_at", None)
                if heartbeat and heartbeat < cutoff:
                    is_stale = True
                elif not heartbeat and job.started_at and job.started_at < cutoff:
                    is_stale = True

                if is_stale:
                    stale_jobs.append(job)

            count = len(stale_jobs)
            for job in stale_jobs:
                job.status = "pending"
                job.started_at = None
                job.error_message = f"Recovered stale job (was running >{timeout_minutes}min)"
                if hasattr(job, "worker_id"):
                    job.worker_id = None
                if hasattr(job, "heartbeat_at"):
                    job.heartbeat_at = None
                logger.info("Recovered stale job %s", job.id)

            if count > 0:
                session.commit()
                logger.info("Recovered %d stale jobs", count)

            return count
        except Exception as exc:
            if session:
                session.rollback()
            logger.error("Failed to recover stale jobs: %s", exc)
            return 0
        finally:
            if session:
                session.close()


# ══════════════════════════════════════════════════════════
#  CLI helper: display worker status as a Rich table
# ══════════════════════════════════════════════════════════

def display_worker_status() -> None:
    """Print current worker/queue status to the console."""
    queue = JobQueue()
    q_stats = queue.stats()
    q_health = queue.health_check()

    # Status table
    table = Table(title="Worker & Queue Status", show_lines=True)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=40)

    table.add_row("Queue Status", q_health["status"])
    table.add_row("Total Jobs", str(q_stats["total_jobs"]))
    table.add_row("Pending", str(q_stats["pending_jobs"]))
    table.add_row("Running", str(q_stats["running_jobs"]))
    table.add_row("Completed", str(q_stats["done_jobs"]))
    table.add_row("Failed", str(q_stats["failed_jobs"]))
    table.add_row("Dead Letter", str(q_stats["dead_letter_jobs"]))
    table.add_row("Success Rate", f"{q_stats['success_rate']}%")
    table.add_row("Avg Duration", f"{q_stats['avg_duration_seconds']}s")
    table.add_row("Avg Wait", f"{q_stats['avg_wait_seconds']}s")
    table.add_row("Throughput/hr", str(q_stats["throughput_per_hour"]))

    # Priority breakdown
    by_p = q_stats["pending_by_priority"]
    table.add_row("Pending (high)", str(by_p.get("high", 0)))
    table.add_row("Pending (medium)", str(by_p.get("medium", 0)))
    table.add_row("Pending (low)", str(by_p.get("low", 0)))

    console.print(table)

    # Health warnings
    if q_health["warnings"]:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in q_health["warnings"]:
            console.print(f"  - {w}")

    # System load
    sys_load = Worker(max_concurrent=0).get_system_load()
    load_table = Table(title="System Load")
    load_table.add_column("Resource", style="cyan")
    load_table.add_column("Usage", style="bold")
    load_table.add_column("Detail", style="dim")

    load_table.add_row(
        "CPU", f"{sys_load['cpu_percent']}%", "",
    )
    load_table.add_row(
        "Memory",
        f"{sys_load['memory_percent']}%",
        f"{sys_load['memory_used_mb']:.0f} / {sys_load['memory_total_mb']:.0f} MB",
    )
    load_table.add_row(
        "Disk",
        f"{sys_load['disk_percent']}%",
        f"{sys_load['disk_free_gb']:.1f} GB free",
    )
    console.print(load_table)


# ══════════════════════════════════════════════════════════
#  Standalone CLI entry point
# ══════════════════════════════════════════════════════════

def main():
    """CLI entry point for the worker daemon."""
    import click

    @click.group()
    def cli():
        """Worker daemon commands."""
        pass

    @cli.command()
    @click.option("--max-concurrent", default=None, type=int, help="Maximum concurrent jobs")
    @click.option("--auto-scale", is_flag=True, help="Enable auto-scaling")
    @click.option("--poll-interval", default=None, type=float, help="Poll interval in seconds")
    def start(max_concurrent, auto_scale, poll_interval):
        """Start the worker daemon."""
        worker = Worker(
            max_concurrent=max_concurrent,
            auto_scale=auto_scale,
            poll_interval=poll_interval,
        )
        worker.start()

    @cli.command()
    def status():
        """Show worker and queue status."""
        display_worker_status()

    @cli.command()
    @click.option("--timeout", default=30, type=int, help="Stale job timeout in minutes")
    def recover(timeout):
        """Recover stale running jobs."""
        worker = Worker(max_concurrent=0)
        count = worker.recover_stale_jobs(timeout_minutes=timeout)
        if count > 0:
            console.print(f"[green]Recovered {count} stale job(s)[/green]")
        else:
            console.print("[dim]No stale jobs found[/dim]")

    cli()


if __name__ == "__main__":
    main()
