"""
main.py — Click CLI entry point for yt-shorts-factory.

Provides 16 commands: run, queue, worker, history, stats, verify,
config, analyze, transcribe, cleanup, info, interactive, rate, ai-meta,
presets, patterns, about.
Rich console output with panels, tables, trees, and ASCII banner.

Developed by Abrar Hussain
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.tree import Tree
from sqlalchemy import select, func, desc, and_, or_

from utils.logger import get_logger

logger = get_logger("cli")
console = Console()

# ── ASCII Banner ──────────────────────────────────────────
_BANNER = r"""
[yellow]   __   _____  __    ____  __  ____  ____
  / /  / ___/ / /   / __/ / / / __/ / __/
 / /__/ /__  / /__ / /__ / /_/ /__ / /__
/____/\___/ /____//___//___//___//____/[/yellow]

[cyan]  YouTube Shorts Factory v4.0[/cyan]
[dim]  Automated video intelligence pipeline[/dim]
[bold magenta]  Developed by Abrar Hussain[/bold magenta]
"""


def _print_banner() -> None:
    """Print the startup banner with system info."""
    console.print(_BANNER)

    from config.settings import get_settings
    settings = get_settings()
    info = settings.platform_info

    sys_table = Table(show_header=False, box=None, padding=(0, 2))
    sys_table.add_column("key", style="dim")
    sys_table.add_column("value", style="bold")

    sys_table.add_row("Python", info["python"])
    sys_table.add_row("Platform", info["platform"])
    sys_table.add_row("Whisper", f"{info['whisper_model']} on {info['whisper_device']} ({info['whisper_compute']})")

    try:
        from utils.ffmpeg_utils import check_ffmpeg
        check_ffmpeg()
        sys_table.add_row("FFmpeg", "[green]OK[/green]")
    except RuntimeError as exc:
        sys_table.add_row("FFmpeg", f"[red]{exc}[/red]")

    sys_table.add_row("Developer", "[bold magenta]Abrar Hussain[/bold magenta]")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            sys_table.add_row("GPU", f"[green]{gpu_name}[/green] ({gpu_mem:.1f} GB VRAM)")
            sys_table.add_row("CUDA", f"[green]Available[/green] (compute {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor})")
        else:
            sys_table.add_row("GPU", "[yellow]No CUDA GPU detected[/yellow]")
    except ImportError:
        sys_table.add_row("GPU", "[dim]PyTorch not installed[/dim]")

    console.print(Panel(sys_table, title="System Info", border_style="dim"))


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """YT Shorts Factory - Automated video intelligence pipeline.

    Run without arguments to launch the interactive menu with arrow-key navigation.
    Use 'python main.py menu' for the full interactive experience.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand given — launch interactive menu
        from utils.interactive_menu import interactive_main
        interactive_main()


# ══════════════════════════════════════════════════════════
#  1. RUN — Process URL(s) with full pipeline
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--url", "-u", help="YouTube video URL")
@click.option("--batch", "-b", type=click.Path(exists=True), help="File with one URL per line")
@click.option("--duration", "-d", type=str, help="Clip duration: preset name (quick/25s/standard/45s/extended/3min) or seconds")
@click.option("--preset", type=click.Choice(["quick", "standard", "extended", "25s", "45s", "3min"]), help="Duration preset: quick=25s, standard=45s, extended=3min")
@click.option("--clips", type=int, default=1, help="Number of clips to extract (multi-clip mode)")
@click.option("--no-subs", is_flag=True, help="Skip transcription and subtitles")
@click.option("--no-logo", is_flag=True, help="Skip logo stamping")
@click.option("--whisper-model", type=click.Choice(["tiny", "base", "small", "medium", "large"]), help="Whisper model")
@click.option("--platforms", "-p", multiple=True, type=click.Choice(["youtube", "tiktok", "reels"]), help="Target platforms")
@click.option("--output-dir", "-o", type=click.Path(), help="Override output directory")
@click.option("--animation", "-a", type=click.Choice(["karaoke", "fade", "pop", "glow", "typewriter", "bounce", "wave", "rainbow", "neon", "matrix", "3d_rotate", "none"]), help="Subtitle animation")
@click.option("--enhance-audio", is_flag=True, help="Enable audio enhancement (noise reduction, compression, normalization)")
@click.option("--moderate", is_flag=True, help="Enable content moderation check")
@click.option("--blur-bg", is_flag=True, help="Blur background instead of crop")
@click.option("--format", "aspect_ratio", type=click.Choice(["9:16", "1:1", "4:5"]), default="9:16", help="Output aspect ratio")
@click.option("--quality", type=click.Choice(["fast", "balanced", "high"]), default="balanced", help="Encoding quality preset")
@click.option("--turbo", is_flag=True, help="TURBO MODE: maximum speed, reduced quality (ultrafast encoding, tiny whisper, skip extras)")
@click.option("--superfast", is_flag=True, help="SUPERFAST MODE: single-pass FFmpeg, center crop, minimal analysis (2-4x faster than turbo)")
@click.option("--pattern", type=click.Choice(["viral_hype", "chill_vibes", "news_alert", "educational", "gaming_clips", "motivational", "comedy_clip", "lifestyle", "tech_review", "my_channel", "custom"]), help="Channel branding pattern for consistent styling")
@click.option("--channel-name", type=str, help="Channel name for branding overlays (lower third, outro)")
@click.option("--variants", is_flag=True, help="Export A/B variants for testing")
@click.option("--no-cleanup", is_flag=True, help="Keep intermediate files")
@click.option("--no-resume", is_flag=True, help="Disable checkpoint resume")
@click.option("--dry-run", is_flag=True, help="Validate inputs without processing")
def run(url, batch, duration, preset, clips, no_subs, no_logo, whisper_model, platforms,
        output_dir, animation, enhance_audio, moderate, blur_bg, aspect_ratio,
        quality, turbo, superfast, pattern, channel_name, variants, no_cleanup, no_resume, dry_run):
    """Process a YouTube video into shorts."""
    _print_banner()

    from config.settings import get_settings
    from core.pipeline import run_pipeline

    # Collect URLs
    urls: list[str] = []
    if url:
        urls.append(url)
    if batch:
        batch_path = Path(batch)
        for line in batch_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        console.print("[red]Error: Provide --url or --batch[/red]")
        sys.exit(1)

    # Apply overrides
    settings = get_settings()

    # ── TURBO MODE: maximum speed ─────────────────────────
    if superfast:
        settings.SUPERFAST_MODE = True
        settings.apply_superfast()
        quality = "fast"
        console.print("[bold magenta]SUPERFAST MODE ACTIVATED[/bold magenta] — single-pass pipeline!")
        console.print("[dim]  1 FFmpeg pass | center crop | tiny whisper | minimal analysis | 4-8x faster[/dim]")
    elif turbo:
        settings.TURBO_MODE = True
        settings.apply_turbo()
        quality = "fast"  # Force fast quality
        console.print("[bold yellow]TURBO MODE ACTIVATED[/bold yellow] — maximum speed!")
        console.print("[dim]  ultrafast encoding | tiny whisper | no extras | 4x faster[/dim]")

    # ── Speed Optimizer: auto-detect hardware ────────────
    try:
        from core.parallel_pipeline import SpeedOptimizer
        optimizer = SpeedOptimizer(settings)
        speed_opts = optimizer.apply_speed_settings(turbo=turbo)
        if speed_opts:
            optimizer.print_optimization_report(speed_opts)
    except Exception as exc:
        logger.debug("Speed optimizer not available: %s", exc)

    # ── Channel Pattern ──────────────────────────────────
    if pattern:
        settings.CHANNEL_PATTERN = pattern
        console.print(f"[cyan]Channel Pattern:[/cyan] [bold]{pattern}[/bold]")
    if channel_name:
        settings.CHANNEL_NAME = channel_name

    # Resolve duration from --preset or --duration
    resolved_duration: int | None = None
    if preset:
        # --preset takes priority
        resolved_duration = settings.resolve_duration(preset)
        settings.CLIP_DURATION_PRESET = preset
        settings.CLIP_DURATION = resolved_duration
    elif duration:
        # --duration can be a preset name or raw seconds
        try:
            resolved_duration = int(duration)
        except ValueError:
            # Not a number — try as preset name
            resolved_duration = settings.resolve_duration(duration)
        settings.CLIP_DURATION = resolved_duration

    if whisper_model:
        settings.WHISPER_MODEL = whisper_model
    if platforms:
        settings.EXPORT_YOUTUBE = "youtube" in platforms
        settings.EXPORT_TIKTOK = "tiktok" in platforms
        settings.EXPORT_REELS = "reels" in platforms
    if animation:
        settings.SUBTITLE_ANIMATION = animation
    if no_cleanup:
        settings.CLEANUP_INTERMEDIATES = False
    if enhance_audio:
        settings.AUDIO_NOISE_REDUCTION = True
        settings.AUDIO_COMPRESSION = True
        settings.AUDIO_NORMALIZER = True
    if moderate:
        settings.CONTENT_MODERATION_ENABLED = True
    if output_dir:
        output = Path(output_dir)
        settings.SHORTS_DIR = output / "shorts"
        settings.YOUTUBE_DIR = output / "shorts" / "youtube"
        settings.TIKTOK_DIR = output / "shorts" / "tiktok"
        settings.REELS_DIR = output / "shorts" / "reels"

    # Quality preset
    quality_map = {
        "fast": {"FFMPEG_PRESET": "ultrafast", "FFMPEG_CRF": 28},
        "balanced": {"FFMPEG_PRESET": "fast", "FFMPEG_CRF": 23},
        "high": {"FFMPEG_PRESET": "slow", "FFMPEG_CRF": 18},
    }
    if quality in quality_map:
        for k, v in quality_map[quality].items():
            setattr(settings, k, v)

    # Aspect ratio
    ratio_map = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}
    if aspect_ratio in ratio_map:
        settings.OUTPUT_WIDTH, settings.OUTPUT_HEIGHT = ratio_map[aspect_ratio]

    # Platform duration compatibility check
    if resolved_duration:
        from core.platform_exporter import PLATFORM_SPECS
        enabled_platforms = list(platforms) if platforms else []
        if not enabled_platforms:
            if settings.EXPORT_YOUTUBE:
                enabled_platforms.append("youtube")
            if settings.EXPORT_TIKTOK:
                enabled_platforms.append("tiktok")
            if settings.EXPORT_REELS:
                enabled_platforms.append("instagram")

        for p_key in enabled_platforms:
            spec = PLATFORM_SPECS.get(p_key)
            if spec and resolved_duration > spec.max_duration:
                console.print(
                    f"[yellow]Warning: Duration {resolved_duration}s exceeds {spec.name} "
                    f"max ({spec.max_duration:.0f}s). Clip will be trimmed for this platform.[/yellow]"
                )

    # Dry run
    if dry_run:
        console.print("[yellow]DRY RUN — validating inputs:[/yellow]")
        for u in urls:
            console.print(f"  URL: {u}")
        console.print(f"  Duration: {settings.CLIP_DURATION}s ({settings.current_preset_label})")
        console.print(f"  Clips: {clips}")
        console.print(f"  Aspect: {aspect_ratio}")
        console.print(f"  Quality: {quality}")
        console.print(f"  Turbo: {'YES' if turbo else 'no'}")
        console.print(f"  Superfast: {'YES' if superfast else 'no'}")
        console.print(f"  Whisper: {settings.WHISPER_MODEL}")
        console.print(f"  Platforms: YouTube={settings.EXPORT_YOUTUBE}, TikTok={settings.EXPORT_TIKTOK}, Reels={settings.EXPORT_REELS}")
        console.print(f"  Animation: {settings.SUBTITLE_ANIMATION}")
        console.print(f"  Enhance Audio: {enhance_audio}")
        console.print(f"  Content Moderation: {moderate}")
        console.print(f"  Blur Background: {blur_bg}")
        console.print(f"  A/B Variants: {variants}")
        return

    # Process URLs
    results = []
    for i, u in enumerate(urls, 1):
        console.print(f"\n[bold]Processing {i}/{len(urls)}:[/bold] {u[:80]}")
        try:
            if superfast:
                # SUPERFAST: single-pass pipeline (3-5x faster)
                from core.superfast import superfast_pipeline
                sf_result = superfast_pipeline(
                    url=u,
                    duration=resolved_duration,
                    skip_subs=no_subs,
                    no_logo=no_logo,
                    platforms=list(platforms) if platforms else None,
                    settings=settings,
                    blur_bg=blur_bg,
                )
                # Wrap into a compatible result for the summary table
                from core.pipeline import PipelineResult
                result = PipelineResult(
                    success=sf_result.success,
                    error="" if sf_result.success else "Superfast pipeline failed",
                    total_duration_seconds=sf_result.duration,
                )
                if sf_result.output_path:
                    console.print(f"  [green]Output:[/green] {sf_result.output_path.name} ({sf_result.file_size_human})")
                    console.print(f"  [cyan]FFmpeg passes:[/cyan] {sf_result.ffmpeg_passes}")
                    console.print(f"  [cyan]Total time:[/cyan] {sf_result.duration:.1f}s")
            else:
                result = run_pipeline(
                    url=u,
                    duration=resolved_duration,
                    skip_subs=no_subs,
                    no_logo=no_logo,
                    platforms=list(platforms) if platforms else None,
                    resume=not no_resume,
                    quality=quality,
                )
            results.append((u, result))
        except Exception as exc:
            console.print(f"[red]Pipeline error: {exc}[/red]")
            results.append((u, None))

    # Summary table
    if len(urls) > 1 or results:
        console.print("\n[bold]Batch Summary:[/bold]")
        summary = Table(title="Results")
        summary.add_column("URL", style="cyan", max_width=50)
        summary.add_column("Status", style="bold")
        summary.add_column("Duration", style="dim")
        summary.add_column("Outputs")

        for u, result in results:
            if result is None:
                summary.add_row(u[:50], "[red]ERROR[/red]", "-", "-")
            elif result.success:
                output_count = result.outputs.count if result.outputs else 0
                summary.add_row(u[:50], "[green]OK[/green]", f"{result.total_duration_seconds:.1f}s", str(output_count))
            else:
                summary.add_row(u[:50], "[red]FAILED[/red]", f"{result.total_duration_seconds:.1f}s", result.error[:30])

        console.print(summary)


# ══════════════════════════════════════════════════════════
#  2. QUEUE — Add URLs to queue
# ══════════════════════════════════════════════════════════

@cli.command()
@click.argument("url", required=False)
@click.option("--priority", type=click.Choice(["high", "medium", "low"]), default="medium", help="Job priority")
@click.option("--schedule", type=str, help="Schedule time (ISO format: 2025-01-01T12:00 or +30m for 30 min delay)")
@click.option("--batch", type=click.Path(exists=True), help="File with one URL per line for batch enqueue")
@click.option("--source", "-s", default="", help="Source identifier for rate limiting")
@click.option("--depends-on", type=str, help="Job ID this job depends on")
@click.option("--turbo", is_flag=True, help="Queue job with TURBO mode (maximum speed)")
def queue(url, priority, schedule, batch, source, depends_on, turbo):
    """Add URLs to the job queue."""
    from scheduler.job_queue import JobQueue
    q = JobQueue()

    scheduled_at = None
    if schedule:
        if schedule.startswith("+"):
            # Parse delay like +30m, +2h, +90s
            delay_str = schedule[1:]
            try:
                if delay_str.endswith("m"):
                    scheduled_at = datetime.now(timezone.utc) + timedelta(minutes=int(delay_str[:-1]))
                elif delay_str.endswith("h"):
                    scheduled_at = datetime.now(timezone.utc) + timedelta(hours=int(delay_str[:-1]))
                elif delay_str.endswith("s"):
                    scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=int(delay_str[:-1]))
                else:
                    scheduled_at = datetime.now(timezone.utc) + timedelta(minutes=int(delay_str))
            except ValueError:
                console.print(f"[red]Invalid schedule format: {schedule}[/red]")
                return
        else:
            try:
                scheduled_at = datetime.fromisoformat(schedule).replace(tzinfo=timezone.utc)
            except ValueError:
                console.print(f"[red]Invalid schedule format: {schedule}[/red]")
                return

    if batch:
        # Batch enqueue from file
        batch_path = Path(batch)
        urls = [l.strip() for l in batch_path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        if not urls:
            console.print("[red]No URLs found in batch file[/red]")
            return
        job_ids = q.batch_enqueue(urls, priority=priority, source=source)
        console.print(f"[green]Batch enqueued:[/green] {len(job_ids)}/{len(urls)} jobs")
        console.print(f"  Priority: {priority}")
        console.print(f"  Batch ID: {job_ids[0][:8]}..." if job_ids else "")
        console.print(f"  Queue depth: {q.pending_count()}")
    elif url:
        job_id = q.enqueue(
            url=url,
            priority=priority,
            scheduled_at=scheduled_at,
            depends_on=depends_on,
            source=source,
        )
        console.print(f"[green]Enqueued:[/green] {job_id}")
        console.print(f"  URL: {url}")
        console.print(f"  Priority: {priority}")
        if turbo:
            console.print(f"  Turbo: [bold yellow]YES[/bold yellow]")
        if scheduled_at:
            console.print(f"  Scheduled: {scheduled_at.isoformat()}")
        console.print(f"  Queue depth: {q.pending_count()}")
        console.print(f"  Position: {q.get_job_position(job_id)}")
        console.print(f"  Est. wait: {q.estimate_wait_time(job_id):.0f}s")
    else:
        console.print("[red]Provide a URL or use --batch[/red]")


# ══════════════════════════════════════════════════════════
#  3. WORKER — Manage worker
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--start", "start_worker", is_flag=True, help="Start the worker daemon")
@click.option("--status", "show_status", is_flag=True, help="Show worker and queue status")
@click.option("--health", "show_health", is_flag=True, help="Show worker health check")
@click.option("--recover", is_flag=True, help="Recover stale running jobs")
@click.option("--max-concurrent", default=None, type=int, help="Max concurrent jobs")
@click.option("--auto-scale", is_flag=True, help="Enable auto-scaling based on system load")
@click.option("--poll-interval", default=None, type=float, help="Poll interval in seconds")
def worker(start_worker, show_status, show_health, recover, max_concurrent, auto_scale, poll_interval):
    """Manage the background worker."""
    if show_status:
        from scheduler.worker import display_worker_status
        display_worker_status()
    elif show_health:
        from scheduler.worker import Worker
        w = Worker(max_concurrent=0)
        health = w.health()
        table = Table(title="Worker Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Detail")
        table.add_row("Overall", health["worker_status"],
                       ", ".join(health["issues"]) if health["issues"] else "OK")
        table.add_row("Queue", health["queue_health"]["status"],
                       ", ".join(health["queue_health"].get("warnings", [])) or "OK")
        table.add_row("CPU", f"{health['system_load']['cpu_percent']}%", "")
        table.add_row("Memory", f"{health['system_load']['memory_percent']}%", "")
        table.add_row("Disk", f"{health['system_load']['disk_percent']}%", "")
        console.print(table)
    elif recover:
        from scheduler.worker import Worker
        w = Worker(max_concurrent=0)
        count = w.recover_stale_jobs()
        if count > 0:
            console.print(f"[green]Recovered {count} stale job(s)[/green]")
        else:
            console.print("[dim]No stale jobs found[/dim]")
    elif start_worker:
        from scheduler.worker import Worker
        w = Worker(max_concurrent=max_concurrent, auto_scale=auto_scale, poll_interval=poll_interval)
        w.start()
    else:
        console.print("Use --start, --status, --health, or --recover")


# ══════════════════════════════════════════════════════════
#  4. HISTORY — Show job history
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--limit", "-l", default=20, help="Number of jobs to show")
@click.option("--status", "-s", type=click.Choice(["pending", "running", "done", "failed"]), help="Filter by status")
@click.option("--search", type=str, help="Search URL or error message")
@click.option("--date-range", type=str, help="Date range: today / 7d / 30d / YYYY-MM-DD,YYYY-MM-DD")
@click.option("--format", "output_format", type=click.Choice(["table", "json", "csv"]), default="table", help="Output format")
def history(limit, status, search, date_range, output_format):
    """Show recent pipeline job history."""
    from database.db import init_db, _new_session, Job
    init_db()

    session = _new_session()
    try:
        stmt = select(Job).order_by(desc(Job.created_at))

        if status:
            stmt = stmt.where(Job.status == status)

        # Date range filtering
        if date_range:
            now = datetime.now(timezone.utc)
            if date_range == "today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
                stmt = stmt.where(Job.created_at >= cutoff)
            elif date_range == "7d":
                stmt = stmt.where(Job.created_at >= now - timedelta(days=7))
            elif date_range == "30d":
                stmt = stmt.where(Job.created_at >= now - timedelta(days=30))
            elif "," in date_range:
                parts = date_range.split(",")
                try:
                    start = datetime.fromisoformat(parts[0]).replace(tzinfo=timezone.utc)
                    end = datetime.fromisoformat(parts[1]).replace(tzinfo=timezone.utc)
                    stmt = stmt.where(Job.created_at.between(start, end))
                except ValueError:
                    console.print(f"[red]Invalid date range: {date_range}[/red]")

        # Text search
        if search:
            search_term = f"%{search}%"
            stmt = stmt.where(
                or_(
                    Job.url.like(search_term),
                    Job.error_message.like(search_term),
                )
            )

        stmt = stmt.limit(limit)
        jobs = list(session.scalars(stmt).all())
    finally:
        session.close()

    if output_format == "json":
        data = []
        for job in jobs:
            data.append({
                "id": job.id,
                "url": job.url,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "duration": job.duration_seconds,
                "retry_count": job.retry_count,
                "error": job.error_message,
            })
        console.print_json(json.dumps(data, indent=2))
        return

    if output_format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "url", "status", "created_at", "duration", "retries", "error"])
        for job in jobs:
            writer.writerow([
                job.id, job.url, job.status,
                job.created_at.isoformat() if job.created_at else "",
                job.duration_seconds or 0,
                job.retry_count or 0,
                job.error_message or "",
            ])
        console.print(output.getvalue())
        return

    # Table format
    table = Table(title="Job History")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("URL", style="cyan", max_width=40)
    table.add_column("Status", style="bold")
    table.add_column("Created", style="dim")
    table.add_column("Duration", style="dim")
    table.add_column("Retries", style="dim")

    status_styles = {
        "pending": "[yellow]pending[/yellow]",
        "running": "[blue]running[/blue]",
        "done": "[green]done[/green]",
        "failed": "[red]failed[/red]",
        "retrying": "[yellow]retrying[/yellow]",
    }

    for job in jobs:
        created = job.created_at.strftime("%Y-%m-%d %H:%M") if job.created_at else "-"
        dur = f"{job.duration_seconds:.1f}s" if job.duration_seconds else "-"
        status_str = status_styles.get(job.status, job.status)
        table.add_row(
            job.id[:12], job.url[:40], status_str, created, dur,
            str(job.retry_count or 0),
        )

    console.print(table)


# ══════════════════════════════════════════════════════════
#  5. STATS — Show analytics
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--daily", is_flag=True, help="Show daily breakdown")
@click.option("--channels", is_flag=True, help="Show top channels")
@click.option("--performance", is_flag=True, help="Show processing time analysis")
@click.option("--export", "export_path", type=click.Path(), help="Export stats to CSV")
def stats(daily, channels, performance, export_path):
    """Show aggregate pipeline statistics."""
    from database.db import init_db, _new_session, Job, Video
    init_db()

    session = _new_session()
    try:
        # Base stats
        total = session.scalar(select(func.count(Job.id))) or 0
        done = session.scalar(select(func.count(Job.id)).where(Job.status == "done")) or 0
        failed = session.scalar(select(func.count(Job.id)).where(Job.status == "failed")) or 0
        running = session.scalar(select(func.count(Job.id)).where(Job.status == "running")) or 0
        pending = session.scalar(select(func.count(Job.id)).where(Job.status == "pending")) or 0
        avg_dur = session.scalar(select(func.avg(Job.duration_seconds)).where(Job.status == "done")) or 0.0
        total_videos = session.scalar(select(func.count(Video.id))) or 0
        success_rate = round(done / total * 100, 1) if total > 0 else 0.0

        table = Table(title="Pipeline Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_row("Total Jobs", str(total))
        table.add_row("Completed", str(done))
        table.add_row("Failed", str(failed))
        table.add_row("Success Rate", f"{success_rate}%")
        table.add_row("Avg Duration", f"{avg_dur:.1f}s")
        table.add_row("Videos Created", str(total_videos))
        table.add_row("Pending", str(pending))
        table.add_row("Running", str(running))
        console.print(table)

        # Daily breakdown
        if daily:
            console.print("\n[bold]Daily Breakdown[/bold]")
            daily_table = Table()
            daily_table.add_column("Date", style="cyan")
            daily_table.add_column("Jobs", style="green")
            daily_table.add_column("Done", style="green")
            daily_table.add_column("Failed", style="red")
            daily_table.add_column("Avg Duration", style="dim")

            for days_ago in range(7):
                day_start = datetime.now(timezone.utc) - timedelta(days=days_ago + 1)
                day_end = datetime.now(timezone.utc) - timedelta(days=days_ago)
                day_total = session.scalar(
                    select(func.count(Job.id)).where(
                        Job.created_at.between(day_start, day_end)
                    )
                ) or 0
                day_done = session.scalar(
                    select(func.count(Job.id)).where(
                        and_(Job.status == "done", Job.finished_at.between(day_start, day_end))
                    )
                ) or 0
                day_failed = session.scalar(
                    select(func.count(Job.id)).where(
                        and_(Job.status == "failed", Job.finished_at.between(day_start, day_end))
                    )
                ) or 0
                day_avg = session.scalar(
                    select(func.avg(Job.duration_seconds)).where(
                        and_(Job.status == "done", Job.finished_at.between(day_start, day_end))
                    )
                ) or 0.0
                daily_table.add_row(
                    day_start.strftime("%Y-%m-%d"),
                    str(day_total), str(day_done), str(day_failed),
                    f"{day_avg:.1f}s",
                )
            console.print(daily_table)

        # Top channels
        if channels:
            console.print("\n[bold]Top Channels[/bold]")
            ch_table = Table()
            ch_table.add_column("Channel", style="cyan")
            ch_table.add_column("Videos", style="green")
            ch_table.add_column("Avg Energy", style="dim")

            top_channels = session.query(
                Video.channel, func.count(Video.id), func.avg(Video.energy_score)
            ).group_by(Video.channel).order_by(desc(func.count(Video.id))).limit(10).all()

            for ch, count, avg_energy in top_channels:
                ch_table.add_row(ch or "Unknown", str(count), f"{avg_energy:.4f}" if avg_energy else "-")
            console.print(ch_table)

        # Performance analysis
        if performance:
            console.print("\n[bold]Processing Time Analysis[/bold]")
            perf_table = Table()
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="green")

            min_dur = session.scalar(
                select(func.min(Job.duration_seconds)).where(
                    and_(Job.status == "done", Job.duration_seconds > 0)
                )
            ) or 0
            max_dur = session.scalar(
                select(func.max(Job.duration_seconds)).where(Job.status == "done")
            ) or 0
            p50 = session.scalar(
                select(Job.duration_seconds).where(Job.status == "done").order_by(Job.duration_seconds).offset(int(done * 0.5)).limit(1)
            ) or 0
            p90 = session.scalar(
                select(Job.duration_seconds).where(Job.status == "done").order_by(Job.duration_seconds).offset(int(done * 0.9)).limit(1)
            ) or 0

            perf_table.add_row("Min Duration", f"{min_dur:.1f}s")
            perf_table.add_row("Max Duration", f"{max_dur:.1f}s")
            perf_table.add_row("P50 Duration", f"{float(p50):.1f}s")
            perf_table.add_row("P90 Duration", f"{float(p90):.1f}s")
            perf_table.add_row("Avg Duration", f"{avg_dur:.1f}s")
            console.print(perf_table)

        # CSV export
        if export_path:
            export_file = Path(export_path)
            with open(export_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["total_jobs", total])
                writer.writerow(["done_jobs", done])
                writer.writerow(["failed_jobs", failed])
                writer.writerow(["success_rate", success_rate])
                writer.writerow(["avg_duration", round(avg_dur, 1)])
                writer.writerow(["videos_created", total_videos])
            console.print(f"[green]Stats exported to {export_file}[/green]")

    finally:
        session.close()


# ══════════════════════════════════════════════════════════
#  6. VERIFY — Check dependencies
# ══════════════════════════════════════════════════════════

@cli.command()
def verify():
    """Verify all system dependencies are installed."""
    console.print("[bold]Verifying system dependencies...[/bold]\n")

    checks = []

    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), sys.version))

    try:
        from utils.ffmpeg_utils import check_ffmpeg
        check_ffmpeg()
        checks.append(("FFmpeg >= 4.0", True, "installed"))
    except RuntimeError as exc:
        checks.append(("FFmpeg >= 4.0", False, str(exc)))

    try:
        import torch
        cuda = "CUDA" if torch.cuda.is_available() else "CPU"
        checks.append(("PyTorch", True, f"{torch.__version__} ({cuda})"))
    except ImportError:
        checks.append(("PyTorch", False, "not installed"))

    try:
        import whisper
        checks.append(("Whisper", True, "installed"))
    except ImportError:
        checks.append(("Whisper", False, "not installed"))

    try:
        import numpy as np
        checks.append(("NumPy", True, np.__version__))
    except ImportError:
        checks.append(("NumPy", False, "not installed"))

    try:
        from PIL import Image
        checks.append(("Pillow", True, Image.__version__))
    except ImportError:
        checks.append(("Pillow", False, "not installed"))

    try:
        import sqlalchemy
        checks.append(("SQLAlchemy", True, sqlalchemy.__version__))
    except ImportError:
        checks.append(("SQLAlchemy", False, "not installed"))

    try:
        import importlib.metadata
        rich_ver = importlib.metadata.version("rich")
        checks.append(("Rich", True, rich_ver))
    except (ImportError, Exception):
        checks.append(("Rich", False, "not installed"))

    try:
        import yt_dlp
        checks.append(("yt-dlp", True, "installed"))
    except ImportError:
        checks.append(("yt-dlp", False, "not installed"))

    try:
        import click
        checks.append(("Click", True, click.__version__))
    except (ImportError, AttributeError):
        checks.append(("Click", False, "not installed"))

    try:
        from config.settings import get_settings
        settings = get_settings()
        checks.append(("Config", True, f"model={settings.WHISPER_MODEL}, device={settings.WHISPER_DEVICE}"))
    except Exception as exc:
        checks.append(("Config", False, str(exc)))

    from config.settings import get_settings
    settings = get_settings()
    dirs_ok = all(d.exists() for d in [settings.SHORTS_DIR, settings.LOGS_DIR, settings.METADATA_DIR])
    checks.append(("Output Dirs", dirs_ok, "created" if dirs_ok else "missing"))

    table = Table(title="Dependency Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Detail")

    all_ok = True
    for name, ok, detail in checks:
        status = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
        if not ok:
            all_ok = False
        table.add_row(name, status, detail)

    console.print(table)

    if all_ok:
        console.print("\n[bold green]All dependencies verified![/bold green]")
    else:
        console.print("\n[bold red]Some dependencies are missing![/bold red]")
        console.print("[dim]Run: pip install -r requirements.txt[/dim]")


# ══════════════════════════════════════════════════════════
#  6b. PRESETS — Show available duration presets
# ══════════════════════════════════════════════════════════

@cli.command()
def presets():
    """Show available clip duration presets."""
    from config.settings import get_settings
    settings = get_settings()

    table = Table(title="Duration Presets", show_lines=True)
    table.add_column("Preset", style="cyan bold")
    table.add_column("Duration", style="green bold")
    table.add_column("Label", style="yellow")
    table.add_column("Best For", style="dim")
    table.add_column("CLI Usage", style="blue")

    preset_info = [
        ("quick", "25s", "Quick Hook", "TikTok fast scroll, attention grabbers", "--preset quick  or  -d 25s"),
        ("standard", "45s", "Standard Short", "YouTube Shorts & Reels sweet spot", "--preset standard  or  -d 45s"),
        ("extended", "180s (3 min)", "Extended Clip", "Long-form Shorts, storytelling", "--preset extended  or  -d 3min"),
    ]

    for name, dur, label, best_for, cli in preset_info:
        is_current = settings.DURATION_PRESETS.get(name) == settings.CLIP_DURATION
        marker = " [bold green]<- current[/bold green]" if is_current else ""
        table.add_row(f"{name}{marker}", dur, label, best_for, cli)

    # Also show shorthand aliases
    alias_table = Table(title="Shorthand Aliases")
    alias_table.add_column("Alias", style="cyan")
    alias_table.add_column("Seconds", style="green")
    alias_table.add_column("Same As", style="dim")
    alias_table.add_row("25s", "25", "quick")
    alias_table.add_row("45s", "45", "standard")
    alias_table.add_row("3min", "180", "extended")
    alias_table.add_row("3minute", "180", "extended")

    console.print(table)
    console.print(alias_table)
    console.print(f"\n[dim]Custom durations also work: -d 30  (30 seconds), -d 120  (2 minutes)[/dim]")
    console.print(f"[dim]Current preset: {settings.current_preset_label}[/dim]")


# ══════════════════════════════════════════════════════════
#  6c. PATTERNS — Show available channel patterns
# ══════════════════════════════════════════════════════════

@cli.command()
def patterns():
    """Show available channel branding patterns."""
    try:
        from core.channel_pattern import list_patterns, BUILTIN_PATTERNS
    except ImportError:
        console.print("[red]Error: core.channel_pattern module not found[/red]")
        return

    all_patterns = list_patterns()

    table = Table(title="Channel Branding Patterns", show_lines=True)
    table.add_column("Pattern", style="cyan bold")
    table.add_column("Display Name", style="yellow bold")
    table.add_column("Category", style="green")
    table.add_column("Description", style="dim")
    table.add_column("Features", style="blue")

    for p in all_patterns:
        features = []
        if p.hook.enabled:
            features.append("Hook")
        if p.cta.enabled:
            features.append("CTA")
        if p.intro.enabled:
            features.append("Intro")
        if p.outro.enabled:
            features.append("Outro")
        if p.lower_third.enabled:
            features.append("Lower3rd")
        features_str = ", ".join(features) or "None"

        table.add_row(
            p.name,
            p.display_name,
            p.category,
            p.description[:60] + ("..." if len(p.description) > 60 else ""),
            features_str,
        )

    # Add my_channel custom pattern entry
    table.add_row(
        "my_channel",
        "My Channel",
        "custom",
        "Your personal channel branding pattern (configure in .env)",
        "Hook, CTA, Intro, Outro",
    )

    console.print(table)

    # Show usage
    usage_table = Table(title="Pattern Usage")
    usage_table.add_column("Method", style="cyan")
    usage_table.add_column("Command", style="green")
    usage_table.add_row("CLI flag", "python main.py run --url URL --pattern viral_hype")
    usage_table.add_row("With channel name", "python main.py run --url URL --pattern news_alert --channel-name 'Tech News'")
    usage_table.add_row("Env variable", "CHANNEL_PATTERN=viral_hype  (in .env)")
    usage_table.add_row("Custom pattern", "Create assets/patterns/mypattern.json")

    console.print(usage_table)
    console.print("\n[dim]Each pattern configures: intro animation, hook style, CTA overlay, outro card,[/dim]")
    console.print("[dim]subtitle style, color palette, transition effects, and watermark positioning.[/dim]")
    console.print("[dim]Create custom patterns in assets/patterns/<name>.json[/dim]")


# ══════════════════════════════════════════════════════════
#  7. CONFIG — Show/edit configuration
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--set", "set_values", multiple=True, help="Set KEY=VALUE")
@click.option("--reset", is_flag=True, help="Reset to defaults")
def config(set_values, reset):
    """Display or edit configuration."""
    from config.settings import get_settings, BASE_DIR

    if reset:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            backup = env_file.with_suffix(".env.bak")
            env_file.rename(backup)
            console.print(f"[green]Reset config. Backup saved to {backup}[/green]")
        else:
            console.print("[dim]No .env file to reset[/dim]")
        return

    if set_values:
        env_file = BASE_DIR / ".env"
        lines: list[str] = []
        if env_file.exists():
            lines = env_file.read_text(encoding="utf-8").splitlines()

        existing_keys = {}
        for i, line in enumerate(lines):
            if "=" in line and not line.startswith("#"):
                key = line.split("=")[0].strip()
                existing_keys[key] = i

        for kv in set_values:
            if "=" not in kv:
                console.print(f"[red]Invalid format: {kv} (use KEY=VALUE)[/red]")
                continue
            key, value = kv.split("=", 1)
            key = key.strip().upper()
            value = value.strip()

            if key in existing_keys:
                lines[existing_keys[key]] = f"{key}={value}"
            else:
                lines.append(f"{key}={value}")
            console.print(f"[green]Set {key}={value}[/green]")

        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # Display config
    settings = get_settings()
    tree = Tree("[bold]Configuration[/bold]")

    whisper_branch = tree.add("[cyan]Whisper[/cyan]")
    whisper_branch.add(f"Model: {settings.WHISPER_MODEL}")
    whisper_branch.add(f"Device: {settings.WHISPER_DEVICE}")
    whisper_branch.add(f"Compute Type: {settings.WHISPER_COMPUTE_TYPE}")
    whisper_branch.add(f"Language: {settings.WHISPER_LANGUAGE}")
    whisper_branch.add(f"Task: {settings.WHISPER_TASK}")

    output_branch = tree.add("[cyan]Output[/cyan]")
    output_branch.add(f"Resolution: {settings.OUTPUT_WIDTH}x{settings.OUTPUT_HEIGHT}")
    output_branch.add(f"Clip Duration: {settings.CLIP_DURATION}s ({settings.current_preset_label})")
    output_branch.add(f"Duration Preset: {settings.CLIP_DURATION_PRESET}")
    output_branch.add(f"Available Presets: quick(25s) / standard(45s) / extended(180s)")
    output_branch.add(f"Turbo Mode: {settings.TURBO_MODE}")

    logo_branch = tree.add("[cyan]Logo[/cyan]")
    logo_branch.add(f"Path: {settings.LOGO_PATH}")
    logo_branch.add(f"Position: {settings.LOGO_POSITION}")
    logo_branch.add(f"Opacity: {settings.LOGO_OPACITY}")
    logo_branch.add(f"Scale: {settings.LOGO_SCALE}")

    sub_branch = tree.add("[cyan]Subtitles[/cyan]")
    sub_branch.add(f"Font: {settings.SUBTITLE_FONT}")
    sub_branch.add(f"Animation: {settings.SUBTITLE_ANIMATION}")
    sub_branch.add(f"Max Words: {settings.SUBTITLE_MAX_WORDS}")

    ffmpeg_branch = tree.add("[cyan]FFmpeg[/cyan]")
    ffmpeg_branch.add(f"Preset: {settings.FFMPEG_PRESET}")
    ffmpeg_branch.add(f"CRF: {settings.FFMPEG_CRF}")
    ffmpeg_branch.add(f"Codec: {settings.FFMPEG_VIDEO_CODEC}")
    ffmpeg_branch.add(f"HW Accel: {settings.FFMPEG_HW_ACCEL}")

    platform_branch = tree.add("[cyan]Platforms[/cyan]")
    platform_branch.add(f"YouTube: {settings.EXPORT_YOUTUBE}")
    platform_branch.add(f"TikTok: {settings.EXPORT_TIKTOK}")
    platform_branch.add(f"Reels: {settings.EXPORT_REELS}")

    queue_branch = tree.add("[cyan]Queue / Worker[/cyan]")
    queue_branch.add(f"Max Concurrent: {settings.MAX_CONCURRENT_JOBS}")
    queue_branch.add(f"Retry Attempts: {settings.JOB_RETRY_ATTEMPTS}")
    queue_branch.add(f"Poll Interval: {settings.WORKER_POLL_INTERVAL}s")

    path_branch = tree.add("[cyan]Paths[/cyan]")
    path_branch.add(f"Base: {BASE_DIR}")
    path_branch.add(f"Downloads: {settings.DOWNLOADS_DIR}")
    path_branch.add(f"Shorts: {settings.SHORTS_DIR}")
    path_branch.add(f"Database: {settings.DB_PATH}")
    path_branch.add(f"Logs: {settings.LOGS_DIR}")

    console.print(tree)


# ══════════════════════════════════════════════════════════
#  8. ANALYZE — Analyze video without full pipeline
# ══════════════════════════════════════════════════════════

@cli.command()
@click.argument("source", required=True)
@click.option("--visualize", is_flag=True, help="Generate charts for the analysis")
@click.option("--duration", "-d", type=str, default=None, help="Target clip duration: preset (quick/standard/extended/25s/45s/3min) or seconds")
def analyze(source, visualize, duration):
    """Analyze a video (URL or local file) without full pipeline."""
    _print_banner()

    from config.settings import get_settings
    settings = get_settings()

    # Resolve duration
    if duration:
        try:
            resolved_dur = int(duration)
        except ValueError:
            resolved_dur = settings.resolve_duration(duration)
    else:
        resolved_dur = settings.CLIP_DURATION

    video_path: Path | None = None
    is_url = source.startswith("http")

    if is_url:
        console.print(f"[cyan]Downloading video for analysis...[/cyan]")
        try:
            from core.downloader import download_video
            video_path, info = download_video(source, settings.DOWNLOADS_DIR)
        except Exception as exc:
            console.print(f"[red]Download failed: {exc}[/red]")
            return
    else:
        video_path = Path(source)
        if not video_path.exists():
            console.print(f"[red]File not found: {video_path}[/red]")
            return

    # Probe video
    try:
        from utils.ffmpeg_utils import probe_video
        video_info = probe_video(video_path)
    except Exception as exc:
        console.print(f"[red]Failed to probe video: {exc}[/red]")
        return

    console.print(f"\n[bold]Video Info:[/bold]")
    info_table = Table()
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    info_table.add_row("Duration", f"{video_info.duration:.1f}s")
    info_table.add_row("Resolution", f"{video_info.width}x{video_info.height}")
    info_table.add_row("Codec", video_info.codec or "unknown")
    info_table.add_row("FPS", str(video_info.fps or "unknown"))
    console.print(info_table)

    # Run engagement analysis
    console.print(f"\n[bold]Engagement Analysis[/bold] (target={resolved_dur}s)...")
    try:
        from core.analyzer import EngagementAnalyzer
        analyzer = EngagementAnalyzer(video_path, resolved_dur)
        segment = analyzer.analyze()

        result_table = Table(title="Best Clip Segment")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", style="bold green")
        result_table.add_row("Start Time", f"{segment.start_time:.1f}s")
        result_table.add_row("End Time", f"{segment.end_time:.1f}s")
        result_table.add_row("Energy Score", f"{segment.energy_score:.4f}")
        result_table.add_row("Quality Grade", segment.overall_quality_grade)
        result_table.add_row("Confidence", f"{segment.confidence:.2f}")
        result_table.add_row("Silence Ratio", f"{segment.silence_ratio:.1%}")
        result_table.add_row("Speech Rate", f"{segment.speech_rate_estimate:.0f} WPM")
        result_table.add_row("Music Likelihood", f"{segment.music_likelihood:.1%}")
        result_table.add_row("Visual Complexity", f"{segment.visual_complexity:.1%}")
        result_table.add_row("Method", segment.method_used)
        console.print(result_table)

        # Multi-clip analysis
        if video_info.duration > resolved_dur * 2:
            console.print(f"\n[bold]Multi-Clip Analysis[/bold]...")
            multi = analyzer.analyze_multiple_clips(num_clips=3)
            multi_table = Table(title="Top Clips")
            multi_table.add_column("#", style="dim")
            multi_table.add_column("Start", style="cyan")
            multi_table.add_column("End", style="cyan")
            multi_table.add_column("Score", style="green")
            multi_table.add_column("Grade", style="bold")
            multi_table.add_column("Confidence", style="dim")
            for i, seg in enumerate(multi.segments, 1):
                multi_table.add_row(
                    str(i),
                    f"{seg.start_time:.1f}s", f"{seg.end_time:.1f}s",
                    f"{seg.energy_score:.4f}", seg.overall_quality_grade,
                    f"{seg.confidence:.2f}",
                )
            console.print(multi_table)

    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")

    # Visualization
    if visualize:
        console.print("\n[bold]Generating visualization...[/bold]")
        try:
            from core.analyzer import EngagementAnalyzer
            analyzer = EngagementAnalyzer(video_path, resolved_dur)
            profile = analyzer._build_energy_profile(video_info.duration)
            import numpy as np
            times = np.arange(len(profile.composite)) * analyzer.sample_interval
            data = list(zip(times.tolist(), profile.composite.tolist()))
            console.print(f"  Composite score samples: {len(data)}")
            console.print(f"  Peak score: {max(profile.composite):.4f}")
            console.print(f"  Avg score: {np.mean(profile.composite):.4f}")
            console.print("[dim](Full chart generation requires matplotlib)[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Visualization error: {exc}[/yellow]")


# ══════════════════════════════════════════════════════════
#  9. TRANSCRIBE — Transcribe video only
# ══════════════════════════════════════════════════════════

@cli.command()
@click.argument("file_path", required=True, type=click.Path(exists=True))
@click.option("--model", type=click.Choice(["tiny", "base", "small", "medium", "large"]), help="Whisper model")
@click.option("--language", "-l", type=str, help="Language code (e.g. en, es, fr)")
@click.option("--output-format", type=click.Choice(["srt", "vtt", "text"]), default="text", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def transcribe(file_path, model, language, output_format, output):
    """Transcribe a local video file."""
    _print_banner()

    from config.settings import get_settings, Settings
    settings = get_settings()

    if model:
        settings.WHISPER_MODEL = model
    if language:
        settings.WHISPER_LANGUAGE = language

    video_path = Path(file_path)
    if not video_path.exists():
        console.print(f"[red]File not found: {video_path}[/red]")
        return

    console.print(f"[cyan]Transcribing:[/cyan] {video_path.name}")
    console.print(f"  Model: {settings.WHISPER_MODEL}")
    console.print(f"  Language: {settings.WHISPER_LANGUAGE}")
    console.print(f"  Format: {output_format}")

    try:
        from core.transcriber import transcribe as do_transcribe
        result = do_transcribe(video_path, settings)

        if result.is_empty:
            console.print("[yellow]No speech detected in the video.[/yellow]")
            return

        console.print(f"\n[green]Transcription complete:[/green]")
        console.print(f"  Language: {result.language} ({result.language_confidence:.0%})")
        console.print(f"  Words: {result.word_count}")
        console.print(f"  Duration: {result.duration:.1f}s")
        console.print(f"  Avg Confidence: {result.average_confidence:.1%}")

        # Generate output
        if output_format == "text":
            text_output = result.text
        elif output_format == "srt":
            lines = []
            for i, seg in enumerate(result.segments, 1):
                start_h, rem = divmod(seg.start, 3600)
                start_m, start_s = divmod(rem, 60)
                end_h, rem = divmod(seg.end, 3600)
                end_m, end_s = divmod(rem, 60)
                lines.append(str(i))
                lines.append(f"{int(start_h):02d}:{int(start_m):02d}:{start_s:06.3f} --> {int(end_h):02d}:{int(end_m):02d}:{end_s:06.3f}")
                lines.append(seg.text.strip())
                lines.append("")
            text_output = "\n".join(lines)
        elif output_format == "vtt":
            lines = ["WEBVTT", ""]
            for seg in result.segments:
                start_h, rem = divmod(seg.start, 3600)
                start_m, start_s = divmod(rem, 60)
                end_h, rem = divmod(seg.end, 3600)
                end_m, end_s = divmod(rem, 60)
                lines.append(f"{int(start_h):02d}:{int(start_m):02d}:{start_s:06.3f} --> {int(end_h):02d}:{int(end_m):02d}:{end_s:06.3f}")
                lines.append(seg.text.strip())
                lines.append("")
            text_output = "\n".join(lines)
        else:
            text_output = result.text

        if output:
            out_path = Path(output)
            out_path.write_text(text_output, encoding="utf-8")
            console.print(f"[green]Written to: {out_path}[/green]")
        else:
            console.print(Panel(text_output[:2000], title="Transcription", border_style="cyan"))

    except Exception as exc:
        console.print(f"[red]Transcription failed: {exc}[/red]")


# ══════════════════════════════════════════════════════════
#  10. CLEANUP — Clean up old files and database
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--days", type=int, default=30, help="Age threshold in days")
@click.option("--dry-run", is_flag=True, help="Preview without deleting")
@click.option("--all", "cleanup_all", is_flag=True, help="Comprehensive cleanup (downloads, logs, temp, db)")
def cleanup(days, dry_run, cleanup_all):
    """Clean up old files and database entries."""
    from config.settings import get_settings
    from database.db import init_db, _new_session, Job

    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    console.print(f"[bold]Cleanup[/bold] (older than {days} days)")
    if dry_run:
        console.print("[yellow]DRY RUN — nothing will be deleted[/yellow]")

    total_cleaned = 0

    # Database cleanup
    init_db()
    session = _new_session()
    try:
        old_jobs = session.scalar(
            select(func.count(Job.id)).where(
                and_(
                    Job.created_at < cutoff,
                    Job.status.in_(["done", "failed"]),
                )
            )
        ) or 0
        console.print(f"  Old completed/failed jobs: {old_jobs}")
        if not dry_run and old_jobs > 0:
            session.query(Job).where(
                and_(Job.created_at < cutoff, Job.status.in_(["done", "failed"]))
            ).delete()
            session.commit()
            total_cleaned += old_jobs
            console.print(f"  [green]Deleted {old_jobs} old jobs[/green]")
    except Exception as exc:
        session.rollback()
        console.print(f"  [red]DB cleanup error: {exc}[/red]")
    finally:
        session.close()

    # Expired pending jobs
    from scheduler.job_queue import JobQueue
    q = JobQueue()
    expired = q.cleanup_expired(max_age_hours=days * 24)
    console.print(f"  Expired pending jobs: {expired}")
    total_cleaned += expired

    # File cleanup
    dirs_to_clean = [settings.SHORTS_DIR, settings.METADATA_DIR]
    if cleanup_all:
        dirs_to_clean.extend([settings.DOWNLOADS_DIR, settings.LOGS_DIR])

    for target_dir in dirs_to_clean:
        if not target_dir.exists():
            continue
        file_count = 0
        for f in target_dir.rglob("*"):
            if f.is_file():
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                    if mtime < cutoff:
                        file_count += 1
                        size_mb = f.stat().st_size / (1024 * 1024)
                        if dry_run:
                            console.print(f"  [dim]Would delete: {f} ({size_mb:.1f} MB)[/dim]")
                        else:
                            f.unlink()
                except OSError:
                    pass
        if file_count > 0:
            console.print(f"  {target_dir.name}: {file_count} old file(s)")
            if not dry_run:
                total_cleaned += file_count

    # Temp files
    temp_files_count = 0
    for target_dir in [settings.SHORTS_DIR, settings.DOWNLOADS_DIR]:
        if target_dir.exists():
            for f in target_dir.rglob("*.part"):
                temp_files_count += 1
                if not dry_run:
                    try:
                        f.unlink()
                    except OSError:
                        pass
            for f in target_dir.rglob("*.ytdl"):
                temp_files_count += 1
                if not dry_run:
                    try:
                        f.unlink()
                    except OSError:
                        pass

    if temp_files_count > 0:
        console.print(f"  Temp/partial files: {temp_files_count}")

    console.print(f"\n[bold]Total items {'would be ' if dry_run else ''}cleaned: {total_cleaned}[/bold]")


# ══════════════════════════════════════════════════════════
#  11. INFO — Show info about a video file or URL
# ══════════════════════════════════════════════════════════

@cli.command()
@click.argument("source", required=True)
def info(source):
    """Show info about a video file or URL."""
    _print_banner()

    is_url = source.startswith("http")

    if is_url:
        # URL info — fetch metadata without downloading
        console.print(f"[cyan]Fetching video info...[/cyan]")
        try:
            from core.downloader import _fetch_metadata, get_available_formats
            meta = _fetch_metadata(source)

            table = Table(title="Video Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Title", meta.get("title", "Unknown"))
            table.add_row("ID", meta.get("id", ""))
            table.add_row("Channel", meta.get("uploader", ""))
            table.add_row("Duration", f"{meta.get('duration', 0):.0f}s ({meta.get('duration', 0) / 60:.1f} min)")
            table.add_row("Views", f"{meta.get('view_count', 0):,}")
            table.add_row("Likes", f"{meta.get('like_count', 0):,}")
            table.add_row("Upload Date", meta.get("upload_date", ""))
            table.add_row("Description", (meta.get("description", "")[:200] + "...") if len(meta.get("description", "")) > 200 else meta.get("description", ""))
            console.print(table)

            # Available formats
            console.print("\n[bold]Available Formats[/bold]")
            try:
                formats = get_available_formats(source)
                fmt_table = Table()
                fmt_table.add_column("Format ID", style="dim")
                fmt_table.add_column("Resolution", style="cyan")
                fmt_table.add_column("Ext", style="dim")
                fmt_table.add_column("Size", style="green")
                fmt_table.add_column("Note", style="dim")

                for fmt in formats[:20]:  # Show top 20
                    fmt_table.add_row(
                        fmt.format_id, fmt.resolution, fmt.ext,
                        fmt.filesize_human, fmt.format_note,
                    )
                console.print(fmt_table)
            except Exception as exc:
                console.print(f"[yellow]Could not list formats: {exc}[/yellow]")

            # Available subtitles
            subtitles = meta.get("subtitles", {})
            auto_subs = meta.get("automatic_captions", {})
            if subtitles or auto_subs:
                console.print(f"\n[bold]Subtitles:[/bold] {', '.join(subtitles.keys()) or 'none'}")
                console.print(f"[bold]Auto Captions:[/bold] {', '.join(auto_subs.keys()) or 'none'}")

        except Exception as exc:
            console.print(f"[red]Failed to fetch info: {exc}[/red]")

    else:
        # Local file info
        video_path = Path(source)
        if not video_path.exists():
            console.print(f"[red]File not found: {video_path}[/red]")
            return

        try:
            from utils.ffmpeg_utils import probe_video
            vi = probe_video(video_path)

            table = Table(title="Video File Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            table.add_row("File", str(video_path))
            table.add_row("Size", f"{file_size_mb:.1f} MB")
            table.add_row("Duration", f"{vi.duration:.1f}s ({vi.duration / 60:.1f} min)")
            table.add_row("Resolution", f"{vi.width}x{vi.height}")
            table.add_row("Codec", vi.codec or "unknown")
            table.add_row("FPS", str(vi.fps or "unknown"))
            table.add_row("Bitrate", f"{vi.bitrate or 0} bps")
            console.print(table)
        except Exception as exc:
            console.print(f"[red]Failed to probe video: {exc}[/red]")


# ══════════════════════════════════════════════════════════
#  12. INTERACTIVE — Interactive mode with guided workflow
# ══════════════════════════════════════════════════════════

@cli.command()
@click.option("--url", "-u", help="Start with a URL pre-loaded")
def interactive(url):
    """Interactive mode with guided workflow."""
    _print_banner()
    console.print(Panel(
        "[bold]Interactive Mode[/bold]\n"
        "Step-by-step guided processing with previews.\n"
        "[dim]Press Ctrl+C at any time to exit.[/dim]",
        border_style="cyan",
    ))

    try:
        # Step 1: Get URL
        if not url:
            url = Prompt.ask("[bold cyan]Step 1:[/bold cyan] Enter YouTube URL")

        if not url:
            console.print("[red]No URL provided. Exiting.[/red]")
            return

        console.print(f"[green]URL:[/green] {url}")

        # Step 2: Choose processing options
        console.print("\n[bold cyan]Step 2:[/bold cyan] Processing Options")

        duration = IntPrompt.ask("Clip duration (seconds)", default=60)
        clips = IntPrompt.ask("Number of clips", default=1)

        do_subs = Confirm.ask("Add subtitles?", default=True)
        do_logo = Confirm.ask("Add logo?", default=True)
        do_enhance = Confirm.ask("Enhance audio?", default=False)

        # Step 3: Choose platform targets
        console.print("\n[bold cyan]Step 3:[/bold cyan] Target Platforms")
        do_youtube = Confirm.ask("YouTube Shorts?", default=True)
        do_tiktok = Confirm.ask("TikTok?", default=True)
        do_reels = Confirm.ask("Instagram Reels?", default=True)

        # Step 4: Choose quality
        console.print("\n[bold cyan]Step 4:[/bold cyan] Quality Preset")
        console.print("  1. Fast (ultrafast, lower quality)")
        console.print("  2. Balanced (fast, good quality)")
        console.print("  3. High (slow, best quality)")
        quality_choice = IntPrompt.ask("Choose quality (1-3)", default=2)
        quality_map = {1: "fast", 2: "balanced", 3: "high"}
        quality = quality_map.get(quality_choice, "balanced")

        # Summary
        console.print("\n" + Panel(
            f"[bold]Processing Summary[/bold]\n\n"
            f"  URL: {url[:80]}\n"
            f"  Duration: {duration}s\n"
            f"  Clips: {clips}\n"
            f"  Subtitles: {'Yes' if do_subs else 'No'}\n"
            f"  Logo: {'Yes' if do_logo else 'No'}\n"
            f"  Audio Enhancement: {'Yes' if do_enhance else 'No'}\n"
            f"  Platforms: {'YouTube ' if do_youtube else ''}{'TikTok ' if do_tiktok else ''}{'Reels' if do_reels else ''}\n"
            f"  Quality: {quality}",
            border_style="green",
        ))

        if not Confirm.ask("\nProceed with processing?", default=True):
            console.print("[yellow]Cancelled.[/yellow]")
            return

        # Step 5: Run pipeline
        console.print("\n[bold cyan]Step 5:[/bold cyan] Processing...")
        from core.pipeline import run_pipeline

        platforms = []
        if do_youtube:
            platforms.append("youtube")
        if do_tiktok:
            platforms.append("tiktok")
        if do_reels:
            platforms.append("reels")

        result = run_pipeline(
            url=url,
            duration=duration,
            skip_subs=not do_subs,
            no_logo=not do_logo,
            platforms=platforms,
            resume=True,
        )

        # Step 6: Show results
        if result.success:
            console.print("\n[bold green]Processing Complete![/bold green]")
            if result.outputs:
                output_table = Table(title="Output Files")
                output_table.add_column("Platform", style="cyan")
                output_table.add_column("File", style="green")

                if result.outputs.youtube_path:
                    output_table.add_row("YouTube", str(result.outputs.youtube_path))
                if result.outputs.tiktok_path:
                    output_table.add_row("TikTok", str(result.outputs.tiktok_path))
                if result.outputs.reels_path:
                    output_table.add_row("Reels", str(result.outputs.reels_path))
                console.print(output_table)
        else:
            console.print(f"\n[red]Processing failed: {result.error}[/red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive mode cancelled.[/yellow]")


# ══════════════════════════════════════════════════════════
#  13. RATE — Rate and rank clips for viral potential
# ══════════════════════════════════════════════════════════

@cli.command()
@click.argument("source", required=True)
@click.option("--duration", "-d", type=int, default=60, help="Target clip duration")
@click.option("--clips", type=int, default=3, help="Number of clips to rate")
@click.option("--platform", type=click.Choice(["youtube", "tiktok", "reels"]), default="youtube", help="Target platform for ranking")
@click.option("--transcribe", "do_transcribe", is_flag=True, help="Include transcription in rating")
def rate(source, duration, clips, platform, do_transcribe):
    """Rate and rank clips from a video for viral potential."""
    _print_banner()

    from config.settings import get_settings
    from core.analyzer import EngagementAnalyzer
    from core.clip_rater import rate_clip, compare_clips, rank_clips

    settings = get_settings()

    # Download if URL, or use local file
    video_path: Path | None = None
    is_url = source.startswith("http")

    if is_url:
        console.print(f"[cyan]Downloading video for rating...[/cyan]")
        try:
            from core.downloader import download_video
            video_path, info = download_video(source, settings.DOWNLOADS_DIR)
        except Exception as exc:
            console.print(f"[red]Download failed: {exc}[/red]")
            return
    else:
        video_path = Path(source)
        if not video_path.exists():
            console.print(f"[red]File not found: {video_path}[/red]")
            return

    # Probe video
    try:
        from utils.ffmpeg_utils import probe_video
        video_info = probe_video(video_path)
    except Exception as exc:
        console.print(f"[red]Failed to probe video: {exc}[/red]")
        return

    console.print(f"\n[bold]Video Info:[/bold]")
    console.print(f"  Duration: {video_info.duration:.1f}s")
    console.print(f"  Resolution: {video_info.width}x{video_info.height}")
    console.print(f"  Target clips: {clips} x {duration}s")
    console.print(f"  Platform: {platform}")

    # Analyze multi-clip segments
    console.print(f"\n[bold]Analyzing clip segments...[/bold]")
    try:
        analyzer = EngagementAnalyzer(video_path, duration)

        if video_info.duration > duration * 2 and clips > 1:
            multi = analyzer.analyze_multiple_clips(num_clips=clips)
            segments = multi.segments
        else:
            segment = analyzer.analyze()
            segments = [segment]
    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")
        return

    # Optionally transcribe for better rating
    transcriptions = [None] * len(segments)
    if do_transcribe:
        console.print(f"\n[bold]Transcribing for rating enrichment...[/bold]")
        try:
            from core.transcriber import transcribe_video
            for i, seg in enumerate(segments):
                try:
                    result = transcribe_video(
                        video_path,
                        start_time=seg.start_time,
                        duration=seg.end_time - seg.start_time,
                    )
                    transcriptions[i] = result
                except Exception:
                    pass  # Keep None, rating still works without it
        except Exception as exc:
            console.print(f"[yellow]Transcription skipped: {exc}[/yellow]")

    # Rate each clip
    console.print(f"\n[bold]Rating {len(segments)} clip(s)...[/bold]")
    rated = []
    for i, seg in enumerate(segments):
        transcription = transcriptions[i] if i < len(transcriptions) else None
        rating = rate_clip(seg, transcription, settings)
        rated.append((seg, rating))

    # Show rating table with grades, scores, strengths/weaknesses
    grade_styles = {"A": "[bold green]", "B": "[green]", "C": "[yellow]", "D": "[red]", "F": "[bold red]"}

    rating_table = Table(title="Clip Ratings")
    rating_table.add_column("#", style="dim")
    rating_table.add_column("Time", style="cyan")
    rating_table.add_column("Overall", style="bold")
    rating_table.add_column("Engage", style="green")
    rating_table.add_column("Speech", style="green")
    rating_table.add_column("Visual", style="green")
    rating_table.add_column("Grade", style="bold")
    rating_table.add_column("Style", style="dim")

    for i, (seg, rating) in enumerate(rated, 1):
        grade_prefix = grade_styles.get(rating.grade, "")
        rating_table.add_row(
            str(i),
            f"{seg.start_time:.1f}-{seg.end_time:.1f}s",
            f"{rating.overall_score:.1f}",
            f"{rating.engagement_score:.1f}",
            f"{rating.speech_score:.1f}",
            f"{rating.visual_score:.1f}",
            f"{grade_prefix}{rating.grade}[/]",
            rating.title_style,
        )
    console.print(rating_table)

    # Show strengths/weaknesses for each clip
    for i, (seg, rating) in enumerate(rated, 1):
        console.print(f"\n[bold]Clip {i}[/bold] ({seg.start_time:.1f}-{seg.end_time:.1f}s):")
        if rating.strengths:
            console.print(f"  [green]Strengths:[/green] {', '.join(rating.strengths)}")
        if rating.weaknesses:
            console.print(f"  [red]Weaknesses:[/red] {', '.join(rating.weaknesses)}")

    # Show ranked clips for the target platform
    if len(rated) > 1:
        ranked = rank_clips(segments, platform=platform, transcriptions=transcriptions, settings=settings)

        rank_table = Table(title=f"Platform Ranking ({platform.title()})")
        rank_table.add_column("Rank", style="bold")
        rank_table.add_column("Clip", style="cyan")
        rank_table.add_column("Platform Fit", style="green")
        rank_table.add_column("Grade", style="bold")

        for rank_num, (seg, rating) in enumerate(ranked, 1):
            platform_fit = 100 - rating.platform_rankings.get(platform, 100)
            grade_prefix = grade_styles.get(rating.grade, "")
            rank_table.add_row(
                f"#{rank_num}",
                f"{seg.start_time:.1f}-{seg.end_time:.1f}s",
                f"{platform_fit:.1f}",
                f"{grade_prefix}{rating.grade}[/]",
            )
        console.print(rank_table)

        # Show comparison results
        console.print(f"\n[bold]Comparison Summary[/bold]")
        best_seg, best_rating = ranked[0]
        console.print(f"  Best clip for {platform}: {best_seg.start_time:.1f}-{best_seg.end_time:.1f}s (score={best_rating.overall_score:.1f}, grade={best_rating.grade})")
        if len(ranked) > 1:
            worst_seg, worst_rating = ranked[-1]
            gap = best_rating.overall_score - worst_rating.overall_score
            console.print(f"  Score gap: {gap:.1f} points between best and worst clip")


# ══════════════════════════════════════════════════════════
#  14. AI-META — Generate AI-powered metadata
# ══════════════════════════════════════════════════════════

@cli.command("ai-meta")
@click.argument("source", required=True)
@click.option("--platform", type=click.Choice(["youtube", "tiktok", "reels", "all"]), default="all", help="Target platform")
@click.option("--transcribe", "do_transcribe", is_flag=True, help="Transcribe video first for better metadata")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
def ai_meta(source, platform, do_transcribe, output):
    """Generate AI-powered titles, descriptions, and hashtags."""
    _print_banner()

    from config.settings import get_settings
    from core.ai_metadata import generate_ai_metadata

    settings = get_settings()

    # Download if URL, or use local file
    video_path: Path | None = None
    video_title = ""
    is_url = source.startswith("http")

    if is_url:
        console.print(f"[cyan]Downloading video...[/cyan]")
        try:
            from core.downloader import download_video
            video_path, info = download_video(source, settings.DOWNLOADS_DIR)
            video_title = info.get("title", "") if isinstance(info, dict) else str(info)
        except Exception as exc:
            console.print(f"[red]Download failed: {exc}[/red]")
            return
    else:
        video_path = Path(source)
        if not video_path.exists():
            console.print(f"[red]File not found: {video_path}[/red]")
            return
        video_title = video_path.stem

    # Optionally transcribe for better results
    transcription_text = ""
    if do_transcribe:
        console.print(f"\n[bold]Transcribing video for metadata generation...[/bold]")
        try:
            from core.transcriber import transcribe_video
            result = transcribe_video(video_path)
            transcription_text = result.text if hasattr(result, "text") else str(result)
            word_count = len(transcription_text.split()) if transcription_text else 0
            console.print(f"  Transcription: {word_count} words")
        except Exception as exc:
            console.print(f"[yellow]Transcription failed: {exc} — using title only[/yellow]")

    if not transcription_text:
        console.print(f"[dim]No transcription available. Generating metadata from title only.[/dim]")
        transcription_text = video_title

    # Generate AI metadata using the local fallback or AI
    console.print(f"\n[bold]Generating AI metadata[/bold] (platform={platform})...")
    try:
        result = generate_ai_metadata(
            transcription_text=transcription_text,
            video_title=video_title,
            platform=platform,
            settings=settings,
        )
    except Exception as exc:
        console.print(f"[red]Metadata generation failed: {exc}[/red]")
        return

    # Display results in a nice table
    meta_table = Table(title="AI-Generated Metadata")
    meta_table.add_column("Field", style="cyan")
    meta_table.add_column("Value", style="green")

    if result.youtube_title:
        meta_table.add_row("YouTube Title", result.youtube_title)
    if result.youtube_description:
        meta_table.add_row("YouTube Description", result.youtube_description[:200] + ("..." if len(result.youtube_description) > 200 else ""))
    if result.tiktok_caption:
        meta_table.add_row("TikTok Caption", result.tiktok_caption)
    if result.reels_caption:
        meta_table.add_row("Reels Caption", result.reels_caption[:200] + ("..." if len(result.reels_caption) > 200 else ""))
    if result.hashtags:
        meta_table.add_row("Hashtags", " ".join(f"#{t}" for t in result.hashtags))
    if result.keywords:
        meta_table.add_row("Keywords", ", ".join(result.keywords))

    meta_table.add_row("SEO Score", f"{result.seo_score:.1f}/100")
    meta_table.add_row("Viral Score", f"{result.viral_score:.1f}/100")
    console.print(meta_table)

    # Save to JSON if --output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Metadata saved to {output_path}[/green]")


# ══════════════════════════════════════════════════════════
#  15. SPEED — Show speed mode comparison
# ══════════════════════════════════════════════════════════

@cli.command()
def speed():
    """Show speed modes and estimated processing times."""
    console.print(_BANNER)

    table = Table(title="Speed Modes", show_lines=True)
    table.add_column("Mode", style="bold")
    table.add_column("CLI Flag", style="cyan")
    table.add_column("Download", style="dim")
    table.add_column("Analysis", style="dim")
    table.add_column("Encoding", style="dim")
    table.add_column("Whisper", style="dim")
    table.add_column("Extras", style="dim")
    table.add_column("Est. Speed", style="green bold")

    table.add_row(
        "[bold yellow]TURBO[/bold yellow]",
        "--turbo --quality fast",
        "720p, 8 fragments",
        "Audio + silence only",
        "ultrafast, CRF 30",
        "tiny (beam=1)",
        "No extras",
        "[bold green]~4x faster[/bold green]",
    )
    table.add_row(
        "Fast",
        "--quality fast",
        "1080p, 4 fragments",
        "3 signals (audio+scene+silence)",
        "ultrafast, CRF 28",
        "base",
        "No face tracking",
        "~2x faster",
    )
    table.add_row(
        "Balanced",
        "--quality balanced",
        "1080p, 4 fragments",
        "7 signals (full Bayesian)",
        "fast, CRF 23",
        "base/small",
        "Face tracking, karaoke subs",
        "1x (default)",
    )
    table.add_row(
        "High",
        "--quality high",
        "1080p, 4 fragments",
        "7 signals (full Bayesian)",
        "slow, CRF 18",
        "medium",
        "Face tracking, 2-pass, film grain",
        "~0.5x (slow)",
    )

    console.print(table)

    # Duration presets
    dur_table = Table(title="Duration Presets")
    dur_table.add_column("Preset", style="cyan")
    dur_table.add_column("Seconds", style="green")
    dur_table.add_column("Best For", style="dim")
    dur_table.add_column("CLI", style="blue")

    dur_table.add_row("quick / 25s", "25s", "TikTok fast scroll", "-d quick  or  -d 25s")
    dur_table.add_row("standard / 45s", "45s", "YouTube Shorts sweet spot", "-d standard  or  -d 45s")
    dur_table.add_row("extended / 3min", "180s", "Long-form Shorts", "-d extended  or  -d 3min")

    console.print(dur_table)

    # Turbo recommendations
    console.print("\n[bold yellow]TURBO MODE Quick Commands:[/bold yellow]")
    console.print("[dim]  python main.py run -u <URL> --turbo                    # Fastest possible[/dim]")
    console.print("[dim]  python main.py run -u <URL> --turbo -d 25s             # Turbo + 25s clip[/dim]")
    console.print("[dim]  python main.py run -u <URL> --turbo -d 45s --no-subs   # Turbo + 45s + no subs[/dim]")
    console.print("[dim]  python main.py run -u <URL> --turbo -d 3min -p youtube  # Turbo + 3min YouTube only[/dim]")
    console.print("[dim]  python main.py run -u <URL> --turbo --no-subs --no-logo # Maximum speed[/dim]")


# ══════════════════════════════════════════════════════════
#  ABOUT — Show developer and project info
# ══════════════════════════════════════════════════════════

@cli.command()
def about():
    """Show project info, credits, and developer details."""
    console.print(_BANNER)

    about_table = Table(title="About YT Shorts Factory", show_lines=True)
    about_table.add_column("Field", style="cyan bold")
    about_table.add_column("Value", style="bold")

    about_table.add_row("Project", "YT Shorts Factory")
    about_table.add_row("Version", "4.0.0")
    about_table.add_row("Developer", "[bold magenta]Abrar Hussain[/bold magenta]")
    about_table.add_row("License", "MIT")
    about_table.add_row("Python", ">= 3.10")
    about_table.add_row("GPU", "NVIDIA GTX 1650 (CUDA + NVENC)")
    about_table.add_row("Description", "Automated video intelligence pipeline for YouTube Shorts, TikTok, Instagram Reels")
    about_table.add_row("Features", "Smart cropping, Whisper AI, FFmpeg, face tracking, channel patterns, multi-platform export")
    about_table.add_row("Speed Modes", "Superfast, Turbo, Fast, Balanced, High Quality")
    about_table.add_row("Platforms", "YouTube Shorts, TikTok, Instagram Reels")
    about_table.add_row("CLI Modes", "Command-line flags + Interactive arrow-key menus")
    about_table.add_row("GPU Accel", "CUDA (Whisper) + NVENC h264_nvenc (FFmpeg encoding)")

    console.print(about_table)

    console.print("\n[bold]Key Technologies:[/bold]")
    tech_tree = Tree("[cyan]yt-shorts-factory[/cyan]")
    gpu = tech_tree.add("[bold green]GPU — NVIDIA GTX 1650[/bold green]")
    gpu.add("CUDA — Whisper AI on GPU (4-8x faster than CPU)")
    gpu.add("NVENC h264_nvenc — FFmpeg hardware encoding (3-5x faster)")
    gpu.add("float16 compute — Optimized for 4GB VRAM")
    gpu.add("faster-whisper — CTranslate2 on CUDA (additional 4x speedup)")

    ai = tech_tree.add("[yellow]AI / ML[/yellow]")
    ai.add("OpenAI Whisper — Speech recognition & transcription")
    ai.add("MediaPipe — Face detection & tracking")

    video = tech_tree.add("[yellow]Video Processing[/yellow]")
    video.add("FFmpeg — Encoding, cropping, effects, subtitles")
    video.add("yt-dlp — Video downloading")
    video.add("OpenCV — Computer vision, smart crop, motion detection")

    ui = tech_tree.add("[yellow]CLI & UI[/yellow]")
    ui.add("Click — Command-line interface framework")
    ui.add("Rich — Terminal formatting, tables, panels, progress bars")
    ui.add("questionary — Interactive arrow-key navigation menus")

    data = tech_tree.add("[yellow]Data & Infrastructure[/yellow]")
    data.add("SQLAlchemy 2.0 — Database ORM & job tracking")
    data.add("Pydantic — Settings validation & configuration")
    data.add("SQLite — Embedded job queue database")

    console.print(tech_tree)

    console.print(f"\n[bold magenta]Developed by Abrar Hussain[/bold magenta]")
    console.print("[dim]  Run 'python main.py' for interactive menu or 'python main.py --help' for all commands[/dim]")


# ══════════════════════════════════════════════════════════
#  GPU — Check GPU status and acceleration
# ══════════════════════════════════════════════════════════

@cli.command()
def gpu():
    """Check GPU status and hardware acceleration availability."""
    console.print("[bold]GPU Status Check[/bold]\n")

    gpu_table = Table(title="GPU Information", show_lines=True)
    gpu_table.add_column("Component", style="cyan bold")
    gpu_table.add_column("Status", style="bold")
    gpu_table.add_column("Detail")

    # PyTorch / CUDA
    try:
        import torch
        gpu_table.add_row("PyTorch", "[green]Installed[/green]", f"v{torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            gpu_table.add_row("CUDA", "[green]Available[/green]", f"CUDA {torch.version.cuda}")
            gpu_table.add_row("GPU", f"[green]{gpu_name}[/green]", f"{gpu_mem:.1f} GB VRAM")
            compute = torch.cuda.get_device_properties(0)
            gpu_table.add_row("Compute Capability", "[green]OK[/green]", f"{compute.major}.{compute.minor}")

            # Recommended Whisper model based on VRAM
            if gpu_mem >= 8.0:
                rec_model = "small or medium (8GB+ VRAM)"
            elif gpu_mem >= 4.0:
                rec_model = "base or small (4GB VRAM)"
            else:
                rec_model = "tiny (< 4GB VRAM)"
            gpu_table.add_row("Recommended Whisper", "[cyan]Suggested[/cyan]", rec_model)
        else:
            gpu_table.add_row("CUDA", "[red]Not available[/red]", "CPU-only mode")
            gpu_table.add_row("Note", "[yellow]Warning[/yellow]", "Install NVIDIA drivers + CUDA toolkit")
    except ImportError:
        gpu_table.add_row("PyTorch", "[red]Not installed[/red]", "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    # faster-whisper
    try:
        import faster_whisper
        gpu_table.add_row("faster-whisper", "[green]Installed[/green]", f"v{getattr(faster_whisper, '__version__', 'unknown')} (CTranslate2, 4x faster)")
    except ImportError:
        gpu_table.add_row("faster-whisper", "[yellow]Not installed[/yellow]", "pip install faster-whisper ctranslate2 (4x faster transcription on GPU)")

    # FFmpeg NVENC
    try:
        import subprocess, shutil
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            result = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-y",
                 "-f", "lavfi", "-i", "color=black:size=64x64:duration=0.1:rate=1",
                 "-c:v", "h264_nvenc", "-frames:v", "1",
                 "-f", "null", "-"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                gpu_table.add_row("NVENC (h264_nvenc)", "[green]Working[/green]", "FFmpeg hardware encoding available (3-5x faster)")
            else:
                gpu_table.add_row("NVENC (h264_nvenc)", "[red]Failed[/red]", "GPU driver may not support NVENC")
        else:
            gpu_table.add_row("FFmpeg", "[red]Not found[/red]", "Install FFmpeg")
    except Exception as exc:
        gpu_table.add_row("NVENC check", "[yellow]Error[/yellow]", str(exc)[:60])

    console.print(gpu_table)

    # Speed recommendations
    console.print("\n[bold]Speed Recommendations for GTX 1650:[/bold]")
    rec_table = Table()
    rec_table.add_column("Mode", style="cyan")
    rec_table.add_column("Whisper", style="green")
    rec_table.add_column("Encoding", style="yellow")
    rec_table.add_column("Est. Speed", style="bold")
    rec_table.add_row("Superfast", "tiny (CUDA)", "NVENC ultrafast", "2-4 min per short")
    rec_table.add_row("Turbo", "tiny (CUDA)", "NVENC ultrafast", "4-8 min per short")
    rec_table.add_row("Fast", "base (CUDA)", "NVENC veryfast", "6-12 min per short")
    rec_table.add_row("Balanced", "base (CUDA)", "NVENC fast", "8-15 min per short")
    rec_table.add_row("High Quality", "small (CUDA)", "NVENC medium", "15-25 min per short")
    console.print(rec_table)

    console.print("\n[dim]Tip: Use --turbo or --superfast for maximum speed on your GTX 1650![/dim]")
    console.print("[dim]  python main.py run -u URL --turbo       # Fast + GPU[/dim]")
    console.print("[dim]  python main.py run -u URL --superfast   # Fastest possible[/dim]")


# ══════════════════════════════════════════════════════════
#  MENU — Interactive menu with arrow-key navigation
# ══════════════════════════════════════════════════════════

@cli.command()
def menu():
    """Launch interactive menu with arrow-key navigation (easy mode)."""
    from utils.interactive_menu import interactive_main
    interactive_main()


if __name__ == "__main__":
    cli()
