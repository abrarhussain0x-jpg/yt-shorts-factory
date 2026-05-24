"""
core/pipeline.py — Master orchestrator for the full shorts creation pipeline.

Coordinates all 13 pipeline steps with timing, error handling, progress
tracking, checkpoint recovery, smart caching, multi-clip support, audio
enhancement, face tracking, content moderation, thumbnail generation,
SRT/VTT export, and intermediate cleanup. Each step is independent and
can be resumed from its checkpoint.

Steps (fatal on failure marked with *):
  1.  DOWNLOAD*           Download video from URL
  2.  ANALYZE*            Multi-signal engagement analysis → SegmentResult(s)
  3.  ENHANCE AUDIO       AudioEnhancer: denoise + normalize + compress
  4.  CONVERT*            9:16 smart crop with face tracking / blur background
  5.  TRANSCRIBE          Whisper transcription with VAD, caching, word timestamps
  6.  BURN SUBTITLES      Generate ASS subtitles + burn into video
  7.  STAMP LOGO          Logo overlay with optional animation
  8.  ADD EFFECTS         Fade in/out, film grain
  9.  MODERATE            Content moderation via ContentModerator
  10. EXPORT PLATFORMS    Multi-platform export (YouTube / TikTok / Reels)
  11. GENERATE METADATA   Keywords, titles, descriptions
  12. GENERATE THUMBNAILS Extract / compose thumbnails
  13. CLEANUP             Remove intermediate files

Non-fatal steps (3, 5, 6, 7, 8, 9) warn and continue on failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from config.settings import Settings, get_settings
from core.analyzer import EngagementAnalyzer, SegmentResult, MultiClipResult
from core.downloader import download_video, DownloadError
from core.logo_stamper import stamp_logo
from core.metadata_generator import generate_metadata, MetadataResult
from core.platform_exporter import export_for_platforms, PlatformExports, ExportReport
from core.shorts_converter import convert_to_shorts, ConverterError
from core.subtitle_engine import generate_subtitles, burn_subtitles
from core.transcriber import (
    transcribe,
    TranscriptionResult,
    export_srt,
    export_vtt,
)
from database.db import init_db, create_job, update_job_status, save_video_record
from utils.file_utils import (
    make_output_path,
    safe_delete,
    cleanup_intermediates,
    sanitize_filename,
    get_file_size_human,
    compute_output_basename,
)
from utils.ffmpeg_utils import FFmpegError, probe_video, get_video_thumbnail, FFmpegProgress
from utils.logger import get_logger
from utils.progress import PipelineProgress

# ── Optional heavy imports ─────────────────────────────────
try:
    from core.face_tracker import FaceTracker
except ImportError as exc:
    raise ImportError(
        "core.face_tracker is required but could not be imported. "
        "Ensure OpenCV (cv2) is installed: pip install opencv-python"
    ) from exc

try:
    from core.motion_detector import MotionDetector
except ImportError as exc:
    raise ImportError(
        "core.motion_detector is required but could not be imported. "
        "Check that the module exists and its dependencies are installed."
    ) from exc

try:
    from core.audio_enhancer import AudioEnhancer, AudioEnhanceError
except ImportError as exc:
    raise ImportError(
        "core.audio_enhancer is required but could not be imported. "
        "Ensure FFmpeg is available on PATH."
    ) from exc

try:
    from core.content_moderator import ContentModerator, FullModerationResult
except ImportError as exc:
    raise ImportError(
        "core.content_moderator is required but could not be imported. "
        "Check that the module and its dependencies are installed."
    ) from exc

try:
    from core.thumbnail_generator import ThumbnailGenerator
except ImportError as exc:
    raise ImportError(
        "core.thumbnail_generator is required but could not be imported. "
        "Ensure Pillow (PIL) is installed: pip install Pillow"
    ) from exc

logger = get_logger("pipeline")
console = Console()


# ═══════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """Result of a single pipeline step."""

    name: str
    status: str  # done | failed | skipped | cached
    duration: float = 0.0
    output_path: str = ""
    detail: str = ""
    error: str = ""
    sub_steps: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Per-run configuration overrides for the pipeline."""

    url: str = ""
    duration: int | None = None
    skip_subs: bool = False
    no_logo: bool = False
    enhance_audio: bool = True
    moderate_content: bool = True
    blur_background: bool = False
    platforms: list[str] = field(default_factory=list)
    animation: str | None = None
    quality: str = "balanced"  # fast / balanced / high
    max_clips: int = 1
    aspect_format: str = "9:16"  # 9:16, 1:1, 4:5


@dataclass
class PipelineResult:
    """Complete result of a pipeline run."""

    success: bool = False
    job_id: str = ""
    outputs: Optional[PlatformExports] = None
    metadata: Optional[MetadataResult] = None
    total_duration_seconds: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    error: str = ""
    quality_metrics: dict = field(default_factory=dict)
    moderation_result: Optional[Any] = None


# ═══════════════════════════════════════════════════════════
#  Step Names (canonical order)
# ═══════════════════════════════════════════════════════════

STEP_NAMES: list[str] = [
    "Download",
    "Analyze",
    "Enhance Audio",
    "Convert",
    "Transcribe",
    "Burn Subtitles",
    "Stamp Logo",
    "Add Effects",
    "Moderate",
    "Export Platforms",
    "Generate Metadata",
    "Generate Thumbnails",
    "Cleanup",
]

# Steps that abort the pipeline on failure
FATAL_STEPS: set[str] = {"Download", "Analyze", "Convert"}


# ═══════════════════════════════════════════════════════════
#  Checkpoint Helpers
# ═══════════════════════════════════════════════════════════

def _save_checkpoint(job_id: str, step: str, data: dict, settings: Settings) -> None:
    """Persist a checkpoint so the pipeline can resume after a crash.

    Args:
        job_id: Unique job identifier.
        step: Step name to checkpoint.
        data: Serializable dict of step outputs.
        settings: Application settings (controls checkpoint dir).
    """
    if not settings.CHECKPOINT_ENABLED:
        return
    checkpoint_dir = Path(settings.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{job_id}.json"
    checkpoint: dict = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    checkpoint[step] = data
    checkpoint_path.write_text(json.dumps(checkpoint, default=str), encoding="utf-8")
    logger.debug("Checkpoint saved: %s -> %s", job_id, step)


def _load_checkpoint(job_id: str, settings: Settings) -> dict:
    """Load a previously saved checkpoint for the given job.

    Args:
        job_id: Unique job identifier.
        settings: Application settings.

    Returns:
        Dict mapping step names to their checkpoint data.
        Returns empty dict if no checkpoint exists.
    """
    checkpoint_path = Path(settings.CHECKPOINT_DIR) / f"{job_id}.json"
    if checkpoint_path.exists():
        try:
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _clear_checkpoint(job_id: str, settings: Settings) -> None:
    """Remove checkpoint file after successful pipeline completion.

    Args:
        job_id: Unique job identifier.
        settings: Application settings.
    """
    checkpoint_path = Path(settings.CHECKPOINT_DIR) / f"{job_id}.json"
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug("Checkpoint cleared: %s", job_id)
        except OSError:
            pass


def _compute_input_hash(*paths: Path) -> str:
    """Hash the contents of one or more files for caching decisions.

    Uses SHA-256 truncated to 32 hex chars for speed and brevity.
    Reads files in 64 KB chunks to handle large video files efficiently.

    Args:
        *paths: Variable number of file paths to hash.

    Returns:
        Hex digest string (32 chars) of the combined file hashes,
        or empty string if no valid paths were given.
    """
    hasher = hashlib.sha256()
    valid_count = 0
    for p in paths:
        if p is None or not p.exists():
            continue
        try:
            file_hash = hashlib.sha256()
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    file_hash.update(chunk)
            hasher.update(file_hash.hexdigest().encode())
            valid_count += 1
        except OSError:
            continue
    if valid_count == 0:
        return ""
    return hasher.hexdigest()[:32]


def _should_skip_step(step_name: str, checkpoint: dict, input_hash: str) -> bool:
    """Determine whether a step can be skipped due to cached results.

    A step is skippable when:
    1. A checkpoint exists for this step, AND
    2. The checkpoint contains an input_hash that matches the current
       input_hash (i.e., inputs have not changed), AND
    3. The checkpoint output_path (if any) still exists on disk.

    Args:
        step_name: Name of the pipeline step.
        checkpoint: Loaded checkpoint dict for the job.
        input_hash: Hash of current step inputs.

    Returns:
        True if the step should be skipped (cached result is valid).
    """
    if step_name not in checkpoint:
        return False
    cp = checkpoint[step_name]
    if not isinstance(cp, dict):
        return False
    # If input_hash is empty, we cannot verify cache validity
    if input_hash and cp.get("input_hash") and cp["input_hash"] != input_hash:
        return False
    # Verify output file still exists (if checkpoint records one)
    output_path = cp.get("output_path", "")
    if output_path and not Path(output_path).exists():
        return False
    return True


# ═══════════════════════════════════════════════════════════
#  Pre-flight Checks
# ═══════════════════════════════════════════════════════════

def _check_disk_space(settings: Settings, required_mb: int = 2048) -> bool:
    """Verify sufficient disk space is available in the output directory.

    Args:
        settings: Application settings.
        required_mb: Minimum required free space in megabytes.

    Returns:
        True if enough disk space is available.
    """
    try:
        target_dir = settings.SHORTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target_dir)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < required_mb:
            logger.error(
                "Insufficient disk space: %.0f MB free, %d MB required",
                free_mb, required_mb,
            )
            return False
        logger.debug("Disk space OK: %.0f MB free", free_mb)
        return True
    except OSError as exc:
        logger.warning("Could not check disk space: %s", exc)
        return True  # Proceed optimistically


def _check_memory(settings: Settings) -> bool:
    """Check if system memory is sufficient for processing.

    Verifies available memory against the configured memory limit.
    Uses psutil if available, otherwise skips the check.

    Args:
        settings: Application settings with PERFORMANCE_MEMORY_LIMIT_MB.

    Returns:
        True if memory is sufficient or cannot be checked.
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        required_mb = settings.PERFORMANCE_MEMORY_LIMIT_MB
        if available_mb < required_mb * 0.5:
            logger.warning(
                "Low memory: %.0f MB available, %d MB recommended",
                available_mb, required_mb,
            )
            return False
        logger.debug("Memory OK: %.0f MB available", available_mb)
        return True
    except ImportError:
        logger.debug("psutil not installed, skipping memory check")
        return True


def _quality_preset(quality: str) -> dict[str, Any]:
    """Return processing parameters for a quality preset.

    Args:
        quality: One of 'fast', 'balanced', 'high', 'turbo'.

    Returns:
        Dict of processing parameters for the given quality level.
    """
    presets: dict[str, dict[str, Any]] = {
        "turbo": {
            "ffmpeg_preset": "ultrafast",
            "ffmpeg_crf": 30,
            "denoise_strength": "light",
            "face_tracking": False,
            "film_grain": 0,
            "two_pass": False,
            "subtitle_animation": "none",
        },
        "fast": {
            "ffmpeg_preset": "ultrafast",
            "ffmpeg_crf": 28,
            "denoise_strength": "light",
            "face_tracking": False,
            "film_grain": 0,
            "two_pass": False,
            "subtitle_animation": "none",
        },
        "balanced": {
            "ffmpeg_preset": "fast",
            "ffmpeg_crf": 23,
            "denoise_strength": "medium",
            "face_tracking": True,
            "film_grain": 0,
            "two_pass": False,
            "subtitle_animation": "karaoke",
        },
        "high": {
            "ffmpeg_preset": "slow",
            "ffmpeg_crf": 18,
            "denoise_strength": "medium",
            "face_tracking": True,
            "film_grain": 4,
            "two_pass": True,
            "subtitle_animation": "karaoke",
        },
    }
    return presets.get(quality, presets["balanced"])


def _aspect_resolution(aspect_format: str) -> tuple[int, int]:
    """Map aspect format string to output resolution.

    Args:
        aspect_format: One of '9:16', '1:1', '4:5'.

    Returns:
        Tuple of (width, height) in pixels.
    """
    mapping: dict[str, tuple[int, int]] = {
        "9:16": (1080, 1920),
        "1:1": (1080, 1080),
        "4:5": (1080, 1350),
    }
    return mapping.get(aspect_format, (1080, 1920))


# ═══════════════════════════════════════════════════════════
#  SegmentResult Serialization for Checkpoints
# ═══════════════════════════════════════════════════════════

def _segment_to_dict(seg: SegmentResult) -> dict:
    """Convert a SegmentResult to a JSON-serializable dict.

    Args:
        seg: SegmentResult instance.

    Returns:
        Dict of all SegmentResult fields.
    """
    return {
        "start_time": seg.start_time,
        "end_time": seg.end_time,
        "energy_score": seg.energy_score,
        "method_used": seg.method_used,
        "audio_peak_time": seg.audio_peak_time,
        "scene_peak_time": seg.scene_peak_time,
        "silence_ratio": seg.silence_ratio,
        "speech_detected": seg.speech_detected,
        "best_crop_x": seg.best_crop_x,
        "best_crop_y": seg.best_crop_y,
        "confidence": seg.confidence,
        "speech_rate_estimate": seg.speech_rate_estimate,
        "music_likelihood": seg.music_likelihood,
        "visual_complexity": seg.visual_complexity,
        "overall_quality_grade": seg.overall_quality_grade,
        "has_emphasis": seg.has_emphasis,
        "motion_energy": seg.motion_energy,
        "spectral_centroid_avg": seg.spectral_centroid_avg,
    }


def _dict_to_segment(d: dict, default_duration: float = 60.0) -> SegmentResult:
    """Reconstruct a SegmentResult from a checkpoint dict.

    Args:
        d: Dict from checkpoint.
        default_duration: Fallback clip duration if end_time missing.

    Returns:
        SegmentResult with restored field values.
    """
    return SegmentResult(
        start_time=d.get("start_time", 0.0),
        end_time=d.get("end_time", default_duration),
        energy_score=d.get("energy_score", 0.0),
        method_used=d.get("method_used", "resumed"),
        audio_peak_time=d.get("audio_peak_time", 0.0),
        scene_peak_time=d.get("scene_peak_time", 0.0),
        silence_ratio=d.get("silence_ratio", 0.0),
        speech_detected=d.get("speech_detected", True),
        best_crop_x=d.get("best_crop_x", -1),
        best_crop_y=d.get("best_crop_y", -1),
        confidence=d.get("confidence", 0.0),
        speech_rate_estimate=d.get("speech_rate_estimate", 0.0),
        music_likelihood=d.get("music_likelihood", 0.0),
        visual_complexity=d.get("visual_complexity", 0.0),
        overall_quality_grade=d.get("overall_quality_grade", "C"),
        has_emphasis=d.get("has_emphasis", False),
        motion_energy=d.get("motion_energy", 0.0),
        spectral_centroid_avg=d.get("spectral_centroid_avg", 0.0),
    )


# ═══════════════════════════════════════════════════════════
#  Helper: run a pipeline step with error handling
# ═══════════════════════════════════════════════════════════

def _run_step(
    step_name: str,
    step_fn,
    is_fatal: bool,
    result: PipelineResult,
    progress: PipelineProgress,
    checkpoint: dict,
    settings: Settings,
    job_id: str,
    input_hash: str = "",
) -> Any:
    """Execute a single pipeline step with timing, checkpointing, and error recovery.

    If the step's checkpoint is valid and inputs haven't changed, the step
    is skipped (status='cached'). On failure, fatal steps abort the pipeline;
    non-fatal steps log a warning and return None.

    Args:
        step_name: Canonical step name (e.g. "Download").
        step_fn: Callable that performs the step. Should return step output.
        is_fatal: If True, failure aborts the entire pipeline.
        result: PipelineResult to append StepResult to.
        progress: PipelineProgress for UI updates.
        checkpoint: Loaded checkpoint dict.
        settings: Application settings.
        job_id: Current job identifier.
        input_hash: Hash of step inputs for cache validation.

    Returns:
        The return value of step_fn, or None on failure/skip.
    """
    step_t = time.time()
    progress.step_start(step_name)

    # Check cache
    if _should_skip_step(step_name, checkpoint, input_hash):
        cp = checkpoint[step_name]
        detail = "cached"
        output_path = cp.get("output_path", "")
        if output_path:
            detail = f"cached ({get_file_size_human(Path(output_path))})"
        progress.step_done(step_name, 0.0, detail)
        result.steps.append(
            StepResult(name=step_name, status="cached", detail=detail, output_path=output_path)
        )
        logger.info("Step %s: cached (skipping)", step_name)
        return cp.get("return_value")

    # Execute the step
    try:
        step_output = step_fn()
        step_duration = time.time() - step_t

        # Save checkpoint
        cp_data = {
            "input_hash": input_hash,
            "timestamp": time.time(),
            "duration": step_duration,
        }
        if isinstance(step_output, dict):
            cp_data.update(step_output)
        elif isinstance(step_output, Path):
            cp_data["output_path"] = str(step_output)
        elif isinstance(step_output, str):
            cp_data["return_value"] = step_output

        _save_checkpoint(job_id, step_name, cp_data, settings)

        detail = cp_data.get("detail", "")
        output_path = cp_data.get("output_path", "")
        if output_path:
            try:
                detail = detail or get_file_size_human(Path(output_path))
            except OSError:
                pass

        progress.step_done(step_name, step_duration, detail)
        result.steps.append(
            StepResult(
                name=step_name,
                status="done",
                duration=step_duration,
                output_path=output_path,
                detail=detail,
            )
        )
        logger.info("Step %s: done (%.1fs)", step_name, step_duration)
        return step_output

    except Exception as exc:
        step_duration = time.time() - step_t
        error_msg = str(exc)

        if is_fatal:
            progress.step_failed(step_name, error_msg[:80])
            result.steps.append(
                StepResult(name=step_name, status="failed", duration=step_duration, error=error_msg)
            )
            result.success = False
            result.error = error_msg
            result.total_duration_seconds = time.time()  # Will be fixed by caller
            logger.error("Step %s FATAL: %s", step_name, error_msg)
        else:
            progress.step_failed(step_name, error_msg[:60])
            result.steps.append(
                StepResult(name=step_name, status="failed", duration=step_duration, error=error_msg)
            )
            logger.warning("Step %s failed (continuing): %s", step_name, error_msg)

        return None


# ═══════════════════════════════════════════════════════════
#  Main Pipeline Entry Point
# ═══════════════════════════════════════════════════════════

def run_pipeline(
    url: str,
    duration: int | None = None,
    skip_subs: bool = False,
    no_logo: bool = False,
    enhance_audio: bool = True,
    moderate_content: bool = True,
    blur_background: bool = False,
    platforms: list[str] | None = None,
    job_id: str | None = None,
    settings: Settings | None = None,
    resume: bool = True,
    animation: str | None = None,
    quality: str = "balanced",
    max_clips: int = 1,
    aspect_format: str = "9:16",
) -> PipelineResult:
    """Execute the full shorts creation pipeline for a single URL.

    Supports checkpoint-based resume: if a previous run crashed partway,
    it will skip already-completed steps and resume from the last checkpoint.
    Non-fatal steps (Enhance Audio, Transcribe, Burn Subtitles, Stamp Logo,
    Add Effects, Moderate) are allowed to fail without stopping the pipeline.

    Args:
        url: YouTube video URL.
        duration: Override clip duration in seconds.
        skip_subs: Skip transcription and subtitle burning.
        no_logo: Skip logo stamping.
        enhance_audio: Apply audio enhancement chain.
        moderate_content: Run content moderation checks.
        blur_background: Use blurred background instead of crop.
        platforms: List of target platforms (youtube, tiktok, reels).
        job_id: Optional existing job ID for resume.
        settings: Optional Settings override.
        resume: Whether to attempt checkpoint-based resume.
        animation: Logo animation type override.
        quality: Processing quality preset (fast/balanced/high).
        max_clips: Maximum number of clips to extract per video.
        aspect_format: Target aspect ratio (9:16, 1:1, 4:5).

    Returns:
        PipelineResult with success flag, outputs, step results, and metrics.
    """
    start_time = time.time()
    result = PipelineResult()
    config = PipelineConfig(
        url=url,
        duration=duration,
        skip_subs=skip_subs,
        no_logo=no_logo,
        enhance_audio=enhance_audio,
        moderate_content=moderate_content,
        blur_background=blur_background,
        platforms=platforms or [],
        animation=animation,
        quality=quality,
        max_clips=max_clips,
        aspect_format=aspect_format,
    )

    # ── Resolve settings ────────────────────────────────────
    if settings is None:
        settings = get_settings()

    # ── Apply turbo mode if enabled ────────────────────────
    is_turbo = settings.TURBO_MODE or settings.SUPERFAST_MODE
    if is_turbo:
        settings.apply_turbo()
        # In turbo mode, force skip non-essential steps
        enhance_audio = False
        moderate_content = False
        logger.info("TURBO MODE: skipping audio enhancement, moderation, thumbnails for speed")

    if config.duration is None:
        config.duration = settings.CLIP_DURATION
    if not config.platforms:
        if settings.EXPORT_YOUTUBE:
            config.platforms.append("youtube")
        if settings.EXPORT_TIKTOK:
            config.platforms.append("tiktok")
        if settings.EXPORT_REELS:
            config.platforms.append("reels")
    if max_clips <= 1 and settings.MULTI_CLIP_MAX_PER_VIDEO > 1:
        config.max_clips = settings.MULTI_CLIP_MAX_PER_VIDEO

    # ── Pre-flight checks ───────────────────────────────────
    if not _check_disk_space(settings, settings.PERFORMANCE_DISK_SPACE_RESERVE_MB):
        result.success = False
        result.error = "Insufficient disk space"
        result.total_duration_seconds = time.time() - start_time
        return result

    _check_memory(settings)  # Log warning but don't abort

    # ── Initialise database and job ─────────────────────────
    init_db()
    if not job_id:
        job = create_job(url=url)
        result.job_id = job.id
    else:
        from database.db import get_job as _get_job
        job = _get_job(job_id)
        if job is None:
            job = create_job(url=url)
            result.job_id = job.id
        else:
            result.job_id = job_id

    update_job_status(result.job_id, "running")

    # ── Load checkpoint for resume ──────────────────────────
    checkpoint = _load_checkpoint(result.job_id, settings) if resume else {}

    # ── Initialise progress tracker ─────────────────────────
    progress = PipelineProgress(
        title=f"Pipeline: {url[:60]}",
        steps=STEP_NAMES,
    )
    progress.start()

    # ── Quality preset ──────────────────────────────────────
    qpreset = _quality_preset(config.quality)

    # ── Working state ───────────────────────────────────────
    intermediates: list[Path] = []
    raw_path: Path | None = None
    video_info_dict: dict[str, Any] = {}
    segments: list[SegmentResult] = []
    transcription: TranscriptionResult | None = None
    current_video: Path | None = None  # Tracks the latest video in the chain

    # Helper to check if a step is done in checkpoint
    def is_step_done(step_name: str) -> bool:
        return step_name in checkpoint

    def fatal_abort(step_name: str, exc: Exception) -> PipelineResult:
        """Record a fatal failure and return the result immediately."""
        step_duration = time.time() - step_t
        progress.step_failed(step_name, str(exc)[:80])
        result.steps.append(
            StepResult(name=step_name, status="failed", duration=step_duration, error=str(exc))
        )
        result.success = False
        result.error = str(exc)
        result.total_duration_seconds = time.time() - start_time
        update_job_status(result.job_id, "failed", error_message=str(exc))
        progress.finish({"Status": "Failed", "Error": str(exc)[:200]})
        return result

    # ══════════════════════════════════════════════════════
    # STEP 1: DOWNLOAD (fatal)
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Download")

    if is_step_done("Download"):
        cp = checkpoint["Download"]
        raw_path = Path(cp.get("path", ""))
        video_info_dict = cp.get("info", {})
        if raw_path and raw_path.exists():
            logger.info("Resuming: Download already complete (%s)", raw_path.name)
            progress.step_done("Download", 0.0, get_file_size_human(raw_path))
            result.steps.append(
                StepResult(name="Download", status="cached", output_path=str(raw_path),
                          detail=get_file_size_human(raw_path))
            )
        else:
            # File gone, re-download
            checkpoint.pop("Download", None)
            raw_path = None

    if not is_step_done("Download") or raw_path is None:
        try:
            raw_path, video_info_dict = download_video(url, settings.DOWNLOADS_DIR, turbo=is_turbo)
            step_duration = time.time() - step_t
            progress.step_done("Download", step_duration, get_file_size_human(raw_path))
            _save_checkpoint(result.job_id, "Download", {
                "path": str(raw_path), "info": video_info_dict,
            }, settings)
            result.steps.append(
                StepResult(name="Download", status="done", duration=step_duration,
                          output_path=str(raw_path), detail=get_file_size_human(raw_path))
            )
        except DownloadError as exc:
            result = fatal_abort("Download", exc)
            return result

    current_video = raw_path

    # ══════════════════════════════════════════════════════
    # STEP 2: ANALYZE (fatal) — multi-clip support
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Analyze")

    if is_step_done("Analyze") and raw_path:
        cp = checkpoint["Analyze"]
        stored_segments = cp.get("segments", [])
        if stored_segments:
            segments = [_dict_to_segment(s, float(config.duration)) for s in stored_segments]
            logger.info(
                "Resuming: Analysis complete (%d segment(s), top: %.1f-%.1f)",
                len(segments), segments[0].start_time, segments[0].end_time,
            )
            detail = f"{len(segments)} clip(s), best score={segments[0].energy_score:.4f}"
            progress.step_done("Analyze", 0.0, detail)
            result.steps.append(StepResult(name="Analyze", status="cached", detail=detail))
        else:
            # Fallback: single segment from old checkpoint format
            seg = _dict_to_segment(cp, float(config.duration))
            segments = [seg]
            progress.step_done("Analyze", 0.0, f"score={seg.energy_score:.4f}")
            result.steps.append(StepResult(name="Analyze", status="cached", detail="resumed"))
    elif raw_path:
        try:
            analyzer = EngagementAnalyzer(raw_path, config.duration)
            num_clips = max(1, config.max_clips)

            if is_turbo:
                # TURBO: Use fast analysis (audio + silence only)
                segments = [analyzer.analyze_fast()]
                logger.info("Turbo fast analysis: 1 segment found")
            elif num_clips > 1:
                multi_result: MultiClipResult = analyzer.analyze_multiple_clips(
                    num_clips=num_clips,
                    min_gap_seconds=settings.MULTI_CLIP_MIN_GAP_SECONDS,
                )
                segments = multi_result.segments
                if not segments:
                    # Fallback: use single analysis
                    segments = [analyzer.analyze()]
                logger.info("Multi-clip analysis: %d segments found", len(segments))
            else:
                segments = [analyzer.analyze()]

            step_duration = time.time() - step_t
            detail = f"{len(segments)} clip(s), best score={segments[0].energy_score:.4f}"
            progress.step_done("Analyze", step_duration, detail)
            _save_checkpoint(result.job_id, "Analyze", {
                "segments": [_segment_to_dict(s) for s in segments],
            }, settings)
            result.steps.append(
                StepResult(name="Analyze", status="done", duration=step_duration, detail=detail)
            )
        except Exception as exc:
            result = fatal_abort("Analyze", exc)
            return result

    if not segments:
        logger.warning("No segments found; using fallback 0-%d", config.duration)
        segments = [SegmentResult(start_time=0.0, end_time=float(config.duration), energy_score=0.0)]

    # ══════════════════════════════════════════════════════
    # From here, we process each segment (single or multi-clip)
    # The pipeline continues with the BEST segment for the main
    # output chain, but all segments get converted.
    # ══════════════════════════════════════════════════════
    primary_segment = segments[0]
    title = video_info_dict.get("title", "untitled")
    safe_title = sanitize_filename(title)
    target_w, target_h = _aspect_resolution(config.aspect_format)

    # ══════════════════════════════════════════════════════
    # STEP 3: ENHANCE AUDIO (non-fatal)
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Enhance Audio")
    enhanced_path: Path | None = None

    if not config.enhance_audio or not raw_path:
        reason = "disabled" if not config.enhance_audio else "no input"
        progress.step_skipped("Enhance Audio", reason)
        result.steps.append(StepResult(name="Enhance Audio", status="skipped", detail=reason))
    else:
        input_hash = _compute_input_hash(raw_path)
        if _should_skip_step("Enhance Audio", checkpoint, input_hash):
            cp = checkpoint["Enhance Audio"]
            enhanced_path = Path(cp.get("output_path", ""))
            if enhanced_path.exists():
                progress.step_done("Enhance Audio", 0.0, "cached")
                result.steps.append(
                    StepResult(name="Enhance Audio", status="cached", output_path=str(enhanced_path))
                )
            else:
                enhanced_path = None  # Re-process

        if enhanced_path is None:
            try:
                enhancer = AudioEnhancer(timeout=settings.FFMPEG_TIMEOUT)
                enhanced_path = make_output_path(settings.SHORTS_DIR, safe_title, "enhanced")
                intermediates.append(enhanced_path)

                # Build enhancement chain: denoise → compress → normalize
                denoise_strength = qpreset.get("denoise_strength", "medium")
                temp_denoised = make_output_path(settings.SHORTS_DIR, safe_title, "denoised")
                intermediates.append(temp_denoised)

                enhancer.reduce_noise(raw_path, temp_denoised, strength=denoise_strength)

                temp_compressed = make_output_path(settings.SHORTS_DIR, safe_title, "compressed")
                intermediates.append(temp_compressed)

                threshold_db = float(settings.AUDIO_COMPRESSION_THRESHOLD.replace("dB", ""))
                enhancer.compress_dynamic_range(
                    temp_denoised, temp_compressed,
                    threshold=threshold_db,
                    ratio=float(settings.AUDIO_COMPRESSION_RATIO),
                )

                enhancer.normalize_loudness(
                    temp_compressed, enhanced_path,
                    target_lufs=settings.AUDIO_NORMALIZER_TARGET_LUFS,
                )

                step_duration = time.time() - step_t
                progress.step_done("Enhance Audio", step_duration, get_file_size_human(enhanced_path))
                _save_checkpoint(result.job_id, "Enhance Audio", {
                    "output_path": str(enhanced_path),
                    "input_hash": input_hash,
                }, settings)
                result.steps.append(
                    StepResult(
                        name="Enhance Audio", status="done", duration=step_duration,
                        output_path=str(enhanced_path), detail=get_file_size_human(enhanced_path),
                        sub_steps=["denoise", "compress", "normalize"],
                    )
                )
                current_video = enhanced_path
            except (AudioEnhanceError, FFmpegError, Exception) as exc:
                logger.warning("Audio enhancement failed (continuing without): %s", exc)
                enhanced_path = None
                current_video = raw_path
                step_duration = time.time() - step_t
                progress.step_failed("Enhance Audio", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Enhance Audio", status="failed", duration=step_duration, error=str(exc))
                )

    # ══════════════════════════════════════════════════════
    # STEP 4: CONVERT (fatal) — face tracking + smart crop
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Convert")
    crop_path: Path | None = None

    # Process each segment
    clip_paths: list[Path] = []

    for clip_idx, segment in enumerate(segments):
        suffix = f"_clip{clip_idx}" if len(segments) > 1 else ""
        clip_output = make_output_path(settings.SHORTS_DIR, safe_title, f"cropped{suffix}")
        intermediates.append(clip_output)

        step_key = f"Convert_clip{clip_idx}" if len(segments) > 1 else "Convert"
        input_hash = _compute_input_hash(current_video or raw_path)

        if _should_skip_step(step_key, checkpoint, input_hash):
            cp = checkpoint.get(step_key, {})
            cached_path = Path(cp.get("output_path", ""))
            if cached_path.exists():
                clip_paths.append(cached_path)
                if clip_idx == 0:
                    crop_path = cached_path
                    logger.info("Resuming: Convert already complete (clip %d)", clip_idx)
                continue

        try:
            # Face tracking for smart crop
            use_face_tracking = qpreset.get("face_tracking", True) and not config.blur_background
            if use_face_tracking and current_video:
                try:
                    tracker = FaceTracker(settings=settings)
                    smart_crop = tracker.compute_smart_crop(
                        current_video or raw_path,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        target_w=target_w,
                        target_h=target_h,
                    )
                    if smart_crop.crop_positions:
                        # Use the first crop position as a hint
                        avg_x = sum(cp.x for cp in smart_crop.crop_positions) // len(smart_crop.crop_positions)
                        segment.best_crop_x = avg_x
                        logger.info(
                            "Face tracking: avg crop_x=%d, detection_rate=%.0f%%",
                            avg_x, smart_crop.face_detection_rate * 100,
                        )
                except Exception as exc:
                    logger.debug("Face tracking failed, using center crop: %s", exc)

            # Convert with smart crop
            fade_dur = 0.5 if config.quality == "high" else 0.0
            convert_to_shorts(
                current_video or raw_path,
                segment,
                clip_output,
                settings,
                use_face_tracking=use_face_tracking,
                blur_background=config.blur_background,
                fade_in=fade_dur,
                fade_out=fade_dur,
                target_fps=30.0,
            )

            clip_paths.append(clip_output)
            if clip_idx == 0:
                crop_path = clip_output

            _save_checkpoint(result.job_id, step_key, {
                "output_path": str(clip_output),
                "input_hash": input_hash,
                "segment_index": clip_idx,
            }, settings)

        except (ConverterError, FFmpegError) as exc:
            if len(segments) > 1:
                logger.warning("Convert failed for clip %d (skipping): %s", clip_idx, exc)
                continue
            # Fatal for single-clip
            result = fatal_abort("Convert", exc)
            return result

    if not clip_paths:
        result = fatal_abort("Convert", Exception("No clips were successfully converted"))
        return result

    crop_path = clip_paths[0]
    current_video = crop_path
    step_duration = time.time() - step_t
    detail = f"{len(clip_paths)} clip(s), {get_file_size_human(crop_path)}"
    progress.step_done("Convert", step_duration, detail)
    result.steps.append(
        StepResult(
            name="Convert", status="done", duration=step_duration,
            output_path=str(crop_path), detail=detail,
            sub_steps=[f"clip_{i}" for i in range(len(clip_paths))],
        )
    )

    # ══════════════════════════════════════════════════════
    # STEP 5: TRANSCRIBE (non-fatal)
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Transcribe")
    transcription_cache_key: str = ""

    if config.skip_subs or not crop_path:
        reason = "disabled" if config.skip_subs else "no input"
        progress.step_skipped("Transcribe", reason)
        result.steps.append(StepResult(name="Transcribe", status="skipped", detail=reason))
    else:
        input_hash = _compute_input_hash(crop_path)
        if _should_skip_step("Transcribe", checkpoint, input_hash):
            cp = checkpoint["Transcribe"]
            transcription_cache_key = cp.get("cache_key", "")
            progress.step_done("Transcribe", 0.0, "cached")
            result.steps.append(
                StepResult(name="Transcribe", status="cached", detail=f"cache_key={transcription_cache_key}")
            )
        else:
            try:
                transcription = transcribe(crop_path, settings)
                transcription_cache_key = _compute_input_hash(crop_path)
                step_duration = time.time() - step_t
                detail = f"{transcription.word_count} words ({transcription.language})"

                # Export SRT and VTT
                srt_sub_steps: list[str] = []
                if transcription and not transcription.is_empty:
                    try:
                        srt_path = settings.SHORTS_DIR / f"{compute_output_basename(title)}.srt"
                        export_srt(transcription, srt_path)
                        srt_sub_steps.append(f"srt:{srt_path.name}")
                        logger.info("SRT exported: %s", srt_path.name)
                    except Exception as exc:
                        logger.debug("SRT export failed: %s", exc)

                    try:
                        vtt_path = settings.SHORTS_DIR / f"{compute_output_basename(title)}.vtt"
                        export_vtt(transcription, vtt_path)
                        srt_sub_steps.append(f"vtt:{vtt_path.name}")
                        logger.info("VTT exported: %s", vtt_path.name)
                    except Exception as exc:
                        logger.debug("VTT export failed: %s", exc)

                progress.step_done("Transcribe", step_duration, detail)
                _save_checkpoint(result.job_id, "Transcribe", {
                    "cache_key": transcription_cache_key,
                    "input_hash": input_hash,
                    "word_count": transcription.word_count,
                    "language": transcription.language,
                }, settings)
                result.steps.append(
                    StepResult(
                        name="Transcribe", status="done", duration=step_duration,
                        detail=detail, sub_steps=srt_sub_steps,
                    )
                )
            except Exception as exc:
                logger.warning("Transcription failed (continuing without subtitles): %s", exc)
                transcription = None
                step_duration = time.time() - step_t
                progress.step_failed("Transcribe", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Transcribe", status="failed", duration=step_duration, error=str(exc))
                )

    # ══════════════════════════════════════════════════════
    # STEP 6: BURN SUBTITLES (non-fatal)
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Burn Subtitles")
    subbed_path: Path = crop_path or Path()

    if config.skip_subs or transcription is None or not transcription.words or not crop_path:
        reason = "disabled" if config.skip_subs else (
            "no transcription" if not transcription else "no words"
        )
        progress.step_skipped("Burn Subtitles", reason)
        result.steps.append(StepResult(name="Burn Subtitles", status="skipped", detail=reason))
    else:
        input_hash = _compute_input_hash(crop_path)
        if _should_skip_step("Burn Subtitles", checkpoint, input_hash):
            cp = checkpoint["Burn Subtitles"]
            cached_path = Path(cp.get("output_path", ""))
            if cached_path.exists():
                subbed_path = cached_path
                progress.step_done("Burn Subtitles", 0.0, "cached")
                result.steps.append(
                    StepResult(name="Burn Subtitles", status="cached", output_path=str(subbed_path))
                )
            else:
                # Re-process
                pass

        if subbed_path == crop_path or subbed_path == Path():
            try:
                ass_path = Path(tempfile.mktemp(suffix=".ass", prefix="subs_"))
                intermediates.append(ass_path)
                generate_subtitles(transcription, ass_path, settings)

                subbed_path = make_output_path(settings.SHORTS_DIR, safe_title, "subbed")
                intermediates.append(subbed_path)

                burn_subtitles(crop_path, ass_path, subbed_path, settings)
                step_duration = time.time() - step_t
                progress.step_done("Burn Subtitles", step_duration, get_file_size_human(subbed_path))
                _save_checkpoint(result.job_id, "Burn Subtitles", {
                    "output_path": str(subbed_path),
                    "input_hash": input_hash,
                }, settings)
                result.steps.append(
                    StepResult(
                        name="Burn Subtitles", status="done", duration=step_duration,
                        output_path=str(subbed_path), detail=get_file_size_human(subbed_path),
                    )
                )
            except FFmpegError as exc:
                logger.warning("Subtitle burning failed (using unsubtitled video): %s", exc)
                subbed_path = crop_path
                step_duration = time.time() - step_t
                progress.step_failed("Burn Subtitles", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Burn Subtitles", status="failed", duration=step_duration, error=str(exc))
                )

    current_video = subbed_path if subbed_path and subbed_path.exists() else crop_path

    # ══════════════════════════════════════════════════════
    # STEP 7: STAMP LOGO (non-fatal)
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Stamp Logo")
    logoed_path: Path = current_video or Path()

    if config.no_logo or not current_video or not current_video.exists():
        reason = "disabled" if config.no_logo else "no input"
        progress.step_skipped("Stamp Logo", reason)
        result.steps.append(StepResult(name="Stamp Logo", status="skipped", detail=reason))
    else:
        input_hash = _compute_input_hash(current_video)
        if _should_skip_step("Stamp Logo", checkpoint, input_hash):
            cp = checkpoint["Stamp Logo"]
            cached_path = Path(cp.get("output_path", ""))
            if cached_path.exists():
                logoed_path = cached_path
                progress.step_done("Stamp Logo", 0.0, "cached")
                result.steps.append(
                    StepResult(name="Stamp Logo", status="cached", output_path=str(logoed_path))
                )
            else:
                pass  # Re-process below

        if logoed_path == current_video:
            try:
                logoed_path = make_output_path(settings.SHORTS_DIR, safe_title, "logoed")
                intermediates.append(logoed_path)

                stamp_logo(
                    current_video,
                    output_path=logoed_path,
                    settings=settings,
                    animation=config.animation,
                )
                step_duration = time.time() - step_t
                progress.step_done("Stamp Logo", step_duration, get_file_size_human(logoed_path))
                _save_checkpoint(result.job_id, "Stamp Logo", {
                    "output_path": str(logoed_path),
                    "input_hash": input_hash,
                }, settings)
                result.steps.append(
                    StepResult(
                        name="Stamp Logo", status="done", duration=step_duration,
                        output_path=str(logoed_path), detail=get_file_size_human(logoed_path),
                    )
                )
            except Exception as exc:
                logger.warning("Logo stamping failed (using un-logoed video): %s", exc)
                logoed_path = current_video
                step_duration = time.time() - step_t
                progress.step_failed("Stamp Logo", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Stamp Logo", status="failed", duration=step_duration, error=str(exc))
                )

    current_video = logoed_path if logoed_path and logoed_path.exists() else current_video

    # ══════════════════════════════════════════════════════
    # STEP 7b: CHANNEL PATTERN (non-fatal)
    #   Apply hook, lower-third, CTA, and outro from pattern
    # ══════════════════════════════════════════════════════
    pattern_path: Path = current_video or Path()

    if settings.CHANNEL_PATTERN and current_video and current_video.exists():
        try:
            from core.channel_pattern import get_pattern, apply_channel_pattern, extract_hook_text

            pattern = get_pattern(settings.CHANNEL_PATTERN)
            channel_name = settings.CHANNEL_NAME or video_info_dict.get("channel", "")

            # Override pattern element settings from CLI flags
            if not settings.CHANNEL_HOOK_ENABLED:
                pattern.hook.enabled = False
            if not settings.CHANNEL_CTA_ENABLED:
                pattern.cta.enabled = False
            if not settings.CHANNEL_OUTRO_ENABLED:
                pattern.outro.enabled = False
            if not settings.CHANNEL_LOWER_THIRD_ENABLED:
                pattern.lower_third.enabled = False

            # Apply pattern subtitle style to settings
            if hasattr(pattern, 'subtitle_style'):
                ss = pattern.subtitle_style
                if ss.font:
                    settings.SUBTITLE_FONT = ss.font
                if ss.animation:
                    settings.SUBTITLE_ANIMATION = ss.animation
                if ss.max_words:
                    settings.SUBTITLE_MAX_WORDS = ss.max_words
                if ss.font_size:
                    settings.SUBTITLE_FONT_SIZE = ss.font_size

            # Extract hook text from transcription
            hook_text = ""
            if pattern.hook.enabled and transcription:
                hook_text = extract_hook_text(transcription)

            logger.info(
                "Applying channel pattern: %s (hook=%s, cta=%s, outro=%s)",
                settings.CHANNEL_PATTERN,
                pattern.hook.enabled, pattern.cta.enabled, pattern.outro.enabled,
            )

            pattern_result = apply_channel_pattern(
                current_video, pattern, settings,
                channel_name=channel_name,
                hook_text=hook_text,
            )

            if pattern_result and pattern_result.exists() and pattern_result != current_video:
                current_video = pattern_result
                logger.info("Channel pattern applied: %s", settings.CHANNEL_PATTERN)
            else:
                logger.debug("Channel pattern made no changes")
        except Exception as exc:
            logger.warning("Channel pattern failed (continuing without): %s", exc)
    elif settings.CHANNEL_PATTERN:
        logger.debug("Channel pattern requested but no video to apply to")

    # ══════════════════════════════════════════════════════
    # STEP 8: ADD EFFECTS (non-fatal)
    #   Fade in/out and optional film grain
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Add Effects")
    effects_path: Path = current_video or Path()
    film_grain_amount = qpreset.get("film_grain", 0)

    if not current_video or not current_video.exists():
        progress.step_skipped("Add Effects", "no input")
        result.steps.append(StepResult(name="Add Effects", status="skipped"))
    elif film_grain_amount == 0 and config.quality != "high":
        # No effects needed — skip step entirely
        progress.step_skipped("Add Effects", "no effects requested")
        result.steps.append(StepResult(name="Add Effects", status="skipped", detail="no effects"))
    else:
        input_hash = _compute_input_hash(current_video)
        if _should_skip_step("Add Effects", checkpoint, input_hash):
            cp = checkpoint["Add Effects"]
            cached_path = Path(cp.get("output_path", ""))
            if cached_path.exists():
                effects_path = cached_path
                progress.step_done("Add Effects", 0.0, "cached")
                result.steps.append(
                    StepResult(name="Add Effects", status="cached", output_path=str(effects_path))
                )

        if effects_path == current_video:
            try:
                effects_path = make_output_path(settings.SHORTS_DIR, safe_title, "effects")
                intermediates.append(effects_path)

                # Build FFmpeg filter chain for effects
                video_info = probe_video(current_video)
                duration = video_info.duration

                vfilters: list[str] = []

                # Fade in/out
                fade_in_dur = 0.5
                fade_out_dur = 0.5
                fade_out_start = max(0.0, duration - fade_out_dur)
                vfilters.append(f"fade=t=in:st=0:d={fade_in_dur}")
                vfilters.append(f"fade=t=out:st={fade_out_start:.3f}:d={fade_out_dur}")

                # Film grain
                if film_grain_amount > 0:
                    vfilters.append(f"noise=alls={film_grain_amount}:allf=t")

                vf_str = ",".join(vfilters)

                from utils.ffmpeg_utils import run_ffmpeg as _run_ffmpeg
                _run_ffmpeg(
                    [
                        "-i", str(current_video),
                        "-vf", vf_str,
                        "-af", "afade=t=in:st=0:d=0.3,afade=t=out:st={:.3f}:d=0.3".format(
                            max(0.0, duration - 0.3)
                        ),
                        "-c:v", settings.FFMPEG_VIDEO_CODEC,
                        "-c:a", settings.FFMPEG_AUDIO_CODEC,
                        "-preset", settings.FFMPEG_PRESET,
                        "-crf", str(settings.FFMPEG_CRF),
                        str(effects_path),
                    ],
                    timeout=settings.FFMPEG_TIMEOUT,
                )

                step_duration = time.time() - step_t
                progress.step_done("Add Effects", step_duration, get_file_size_human(effects_path))
                _save_checkpoint(result.job_id, "Add Effects", {
                    "output_path": str(effects_path),
                    "input_hash": input_hash,
                    "film_grain": film_grain_amount,
                }, settings)
                result.steps.append(
                    StepResult(
                        name="Add Effects", status="done", duration=step_duration,
                        output_path=str(effects_path), detail=get_file_size_human(effects_path),
                        sub_steps=["fade_in", "fade_out"] + (
                            ["film_grain"] if film_grain_amount > 0 else []
                        ),
                    )
                )
            except (FFmpegError, Exception) as exc:
                logger.warning("Effects failed (using video without effects): %s", exc)
                effects_path = current_video
                step_duration = time.time() - step_t
                progress.step_failed("Add Effects", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Add Effects", status="failed", duration=step_duration, error=str(exc))
                )

    current_video = effects_path if effects_path and effects_path.exists() else current_video

    # ══════════════════════════════════════════════════════
    # STEP 9: MODERATE (non-fatal)
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Moderate")
    moderation_result: FullModerationResult | None = None

    if not config.moderate_content or not current_video:
        reason = "disabled" if not config.moderate_content else "no input"
        progress.step_skipped("Moderate", reason)
        result.steps.append(StepResult(name="Moderate", status="skipped", detail=reason))
    else:
        input_hash = _compute_input_hash(current_video)
        if _should_skip_step("Moderate", checkpoint, input_hash):
            cp = checkpoint["Moderate"]
            progress.step_done("Moderate", 0.0, "cached")
            result.steps.append(StepResult(name="Moderate", status="cached", detail="cached"))
            result.moderation_result = cp
        else:
            try:
                sensitivity = settings.CONTENT_MODERATION_SENSITIVITY
                moderator = ContentModerator(settings=settings, sensitivity=sensitivity)

                # Build transcription text for moderation
                trans_text = ""
                if transcription and transcription.words:
                    trans_text = " ".join(w.word for w in transcription.words)

                moderation_result = moderator.moderate_full(
                    current_video,
                    transcription=trans_text,
                )

                step_duration = time.time() - step_t
                detail = f"severity={moderation_result.overall_severity}, rating={moderation_result.content_rating}"
                progress.step_done("Moderate", step_duration, detail)
                _save_checkpoint(result.job_id, "Moderate", {
                    "input_hash": input_hash,
                    "overall_severity": moderation_result.overall_severity,
                    "overall_clean": moderation_result.overall_clean,
                    "content_rating": moderation_result.content_rating,
                    "warning_count": len(moderation_result.warnings),
                    "recommendation_count": len(moderation_result.recommendations),
                }, settings)
                result.moderation_result = moderation_result
                result.quality_metrics["content_rating"] = moderation_result.content_rating
                result.quality_metrics["content_severity"] = moderation_result.overall_severity

                # Log warnings
                for warning in moderation_result.warnings:
                    logger.info("Moderation warning: %s", warning)

                result.steps.append(
                    StepResult(
                        name="Moderate", status="done", duration=step_duration,
                        detail=detail,
                        sub_steps=[
                            f"severity:{moderation_result.overall_severity}",
                            f"rating:{moderation_result.content_rating}",
                            f"warnings:{len(moderation_result.warnings)}",
                        ],
                    )
                )
            except Exception as exc:
                logger.warning("Content moderation failed (continuing): %s", exc)
                step_duration = time.time() - step_t
                progress.step_failed("Moderate", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Moderate", status="failed", duration=step_duration, error=str(exc))
                )

    # ══════════════════════════════════════════════════════
    # STEP 10: EXPORT PLATFORMS
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Export Platforms")

    if not current_video or not current_video.exists():
        progress.step_skipped("Export Platforms", "no input")
        result.steps.append(StepResult(name="Export Platforms", status="skipped"))
    else:
        input_hash = _compute_input_hash(current_video)
        if _should_skip_step("Export Platforms", checkpoint, input_hash):
            cp = checkpoint["Export Platforms"]
            progress.step_done("Export Platforms", 0.0, "cached")
            result.steps.append(StepResult(name="Export Platforms", status="cached"))
        else:
            try:
                exports, export_report = export_for_platforms(current_video, title, settings)
                step_duration = time.time() - step_t
                detail = f"{exports.count} variants"
                progress.step_done("Export Platforms", step_duration, detail)
                result.outputs = exports
                _save_checkpoint(result.job_id, "Export Platforms", {
                    "input_hash": input_hash,
                    "export_count": exports.count,
                    "paths": [str(p) for p in exports.paths if p],
                }, settings)
                result.steps.append(
                    StepResult(
                        name="Export Platforms", status="done", duration=step_duration,
                        detail=detail,
                        sub_steps=[p.parent.name for p in exports.paths if p],
                    )
                )
            except Exception as exc:
                logger.error("Platform export failed: %s", exc)
                step_duration = time.time() - step_t
                progress.step_failed("Export Platforms", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Export Platforms", status="failed", duration=step_duration, error=str(exc))
                )

    # ══════════════════════════════════════════════════════
    # STEP 11: GENERATE METADATA
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Generate Metadata")

    if not settings.GENERATE_METADATA:
        progress.step_skipped("Generate Metadata", "disabled")
        result.steps.append(StepResult(name="Generate Metadata", status="skipped"))
    else:
        try:
            metadata = generate_metadata(
                video_info_dict,
                transcription or TranscriptionResult(),
                settings,
            )
            step_duration = time.time() - step_t

            _save_checkpoint(result.job_id, "Generate Metadata", {
                "keywords_count": len(metadata.keywords),
                "title": metadata.title,
            }, settings)

            progress.step_done("Generate Metadata", step_duration, f"{len(metadata.keywords)} keywords")
            result.metadata = metadata
            result.steps.append(
                StepResult(
                    name="Generate Metadata", status="done", duration=step_duration,
                    detail=f"{len(metadata.keywords)} keywords",
                )
            )
        except Exception as exc:
            logger.warning("Metadata generation failed: %s", exc)
            step_duration = time.time() - step_t
            progress.step_failed("Generate Metadata", str(exc)[:60])
            result.steps.append(
                StepResult(name="Generate Metadata", status="failed", duration=step_duration, error=str(exc))
            )

    # ══════════════════════════════════════════════════════
    # STEP 12: GENERATE THUMBNAILS
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Generate Thumbnails")
    thumbnail_paths: list[Path] = []

    if is_turbo or settings.THUMBNAIL_COUNT <= 0:
        # TURBO: Skip thumbnail generation entirely
        reason = "turbo mode" if is_turbo else "disabled"
        progress.step_skipped("Generate Thumbnails", reason)
        result.steps.append(StepResult(name="Generate Thumbnails", status="skipped", detail=reason))
    elif not current_video or not current_video.exists():
        progress.step_skipped("Generate Thumbnails", "no input")
        result.steps.append(StepResult(name="Generate Thumbnails", status="skipped"))
    else:
        input_hash = _compute_input_hash(current_video)
        if _should_skip_step("Generate Thumbnails", checkpoint, input_hash):
            cp = checkpoint["Generate Thumbnails"]
            cached_paths = cp.get("paths", [])
            if cached_paths:
                thumbnail_paths = [Path(p) for p in cached_paths if Path(p).exists()]
            progress.step_done("Generate Thumbnails", 0.0, "cached")
            result.steps.append(
                StepResult(name="Generate Thumbnails", status="cached", detail=f"{len(thumbnail_paths)} thumbs")
            )
        else:
            try:
                thumb_gen = ThumbnailGenerator(settings=settings)
                count = settings.THUMBNAIL_COUNT
                strategy = settings.THUMBNAIL_SELECTION_STRATEGY.replace("-", "_")

                # Extract best frames
                extracted_frames = thumb_gen.extract_best_frames(
                    current_video,
                    count=count,
                    strategy=strategy,
                )

                # Generate styled thumbnails from extracted frames
                basename = compute_output_basename(title)
                for i, frame_path in enumerate(extracted_frames):
                    thumb_out = settings.THUMBNAILS_DIR / f"{basename}_thumb_{i}.jpg"
                    try:
                        thumb_gen.generate_thumbnail(
                            frame_path,
                            title_text=title,
                            output_path=thumb_out,
                            style="modern",
                            platform="youtube_shorts",
                        )
                        thumbnail_paths.append(thumb_out)
                    except Exception as exc:
                        logger.debug("Thumbnail %d generation failed: %s", i, exc)
                    finally:
                        # Clean up extracted frame
                        safe_delete(frame_path)

                # Also try to extract a thumbnail from the raw video at the best segment time
                if raw_path and primary_segment:
                    try:
                        thumb_ts = primary_segment.start_time + (
                            primary_segment.end_time - primary_segment.start_time
                        ) * 0.3
                        raw_thumb = settings.THUMBNAILS_DIR / f"{basename}_raw_thumb.jpg"
                        get_video_thumbnail(raw_path, thumb_ts, raw_thumb)
                        if raw_thumb.exists():
                            thumbnail_paths.append(raw_thumb)
                            logger.info("Raw thumbnail extracted at %.1fs", thumb_ts)
                    except Exception as exc:
                        logger.debug("Raw thumbnail extraction failed: %s", exc)

                step_duration = time.time() - step_t
                detail = f"{len(thumbnail_paths)} thumbnails"
                progress.step_done("Generate Thumbnails", step_duration, detail)
                _save_checkpoint(result.job_id, "Generate Thumbnails", {
                    "input_hash": input_hash,
                    "paths": [str(p) for p in thumbnail_paths],
                }, settings)
                result.steps.append(
                    StepResult(
                        name="Generate Thumbnails", status="done", duration=step_duration,
                        detail=detail,
                        sub_steps=[p.name for p in thumbnail_paths],
                    )
                )
                result.quality_metrics["thumbnail_count"] = len(thumbnail_paths)
            except Exception as exc:
                logger.warning("Thumbnail generation failed: %s", exc)
                step_duration = time.time() - step_t
                progress.step_failed("Generate Thumbnails", str(exc)[:60])
                result.steps.append(
                    StepResult(name="Generate Thumbnails", status="failed", duration=step_duration, error=str(exc))
                )

    # ══════════════════════════════════════════════════════
    # STEP 13: CLEANUP
    # ══════════════════════════════════════════════════════
    step_t = time.time()
    progress.step_start("Cleanup")

    if settings.CLEANUP_INTERMEDIATES:
        cleaned = cleanup_intermediates(intermediates)
        step_duration = time.time() - step_t
        progress.step_done("Cleanup", step_duration, f"{cleaned} files")
        result.steps.append(
            StepResult(name="Cleanup", status="done", duration=step_duration, detail=f"{cleaned} files")
        )
    else:
        progress.step_skipped("Cleanup", "disabled")
        result.steps.append(StepResult(name="Cleanup", status="skipped"))

    # ══════════════════════════════════════════════════════
    # FINALISE
    # ══════════════════════════════════════════════════════
    result.success = True
    result.total_duration_seconds = time.time() - start_time

    # Compute quality metrics
    done_steps = sum(1 for s in result.steps if s.status in ("done", "cached"))
    total_steps = len(result.steps)
    result.quality_metrics["step_success_rate"] = round(done_steps / max(total_steps, 1), 2)
    result.quality_metrics["total_duration_s"] = round(result.total_duration_seconds, 1)

    if primary_segment:
        result.quality_metrics["energy_score"] = primary_segment.energy_score
        result.quality_metrics["quality_grade"] = primary_segment.overall_quality_grade
        result.quality_metrics["confidence"] = primary_segment.confidence
        result.quality_metrics["silence_ratio"] = primary_segment.silence_ratio

    if transcription:
        result.quality_metrics["word_count"] = transcription.word_count
        result.quality_metrics["language"] = transcription.language
        result.quality_metrics["avg_confidence"] = round(transcription.average_confidence, 2)

    # ── Update database ────────────────────────────────────
    output_paths_str = ""
    if result.outputs:
        paths = [str(p) for p in result.outputs.paths if p]
        output_paths_str = ";".join(paths)

    save_video_record(
        result.job_id,
        data={
            "youtube_id": video_info_dict.get("id", ""),
            "title": title,
            "channel": video_info_dict.get("uploader", ""),
            "duration": video_info_dict.get("duration", 0.0),
            "view_count": video_info_dict.get("view_count", 0),
            "clip_start": primary_segment.start_time if primary_segment else 0.0,
            "clip_end": primary_segment.end_time if primary_segment else 0.0,
            "energy_score": primary_segment.energy_score if primary_segment else 0.0,
            "whisper_model_used": settings.WHISPER_MODEL,
            "word_count": transcription.word_count if transcription else 0,
            "output_youtube": str(result.outputs.youtube_path) if result.outputs and result.outputs.youtube_path else "",
            "output_tiktok": str(result.outputs.tiktok_path) if result.outputs and result.outputs.tiktok_path else "",
            "output_reels": str(result.outputs.reels_path) if result.outputs and result.outputs.reels_path else "",
            "quality_grade": primary_segment.overall_quality_grade if primary_segment else "N/A",
            "content_rating": result.quality_metrics.get("content_rating", ""),
            "pipeline_duration_s": round(result.total_duration_seconds, 1),
            "clip_count": len(segments),
        },
    )
    update_job_status(result.job_id, "done", output_path=output_paths_str)
    _clear_checkpoint(result.job_id, settings)

    # ── Build summary ──────────────────────────────────────
    summary = {
        "Status": "Complete",
        "Steps": f"{done_steps}/{total_steps}",
        "Duration": f"{result.total_duration_seconds:.1f}s",
        "Clips": str(len(segments)),
        "Grade": primary_segment.overall_quality_grade if primary_segment else "N/A",
    }
    if result.outputs:
        for p in result.outputs.paths:
            if p:
                summary[p.parent.name] = p.name
    if thumbnail_paths:
        summary["Thumbnails"] = str(len(thumbnail_paths))

    progress.finish(summary)

    logger.info(
        "Pipeline complete: job=%s, %.1fs, %d/%d steps done, %d clips, grade=%s",
        result.job_id,
        result.total_duration_seconds,
        done_steps,
        total_steps,
        len(segments),
        primary_segment.overall_quality_grade if primary_segment else "N/A",
    )

    # ── Audit trail ────────────────────────────────────────
    try:
        audit_dir = settings.LOGS_DIR / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{result.job_id}_pipeline.json"
        audit_data = {
            "job_id": result.job_id,
            "url": url,
            "timestamp": time.time(),
            "total_duration_s": round(result.total_duration_seconds, 1),
            "success": result.success,
            "clip_count": len(segments),
            "segments": [_segment_to_dict(s) for s in segments],
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration": round(s.duration, 2),
                    "detail": s.detail,
                    "error": s.error,
                    "sub_steps": s.sub_steps,
                }
                for s in result.steps
            ],
            "quality_metrics": result.quality_metrics,
            "config": {
                "quality": config.quality,
                "max_clips": config.max_clips,
                "aspect_format": config.aspect_format,
                "enhance_audio": config.enhance_audio,
                "blur_background": config.blur_background,
                "moderate_content": config.moderate_content,
            },
        }
        audit_path.write_text(
            json.dumps(audit_data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Audit trail saved: %s", audit_path.name)
    except OSError as exc:
        logger.debug("Could not save audit trail: %s", exc)

    return result
