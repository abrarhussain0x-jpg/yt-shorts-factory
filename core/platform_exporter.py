"""
core/platform_exporter.py — Platform-optimised video export with parallel encoding,
two-pass VBR, A/B variants, thumbnail/metadata/subtitle generation, resolution and
codec variants, export validation, and comprehensive reporting.

Generates upload-ready variants for YouTube Shorts, TikTok, Instagram Reels,
Twitter/X, Facebook Reels, and Snapchat with platform-specific codec, profile,
duration, and bitrate requirements.  Supports square (1:1) and 4:5 aspect ratios
alongside the default 9:16 vertical format.

Uses ThreadPoolExecutor for concurrent platform exports and provides detailed
per-platform progress tracking, validation, and reporting.
"""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.settings import Settings, get_settings
from utils.ffmpeg_utils import run_ffmpeg, probe_video, detect_hw_encoder, FFmpegError
from utils.file_utils import sanitize_filename, make_output_path, get_file_size_human
from utils.logger import get_logger

logger = get_logger("platform_exporter")


# ═══════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class PlatformSpec:
    """Full specification for a single platform's encoding requirements.

    Encapsulates every tunable parameter so that adding a new platform
    is a one-liner — no conditional logic required in the encoder.

    Attributes:
        name: Human-readable platform name (e.g. 'YouTube Shorts').
        key: Short identifier used in filenames and settings (e.g. 'youtube').
        max_duration: Maximum allowed clip duration in seconds.
        resolution: (width, height) tuple for the primary output.
        codec: FFmpeg video codec name (e.g. 'libx264').
        profile: H.264 / H.265 profile string (e.g. 'high', 'baseline').
        crf: Constant Rate Factor (0-51, lower = higher quality).
        audio_codec: FFmpeg audio codec name (e.g. 'aac').
        audio_bitrate: Target audio bitrate string (e.g. '192k').
        pixel_format: FFmpeg pixel format (e.g. 'yuv420p').
        extra_ffmpeg_args: Additional FFmpeg arguments specific to this platform.
        max_bitrate: Maximum video bitrate in bits/sec (0 = unlimited).
        output_dir_name: Subdirectory name under output/shorts/.
    """

    name: str
    key: str
    max_duration: float
    resolution: tuple[int, int]
    codec: str = "libx264"
    profile: str = "high"
    crf: int = 23
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    pixel_format: str = "yuv420p"
    extra_ffmpeg_args: list[str] = field(default_factory=list)
    max_bitrate: int = 0
    output_dir_name: str = ""


@dataclass
class ExportResult:
    """Result of a single platform export operation.

    Attributes:
        platform: Platform key (e.g. 'youtube').
        path: Path to the exported video file.
        file_size: File size in bytes.
        duration: Actual duration in seconds of the exported video.
        resolution: (width, height) of the exported video.
        codec: Video codec used in the export.
        crf: CRF value used for encoding.
        validated: Whether the export passed platform validation.
        validation_errors: List of validation error messages (empty if validated).
        variant: Variant label (e.g. 'A', 'B', or '' for single export).
        resolution_label: Label like '1080p' or '720p'.
        codec_label: Label like 'h264' or 'hevc'.
    """

    platform: str
    path: Optional[Path] = None
    file_size: int = 0
    duration: float = 0.0
    resolution: tuple[int, int] = (0, 0)
    codec: str = ""
    crf: int = 0
    validated: bool = False
    validation_errors: list[str] = field(default_factory=list)
    variant: str = ""
    resolution_label: str = "1080p"
    codec_label: str = "h264"

    @property
    def file_size_human(self) -> str:
        """Return human-readable file size string."""
        if self.path and self.path.exists():
            return get_file_size_human(self.path)
        size = self.file_size
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def is_success(self) -> bool:
        """Return True if the export produced a valid file."""
        return self.path is not None and self.path.exists() and self.file_size > 0


@dataclass
class ABVariant:
    """Pair of A/B export variants for quality testing.

    Attributes:
        variant_a_path: Path to Variant A (standard CRF).
        variant_b_path: Path to Variant B (lower CRF / higher quality).
        variant_a_crf: CRF used for Variant A.
        variant_b_crf: CRF used for Variant B.
        variant_a_size: File size of Variant A in bytes.
        variant_b_size: File size of Variant B in bytes.
    """

    variant_a_path: Optional[Path] = None
    variant_b_path: Optional[Path] = None
    variant_a_crf: int = 0
    variant_b_crf: int = 0
    variant_a_size: int = 0
    variant_b_size: int = 0

    @property
    def size_difference_bytes(self) -> int:
        """Return the file size difference (B - A) in bytes."""
        return self.variant_b_size - self.variant_a_size

    @property
    def quality_improvement_estimate(self) -> str:
        """Return a qualitative estimate of quality difference."""
        crf_diff = self.variant_a_crf - self.variant_b_crf
        if crf_diff <= 0:
            return "No improvement"
        elif crf_diff <= 2:
            return "Subtle"
        elif crf_diff <= 5:
            return "Noticeable"
        elif crf_diff <= 8:
            return "Significant"
        else:
            return "Major"


@dataclass
class PlatformExports:
    """All outputs from a multi-platform export run.

    Attributes:
        results: All ExportResult objects (one per platform/variant/resolution).
        thumbnail_paths: Paths to generated thumbnail images per platform.
        metadata_paths: Paths to generated metadata JSON files per platform.
        subtitle_paths: Paths to generated subtitle (SRT/VTT) files per platform.
        ab_variants: A/B variant pairs per platform key.
    """

    results: list[ExportResult] = field(default_factory=list)
    thumbnail_paths: dict[str, list[Path]] = field(default_factory=dict)
    metadata_paths: dict[str, Path] = field(default_factory=dict)
    subtitle_paths: dict[str, list[Path]] = field(default_factory=dict)
    ab_variants: dict[str, ABVariant] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Return number of successful exports."""
        return sum(1 for r in self.results if r.is_success)

    @property
    def total_results(self) -> int:
        """Return total number of export results (including failures)."""
        return len(self.results)

    @property
    def paths(self) -> list[Path]:
        """Return list of all successful export paths."""
        return [r.path for r in self.results if r.is_success and r.path is not None]

    @property
    def failed_platforms(self) -> list[str]:
        """Return list of platform keys that failed export."""
        return [r.platform for r in self.results if not r.is_success]

    @property
    def validated_platforms(self) -> list[str]:
        """Return list of platform keys that passed validation."""
        return [r.platform for r in self.results if r.validated]

    @property
    def validation_failed_platforms(self) -> list[str]:
        """Return list of platform keys that failed validation."""
        return [r.platform for r in self.results if r.is_success and not r.validated]

    def get_results_for_platform(self, platform_key: str) -> list[ExportResult]:
        """Return all ExportResult objects for a given platform key."""
        return [r for r in self.results if r.platform == platform_key]

    def get_primary_paths(self) -> dict[str, Path]:
        """Return a mapping of platform key to primary (first successful) export path."""
        primary: dict[str, Path] = {}
        for r in self.results:
            if r.is_success and r.platform not in primary:
                primary[r.platform] = r.path  # type: ignore[assignment]
        return primary


@dataclass
class ExportReport:
    """Comprehensive summary of a multi-platform export run.

    Attributes:
        timestamp: ISO timestamp of when the export was initiated.
        source_path: Path to the source video.
        title: Video title used for output filenames.
        total_results: Total number of export results.
        successful: Number of successful exports.
        failed: Number of failed exports.
        validated: Number of exports that passed validation.
        results: All individual ExportResult objects.
        platform_summaries: Per-platform summary dicts.
        total_output_size_bytes: Combined size of all successful exports.
        export_duration_seconds: Wall-clock time for the full export run.
    """

    timestamp: str = ""
    source_path: str = ""
    title: str = ""
    total_results: int = 0
    successful: int = 0
    failed: int = 0
    validated: int = 0
    results: list[ExportResult] = field(default_factory=list)
    platform_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_output_size_bytes: int = 0
    export_duration_seconds: float = 0.0

    @property
    def total_output_size_human(self) -> str:
        """Return human-readable total output size."""
        size = self.total_output_size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serialisable dictionary."""
        return {
            "timestamp": self.timestamp,
            "source_path": self.source_path,
            "title": self.title,
            "total_results": self.total_results,
            "successful": self.successful,
            "failed": self.failed,
            "validated": self.validated,
            "total_output_size_bytes": self.total_output_size_bytes,
            "total_output_size_human": self.total_output_size_human,
            "export_duration_seconds": round(self.export_duration_seconds, 2),
            "platform_summaries": self.platform_summaries,
            "results": [
                {
                    "platform": r.platform,
                    "path": str(r.path) if r.path else "",
                    "file_size": r.file_size,
                    "file_size_human": r.file_size_human,
                    "duration": r.duration,
                    "resolution": f"{r.resolution[0]}x{r.resolution[1]}",
                    "codec": r.codec,
                    "crf": r.crf,
                    "validated": r.validated,
                    "validation_errors": r.validation_errors,
                    "variant": r.variant,
                    "resolution_label": r.resolution_label,
                    "codec_label": r.codec_label,
                }
                for r in self.results
            ],
        }

    def to_text(self) -> str:
        """Generate a human-readable report string."""
        lines: list[str] = [
            "=" * 60,
            "  EXPORT REPORT",
            "=" * 60,
            f"  Timestamp:     {self.timestamp}",
            f"  Source:        {self.source_path}",
            f"  Title:         {self.title}",
            f"  Duration:      {self.export_duration_seconds:.1f}s",
            "-" * 60,
            f"  Total exports: {self.total_results}",
            f"  Successful:    {self.successful}",
            f"  Failed:        {self.failed}",
            f"  Validated:     {self.validated}",
            f"  Total size:    {self.total_output_size_human}",
            "-" * 60,
        ]

        for platform_key, summary in self.platform_summaries.items():
            lines.append(f"  [{platform_key.upper()}]")
            for k, v in summary.items():
                lines.append(f"    {k}: {v}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Platform Specifications
# ═══════════════════════════════════════════════════════════════

PLATFORM_SPECS: dict[str, PlatformSpec] = {
    "youtube": PlatformSpec(
        name="YouTube Shorts",
        key="youtube",
        max_duration=180.0,  # YouTube Shorts now supports up to 3 minutes
        resolution=(1080, 1920),
        codec="libx264",
        profile="high",
        crf=23,
        audio_codec="aac",
        audio_bitrate="192k",
        pixel_format="yuv420p",
        extra_ffmpeg_args=["-movflags", "+faststart"],
        output_dir_name="youtube",
    ),
    "tiktok": PlatformSpec(
        name="TikTok",
        key="tiktok",
        max_duration=180.0,  # TikTok supports up to 3 minutes (10 min for some accounts)
        resolution=(1080, 1920),
        codec="libx264",
        profile="baseline",
        crf=22,
        audio_codec="aac",
        audio_bitrate="192k",
        pixel_format="yuv420p",
        extra_ffmpeg_args=["-bf", "0", "-movflags", "+faststart"],
        output_dir_name="tiktok",
    ),
    "instagram": PlatformSpec(
        name="Instagram Reels",
        key="instagram",
        max_duration=90.0,  # Instagram Reels up to 90 seconds
        resolution=(1080, 1920),
        codec="libx264",
        profile="main",
        crf=22,
        audio_codec="aac",
        audio_bitrate="128k",
        pixel_format="yuv420p",
        extra_ffmpeg_args=["-movflags", "+faststart"],
        output_dir_name="reels",
    ),
    "twitter": PlatformSpec(
        name="Twitter/X",
        key="twitter",
        max_duration=140.0,
        resolution=(1080, 1920),
        codec="libx264",
        profile="main",
        crf=24,
        audio_codec="aac",
        audio_bitrate="192k",
        pixel_format="yuv420p",
        extra_ffmpeg_args=["-movflags", "+faststart", "-maxrate", "25M", "-bufsize", "25M"],
        max_bitrate=25_000_000,
        output_dir_name="twitter",
    ),
    "facebook": PlatformSpec(
        name="Facebook Reels",
        key="facebook",
        max_duration=60.0,
        resolution=(1080, 1920),
        codec="libx264",
        profile="high",
        crf=23,
        audio_codec="aac",
        audio_bitrate="192k",
        pixel_format="yuv420p",
        extra_ffmpeg_args=["-movflags", "+faststart"],
        output_dir_name="facebook",
    ),
    "snapchat": PlatformSpec(
        name="Snapchat",
        key="snapchat",
        max_duration=60.0,
        resolution=(1080, 1920),
        codec="libx264",
        profile="baseline",
        crf=22,
        audio_codec="aac",
        audio_bitrate="192k",
        pixel_format="yuv420p",
        extra_ffmpeg_args=["-bf", "0", "-movflags", "+faststart"],
        output_dir_name="snapchat",
    ),
}

# Alternative aspect ratio resolutions
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),   # Default vertical
    "1:1": (1080, 1080),    # Square
    "4:5": (1080, 1350),    # Instagram portrait
}

# Resolution variant presets
RESOLUTION_VARIANTS: dict[str, tuple[int, int]] = {
    "1080p": (1080, 1920),
    "720p": (720, 1280),
}

# Codec variant presets with FFmpeg codec names
CODEC_VARIANTS: dict[str, dict[str, str]] = {
    "h264": {
        "codec": "libx264",
        "ext": "mp4",
        "pixel_format": "yuv420p",
    },
    "hevc": {
        "codec": "libx265",
        "ext": "mp4",
        "pixel_format": "yuv420p",
    },
    "vp9": {
        "codec": "libvpx-vp9",
        "ext": "webm",
        "pixel_format": "yuv420p",
    },
}


# ═══════════════════════════════════════════════════════════════
#  Main Export Function
# ═══════════════════════════════════════════════════════════════

def export_for_platforms(
    source_path: Path,
    title: str,
    settings: Settings | None = None,
    platforms: list[str] | None = None,
    aspect_ratio: str = "9:16",
    enable_ab_variants: bool = False,
    ab_crf_offset: int = 4,
    enable_two_pass: bool = False,
    resolution_variants: list[str] | None = None,
    codec_variants: list[str] | None = None,
    generate_thumbnails: bool = True,
    generate_metadata: bool = True,
    generate_subtitles: bool = True,
    validate_exports: bool = True,
    subtitle_srt_path: Path | None = None,
    subtitle_vtt_path: Path | None = None,
    metadata_dict: dict[str, Any] | None = None,
) -> tuple[PlatformExports, ExportReport]:
    """Export a video file for each enabled platform concurrently.

    This is the primary entry point. It orchestrates parallel platform
    exports, optional A/B variants, two-pass encoding, resolution and
    codec variants, thumbnail/metadata/subtitle generation, validation,
    and reporting.

    Args:
        source_path: Path to the source video (already vertical/square).
        title: Video title for output filenames.
        settings: Optional Settings override.
        platforms: List of platform keys to export for. If None, uses
            settings to determine enabled platforms.
        aspect_ratio: Target aspect ratio ('9:16', '1:1', '4:5').
        enable_ab_variants: Generate A/B CRF variants for testing.
        ab_crf_offset: CRF offset for Variant B (lower = higher quality).
        enable_two_pass: Use two-pass VBR encoding for higher quality.
        resolution_variants: List of resolution labels (e.g. ['1080p', '720p']).
        codec_variants: List of codec labels (e.g. ['h264', 'hevc']).
        generate_thumbnails: Export thumbnails for each platform.
        generate_metadata: Export metadata JSON for each platform.
        generate_subtitles: Export SRT/VTT alongside videos.
        validate_exports: Validate exported files against platform specs.
        subtitle_srt_path: Existing SRT file to copy alongside exports.
        subtitle_vtt_path: Existing VTT file to copy alongside exports.
        metadata_dict: Existing metadata dict to write as JSON.

    Returns:
        Tuple of (PlatformExports, ExportReport).
    """
    import time as _time

    start_ts = _time.monotonic()

    if settings is None:
        settings = get_settings()

    if not source_path.exists():
        raise FFmpegError(f"Source video not found: {source_path}")

    exports = PlatformExports()
    safe_title = sanitize_filename(title, max_length=80)

    # Probe source for duration and dimensions
    video_info = probe_video(source_path)
    logger.info(
        "Source: %s (%dx%d, %.1fs, %s)",
        source_path.name, video_info.width, video_info.height,
        video_info.duration, video_info.video_codec,
    )

    # Determine enabled platforms
    if platforms is None:
        platforms = _get_enabled_platforms(settings)

    if not platforms:
        logger.warning("No platforms enabled for export")
        report = _build_report(exports, source_path, title, start_ts)
        return exports, report

    # Default resolution/codec variants if not specified
    if resolution_variants is None:
        resolution_variants = ["1080p"]
    if codec_variants is None:
        codec_variants = ["h264"]

    # Build export task list
    tasks: list[dict[str, Any]] = []
    for platform_key in platforms:
        spec = PLATFORM_SPECS.get(platform_key)
        if spec is None:
            logger.warning("Unknown platform key: %s — skipping", platform_key)
            continue

        for res_label in resolution_variants:
            res_tuple = RESOLUTION_VARIANTS.get(res_label, spec.resolution)
            for codec_label in codec_variants:
                codec_info = CODEC_VARIANTS.get(codec_label, CODEC_VARIANTS["h264"])

                # Primary export
                tasks.append({
                    "platform_key": platform_key,
                    "spec": spec,
                    "safe_title": safe_title,
                    "resolution": res_tuple,
                    "resolution_label": res_label,
                    "codec_info": codec_info,
                    "codec_label": codec_label,
                    "variant": "",
                    "crf_override": None,
                    "aspect_ratio": aspect_ratio,
                    "two_pass": enable_two_pass,
                })

                # A/B variant B (lower CRF)
                if enable_ab_variants:
                    tasks.append({
                        "platform_key": platform_key,
                        "spec": spec,
                        "safe_title": safe_title,
                        "resolution": res_tuple,
                        "resolution_label": res_label,
                        "codec_info": codec_info,
                        "codec_label": codec_label,
                        "variant": "B",
                        "crf_override": max(0, spec.crf - ab_crf_offset),
                        "aspect_ratio": aspect_ratio,
                        "two_pass": enable_two_pass,
                    })

    if not tasks:
        logger.warning("No export tasks generated")
        report = _build_report(exports, source_path, title, start_ts)
        return exports, report

    logger.info("Exporting %d task(s) across %d platform(s)", len(tasks), len(platforms))

    # Execute exports — parallel if multiple tasks, serial if one
    if len(tasks) <= 1:
        for task in tasks:
            result = _execute_export_task(source_path, task, settings, video_info)
            exports.results.append(result)
    else:
        max_workers = min(len(tasks), settings.MAX_CONCURRENT_JOBS + 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task in tasks:
                future = executor.submit(
                    _execute_export_task, source_path, task, settings, video_info,
                )
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    exports.results.append(result)
                    logger.info(
                        "Completed: %s/%s/%s %s (%s)",
                        result.platform, result.resolution_label,
                        result.codec_label, result.variant or "primary",
                        result.file_size_human,
                    )
                except Exception as exc:
                    platform_key = task["platform_key"]
                    logger.error(
                        "%s export failed: %s", platform_key.title(), exc,
                    )
                    exports.results.append(ExportResult(
                        platform=platform_key,
                        validation_errors=[str(exc)],
                    ))

    # Post-processing: thumbnails, metadata, subtitles, validation, A/B pairing
    _post_process_exports(
        exports=exports,
        source_path=source_path,
        safe_title=safe_title,
        settings=settings,
        enable_ab_variants=enable_ab_variants,
        generate_thumbnails=generate_thumbnails,
        generate_metadata=generate_metadata,
        generate_subtitles=generate_subtitles,
        validate_exports=validate_exports,
        subtitle_srt_path=subtitle_srt_path,
        subtitle_vtt_path=subtitle_vtt_path,
        metadata_dict=metadata_dict,
        title=title,
        video_info=video_info,
        aspect_ratio=aspect_ratio,
    )

    report = _build_report(exports, source_path, title, start_ts)
    logger.info(
        "Export complete: %d/%d successful, %d validated, total %s",
        exports.count, exports.total_results,
        len(exports.validated_platforms),
        report.total_output_size_human,
    )

    return exports, report


# ═══════════════════════════════════════════════════════════════
#  Export Task Execution
# ═══════════════════════════════════════════════════════════════

def _execute_export_task(
    source_path: Path,
    task: dict[str, Any],
    settings: Settings,
    video_info: Any,
) -> ExportResult:
    """Execute a single platform export task.

    Handles resolution scaling, codec selection, two-pass encoding,
    duration clipping, and hardware-acceleration fallback.

    Args:
        source_path: Path to the source video.
        task: Task descriptor dictionary from the task list.
        settings: Application settings.
        video_info: Probed VideoInfo from the source.

    Returns:
        ExportResult with details of the export.
    """
    platform_key: str = task["platform_key"]
    spec: PlatformSpec = task["spec"]
    safe_title: str = task["safe_title"]
    target_resolution: tuple[int, int] = task["resolution"]
    resolution_label: str = task["resolution_label"]
    codec_info: dict[str, str] = task["codec_info"]
    codec_label: str = task["codec_label"]
    variant: str = task["variant"]
    crf_override: int | None = task["crf_override"]
    aspect_ratio: str = task["aspect_ratio"]
    two_pass: bool = task["two_pass"]

    # Determine effective CRF
    effective_crf = crf_override if crf_override is not None else spec.crf

    # Determine effective resolution from aspect ratio
    base_w, base_h = target_resolution
    if aspect_ratio == "1:1":
        effective_resolution = (base_w, base_w)
    elif aspect_ratio == "4:5":
        effective_resolution = (base_w, int(base_w * 5 / 4))
    else:
        effective_resolution = (base_w, base_h)

    # Ensure dimensions are even (required by H.264)
    eff_w = effective_resolution[0] if effective_resolution[0] % 2 == 0 else effective_resolution[0] + 1
    eff_h = effective_resolution[1] if effective_resolution[1] % 2 == 0 else effective_resolution[1] + 1

    # Output directory
    output_dir_name = spec.output_dir_name or platform_key
    output_dir = settings.SHORTS_DIR / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename
    suffix_parts = [platform_key]
    if resolution_label != "1080p":
        suffix_parts.append(resolution_label)
    if codec_label != "h264":
        suffix_parts.append(codec_label)
    if variant:
        suffix_parts.append(f"var{variant}")
    suffix = "_".join(suffix_parts)

    ext = codec_info.get("ext", "mp4")
    output_path = make_output_path(output_dir, safe_title, suffix, ext=ext)

    # Determine codec — try hardware encoder first
    video_codec = codec_info["codec"]
    pixel_format = codec_info["pixel_format"]
    hw_encoder = ""
    hw_preset = ""

    if video_codec == "libx264":
        hw_encoder, hw_preset = detect_hw_encoder()
        if hw_encoder:
            video_codec = hw_encoder
    effective_preset = hw_preset if hw_encoder else settings.FFMPEG_PRESET

    # Duration clipping
    max_duration = spec.max_duration
    needs_trim = video_info.duration > max_duration

    # Build FFmpeg command
    cmd: list[str] = ["ffmpeg", "-i", str(source_path)]

    if needs_trim:
        cmd.extend(["-t", str(max_duration)])

    # Video filter for scaling if resolution differs from source
    vf_parts: list[str] = []
    if (video_info.width, video_info.height) != (eff_w, eff_h):
        vf_parts.append(f"scale={eff_w}:{eff_h}:force_original_aspect_ratio=decrease,pad={eff_w}:{eff_h}:(ow-iw)/2:(oh-ih)/2")
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    # Two-pass encoding
    if two_pass and video_codec in ("libx264", "libx265"):
        _run_two_pass(
            cmd=cmd,
            source_path=source_path,
            output_path=output_path,
            video_codec=video_codec,
            preset=effective_preset,
            crf=effective_crf,
            profile=spec.profile,
            audio_codec=spec.audio_codec,
            audio_bitrate=spec.audio_bitrate,
            pixel_format=pixel_format,
            extra_args=spec.extra_ffmpeg_args,
            threads=settings.FFMPEG_THREADS,
            timeout=settings.FFMPEG_TIMEOUT,
            platform_name=spec.name,
            safe_title=safe_title,
            needs_trim=needs_trim,
            max_duration=max_duration,
            vf_parts=vf_parts,
        )
    else:
        # Single-pass encoding
        cmd.extend(["-c:v", video_codec])

        # Profile (only for x264/x265)
        if spec.profile and video_codec in ("libx264", "libx265"):
            cmd.extend(["-profile:v", spec.profile])

        if effective_preset:
            cmd.extend(["-preset", effective_preset])

        cmd.extend(["-crf", str(effective_crf)])

        # Bitrate cap
        if spec.max_bitrate > 0:
            cmd.extend(["-maxrate", str(spec.max_bitrate), "-bufsize", str(spec.max_bitrate)])

        cmd.extend([
            "-c:a", spec.audio_codec,
            "-b:a", spec.audio_bitrate,
            "-pix_fmt", pixel_format,
        ])

        # Platform-specific extra args
        cmd.extend(spec.extra_ffmpeg_args)

        cmd.extend(["-threads", str(settings.FFMPEG_THREADS), str(output_path)])

        description = f"Export {spec.name}: {safe_title}"
        if variant:
            description += f" (Variant {variant})"

        try:
            run_ffmpeg(cmd, description=description, timeout=settings.FFMPEG_TIMEOUT)
        except FFmpegError:
            # Fallback to software encoding if HW encoder failed
            if hw_encoder:
                logger.warning(
                    "HW encoder '%s' failed for %s — falling back to software",
                    hw_encoder, spec.name,
                )
                sw_cmd = list(cmd)
                sw_codec = codec_info["codec"]
                idx = sw_cmd.index(video_codec)
                sw_cmd[idx] = sw_codec
                if effective_preset and hw_preset:
                    idx_p = sw_cmd.index(hw_preset)
                    sw_cmd[idx_p] = settings.FFMPEG_PRESET
                run_ffmpeg(sw_cmd, description=f"Export {spec.name} (SW fallback)", timeout=settings.FFMPEG_TIMEOUT)
            else:
                raise

    # Build result
    result = ExportResult(
        platform=platform_key,
        path=output_path,
        file_size=output_path.stat().st_size if output_path.exists() else 0,
        duration=min(video_info.duration, max_duration),
        resolution=(eff_w, eff_h),
        codec=video_codec,
        crf=effective_crf,
        variant=variant,
        resolution_label=resolution_label,
        codec_label=codec_label,
    )

    logger.info(
        "%s: %s (%s, %dx%d, CRF %d)",
        spec.name, output_path.name, result.file_size_human,
        eff_w, eff_h, effective_crf,
    )

    return result


# ═══════════════════════════════════════════════════════════════
#  Two-Pass Encoding
# ═══════════════════════════════════════════════════════════════

def _run_two_pass(
    cmd: list[str],
    source_path: Path,
    output_path: Path,
    video_codec: str,
    preset: str,
    crf: int,
    profile: str,
    audio_codec: str,
    audio_bitrate: str,
    pixel_format: str,
    extra_args: list[str],
    threads: int,
    timeout: int,
    platform_name: str,
    safe_title: str,
    needs_trim: bool,
    max_duration: float,
    vf_parts: list[str],
) -> None:
    """Run two-pass VBR encoding for higher quality at the same bitrate.

    Pass 1 analyses the video and writes statistics to a temporary
    log file.  Pass 2 uses those statistics for optimal bitrate
    allocation.

    Args:
        cmd: Base FFmpeg command list to extend.
        source_path: Path to the source video.
        output_path: Destination path for the encoded video.
        video_codec: Video codec name (e.g. 'libx264').
        preset: Encoding preset (e.g. 'fast').
        crf: Target CRF value.
        profile: H.264/H.265 profile.
        audio_codec: Audio codec name.
        audio_bitrate: Audio bitrate string.
        pixel_format: Pixel format string.
        extra_args: Platform-specific extra FFmpeg arguments.
        threads: Number of threads (0 = auto).
        timeout: FFmpeg timeout in seconds.
        platform_name: Human-readable platform name for logging.
        safe_title: Sanitised video title.
        needs_trim: Whether the video needs duration trimming.
        max_duration: Maximum allowed duration.
        vf_parts: Video filter parts for scaling.
    """
    # Create temp passlog file
    passlog_fd, passlog_path = tempfile.mkstemp(suffix=".log", prefix="ffmpeg2pass_")
    passlog_file = Path(passlog_path)

    try:
        # ── Pass 1: Analysis ────────────────────────────
        pass1_cmd: list[str] = ["ffmpeg", "-i", str(source_path)]
        if needs_trim:
            pass1_cmd.extend(["-t", str(max_duration)])
        if vf_parts:
            pass1_cmd.extend(["-vf", ",".join(vf_parts)])

        pass1_cmd.extend([
            "-c:v", video_codec,
            "-profile:v", profile,
            "-preset", preset,
            "-crf", str(crf),
            "-pass", "1",
            "-passlogfile", str(passlog_file),
            "-an",
            "-f", "null",
            "-",
        ])

        logger.info("Two-pass Pass 1: analysing %s for %s", source_path.name, platform_name)
        run_ffmpeg(
            pass1_cmd,
            description=f"Two-pass P1: {platform_name} — {safe_title}",
            timeout=timeout,
        )

        # ── Pass 2: Encode ──────────────────────────────
        pass2_cmd: list[str] = ["ffmpeg", "-i", str(source_path)]
        if needs_trim:
            pass2_cmd.extend(["-t", str(max_duration)])
        if vf_parts:
            pass2_cmd.extend(["-vf", ",".join(vf_parts)])

        pass2_cmd.extend([
            "-c:v", video_codec,
            "-profile:v", profile,
            "-preset", preset,
            "-crf", str(crf),
            "-pass", "2",
            "-passlogfile", str(passlog_file),
            "-c:a", audio_codec,
            "-b:a", audio_bitrate,
            "-pix_fmt", pixel_format,
        ])
        pass2_cmd.extend(extra_args)
        pass2_cmd.extend(["-threads", str(threads), str(output_path)])

        logger.info("Two-pass Pass 2: encoding %s for %s", source_path.name, platform_name)
        run_ffmpeg(
            pass2_cmd,
            description=f"Two-pass P2: {platform_name} — {safe_title}",
            timeout=timeout,
        )

    finally:
        # Clean up passlog files
        for f in passlog_file.parent.glob(f"{passlog_file.name}*"):
            try:
                f.unlink()
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════
#  Export Validation
# ═══════════════════════════════════════════════════════════════

def validate_export(result: ExportResult, spec: PlatformSpec) -> tuple[bool, list[str]]:
    """Validate an exported video file against platform specifications.

    Checks resolution, duration, codec, bitrate, and audio format.
    Returns a list of error messages for any non-compliant aspects.

    Args:
        result: ExportResult with path to the exported file.
        spec: PlatformSpec to validate against.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: list[str] = []

    if not result.path or not result.path.exists():
        return False, ["Export file does not exist"]

    try:
        info = probe_video(result.path)
    except Exception as exc:
        return False, [f"Cannot probe exported file: {exc}"]

    # Resolution check
    expected_w, expected_h = spec.resolution
    if info.width != expected_w or info.height != expected_h:
        # Allow close match within 2 pixels (rounding)
        if abs(info.width - expected_w) > 2 or abs(info.height - expected_h) > 2:
            errors.append(
                f"Resolution mismatch: expected {expected_w}x{expected_h}, "
                f"got {info.width}x{info.height}"
            )

    # Duration check
    if info.duration > spec.max_duration + 0.5:
        errors.append(
            f"Duration exceeds limit: {info.duration:.1f}s > {spec.max_duration:.1f}s"
        )

    # Codec check (flexible — accept both sw and hw encoders for same codec family)
    expected_codec_family = spec.codec.replace("libx", "").replace("h264_", "").replace("hevc_", "")
    actual_codec_family = info.video_codec.replace("libx", "").replace("h264_", "").replace("hevc_", "")
    # Normalize to common names
    codec_aliases: dict[str, set[str]] = {
        "264": {"264", "h264", "avc"},
        "265": {"265", "h265", "hevc"},
    }
    is_codec_match = False
    for family, aliases in codec_aliases.items():
        if expected_codec_family in aliases and actual_codec_family in aliases:
            is_codec_match = True
            break
    if not is_codec_match and expected_codec_family != actual_codec_family:
        # Not a hard failure — just a warning
        errors.append(
            f"Codec differs from spec: expected {spec.codec}, got {info.video_codec}"
        )

    # Bitrate check
    if spec.max_bitrate > 0 and info.bitrate > spec.max_bitrate:
        errors.append(
            f"Bitrate exceeds limit: {info.bitrate} bps > {spec.max_bitrate} bps"
        )

    # Audio check
    if not info.has_audio:
        errors.append("Missing audio track")
    elif spec.audio_codec == "aac" and info.audio_codec not in ("aac", "aac_lc"):
        errors.append(f"Audio codec mismatch: expected aac, got {info.audio_codec}")

    is_valid = len(errors) == 0
    return is_valid, errors


def auto_fix_export(result: ExportResult, spec: PlatformSpec, settings: Settings) -> ExportResult:
    """Attempt to auto-fix a non-compliant export by re-encoding.

    Addresses common issues like wrong resolution, excessive duration,
    or wrong codec by re-exporting with corrected parameters.

    Args:
        result: The original (non-compliant) ExportResult.
        spec: PlatformSpec to conform to.
        settings: Application settings.

    Returns:
        New ExportResult from the re-encoded file.
    """
    if not result.path or not result.path.exists():
        logger.error("Cannot auto-fix: original export file missing")
        return result

    logger.info("Auto-fixing export for %s: %s", spec.name, result.validation_errors)

    # Build a corrected task
    task = {
        "platform_key": spec.key,
        "spec": spec,
        "safe_title": result.path.stem.split("_")[0] if result.path else "fixed",
        "resolution": spec.resolution,
        "resolution_label": result.resolution_label,
        "codec_info": CODEC_VARIANTS.get(result.codec_label, CODEC_VARIANTS["h264"]),
        "codec_label": result.codec_label,
        "variant": result.variant,
        "crf_override": None,
        "aspect_ratio": "9:16",
        "two_pass": False,
    }

    try:
        video_info = probe_video(result.path)
        new_result = _execute_export_task(result.path, task, settings, video_info)
        is_valid, errors = validate_export(new_result, spec)
        new_result.validated = is_valid
        new_result.validation_errors = errors
        if is_valid:
            logger.info("Auto-fix successful for %s", spec.name)
            # Remove the old non-compliant file
            try:
                result.path.unlink()
            except OSError:
                pass
            return new_result
        else:
            logger.warning("Auto-fix produced non-compliant file: %s", errors)
            return new_result
    except Exception as exc:
        logger.error("Auto-fix failed for %s: %s", spec.name, exc)
        return result


# ═══════════════════════════════════════════════════════════════
#  Thumbnail Export
# ═══════════════════════════════════════════════════════════════

def export_thumbnails(
    source_path: Path,
    platform_key: str,
    safe_title: str,
    settings: Settings,
    video_info: Any,
    aspect_ratio: str = "9:16",
) -> list[Path]:
    """Generate platform-specific thumbnail images.

    Uses ThumbnailGenerator if available, falls back to FFmpeg frame
    extraction. Generates thumbnails at platform-appropriate sizes.

    Args:
        source_path: Path to the source video.
        platform_key: Platform identifier (e.g. 'youtube').
        safe_title: Sanitised video title for filenames.
        settings: Application settings.
        video_info: Probed VideoInfo from source.
        aspect_ratio: Target aspect ratio.

    Returns:
        List of paths to generated thumbnail images.
    """
    thumbnail_paths: list[Path] = []

    # Determine thumbnail sizes for this platform
    spec = PLATFORM_SPECS.get(platform_key)
    if spec is None:
        return thumbnail_paths

    thumbnail_sizes: dict[str, tuple[int, int]] = {
        "youtube": (1280, 720),
        "tiktok": (1080, 1920),
        "instagram": (1080, 1920),
        "twitter": (1280, 720),
        "facebook": (1280, 720),
        "snapchat": (1080, 1920),
    }
    target_size = thumbnail_sizes.get(platform_key, (1280, 720))

    output_dir = settings.THUMBNAILS_DIR / platform_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try ThumbnailGenerator
    try:
        from core.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator(settings)
        frames = gen.extract_best_frames(source_path, count=settings.THUMBNAIL_COUNT)

        for i, frame_path in enumerate(frames):
            thumb_path = make_output_path(
                output_dir, safe_title, f"{platform_key}_thumb_{i}",
                ext=settings.THUMBNAIL_FORMAT,
            )
            try:
                gen.generate_thumbnail(
                    frame_path, safe_title, thumb_path,
                    style="modern", platform=platform_key,
                )
                thumbnail_paths.append(thumb_path)
            except Exception as exc:
                logger.warning("Thumbnail generation failed for %s frame %d: %s", platform_key, i, exc)
            finally:
                # Clean up temp frame
                try:
                    frame_path.unlink()
                except OSError:
                    pass

    except ImportError:
        logger.debug("ThumbnailGenerator not available — using FFmpeg fallback")
    except Exception as exc:
        logger.warning("ThumbnailGenerator error: %s — falling back to FFmpeg", exc)

    # FFmpeg fallback: extract frames at evenly spaced timestamps
    if not thumbnail_paths and video_info.duration > 1.0:
        duration = video_info.duration
        for i in range(settings.THUMBNAIL_COUNT):
            ts = duration * (i + 1) / (settings.THUMBNAIL_COUNT + 1)
            thumb_path = make_output_path(
                output_dir, safe_title, f"{platform_key}_thumb_{i}",
                ext=settings.THUMBNAIL_FORMAT,
            )
            try:
                from utils.ffmpeg_utils import get_video_thumbnail
                get_video_thumbnail(source_path, ts, thumb_path)
                thumbnail_paths.append(thumb_path)
            except Exception as exc:
                logger.warning("FFmpeg thumbnail extraction failed at %.1fs: %s", ts, exc)

    if thumbnail_paths:
        logger.info("Generated %d thumbnail(s) for %s", len(thumbnail_paths), platform_key)

    return thumbnail_paths


# ═══════════════════════════════════════════════════════════════
#  Metadata File Export
# ═══════════════════════════════════════════════════════════════

def export_metadata_file(
    platform_key: str,
    safe_title: str,
    settings: Settings,
    title: str,
    metadata_dict: dict[str, Any] | None = None,
    video_info: Any = None,
    export_result: ExportResult | None = None,
) -> Path:
    """Export a platform-specific metadata JSON file alongside the video.

    The metadata file contains title, description, tags, and an upload
    manifest for batch upload tools.

    Args:
        platform_key: Platform identifier.
        safe_title: Sanitised video title for filename.
        settings: Application settings.
        title: Original video title.
        metadata_dict: Optional pre-existing metadata to merge.
        video_info: Probed VideoInfo from source.
        export_result: The ExportResult for this platform.

    Returns:
        Path to the written metadata JSON file.
    """
    output_dir = settings.METADATA_DIR / platform_key
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = make_output_path(output_dir, safe_title, f"{platform_key}_meta", ext="json")

    spec = PLATFORM_SPECS.get(platform_key)

    # Build platform-specific metadata
    platform_meta: dict[str, Any] = {
        "platform": platform_key,
        "platform_name": spec.name if spec else platform_key,
        "title": title,
        "description": metadata_dict.get("description", "") if metadata_dict else "",
        "tags": metadata_dict.get("tags", []) if metadata_dict else [],
        "hashtags": metadata_dict.get("hashtags", []) if metadata_dict else [],
        "keywords": metadata_dict.get("keywords", []) if metadata_dict else [],
        "category": metadata_dict.get("category", "") if metadata_dict else "",
        "max_duration": spec.max_duration if spec else 0,
        "target_resolution": f"{spec.resolution[0]}x{spec.resolution[1]}" if spec else "",
        "export_timestamp": datetime.now().isoformat(),
    }

    # Add export details if available
    if export_result and export_result.is_success:
        platform_meta["export"] = {
            "file_path": str(export_result.path),
            "file_size": export_result.file_size,
            "file_size_human": export_result.file_size_human,
            "duration": export_result.duration,
            "resolution": f"{export_result.resolution[0]}x{export_result.resolution[1]}",
            "codec": export_result.codec,
            "crf": export_result.crf,
            "validated": export_result.validated,
        }

    # Upload manifest for batch tools
    platform_meta["upload_manifest"] = {
        "video_file": str(export_result.path) if export_result and export_result.path else "",
        "metadata_file": str(meta_path),
        "thumbnail_files": [],
        "subtitle_files": [],
        "ready_for_upload": export_result.validated if export_result else False,
    }

    # Merge any additional metadata
    if metadata_dict:
        for key, value in metadata_dict.items():
            if key not in platform_meta:
                platform_meta[key] = value

    try:
        from utils.file_utils import write_json_safe
        write_json_safe(meta_path, platform_meta, indent=2)
        logger.info("Metadata exported: %s", meta_path.name)
    except Exception as exc:
        logger.error("Failed to write metadata for %s: %s", platform_key, exc)

    return meta_path


# ═══════════════════════════════════════════════════════════════
#  Subtitle File Export
# ═══════════════════════════════════════════════════════════════

def export_subtitle_files(
    platform_key: str,
    safe_title: str,
    settings: Settings,
    source_path: Path,
    subtitle_srt_path: Path | None = None,
    subtitle_vtt_path: Path | None = None,
) -> list[Path]:
    """Export SRT and VTT subtitle files alongside the platform video.

    If existing subtitle files are provided, they are copied to the
    platform output directory.  If not, an attempt is made to extract
    subtitles from the source video.

    Args:
        platform_key: Platform identifier.
        safe_title: Sanitised video title for filename.
        settings: Application settings.
        source_path: Path to the source video.
        subtitle_srt_path: Optional existing SRT file to copy.
        subtitle_vtt_path: Optional existing VTT file to copy.

    Returns:
        List of paths to generated/copied subtitle files.
    """
    subtitle_paths: list[Path] = []
    spec = PLATFORM_SPECS.get(platform_key)
    output_dir_name = spec.output_dir_name if spec else platform_key
    output_dir = settings.SHORTS_DIR / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy or generate SRT
    if subtitle_srt_path and subtitle_srt_path.exists():
        srt_out = make_output_path(output_dir, safe_title, f"{platform_key}_sub", ext="srt")
        try:
            import shutil
            shutil.copy2(str(subtitle_srt_path), str(srt_out))
            subtitle_paths.append(srt_out)
            logger.debug("SRT copied for %s: %s", platform_key, srt_out.name)
        except OSError as exc:
            logger.warning("Failed to copy SRT for %s: %s", platform_key, exc)

    # Copy or generate VTT
    if subtitle_vtt_path and subtitle_vtt_path.exists():
        vtt_out = make_output_path(output_dir, safe_title, f"{platform_key}_sub", ext="vtt")
        try:
            import shutil
            shutil.copy2(str(subtitle_vtt_path), str(vtt_out))
            subtitle_paths.append(vtt_out)
            logger.debug("VTT copied for %s: %s", platform_key, vtt_out.name)
        except OSError as exc:
            logger.warning("Failed to copy VTT for %s: %s", platform_key, exc)

    # If no existing subtitles, try extracting from source video
    if not subtitle_paths:
        try:
            video_info = probe_video(source_path)
            if video_info.has_subtitles:
                srt_out = make_output_path(output_dir, safe_title, f"{platform_key}_sub", ext="srt")
                cmd: list[str] = [
                    "ffmpeg", "-i", str(source_path),
                    "-map", "0:s:0?",
                    "-f", "srt", str(srt_out),
                ]
                run_ffmpeg(cmd, description=f"Extract SRT for {platform_key}", timeout=60)
                if srt_out.exists() and srt_out.stat().st_size > 0:
                    subtitle_paths.append(srt_out)

                    # Convert SRT to VTT
                    vtt_out = make_output_path(output_dir, safe_title, f"{platform_key}_sub", ext="vtt")
                    cmd_vtt: list[str] = [
                        "ffmpeg", "-i", str(srt_out),
                        "-f", "webvtt", str(vtt_out),
                    ]
                    run_ffmpeg(cmd_vtt, description=f"Convert SRT to VTT for {platform_key}", timeout=30)
                    if vtt_out.exists() and vtt_out.stat().st_size > 0:
                        subtitle_paths.append(vtt_out)
        except Exception as exc:
            logger.debug("Subtitle extraction failed for %s: %s", platform_key, exc)

    if subtitle_paths:
        logger.info("Exported %d subtitle file(s) for %s", len(subtitle_paths), platform_key)

    return subtitle_paths


# ═══════════════════════════════════════════════════════════════
#  Post-Processing
# ═══════════════════════════════════════════════════════════════

def _post_process_exports(
    exports: PlatformExports,
    source_path: Path,
    safe_title: str,
    settings: Settings,
    enable_ab_variants: bool,
    generate_thumbnails: bool,
    generate_metadata: bool,
    generate_subtitles: bool,
    validate_exports: bool,
    subtitle_srt_path: Path | None,
    subtitle_vtt_path: Path | None,
    metadata_dict: dict[str, Any] | None,
    title: str,
    video_info: Any,
    aspect_ratio: str,
) -> None:
    """Run all post-export processing: validation, thumbnails, metadata, subtitles, A/B pairing.

    Args:
        exports: PlatformExports to augment with post-processing results.
        source_path: Path to the source video.
        safe_title: Sanitised video title.
        settings: Application settings.
        enable_ab_variants: Whether A/B variant pairing should be done.
        generate_thumbnails: Whether to generate thumbnails.
        generate_metadata: Whether to generate metadata files.
        generate_subtitles: Whether to generate subtitle files.
        validate_exports: Whether to validate exports.
        subtitle_srt_path: Optional SRT path.
        subtitle_vtt_path: Optional VTT path.
        metadata_dict: Optional metadata dictionary.
        title: Original video title.
        video_info: Probed VideoInfo.
        aspect_ratio: Target aspect ratio.
    """
    # ── Validation ─────────────────────────────────────
    if validate_exports:
        for result in exports.results:
            if not result.is_success:
                continue
            spec = PLATFORM_SPECS.get(result.platform)
            if spec is None:
                continue
            is_valid, errors = validate_export(result, spec)
            result.validated = is_valid
            result.validation_errors = errors

            # Auto-fix if validation failed
            if not is_valid:
                logger.warning(
                    "Validation failed for %s: %s — attempting auto-fix",
                    result.platform, errors,
                )
                fixed = auto_fix_export(result, spec, settings)
                if fixed.validated:
                    # Replace the failed result with the fixed one
                    idx = exports.results.index(result)
                    exports.results[idx] = fixed

    # ── Thumbnails ─────────────────────────────────────
    if generate_thumbnails:
        for platform_key in set(r.platform for r in exports.results if r.is_success):
            try:
                thumbs = export_thumbnails(
                    source_path, platform_key, safe_title,
                    settings, video_info, aspect_ratio,
                )
                if thumbs:
                    exports.thumbnail_paths[platform_key] = thumbs
            except Exception as exc:
                logger.warning("Thumbnail export failed for %s: %s", platform_key, exc)

    # ── Metadata ───────────────────────────────────────
    if generate_metadata:
        for platform_key in set(r.platform for r in exports.results if r.is_success):
            try:
                primary = None
                for r in exports.results:
                    if r.platform == platform_key and r.is_success:
                        primary = r
                        break
                meta_path = export_metadata_file(
                    platform_key, safe_title, settings,
                    title, metadata_dict, video_info, primary,
                )
                exports.metadata_paths[platform_key] = meta_path
            except Exception as exc:
                logger.warning("Metadata export failed for %s: %s", platform_key, exc)

    # ── Subtitles ──────────────────────────────────────
    if generate_subtitles:
        for platform_key in set(r.platform for r in exports.results if r.is_success):
            try:
                subs = export_subtitle_files(
                    platform_key, safe_title, settings,
                    source_path, subtitle_srt_path, subtitle_vtt_path,
                )
                if subs:
                    exports.subtitle_paths[platform_key] = subs
            except Exception as exc:
                logger.warning("Subtitle export failed for %s: %s", platform_key, exc)

    # ── A/B Variant Pairing ────────────────────────────
    if enable_ab_variants:
        platform_groups: dict[str, list[ExportResult]] = {}
        for r in exports.results:
            if r.is_success:
                platform_groups.setdefault(r.platform, []).append(r)

        for platform_key, group in platform_groups.items():
            variant_a: list[ExportResult] = [r for r in group if r.variant == ""]
            variant_b: list[ExportResult] = [r for r in group if r.variant == "B"]

            if variant_a and variant_b:
                a = variant_a[0]
                b = variant_b[0]
                spec = PLATFORM_SPECS.get(platform_key)
                exports.ab_variants[platform_key] = ABVariant(
                    variant_a_path=a.path,
                    variant_b_path=b.path,
                    variant_a_crf=a.crf,
                    variant_b_crf=b.crf,
                    variant_a_size=a.file_size,
                    variant_b_size=b.file_size,
                )
                logger.info(
                    "A/B Variant for %s: A (CRF %d, %s) vs B (CRF %d, %s) — %s improvement",
                    platform_key, a.crf, a.file_size_human,
                    b.crf, b.file_size_human,
                    exports.ab_variants[platform_key].quality_improvement_estimate,
                )


# ═══════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════

def _build_report(
    exports: PlatformExports,
    source_path: Path,
    title: str,
    start_ts: float,
) -> ExportReport:
    """Build an ExportReport from the completed exports.

    Args:
        exports: Completed PlatformExports.
        source_path: Path to the source video.
        title: Video title.
        start_ts: Monotonic timestamp when export started.

    Returns:
        Fully populated ExportReport.
    """
    import time as _time

    elapsed = _time.monotonic() - start_ts

    report = ExportReport(
        timestamp=datetime.now().isoformat(),
        source_path=str(source_path),
        title=title,
        total_results=exports.total_results,
        successful=exports.count,
        failed=len(exports.failed_platforms),
        validated=len(exports.validated_platforms),
        results=exports.results,
        total_output_size_bytes=sum(r.file_size for r in exports.results if r.is_success),
        export_duration_seconds=elapsed,
    )

    # Build per-platform summaries
    platform_groups: dict[str, list[ExportResult]] = {}
    for r in exports.results:
        platform_groups.setdefault(r.platform, []).append(r)

    for platform_key, results in platform_groups.items():
        spec = PLATFORM_SPECS.get(platform_key)
        successful = [r for r in results if r.is_success]
        total_size = sum(r.file_size for r in successful)

        summary: dict[str, Any] = {
            "platform_name": spec.name if spec else platform_key,
            "total_exports": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "total_size_human": get_file_size_human(Path(str(total_size))) if total_size else "0 B",
        }

        if successful:
            first = successful[0]
            summary["resolution"] = f"{first.resolution[0]}x{first.resolution[1]}"
            summary["codec"] = first.codec
            summary["crf"] = first.crf
            summary["duration"] = f"{first.duration:.1f}s"
            summary["validated"] = all(r.validated for r in successful)

        # A/B variant info
        if platform_key in exports.ab_variants:
            ab = exports.ab_variants[platform_key]
            summary["ab_variant"] = {
                "a_crf": ab.variant_a_crf,
                "b_crf": ab.variant_b_crf,
                "a_size": ab.variant_a_size,
                "b_size": ab.variant_b_size,
                "quality_improvement": ab.quality_improvement_estimate,
            }

        # Thumbnail info
        if platform_key in exports.thumbnail_paths:
            summary["thumbnails"] = len(exports.thumbnail_paths[platform_key])

        # Metadata info
        if platform_key in exports.metadata_paths:
            summary["metadata_file"] = str(exports.metadata_paths[platform_key])

        # Subtitle info
        if platform_key in exports.subtitle_paths:
            summary["subtitle_files"] = len(exports.subtitle_paths[platform_key])

        report.platform_summaries[platform_key] = summary

    return report


def save_report(
    report: ExportReport,
    output_dir: Path | None = None,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save an ExportReport to disk in the specified formats.

    Args:
        report: ExportReport to save.
        output_dir: Directory for report files. Defaults to METADATA_DIR.
        formats: List of formats ('json', 'txt'). Defaults to both.

    Returns:
        List of paths to written report files.
    """
    if output_dir is None:
        settings = get_settings()
        output_dir = settings.METADATA_DIR
    if formats is None:
        formats = ["json", "txt"]

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    written: list[Path] = []

    if "json" in formats:
        json_path = output_dir / f"export_report_{timestamp_str}.json"
        try:
            from utils.file_utils import write_json_safe
            write_json_safe(json_path, report.to_dict(), indent=2)
            written.append(json_path)
            logger.info("Report saved (JSON): %s", json_path.name)
        except Exception as exc:
            logger.error("Failed to save JSON report: %s", exc)

    if "txt" in formats:
        txt_path = output_dir / f"export_report_{timestamp_str}.txt"
        try:
            txt_path.write_text(report.to_text(), encoding="utf-8")
            written.append(txt_path)
            logger.info("Report saved (TXT): %s", txt_path.name)
        except Exception as exc:
            logger.error("Failed to save TXT report: %s", exc)

    return written


# ═══════════════════════════════════════════════════════════════
#  Platform Helpers
# ═══════════════════════════════════════════════════════════════

def _get_enabled_platforms(settings: Settings) -> list[str]:
    """Determine which platforms are enabled from settings.

    Checks the EXPORT_* boolean flags in settings and returns
    the corresponding platform keys.

    Args:
        settings: Application settings.

    Returns:
        List of enabled platform key strings.
    """
    enabled: list[str] = []
    # Map settings flags to platform keys
    flag_map: dict[str, str] = {
        "EXPORT_YOUTUBE": "youtube",
        "EXPORT_TIKTOK": "tiktok",
        "EXPORT_REELS": "instagram",
    }
    for flag_name, platform_key in flag_map.items():
        if getattr(settings, flag_name, False):
            enabled.append(platform_key)

    # If no platforms are enabled via legacy flags, default to all
    if not enabled:
        enabled = list(PLATFORM_SPECS.keys())

    return enabled


def get_platform_spec(platform_key: str) -> PlatformSpec | None:
    """Look up a PlatformSpec by platform key.

    Args:
        platform_key: Platform identifier (e.g. 'youtube').

    Returns:
        PlatformSpec or None if key not found.
    """
    return PLATFORM_SPECS.get(platform_key)


def list_platforms() -> dict[str, dict[str, Any]]:
    """Return a summary of all supported platform specifications.

    Returns:
        Dictionary mapping platform keys to spec summaries.
    """
    result: dict[str, dict[str, Any]] = {}
    for key, spec in PLATFORM_SPECS.items():
        result[key] = {
            "name": spec.name,
            "max_duration": spec.max_duration,
            "resolution": f"{spec.resolution[0]}x{spec.resolution[1]}",
            "codec": spec.codec,
            "profile": spec.profile,
            "crf": spec.crf,
            "audio_codec": spec.audio_codec,
            "audio_bitrate": spec.audio_bitrate,
            "max_bitrate": spec.max_bitrate,
        }
    return result


def get_aspect_ratio_resolution(aspect_ratio: str, base_width: int = 1080) -> tuple[int, int]:
    """Calculate the resolution for a given aspect ratio and base width.

    Args:
        aspect_ratio: Aspect ratio string ('9:16', '1:1', '4:5').
        base_width: Base width in pixels.

    Returns:
        Tuple of (width, height) with even dimensions.
    """
    if aspect_ratio == "1:1":
        h = base_width
    elif aspect_ratio == "4:5":
        h = int(base_width * 5 / 4)
    else:  # 9:16
        h = int(base_width * 16 / 9)

    # Ensure even dimensions
    w = base_width if base_width % 2 == 0 else base_width + 1
    h = h if h % 2 == 0 else h + 1

    return w, h
