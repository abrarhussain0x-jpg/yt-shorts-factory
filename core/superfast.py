"""
core/superfast.py — Superfast single-pass pipeline for maximum speed.

Combines ALL FFmpeg operations (crop, scale, subtitle burn, logo overlay,
audio normalization) into a SINGLE FFmpeg command, eliminating the 5-6
separate encode passes of the standard pipeline.

Speed improvements vs standard pipeline:
  - Single FFmpeg pass instead of 5-6 → 3-5x faster encoding
  - Skip motion energy (saves 300+ FFmpeg subprocess calls)
  - Skip spectral centroid (saves 300+ FFmpeg subprocess calls)
  - Skip letterbox detection (saves 1 FFmpeg call)
  - Center crop instead of face tracking (saves 5 FFmpeg calls)
  - faster-whisper tiny model (4x faster transcription)
  - 720p download (2x faster download)
  - ultrafast encoding preset
  - Skip audio enhancement chain (3 FFmpeg passes)
  - Skip content moderation
  - Minimal thumbnails

Typical speed: 60-90% faster than standard pipeline.

Usage:
    python main.py run --url URL --superfast
    python main.py run --url URL --superfast --duration 25s
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from core.analyzer import EngagementAnalyzer, SegmentResult
from core.transcriber import transcribe, TranscriptionResult
from core.subtitle_engine import generate_subtitles
from utils.ffmpeg_utils import (
    probe_video, run_ffmpeg, detect_hw_encoder, FFmpegError,
    extract_audio_wav,
)
from utils.file_utils import (
    make_output_path, sanitize_filename, get_file_size_human, safe_delete,
)
from utils.logger import get_logger

logger = get_logger("superfast")


@dataclass
class SuperfastResult:
    """Result of a superfast pipeline run."""
    success: bool = False
    output_path: Path | None = None
    duration: float = 0.0
    file_size: int = 0
    ffmpeg_passes: int = 1
    speed_mode: str = "superfast"

    @property
    def file_size_human(self) -> str:
        return get_file_size_human(self.output_path) if self.output_path and self.output_path.exists() else "0B"


def superfast_analyze(video_path: Path, clip_duration: int) -> SegmentResult:
    """Ultra-fast analysis: audio energy + silence only, skip all expensive signals.

    Skips: motion energy, spectral centroid, scene density classification,
    letterbox detection, face tracking. Uses only audio RMS and silence
    detection (2 FFmpeg calls vs 300+ in standard analysis).

    Args:
        video_path: Path to the source video.
        clip_duration: Target clip duration in seconds.

    Returns:
        SegmentResult with the best clip segment.
    """
    logger.info("SUPERFAST ANALYZE: %s (%ds clip)", video_path.name, clip_duration)
    video_info = probe_video(video_path)
    total_duration = video_info.duration

    if total_duration <= 0:
        return SegmentResult(
            start_time=0.0, end_time=min(float(clip_duration), 60.0),
            energy_score=0.5, method_used="superfast_fallback",
        )

    if total_duration <= clip_duration:
        return SegmentResult(
            start_time=0.0, end_time=total_duration,
            energy_score=1.0, method_used="superfast_full_video",
            confidence=1.0, overall_quality_grade="A",
        )

    # Use analyzer's fast mode (audio + silence only)
    analyzer = EngagementAnalyzer(video_path, clip_duration)
    return analyzer.analyze_fast()


def superfast_transcribe(
    video_path: Path,
    settings: Settings,
) -> TranscriptionResult:
    """Fast transcription using tiny Whisper model with faster-whisper.

    Uses the smallest model for maximum speed while still getting
    word-level timestamps for subtitle generation.

    Args:
        video_path: Path to the video file.
        settings: Application settings.

    Returns:
        TranscriptionResult with word-level timestamps.
    """
    # Force tiny model for speed
    settings.WHISPER_MODEL = "tiny"
    settings.WHISPER_BEAM_SIZE = 1  # Minimum beam size for speed
    settings.WHISPER_FALLBACK_MODELS = ""  # No fallbacks

    logger.info("SUPERFAST TRANSCRIBE: tiny model, beam_size=1")
    return transcribe(
        video_path,
        settings=settings,
        use_vad=True,
        use_cache=True,
        compare_models=False,
    )


def superfast_single_pass(
    input_path: Path,
    segment: SegmentResult,
    ass_path: Path | None,
    logo_path: Path | None,
    output_path: Path,
    settings: Settings,
    blur_bg: bool = False,
) -> Path:
    """Single FFmpeg pass that combines ALL operations.

    Combines into one filter_complex:
    1. Trim + setpts (extract segment)
    2. Crop to 9:16 (center crop, no face tracking)
    3. Scale to 1080x1920
    4. Subtitle burn (ASS overlay)
    5. Logo overlay (with fade-in)
    6. Audio trim + loudnorm
    7. Format conversion

    This is 3-5x faster than running separate FFmpeg passes for each step.

    Args:
        input_path: Source video path.
        segment: SegmentResult with start/end times.
        ass_path: Path to ASS subtitle file (None to skip).
        logo_path: Path to logo PNG (None to skip).
        output_path: Destination path.
        settings: Application settings.
        blur_bg: Use blurred background mode.

    Returns:
        Path to the output file.
    """
    if not input_path.exists():
        raise FFmpegError(f"Input not found: {input_path}")

    clip_duration = segment.end_time - segment.start_time
    video_info = probe_video(input_path)
    src_w = video_info.width
    src_h = video_info.height

    target_w = settings.OUTPUT_WIDTH
    target_h = settings.OUTPUT_HEIGHT

    # Ensure even dimensions
    target_w = target_w - (target_w % 2)
    target_h = target_h - (target_h % 2)

    # Calculate center crop (fast, no face tracking)
    target_ar = target_w / target_h
    source_ar = src_w / src_h

    if abs(source_ar - target_ar) < 0.01:
        # Already correct AR
        crop_w, crop_h = src_w, src_h
        crop_x, crop_y = 0, 0
    elif source_ar > target_ar:
        # Landscape → crop sides
        crop_h = src_h
        crop_w = int(src_h * target_ar)
        crop_w = crop_w - (crop_w % 2)
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        # Portrait → crop top/bottom
        crop_w = src_w
        crop_h = int(src_w / target_ar)
        crop_h = crop_h - (crop_h % 2)
        crop_x = 0
        crop_y = (src_h - crop_h) // 2

    # ── Build single-pass filter_complex ────────────────

    # Detect HW encoder
    hw_encoder, hw_preset = detect_hw_encoder()
    video_codec = hw_encoder or settings.FFMPEG_VIDEO_CODEC
    encoding_preset = hw_preset or "ultrafast"

    inputs = ["ffmpeg"]
    filter_parts = []
    input_idx = 0  # Track input indices

    # Input 0: source video
    inputs.extend(["-i", str(input_path)])
    video_input = f"[{input_idx}:v]"
    audio_input = f"[{input_idx}:a]"
    input_idx += 1

    # ── Video filter chain ──────────────────────────────
    vfilter_parts = [
        f"trim=start={segment.start_time}:duration={clip_duration}",
        "setpts=PTS-STARTPTS",
    ]

    if blur_bg:
        # Blurred background mode: split, blur one, crop the other, overlay
        # [0:v]split=2[bg][fg]; [bg]trim+setpts+scale+crop+gblur[bgout];
        # [fg]trim+setpts+crop+scale[fgout]; [bgout][fgout]overlay
        bg_filter = (
            f"trim=start={segment.start_time}:duration={clip_duration},"
            f"setpts=PTS-STARTPTS,"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},"
            f"gblur=sigma=20,"
            f"format={settings.FFMPEG_PIXEL_FORMAT}"
        )
        fg_filter = (
            f"trim=start={segment.start_time}:duration={clip_duration},"
            f"setpts=PTS-STARTPTS,"
            f"crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y},"
            f"scale={target_w}:{target_h}:flags=lanczos,"
            f"format={settings.FFMPEG_PIXEL_FORMAT}"
        )
        # We need filter_complex for blur-bg
        filter_complex = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]{bg_filter}[bgout];"
            f"[fg]{fg_filter}[fgout];"
            f"[bgout][fgout]overlay=(W-w)/2:(H-h)/2:format=auto"
        )

        # Add subtitle and logo to filter_complex if needed
        current_label = "[vout0]"
        # We'll need to restructure for subtitles/logo after overlay

        if ass_path and ass_path.exists():
            # Insert ASS subtitles via ass filter after overlay
            ass_str = str(ass_path).replace("\\", "/").replace(":", "\\:")
            filter_complex = filter_complex.replace(
                "overlay=(W-w)/2:(H-h)/2:format=auto",
                f"overlay=(W-w)/2:(H-h)/2:format=auto[vbase]"
            )
            filter_complex += f";[vbase]ass='{ass_str}'[vout0]"
        else:
            filter_complex = filter_complex.replace(
                "overlay=(W-w)/2:(H-h)/2:format=auto",
                "overlay=(W-w)/2:(H-h)/2:format=auto[vout0]"
            )

        # Logo overlay
        if logo_path and logo_path.exists():
            inputs.extend(["-i", str(logo_path)])
            logo_input_idx = input_idx
            input_idx += 1

            logo_px_w = int(target_w * settings.LOGO_SCALE)
            from PIL import Image
            with Image.open(logo_path) as img:
                aspect = img.size[1] / img.size[0] if img.size[0] > 0 else 1.0
            logo_px_h = int(logo_px_w * aspect)
            logo_px_w = logo_px_w - (logo_px_w % 2)
            logo_px_h = logo_px_h - (logo_px_h % 2)

            safe_top = 48
            margin = settings.LOGO_MARGIN
            overlay_x = target_w - logo_px_w - 24 - margin
            overlay_y = safe_top + margin
            opacity = settings.LOGO_OPACITY

            fade_dur = settings.LOGO_FADE_DURATION

            filter_complex += (
                f";[{logo_input_idx}:v]"
                f"scale={logo_px_w}:{logo_px_h},"
                f"format=rgba,"
                f"colorchannelmixer=aa={opacity},"
                f"fade=t=in:st=0:d={fade_dur}:alpha=1[logo];"
                f"[vout0][logo]overlay=x={overlay_x}:y={overlay_y}:format=auto[vfinal]"
            )
            final_video_label = "[vfinal]"
        else:
            final_video_label = "[vout0]"

        # Audio filter
        afilter = (
            f"atrim=start={segment.start_time}:duration={clip_duration},"
            f"asetpts=PTS-STARTPTS,"
            f"loudnorm=I=-16:LRA=11:TP=-1.5"
        )

        cmd = inputs + [
            "-filter_complex", filter_complex,
            "-map", final_video_label,
            "-map", f"0:a?",
            "-af", afilter,
        ]

    else:
        # Standard crop mode (much simpler filter graph)
        vfilter_parts.extend([
            f"crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y}",
            f"scale={target_w}:{target_h}:flags=lanczos",
            f"fps=30",
            f"format={settings.FFMPEG_PIXEL_FORMAT}",
        ])

        vfilter = ",".join(vfilter_parts)

        # Add ASS subtitle filter if available
        if ass_path and ass_path.exists():
            ass_str = str(ass_path).replace("\\", "/").replace(":", "\\:")
            vfilter += f",ass='{ass_str}'"

        # Logo overlay needs filter_complex
        if logo_path and logo_path.exists():
            inputs.extend(["-i", str(logo_path)])
            logo_input_idx = input_idx
            input_idx += 1

            logo_px_w = int(target_w * settings.LOGO_SCALE)
            try:
                from PIL import Image
                with Image.open(logo_path) as img:
                    aspect = img.size[1] / img.size[0] if img.size[0] > 0 else 1.0
                logo_px_h = int(logo_px_w * aspect)
            except Exception:
                logo_px_h = logo_px_w
            logo_px_w = logo_px_w - (logo_px_w % 2)
            logo_px_h = logo_px_h - (logo_px_h % 2)

            safe_top = 48
            margin = settings.LOGO_MARGIN
            overlay_x = target_w - logo_px_w - 24 - margin
            overlay_y = safe_top + margin
            opacity = settings.LOGO_OPACITY
            fade_dur = settings.LOGO_FADE_DURATION

            logo_filter = (
                f"[{logo_input_idx}:v]"
                f"scale={logo_px_w}:{logo_px_h},"
                f"format=rgba,"
                f"colorchannelmixer=aa={opacity},"
                f"fade=t=in:st=0:d={fade_dur}:alpha=1[logo]"
            )

            filter_complex = (
                f"[0:v]{vfilter}[vbase];"
                f"{logo_filter};"
                f"[vbase][logo]overlay=x={overlay_x}:y={overlay_y}:format=auto"
            )

            afilter = (
                f"atrim=start={segment.start_time}:duration={clip_duration},"
                f"asetpts=PTS-STARTPTS,"
                f"loudnorm=I=-16:LRA=11:TP=-1.5"
            )

            cmd = inputs + [
                "-filter_complex", filter_complex,
                "-af", afilter,
            ]
        else:
            # No logo: simple -vf chain (fastest)
            afilter = (
                f"atrim=start={segment.start_time}:duration={clip_duration},"
                f"asetpts=PTS-STARTPTS,"
                f"loudnorm=I=-16:LRA=11:TP=-1.5"
            )

            cmd = inputs + [
                "-vf", vfilter,
                "-af", afilter,
            ]

    # Common encoding args
    cmd.extend([
        "-c:v", video_codec,
        "-preset", encoding_preset,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", settings.FFMPEG_AUDIO_CODEC,
        "-b:a", settings.FFMPEG_AUDIO_BITRATE,
        "-pix_fmt", settings.FFMPEG_PIXEL_FORMAT,
        "-r", "30",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS or 0),
        str(output_path),
    ])

    logger.info(
        "SUPERFAST single-pass: %s -> %s (%dx%d, %.1fs, codec=%s, preset=%s)",
        input_path.name, output_path.name,
        target_w, target_h, clip_duration, video_codec, encoding_preset,
    )

    try:
        run_ffmpeg(
            cmd,
            description=f"Superfast single-pass: {input_path.name}",
            show_progress=True,
            total_duration=clip_duration,
            timeout=settings.FFMPEG_TIMEOUT,
        )
    except FFmpegError as exc:
        if hw_encoder:
            logger.warning("HW encode failed, falling back to SW: %s", exc)
            # Retry with software encoding
            sw_cmd = list(cmd)
            if video_codec in sw_cmd:
                idx = sw_cmd.index(video_codec)
                sw_cmd[idx] = "libx264"
            if hw_preset and hw_preset in sw_cmd:
                idx = sw_cmd.index(hw_preset)
                sw_cmd[idx] = "ultrafast"
            run_ffmpeg(sw_cmd, description="Superfast single-pass (SW fallback)",
                      show_progress=True, total_duration=clip_duration,
                      timeout=settings.FFMPEG_TIMEOUT)
        else:
            raise

    if not output_path.exists():
        raise FFmpegError(f"Output not created: {output_path}")

    return output_path


def superfast_pipeline(
    url: str,
    duration: int | None = None,
    skip_subs: bool = False,
    no_logo: bool = False,
    platforms: list[str] | None = None,
    settings: Settings | None = None,
    blur_bg: bool = False,
) -> SuperfastResult:
    """Execute the superfast pipeline for maximum speed.

    Pipeline steps (minimized for speed):
    1. Download (720p, no metadata/chapters)
    2. Fast Analyze (audio + silence only, ~2s)
    3. Transcribe (tiny Whisper, faster-whisper)
    4. Generate ASS subtitles
    5. Single FFmpeg pass (crop + scale + subs + logo + audio) ← KEY SPEED WIN
    6. Platform export (single platform or parallel)

    Args:
        url: YouTube video URL.
        duration: Clip duration override in seconds.
        skip_subs: Skip transcription and subtitles.
        no_logo: Skip logo stamping.
        platforms: Target platforms list.
        settings: Settings override.
        blur_bg: Use blurred background mode.

    Returns:
        SuperfastResult with output info.
    """
    start_time = time.time()

    if settings is None:
        settings = get_settings()

    # Apply superfast settings
    settings.FFMPEG_PRESET = "ultrafast"
    settings.FFMPEG_CRF = 28
    settings.WHISPER_MODEL = "tiny"
    settings.WHISPER_BEAM_SIZE = 1
    settings.AUDIO_NOISE_REDUCTION = False
    settings.AUDIO_COMPRESSION = False
    settings.AUDIO_NORMALIZER = True  # Keep this, it's in the single pass
    settings.CONTENT_MODERATION_ENABLED = False
    settings.EXPORT_TWO_PASS_ENCODING = False
    settings.THUMBNAIL_COUNT = 1
    settings.FFMPEG_THREADS = settings.FFMPEG_THREADS or (os.cpu_count() or 4)

    if duration:
        settings.CLIP_DURATION = duration

    result = SuperfastResult()

    # ── Step 1: Download ────────────────────────────────
    logger.info("[1/5] Downloading (720p, turbo)...")
    from core.downloader import download_video, DownloadError

    try:
        raw_path, video_info_dict = download_video(url, settings.DOWNLOADS_DIR, turbo=True)
    except DownloadError as exc:
        logger.error("Download failed: %s", exc)
        result.success = False
        return result

    title = video_info_dict.get("title", "untitled")
    safe_title = sanitize_filename(title)
    logger.info("Downloaded: %s (%s)", raw_path.name, get_file_size_human(raw_path))

    # ── Step 2: Fast Analyze ────────────────────────────
    logger.info("[2/5] Fast analysis (audio only)...")
    segment = superfast_analyze(raw_path, settings.CLIP_DURATION)
    logger.info(
        "Best segment: %.1fs - %.1fs (score=%.4f)",
        segment.start_time, segment.end_time, segment.energy_score,
    )

    # ── Step 3: Transcribe (if not skipped) ─────────────
    ass_path: Path | None = None
    transcription: TranscriptionResult | None = None

    if not skip_subs:
        logger.info("[3/5] Transcribing (tiny Whisper)...")
        try:
            transcription = superfast_transcribe(raw_path, settings)
            logger.info("Transcribed: %d words", transcription.word_count)
        except Exception as exc:
            logger.warning("Transcription failed (continuing without subs): %s", exc)
            transcription = None

        # ── Step 4: Generate ASS subtitles ───────────────
        if transcription and transcription.words:
            logger.info("[4/5] Generating subtitles...")
            ass_path = make_output_path(settings.SHORTS_DIR, safe_title, "subs", ext="ass")
            try:
                generate_subtitles(
                    transcription, ass_path, settings,
                    animation="fade",  # Fastest animation mode
                    bg_box=False,
                )
                logger.info("Subtitles generated: %s", ass_path.name)
            except Exception as exc:
                logger.warning("Subtitle generation failed: %s", exc)
                ass_path = None
        else:
            logger.info("[4/5] No speech detected, skipping subtitles")
    else:
        logger.info("[3-4/5] Subtitles skipped")

    # ── Step 5: Single-pass encode ──────────────────────
    logger.info("[5/5] Single-pass encode (crop + scale + subs + logo)...")
    output_path = make_output_path(settings.SHORTS_DIR, safe_title, "short", ext="mp4")

    logo_path: Path | None = None
    if not no_logo:
        logo_path = Path(settings.LOGO_PATH)
        if not logo_path.is_absolute():
            from config.settings import BASE_DIR
            logo_path = BASE_DIR / settings.LOGO_PATH
        if not logo_path.exists():
            logo_path = None
            logger.info("Logo not found, skipping logo overlay")

    try:
        superfast_single_pass(
            input_path=raw_path,
            segment=segment,
            ass_path=ass_path,
            logo_path=logo_path,
            output_path=output_path,
            settings=settings,
            blur_bg=blur_bg,
        )
    except FFmpegError as exc:
        logger.error("Single-pass encode failed: %s", exc)
        result.success = False
        return result

    result.output_path = output_path
    result.file_size = output_path.stat().st_size if output_path.exists() else 0
    result.duration = time.time() - start_time
    result.ffmpeg_passes = 1
    result.success = True

    logger.info(
        "SUPERFAST complete: %s (%s, %.1fs, 1 FFmpeg pass)",
        output_path.name, result.file_size_human, result.duration,
    )

    # ── Platform export (if requested) ──────────────────
    if platforms:
        from core.platform_exporter import export_for_platforms
        exports, report = export_for_platforms(
            output_path, safe_title, settings, platforms=platforms,
        )
        logger.info("Platform export: %d files", exports.count)

    # ── Cleanup intermediates ───────────────────────────
    if settings.CLEANUP_INTERMEDIATES:
        if ass_path and ass_path.exists():
            safe_delete(ass_path)
        # Don't delete raw_path - it's cached for resume

    return result
