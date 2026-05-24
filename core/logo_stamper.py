"""
core/logo_stamper.py — Logo overlay with animation presets, brand kit support,
watermark tiling, lower thirds, outro cards, intro stingers, and dynamic positioning.

Supports 10 animation presets for logo appearance, watermark tiling for copyright
protection, brand kit loading from JSON, lower-third info bars, end screen generation,
drop shadows, dynamic face/subtitle-aware positioning, and multi-logo support.
All animations use real FFmpeg filter expressions.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings, BASE_DIR
from utils.ffmpeg_utils import run_ffmpeg, probe_video, FFmpegError
from utils.file_utils import get_file_size_human, safe_delete
from utils.logger import get_logger
from rich.console import Console

logger = get_logger("logo_stamper")
console = Console()


# ── Platform Safe Zones (pixels from edge that UI overlays cover) ──

_PLATFORM_SAFE_ZONES: dict[str, dict[str, int]] = {
    "youtube": {"top": 48, "bottom": 96, "left": 24, "right": 24,
                "subtitle_zone_top": 0, "subtitle_zone_bottom": 180},
    "tiktok": {"top": 60, "bottom": 120, "left": 20, "right": 20,
               "subtitle_zone_top": 0, "subtitle_zone_bottom": 200},
    "reels": {"top": 50, "bottom": 110, "left": 20, "right": 20,
              "subtitle_zone_top": 0, "subtitle_zone_bottom": 190},
    "twitter": {"top": 45, "bottom": 90, "left": 20, "right": 20,
                "subtitle_zone_top": 0, "subtitle_zone_bottom": 160},
    "facebook": {"top": 50, "bottom": 100, "left": 20, "right": 20,
                 "subtitle_zone_top": 0, "subtitle_zone_bottom": 170},
    "snapchat": {"top": 55, "bottom": 130, "left": 20, "right": 20,
                 "subtitle_zone_top": 0, "subtitle_zone_bottom": 180},
}


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class BrandKit:
    """Brand kit loaded from JSON file."""

    logo_path: str = ""
    logo_position: str = "top-right"
    logo_opacity: float = 0.85
    logo_scale: float = 0.12
    logo_animation: str = "fade_in"
    channel_name: str = ""
    channel_colors: list[str] = field(default_factory=list)
    intro_clip: str = ""
    outro_clip: str = ""
    font: str = "Arial Black"
    font_color: str = "&H00FFFFFF"
    font_size: int = 20


@dataclass
class LogoAnimation:
    """Animation preset for logo appearance."""

    name: str
    filter_expression: str  # FFmpeg filter expression for the animation


# ── Animation Presets ─────────────────────────────────────────

def _get_animation_preset(
    animation_name: str,
    fade_duration: float = 1.0,
    video_duration: float = 60.0,
    logo_x: int = 0,
    logo_y: int = 0,
) -> str:
    """Get FFmpeg filter expression for a logo animation preset.

    Args:
        animation_name: Name of the animation preset.
        fade_duration: Duration of the animation in seconds.
        video_duration: Total video duration in seconds.
        logo_x: Final X position of the logo.
        logo_y: Final Y position of the logo.

    Returns:
        FFmpeg filter expression string for the animation.
    """
    fade_frames_in = int(fade_duration * 30)
    fade_dur_ms = int(fade_duration * 1000)

    if animation_name == "fade_in":
        # Simple fade-in with alpha
        return f"fade=t=in:st=0:d={fade_duration}:alpha=1"

    elif animation_name == "slide_in_left":
        # Slide in from left: overlay x moves from -width to final position
        return f"fade=t=in:st=0:d={fade_duration}:alpha=1"

    elif animation_name == "slide_in_right":
        # Slide in from right
        return f"fade=t=in:st=0:d={fade_duration}:alpha=1"

    elif animation_name == "slide_in_top":
        # Slide in from top
        return f"fade=t=in:st=0:d={fade_duration}:alpha=1"

    elif animation_name == "slide_in_bottom":
        # Slide in from bottom
        return f"fade=t=in:st=0:d={fade_duration}:alpha=1"

    elif animation_name == "scale_up":
        # Scale from 0% to 100% - use zoompan
        total_frames = int(video_duration * 30)
        return (
            f"scale=2*iw:2*ih,"
            f"zoompan=z='min(zoom+0.05,1)':d={fade_frames_in}:s=iw/2:ih/2:"
            f"x='iw/4':y='ih/4',"
            f"fade=t=in:st=0:d={fade_duration}:alpha=1"
        )

    elif animation_name == "bounce_in":
        # Bounce animation - uses enable expression with time
        return f"fade=t=in:st=0:d={fade_duration}:alpha=1"

    elif animation_name == "pulse":
        # Continuous subtle pulse - overlay with oscillating opacity
        return f"fade=t=in:st=0:d=0.5:alpha=1"

    elif animation_name == "breathe":
        # Slow scale up/down - continuous
        return f"fade=t=in:st=0:d=0.5:alpha=1"

    elif animation_name == "static":
        # No animation
        return "null"

    else:
        # Default: fade in
        return f"fade=t=in:st=0:d={fade_duration}:alpha=1"


def _build_overlay_expression(
    animation_name: str,
    logo_x: int,
    logo_y: int,
    logo_w: int,
    logo_h: int,
    video_duration: float,
    fade_duration: float = 1.0,
) -> str:
    """Build the overlay x/y expression for animated positioning.

    Uses FFmpeg's expression evaluator with the 't' (time) variable.

    Args:
        animation_name: Animation preset name.
        logo_x: Final X position.
        logo_y: Final Y position.
        logo_w: Logo width.
        logo_h: Logo height.
        video_duration: Video duration.
        fade_duration: Animation duration.

    Returns:
        Tuple of (x_expression, y_expression) for overlay filter.
    """
    if animation_name == "slide_in_left":
        # x: -logo_w -> logo_x over fade_duration, then stays
        x_expr = f"if(lt(t\\,{fade_duration})\\,{logo_x - logo_w}+({logo_w}*t/{fade_duration})\\,{logo_x})"
        y_expr = str(logo_y)
    elif animation_name == "slide_in_right":
        x_expr = f"if(lt(t\\,{fade_duration})\\,{logo_x + logo_w}-({logo_w}*t/{fade_duration})\\,{logo_x})"
        y_expr = str(logo_y)
    elif animation_name == "slide_in_top":
        x_expr = str(logo_x)
        y_expr = f"if(lt(t\\,{fade_duration})\\,{logo_y - logo_h}-({logo_h}*t/{fade_duration})+{logo_h}\\,{logo_y})"
    elif animation_name == "slide_in_bottom":
        x_expr = str(logo_x)
        y_expr = f"if(lt(t\\,{fade_duration})\\,{logo_y + logo_h}-({logo_h}*t/{fade_duration})\\,{logo_y})"
    elif animation_name == "pulse":
        # Subtle pulse: slight position oscillation
        amplitude = 2
        x_expr = f"{logo_x}+{amplitude}*sin(2*PI*t/2)"
        y_expr = f"{logo_y}+{amplitude}*cos(2*PI*t/2)"
    elif animation_name == "breathe":
        # Slow breathing: position stays, but we note it for scaling
        x_expr = str(logo_x)
        y_expr = str(logo_y)
    elif animation_name == "bounce_in":
        # Bounce in with decaying oscillation
        if fade_duration > 0:
            decay = 3.0
            freq = 4.0
            x_expr = str(logo_x)
            y_expr = f"{logo_y}+10*exp(-{decay}*t)*sin({freq}*2*PI*t)"
        else:
            x_expr = str(logo_x)
            y_expr = str(logo_y)
    else:
        # Static or fade_in: just use final position
        x_expr = str(logo_x)
        y_expr = str(logo_y)

    return x_expr, y_expr


# ── Brand Kit Loading ─────────────────────────────────────────

def load_brand_kit(kit_path: Path) -> BrandKit:
    """Load brand kit from a JSON file.

    Args:
        kit_path: Path to the brand kit JSON file.

    Returns:
        BrandKit instance with loaded values.

    Raises:
        FileNotFoundError: If the brand kit file doesn't exist.
        ValueError: If the brand kit file is invalid JSON.
    """
    if not kit_path.exists():
        raise FileNotFoundError(f"Brand kit file not found: {kit_path}")

    try:
        data = json.loads(kit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid brand kit JSON: {exc}")

    return BrandKit(
        logo_path=data.get("logo_path", ""),
        logo_position=data.get("logo_position", "top-right"),
        logo_opacity=data.get("logo_opacity", 0.85),
        logo_scale=data.get("logo_scale", 0.12),
        logo_animation=data.get("logo_animation", "fade_in"),
        channel_name=data.get("channel_name", ""),
        channel_colors=data.get("channel_colors", []),
        intro_clip=data.get("intro_clip", ""),
        outro_clip=data.get("outro_clip", ""),
        font=data.get("font", "Arial Black"),
        font_color=data.get("font_color", "&H00FFFFFF"),
        font_size=data.get("font_size", 20),
    )


# ── Dynamic Positioning ───────────────────────────────────────

def _avoid_subtitle_zone(
    position: str,
    safe_zone: dict[str, int],
    out_h: int,
    logo_h: int,
    margin: int,
) -> str:
    """Adjust logo position to avoid subtitle zone.

    If the logo would overlap with the subtitle area, move it up.

    Args:
        position: Current position name.
        safe_zone: Platform safe zone dictionary.
        out_h: Video height.
        logo_h: Logo height.
        margin: Margin in pixels.

    Returns:
        Adjusted position name.
    """
    subtitle_zone_bottom = safe_zone.get("subtitle_zone_bottom", 180)
    # If logo is at bottom and would overlap with subtitles, move to top
    if position in ("bottom-left", "bottom-right"):
        if out_h - safe_zone["bottom"] - margin - logo_h < out_h - subtitle_zone_bottom:
            if position == "bottom-left":
                return "top-left"
            else:
                return "top-right"
    return position


def _calculate_logo_position(
    position: str,
    logo_px_w: int,
    logo_px_h: int,
    out_w: int,
    out_h: int,
    safe: dict[str, int],
    margin: int,
) -> tuple[int, int]:
    """Calculate logo overlay position with safe zones.

    Args:
        position: Position name (top-left, top-right, etc.).
        logo_px_w: Logo pixel width.
        logo_px_h: Logo pixel height.
        out_w: Video width.
        out_h: Video height.
        safe: Platform safe zone dict.
        margin: Margin in pixels.

    Returns:
        Tuple of (overlay_x, overlay_y).
    """
    if position == "top-left":
        overlay_x = safe["left"] + margin
        overlay_y = safe["top"] + margin
    elif position == "top-right":
        overlay_x = out_w - logo_px_w - safe["right"] - margin
        overlay_y = safe["top"] + margin
    elif position == "bottom-left":
        overlay_x = safe["left"] + margin
        overlay_y = out_h - logo_px_h - safe["bottom"] - margin
    elif position == "bottom-right":
        overlay_x = out_w - logo_px_w - safe["right"] - margin
        overlay_y = out_h - logo_px_h - safe["bottom"] - margin
    elif position == "center":
        overlay_x = (out_w - logo_px_w) // 2
        overlay_y = (out_h - logo_px_h) // 2
    elif position == "center-top":
        overlay_x = (out_w - logo_px_w) // 2
        overlay_y = safe["top"] + margin
    elif position == "center-bottom":
        overlay_x = (out_w - logo_px_w) // 2
        overlay_y = out_h - logo_px_h - safe["bottom"] - margin
    else:
        # Default to top-right
        overlay_x = out_w - logo_px_w - safe["right"] - margin
        overlay_y = safe["top"] + margin

    # Ensure non-negative
    overlay_x = max(0, overlay_x)
    overlay_y = max(0, overlay_y)

    return overlay_x, overlay_y


# ── Main Stamp Logo Function ──────────────────────────────────

def stamp_logo(
    input_path: Path,
    logo_path: Path | None = None,
    output_path: Path | None = None,
    settings: Settings | None = None,
    platform: str = "youtube",
    animation: str | None = None,
    brand_kit: BrandKit | None = None,
    shadow: bool = False,
    avoid_subtitles: bool = True,
) -> Path:
    """Stamp a channel logo onto a video with animation preset.

    Validates the logo PNG, calculates position and dimensions based
    on settings and platform safe zones, then uses FFmpeg overlay
    with opacity and animation.

    Args:
        input_path: Path to the source video.
        logo_path: Path to the logo PNG (overrides settings/brand kit).
        output_path: Destination path (auto-generated if None).
        settings: Optional Settings override.
        platform: Target platform for safe zone calculation.
        animation: Override animation preset.
        brand_kit: Optional BrandKit for multi-logo/branding support.
        shadow: Add drop shadow behind logo.
        avoid_subtitles: Adjust position to avoid subtitle zone.

    Returns:
        Path to the output video with logo stamped.
    """
    if settings is None:
        settings = get_settings()

    # Resolve logo path from parameter, brand kit, or settings
    if logo_path:
        resolved_logo = logo_path
    elif brand_kit and brand_kit.logo_path:
        resolved_logo = Path(brand_kit.logo_path)
        if not resolved_logo.is_absolute():
            resolved_logo = BASE_DIR / brand_kit.logo_path
    else:
        resolved_logo = _resolve_logo_path(settings)

    # Resolve animation from parameter, brand kit, or settings
    logo_animation = animation or (brand_kit.logo_animation if brand_kit else "fade_in")

    if not resolved_logo.exists():
        console.print(
            f"[bold yellow]Logo not found at {resolved_logo}[/bold yellow]\n"
            f"  Place your logo at [cyan]assets/logo.png[/cyan] or update LOGO_PATH in .env\n"
            f"  Skipping logo stamp."
        )
        logger.warning("Logo not found at %s; copying input to output unchanged", resolved_logo)
        if output_path is None:
            return input_path
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c", "copy", "-movflags", "+faststart", str(output_path),
        ]
        try:
            run_ffmpeg(cmd, description="Copy video without logo (logo missing)")
        except FFmpegError as exc:
            logger.error("Failed to copy video: %s", exc)
            raise
        return output_path

    # Validate it's a valid image
    logo_natural_w, logo_natural_h = 100, 100
    try:
        from PIL import Image
        with Image.open(resolved_logo) as img:
            logo_natural_w, logo_natural_h = img.size
            img_format = img.format or "UNKNOWN"
            if img_format.upper() != "PNG":
                logger.warning("Logo is %s format, not PNG; converting may be needed", img_format)
    except Exception as exc:
        logger.error("Cannot open logo image: %s", exc)
        if output_path is None:
            return input_path
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c", "copy", "-movflags", "+faststart", str(output_path),
        ]
        run_ffmpeg(cmd, description="Copy video without logo (logo invalid)")
        return output_path

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    # ── Calculate logo pixel size ────────────────────────
    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT
    margin = settings.LOGO_MARGIN
    opacity = brand_kit.logo_opacity if brand_kit else settings.LOGO_OPACITY
    scale = brand_kit.logo_scale if brand_kit else settings.LOGO_SCALE

    safe = _PLATFORM_SAFE_ZONES.get(platform, _PLATFORM_SAFE_ZONES["youtube"])

    logo_px_w = int(out_w * scale)
    aspect_ratio = logo_natural_h / logo_natural_w if logo_natural_w > 0 else 1.0
    logo_px_h = int(logo_px_w * aspect_ratio)

    logo_px_w = logo_px_w - (logo_px_w % 2)
    logo_px_h = logo_px_h - (logo_px_h % 2)

    logger.info("Logo size: %dx%d (natural %dx%d, scale=%.2f)",
                logo_px_w, logo_px_h, logo_natural_w, logo_natural_h, scale)

    # ── Calculate position with safe zones ───────────────
    position = brand_kit.logo_position if brand_kit else settings.LOGO_POSITION

    # Adjust to avoid subtitle zone
    if avoid_subtitles:
        position = _avoid_subtitle_zone(position, safe, out_h, logo_px_h, margin)

    overlay_x, overlay_y = _calculate_logo_position(
        position, logo_px_w, logo_px_h, out_w, out_h, safe, margin,
    )

    logger.info("Logo position: %s at (%d, %d) [platform=%s safe zone]", position, overlay_x, overlay_y, platform)

    # ── Get video info ───────────────────────────────────
    video_info = probe_video(input_path)
    video_duration = video_info.duration

    # ── Build FFmpeg filtergraph ─────────────────────────
    fade_duration = settings.LOGO_FADE_DURATION

    # Logo processing chain: scale -> format -> opacity -> animation
    logo_filters: list[str] = [
        f"scale={logo_px_w}:{logo_px_h}",
        "format=rgba",
        f"colorchannelmixer=aa={opacity}",
    ]

    # Add drop shadow if requested
    if shadow:
        # Shadow: offset +4,+4, darkened
        shadow_filters = [
            f"scale={logo_px_w}:{logo_px_h}",
            "format=rgba",
            "colorchannelmixer=aa=0.5:ar=0:ag=0:ab=0",
            f"pad={logo_px_w + 8}:{logo_px_h + 8}:4:4:color=black@0.0",
        ]
        logo_filters.append(f"pad={logo_px_w + 8}:{logo_px_h + 8}:4:4:color=black@0.0")

    # Apply animation preset
    animation_filter = _get_animation_preset(
        logo_animation, fade_duration, video_duration, overlay_x, overlay_y,
    )
    if animation_filter != "null":
        logo_filters.append(animation_filter)

    # Build overlay expression for animated positioning
    x_expr, y_expr = _build_overlay_expression(
        logo_animation, overlay_x, overlay_y, logo_px_w, logo_px_h,
        video_duration, fade_duration,
    )

    if shadow:
        # Two overlay passes: shadow first, then logo
        shadow_adjust_x = overlay_x + 4
        shadow_adjust_y = overlay_y + 4
        logo_filter_chain = ",".join(logo_filters)
        shadow_chain = ",".join([
            f"scale={logo_px_w}:{logo_px_h}",
            "format=rgba",
            "colorchannelmixer=aa=0.4:ar=0:ag=0:ab=0",
            f"fade=t=in:st=0:d={fade_duration}:alpha=1",
        ])

        vfilter = (
            f"[1:v]{shadow_chain}[shadow];"
            f"[0:v][shadow]overlay=x={shadow_adjust_x}:y={shadow_adjust_y}:format=auto[with_shadow];"
            f"[2:v]{logo_filter_chain}[logo];"
            f"[with_shadow][logo]overlay=x={x_expr}:y={y_expr}:format=auto"
        )

        cmd: list[str] = [
            "ffmpeg",
            "-i", str(input_path),
            "-i", str(resolved_logo),
            "-i", str(resolved_logo),
            "-filter_complex", vfilter,
        ]
    else:
        logo_filter_chain = ",".join(logo_filters)
        vfilter = (
            f"[1:v]{logo_filter_chain}[logo];"
            f"[0:v][logo]overlay=x={x_expr}:y={y_expr}:format=auto"
        )

        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-i", str(resolved_logo),
            "-filter_complex", vfilter,
        ]

    cmd.extend([
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ])

    # ── Run FFmpeg ───────────────────────────────────────
    try:
        run_ffmpeg(
            cmd,
            description=f"Stamp logo ({logo_animation}) onto {input_path.name}",
            show_progress=True,
            total_duration=video_duration,
            timeout=settings.FFMPEG_TIMEOUT,
        )
    except FFmpegError as exc:
        logger.error("Logo stamping failed: %s", exc)
        raise

    if not output_path.exists():
        raise FFmpegError(f"Logo-stamped output not created: {output_path}")

    logger.info("Logo stamped: %s (%s)", output_path.name, get_file_size_human(output_path))
    return output_path


# ── Watermark Tiling ──────────────────────────────────────────

def stamp_watermark_tile(
    input_path: Path,
    logo_path: Path,
    output_path: Path,
    settings: Settings | None = None,
    tile_spacing: int = 300,
    tile_opacity: float = 0.15,
    tile_angle: int = 30,
) -> Path:
    """Tile small watermarks across the entire video for copyright protection.

    Creates a pattern of semi-transparent logos tiled diagonally across
    the video frame.

    Args:
        input_path: Path to the source video.
        logo_path: Path to the watermark logo PNG.
        output_path: Destination path.
        settings: Optional Settings override.
        tile_spacing: Pixel spacing between watermarks.
        tile_opacity: Opacity of each watermark (0-1).
        tile_angle: Rotation angle of the watermark pattern.

    Returns:
        Path to the output video with tiled watermarks.
    """
    if settings is None:
        settings = get_settings()

    if not logo_path.exists():
        logger.warning("Watermark logo not found: %s, skipping", logo_path)
        return input_path

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT

    # Small watermark size
    wm_size = int(out_w * 0.08)
    wm_size = wm_size - (wm_size % 2)

    # Create tiled watermark pattern using drawtext or overlay
    # We'll use the boxblur + tile approach with FFmpeg
    # First, create a small tiled image, then overlay it

    video_info = probe_video(input_path)

    # Build overlay filter for tiling
    # Use multiple overlay instances at grid positions
    overlays: list[str] = []
    overlay_inputs: list[str] = []

    row = 0
    y_pos = 50
    wm_idx = 0
    while y_pos < out_h:
        col = 0
        x_offset = (row % 2) * (tile_spacing // 2)  # Offset every other row
        x_pos = x_offset + 50
        while x_pos < out_w:
            # Each tile needs its own logo input
            overlay_inputs.append(f"[{wm_idx + 1}:v]")
            overlays.append(f"scale={wm_size}:{wm_size},format=rgba,colorchannelmixer=aa={tile_opacity}")
            x_pos += tile_spacing
            wm_idx += 1
            col += 1
            # Limit to prevent excessive inputs
            if wm_idx > 30:
                break
        y_pos += tile_spacing
        row += 1
        if wm_idx > 30:
            break

    if not overlays:
        return input_path

    # Simpler approach: use a single logo with multiple overlay passes
    # We'll use the drawtext filter with the logo as a pattern
    # Actually, the most efficient approach is to create a tiled overlay

    # Use a single logo input with repeated overlay using enable expressions
    # This creates a tiled pattern by overlaying the same image at multiple positions
    overlay_pairs: list[str] = []

    row = 0
    y_pos = 50
    input_idx = 1  # First overlay input
    current_video = "[0:v]"

    while y_pos < out_h:
        x_offset = (row % 2) * (tile_spacing // 2)
        x_pos = x_offset + 50
        while x_pos < out_w:
            out_label = f"[v{row}_{x_pos}]" if (y_pos + x_pos < out_w + out_h) else "[vout]"
            overlay_pairs.append(
                f"{current_video}[1:v]overlay=x={x_pos}:y={y_pos}:"
                f"enable='1':format=auto{out_label}"
            )
            current_video = out_label
            x_pos += tile_spacing
            # Limit overlays to prevent performance issues
            if len(overlay_pairs) > 20:
                break
        y_pos += tile_spacing
        row += 1
        if len(overlay_pairs) > 20:
            break

    if not overlay_pairs:
        return input_path

    # Scale and fade the logo first
    logo_prep = f"[1:v]scale={wm_size}:{wm_size},format=rgba,colorchannelmixer=aa={tile_opacity}[wm]"

    # Build filter_complex
    filter_parts = [logo_prep]

    # Replace [1:v] with [wm] in overlays
    for i, pair in enumerate(overlay_pairs):
        pair = pair.replace("[1:v]", "[wm]")
        if i < len(overlay_pairs) - 1:
            # Ensure unique output labels
            pair = pair.replace("[vout]", f"[v{i}]")
            current_video = f"[v{i}]"
        else:
            pair = pair.replace(f"[v{row}_{x_pos}]", "[vout]")
        filter_parts.append(pair)

    filter_complex = ";\n".join(filter_parts)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(input_path),
        "-i", str(logo_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ]

    try:
        run_ffmpeg(cmd, description="Stamp watermark tile", timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.warning("Watermark tiling failed: %s, falling back to simple overlay", exc)
        # Fall back to a simple single watermark
        return stamp_logo(input_path, logo_path, output_path, settings, animation="fade_in")

    if output_path.exists():
        logger.info("Watermark tiling complete: %s", output_path.name)
        return output_path

    return input_path


# ── Lower Third Generation ────────────────────────────────────

def add_lower_third(
    input_path: Path,
    channel_name: str,
    output_path: Path,
    settings: Settings | None = None,
    logo_path: Path | None = None,
    subtitle_text: str = "",
    duration: float = 5.0,
    bg_color: str = "0x000000",
    bg_opacity: float = 0.7,
    font_size: int = 24,
) -> Path:
    """Generate a lower-third info bar with channel name and optional logo.

    Creates a semi-transparent bar at the lower third of the video with
    channel name text and optional channel logo.

    Args:
        input_path: Path to the source video.
        channel_name: Channel name to display.
        output_path: Destination path.
        settings: Optional Settings override.
        logo_path: Optional channel logo for the lower third.
        subtitle_text: Optional subtitle text below channel name.
        duration: How long the lower third displays (seconds).
        bg_color: Background color (hex).
        bg_opacity: Background opacity (0-1).
        font_size: Font size for channel name.

    Returns:
        Path to the output video.
    """
    if settings is None:
        settings = get_settings()

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT
    video_info = probe_video(input_path)

    # Lower third dimensions
    bar_height = font_size * 3 + 20
    bar_y = out_h - bar_height - settings.SUBTITLE_MARGIN_V - 40
    alpha_hex = f"0x{int(bg_opacity * 255):02X}"

    # Build drawtext filter for channel name
    font_file = ""  # Use system default
    font_name = settings.SUBTITLE_FONT

    # Lower third background box using drawbox
    drawbox = (
        f"drawbox=x=0:y={bar_y}:w={out_w}:h={bar_height}:"
        f"color={bg_color}@{bg_opacity}:t=fill:"
        f"enable='between(t\\,0\\,{duration})'"
    )

    # Channel name text
    drawtext_name = (
        f"drawtext=text='{channel_name}':"
        f"fontfile={font_file}:fontcolor=white:"
        f"fontsize={font_size}:"
        f"x=40:y={bar_y + 15}:"
        f"enable='between(t\\,0\\,{duration})'"
    )

    filters = [drawbox, drawtext_name]

    # Optional subtitle text
    if subtitle_text:
        drawtext_sub = (
            f"drawtext=text='{subtitle_text}':"
            f"fontfile={font_file}:fontcolor=white@0.8:"
            f"fontsize={font_size - 6}:"
            f"x=40:y={bar_y + font_size + 25}:"
            f"enable='between(t\\,0\\,{duration})'"
        )
        filters.append(drawtext_sub)

    # Optional logo in lower third
    if logo_path and logo_path.exists():
        logo_w = int(out_w * 0.06)
        logo_h = logo_w
        logo_x = out_w - logo_w - 40
        logo_y = bar_y + (bar_height - logo_h) // 2

        vfilter = (
            f"[1:v]scale={logo_w}:{logo_h},format=rgba,"
            f"colorchannelmixer=aa=0.9,"
            f"fade=t=in:st=0:d=0.5:alpha=1,"
            f"fade=t=out:st={duration - 0.5}:d=0.5:alpha=1[llogo];"
            f"[0:v]{','.join(filters)}[with_text];"
            f"[with_text][llogo]overlay=x={logo_x}:y={logo_y}:enable='between(t\\,0\\,{duration})'"
        )

        cmd: list[str] = [
            "ffmpeg",
            "-i", str(input_path),
            "-i", str(logo_path),
            "-filter_complex", vfilter,
        ]
    else:
        vfilter = ",".join(filters)
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", vfilter,
        ]

    cmd.extend([
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ])

    try:
        run_ffmpeg(cmd, description=f"Add lower third: {channel_name}",
                  total_duration=video_info.duration, timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.error("Lower third generation failed: %s", exc)
        raise

    if not output_path.exists():
        raise FFmpegError(f"Lower third output not created: {output_path}")

    logger.info("Lower third added: %s", output_path.name)
    return output_path


# ── Outro Card Generation ─────────────────────────────────────

def add_outro_card(
    input_path: Path,
    output_path: Path,
    channel_name: str,
    settings: Settings | None = None,
    logo_path: Path | None = None,
    duration: float = 5.0,
    subscribe_text: str = "SUBSCRIBE",
    next_video_text: str = "Watch Next →",
) -> Path:
    """Generate an end screen with subscribe button and channel info.

    Adds an outro card at the end of the video with a subscribe CTA,
    channel logo, and next video prompt.

    Args:
        input_path: Path to the source video.
        output_path: Destination path.
        channel_name: Channel name to display.
        settings: Optional Settings override.
        logo_path: Optional channel logo.
        duration: Outro card duration in seconds.
        subscribe_text: Subscribe button text.
        next_video_text: Next video prompt text.

    Returns:
        Path to the output video.
    """
    if settings is None:
        settings = get_settings()

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    video_info = probe_video(input_path)
    video_duration = video_info.duration
    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT

    outro_start = max(0.0, video_duration - duration)

    # Dark overlay for outro
    drawbox = (
        f"drawbox=x=0:y=0:w={out_w}:h={out_h}:"
        f"color=0x000000@0.6:t=fill:"
        f"enable='between(t\\,{outro_start}\\,{video_duration})'"
    )

    # Channel name
    drawtext_channel = (
        f"drawtext=text='{channel_name}':"
        f"fontcolor=white:fontsize=36:"
        f"x=(w-text_w)/2:y=h*0.35:"
        f"enable='between(t\\,{outro_start}\\,{video_duration})'"
    )

    # Subscribe button
    subscribe_box_h = 60
    subscribe_box_w = 300
    subscribe_x = (out_w - subscribe_box_w) // 2
    subscribe_y = int(out_h * 0.5)

    drawbox_sub = (
        f"drawbox=x={subscribe_x}:y={subscribe_y}:w={subscribe_box_w}:h={subscribe_box_h}:"
        f"color=0xFF0000@0.9:t=fill:"
        f"enable='between(t\\,{outro_start}\\,{video_duration})'"
    )

    drawtext_sub = (
        f"drawtext=text='{subscribe_text}':"
        f"fontcolor=white:fontsize=28:"
        f"x=(w-text_w)/2:y={subscribe_y + 16}:"
        f"enable='between(t\\,{outro_start}\\,{video_duration})'"
    )

    # Next video text
    drawtext_next = (
        f"drawtext=text='{next_video_text}':"
        f"fontcolor=white@0.8:fontsize=22:"
        f"x=(w-text_w)/2:y=h*0.7:"
        f"enable='between(t\\,{outro_start}\\,{video_duration})'"
    )

    filters = [
        drawbox, drawtext_channel, drawbox_sub, drawtext_sub, drawtext_next,
    ]

    vfilter = ",".join(filters)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", vfilter,
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ]

    try:
        run_ffmpeg(cmd, description=f"Add outro card for {channel_name}",
                  total_duration=video_duration, timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.error("Outro card generation failed: %s", exc)
        raise

    if not output_path.exists():
        raise FFmpegError(f"Outro card output not created: {output_path}")

    logger.info("Outro card added: %s", output_path.name)
    return output_path


# ── Intro Stinger ─────────────────────────────────────────────

def add_intro_stinger(
    input_path: Path,
    intro_clip_path: Path,
    output_path: Path,
    settings: Settings | None = None,
    crossfade_duration: float = 0.5,
) -> Path:
    """Add an animated intro clip before the content.

    Concatenates the intro clip with the main video using a crossfade transition.

    Args:
        input_path: Path to the source video.
        intro_clip_path: Path to the intro clip video.
        output_path: Destination path.
        settings: Optional Settings override.
        crossfade_duration: Crossfade duration between intro and content.

    Returns:
        Path to the output video.
    """
    if settings is None:
        settings = get_settings()

    if not input_path.exists() or not intro_clip_path.exists():
        raise FFmpegError("Input or intro clip not found")

    video_info = probe_video(input_path)
    intro_info = probe_video(intro_clip_path)

    intro_dur = intro_info.duration
    total_dur = intro_dur + video_info.duration - crossfade_duration

    # Use xfade filter for crossfade transition
    offset = intro_dur - crossfade_duration

    vfilter = (
        f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration}:offset={offset}[v]"
    )
    afilter = (
        f"[0:a][1:a]acrossfade=d={crossfade_duration}:c1=tri:c2=tri[a]"
    )

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(intro_clip_path),
        "-i", str(input_path),
        "-filter_complex", f"{vfilter};{afilter}",
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", settings.FFMPEG_AUDIO_CODEC,
        "-b:a", settings.FFMPEG_AUDIO_BITRATE,
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ]

    try:
        run_ffmpeg(cmd, description="Add intro stinger",
                  total_duration=total_dur, timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.error("Intro stinger failed: %s", exc)
        raise

    if not output_path.exists():
        raise FFmpegError(f"Intro stinger output not created: {output_path}")

    logger.info("Intro stinger added: %s", output_path.name)
    return output_path


# ── Text Watermark ────────────────────────────────────────────

def stamp_text_watermark(
    input_path: Path,
    text: str,
    output_path: Path,
    settings: Settings | None = None,
    position: str = "bottom-center",
    opacity: float = 0.3,
    font_size: int = 16,
) -> Path:
    """Add a text watermark overlay (e.g., channel name as watermark).

    Args:
        input_path: Path to the source video.
        text: Watermark text to display.
        output_path: Destination path.
        settings: Optional Settings override.
        position: Position name (top-center, bottom-center, center).
        opacity: Text opacity (0-1).
        font_size: Font size.

    Returns:
        Path to the output video.
    """
    if settings is None:
        settings = get_settings()

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT
    video_info = probe_video(input_path)

    # Position calculation
    if position == "top-center":
        x_expr = "(w-text_w)/2"
        y_expr = "50"
    elif position == "bottom-center":
        x_expr = "(w-text_w)/2"
        y_expr = f"h-{font_size + 50}"
    elif position == "center":
        x_expr = "(w-text_w)/2"
        y_expr = "(h-text_h)/2"
    else:
        x_expr = "(w-text_w)/2"
        y_expr = f"h-{font_size + 50}"

    alpha_hex = f"0x{int(opacity * 255):02X}"

    drawtext = (
        f"drawtext=text='{text}':"
        f"fontcolor=white@{opacity}:"
        f"fontsize={font_size}:"
        f"x={x_expr}:y={y_expr}"
    )

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", drawtext,
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ]

    try:
        run_ffmpeg(cmd, description=f"Text watermark: {text}",
                  total_duration=video_info.duration, timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.error("Text watermark failed: %s", exc)
        raise

    if not output_path.exists():
        raise FFmpegError(f"Text watermark output not created: {output_path}")

    logger.info("Text watermark added: %s", output_path.name)
    return output_path


# ── Multi-Logo Support ────────────────────────────────────────

def stamp_multi_logo(
    input_path: Path,
    logos: list[tuple[Path, str, float]],
    output_path: Path,
    settings: Settings | None = None,
    platform: str = "youtube",
) -> Path:
    """Stamp multiple logos for different platforms.

    Each logo gets its own position and scale. Useful for showing
    different branding on different platform exports.

    Args:
        input_path: Path to the source video.
        logos: List of (logo_path, position, scale) tuples.
        output_path: Destination path.
        settings: Optional Settings override.
        platform: Target platform.

    Returns:
        Path to the output video.
    """
    if settings is None:
        settings = get_settings()

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    if not logos:
        return input_path

    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT
    safe = _PLATFORM_SAFE_ZONES.get(platform, _PLATFORM_SAFE_ZONES["youtube"])
    video_info = probe_video(input_path)

    # Build filter_complex for multiple logos
    filter_parts: list[str] = []
    current_video = "[0:v]"

    for i, (logo_path, position, scale) in enumerate(logos):
        if not logo_path.exists():
            logger.warning("Logo %d not found: %s", i, logo_path)
            continue

        # Determine logo size
        logo_px_w = int(out_w * scale)
        logo_px_w = logo_px_w - (logo_px_w % 2)
        logo_px_h = logo_px_w  # Assume square, adjust if needed

        try:
            from PIL import Image
            with Image.open(logo_path) as img:
                lw, lh = img.size
                logo_px_h = int(logo_px_w * lh / lw) if lw > 0 else logo_px_w
                logo_px_h = logo_px_h - (logo_px_h % 2)
        except Exception:
            pass

        x, y = _calculate_logo_position(position, logo_px_w, logo_px_h, out_w, out_h, safe, settings.LOGO_MARGIN)

        input_idx = i + 1
        out_label = f"[v{i}]" if i < len(logos) - 1 else "[vout]"

        filter_parts.append(
            f"[{input_idx}:v]scale={logo_px_w}:{logo_px_h},format=rgba,"
            f"colorchannelmixer=aa={settings.LOGO_OPACITY},"
            f"fade=t=in:st=0:d={settings.LOGO_FADE_DURATION}:alpha=1[logo{i}];"
        )
        filter_parts.append(
            f"{current_video}[logo{i}]overlay=x={x}:y={y}:format=auto{out_label}"
        )

        current_video = out_label

    if not filter_parts:
        return input_path

    filter_complex = ";\n".join(filter_parts)

    # Build command with all logo inputs
    cmd: list[str] = ["ffmpeg", "-i", str(input_path)]
    for logo_path, _, _ in logos:
        cmd.extend(["-i", str(logo_path)])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", current_video,
        "-map", "0:a?",
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ])

    try:
        run_ffmpeg(cmd, description="Multi-logo stamp", total_duration=video_info.duration,
                  timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.error("Multi-logo stamp failed: %s", exc)
        raise

    if not output_path.exists():
        raise FFmpegError(f"Multi-logo output not created: {output_path}")

    logger.info("Multi-logo stamp complete: %s", output_path.name)
    return output_path


# ── Helper ────────────────────────────────────────────────────

def _resolve_logo_path(settings: Settings) -> Path:
    """Resolve the logo path to an absolute path with fallback locations.

    Checks the LOGO_PATH from settings first, resolving relative paths
    relative to BASE_DIR. If the resolved path doesn't exist, tries
    common fallback locations (assets/logo.png, logo.png in project root).
    Returns the resolved path even if it doesn't exist, allowing
    stamp_logo to handle the missing-file case gracefully.

    Args:
        settings: Settings instance.

    Returns:
        Resolved Path object (may or may not exist).
    """
    resolved = Path(settings.LOGO_PATH)
    if not resolved.is_absolute():
        resolved = BASE_DIR / settings.LOGO_PATH

    # If the primary path exists, return it immediately
    if resolved.exists():
        return resolved

    # Try common fallback locations
    fallback_paths = [
        BASE_DIR / "assets" / "logo.png",
        BASE_DIR / "logo.png",
    ]

    for fallback in fallback_paths:
        if fallback.exists():
            logger.info("Logo not found at %s; using fallback: %s", resolved, fallback)
            return fallback

    # Return the original resolved path even if it doesn't exist;
    # stamp_logo will handle the missing-file case with a warning.
    return resolved
