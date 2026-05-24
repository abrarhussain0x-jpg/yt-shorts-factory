"""
core/subtitle_engine.py — Advanced ASS subtitle generator with 8+ animation modes.
Generates word-level highlighted ASS subtitles with karaoke, fade, pop,
glow, typewriter, bounce, wave, rainbow, neon, matrix, and 3d_rotate effects.
Includes smart line breaking, reading speed optimization, dynamic font scaling,
multi-line layout, background box, gradient highlight, word emphasis,
and SRT/VTT export.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import Settings, get_settings
from core.transcriber import TranscriptionResult, WordTimestamp
from utils.ffmpeg_utils import run_ffmpeg, probe_video, detect_hw_encoder, FFmpegError
from utils.file_utils import safe_delete
from utils.logger import get_logger

logger = get_logger("subtitle_engine")


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class SubtitleLine:
    """A grouped subtitle line for display."""

    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)
    is_question: bool = False
    is_exclamatory: bool = False
    emphasis_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def reading_speed_wpm(self) -> float:
        """Calculate reading speed in words per minute."""
        if self.duration <= 0:
            return 0.0
        return (self.word_count / self.duration) * 60.0


# ── ASS Format Helpers ────────────────────────────────────────

def ass_timestamp(seconds: float) -> str:
    """Convert float seconds to ASS timestamp format H:MM:SS.cc.

    ASS timestamps use centiseconds (hundredths of a second), not milliseconds.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds >= 100:
        centiseconds = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def ass_escape(text: str) -> str:
    """Escape special characters for ASS subtitle format.

    In ASS, curly braces {} and backslash need special handling.
    Newlines become \\N.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("\n", "\\N")
    return text


def _ass_color_to_rgb(ass_color: str) -> tuple[int, int, int]:
    """Convert ASS color string (&H00BBGGRR) to RGB tuple.

    Args:
        ass_color: ASS color string like &H00FFFF00.

    Returns:
        Tuple of (R, G, B) integers.
    """
    try:
        # Remove &H prefix
        hex_str = ass_color.replace("&H", "").replace("&", "")
        # Pad to 8 chars
        hex_str = hex_str.zfill(8)
        # ASS uses BGR order
        b = int(hex_str[2:4], 16)
        g = int(hex_str[4:6], 16)
        r = int(hex_str[6:8], 16)
        return (r, g, b)
    except (ValueError, IndexError):
        return (255, 255, 0)


def _rgb_to_ass_color(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to ASS color string.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        ASS color string like &H00BBGGRR.
    """
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _interpolate_color(color1: str, color2: str, factor: float) -> str:
    """Interpolate between two ASS colors.

    Args:
        color1: Start ASS color.
        color2: End ASS color.
        factor: Interpolation factor (0.0 = color1, 1.0 = color2).

    Returns:
        Interpolated ASS color string.
    """
    r1, g1, b1 = _ass_color_to_rgb(color1)
    r2, g2, b2 = _ass_color_to_rgb(color2)
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    return _rgb_to_ass_color(
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
    )


# ── Smart Line Breaking ───────────────────────────────────────

def _group_words_into_lines(
    words: list[WordTimestamp],
    max_words: int = 4,
    min_display_time: float = 0.3,
    overlap: float = 0.05,
    target_wpm: float = 250.0,
    max_chars_per_line: int = 32,
) -> list[SubtitleLine]:
    """Group word timestamps into display lines with smart breaking.

    Respects natural phrase boundaries (commas, conjunctions), reading
    speed constraints, screen space constraints, and syllable boundaries.

    Args:
        words: List of word timestamps to group.
        max_words: Maximum words per line.
        min_display_time: Minimum display time per line.
        overlap: Overlap between adjacent lines to prevent flicker.
        target_wpm: Target reading speed in words per minute.
        max_chars_per_line: Maximum characters per line.

    Returns:
        List of SubtitleLine objects.
    """
    if not words:
        return []

    # Conjunctions and phrase boundaries that should trigger breaks
    clause_enders = {",", ";", ":", "—", "–"}
    sentence_enders = {".", "!", "?"}
    conjunctions = {
        "and", "but", "or", "nor", "so", "yet", "for",
        "however", "although", "because", "while", "since",
        "therefore", "moreover", "furthermore", "meanwhile",
    }

    lines: list[SubtitleLine] = []
    current_words: list[WordTimestamp] = []
    current_text_len = 0

    for word in words:
        word_len = len(word.word)
        current_words.append(word)
        current_text_len += word_len + 1  # +1 for space

        word_end_char = word.word.rstrip()[-1:] if word.word else ""
        word_clean = word.word.rstrip(".,!?;:—–").lower()

        # Determine if we should break here
        is_sentence_end = word_end_char in sentence_enders
        is_clause_end = word_end_char in clause_enders
        is_conjunction = word_clean in conjunctions
        at_max_words = len(current_words) >= max_words
        at_max_chars = current_text_len >= max_chars_per_line
        reading_speed_ok = True

        # Check reading speed: if we have enough data, check WPM
        if len(current_words) >= 2:
            line_duration = current_words[-1].end - current_words[0].start
            if line_duration > 0:
                actual_wpm = (len(current_words) / line_duration) * 60.0
                # If reading speed is too fast, force a break earlier
                if actual_wpm > target_wpm * 1.5 and len(current_words) >= 2:
                    reading_speed_ok = False

        should_break = (
            is_sentence_end or
            at_max_words or
            at_max_chars or
            not reading_speed_ok or
            (is_clause_end and len(current_words) >= max_words - 1) or
            (is_conjunction and len(current_words) >= max_words - 1)
        )

        if should_break:
            text = " ".join(w.word for w in current_words)
            start = current_words[0].start
            end = current_words[-1].end

            # Ensure minimum display time
            if end - start < min_display_time:
                end = start + min_display_time

            # Adjust for reading speed if too fast
            min_duration_for_reading = (len(current_words) / target_wpm) * 60.0
            if end - start < min_duration_for_reading:
                end = start + min_duration_for_reading

            # Add overlap to prevent flicker
            end += overlap

            is_question = text.rstrip()[-1:] == "?"
            is_exclamatory = text.rstrip()[-1:] == "!"
            emphasis = sum(1 for w in current_words if w.word.isupper() and len(w.word) > 1) / max(1, len(current_words))

            lines.append(SubtitleLine(
                text=text, start=start, end=end,
                words=list(current_words),
                is_question=is_question,
                is_exclamatory=is_exclamatory,
                emphasis_score=round(emphasis, 2),
            ))
            current_words = []
            current_text_len = 0

    # Handle remaining words
    if current_words:
        text = " ".join(w.word for w in current_words)
        start = current_words[0].start
        end = current_words[-1].end + overlap

        if end - start < min_display_time:
            end = start + min_display_time

        is_question = text.rstrip()[-1:] == "?"
        is_exclamatory = text.rstrip()[-1:] == "!"

        lines.append(SubtitleLine(
            text=text, start=start, end=end,
            words=list(current_words),
            is_question=is_question,
            is_exclamatory=is_exclamatory,
        ))

    return lines


# ── Dynamic Font Scaling ──────────────────────────────────────

def _compute_font_scale(text: str, base_font_size: int, max_chars: int = 32) -> int:
    """Compute dynamic font size based on text length.

    Reduces font size for long lines to fit on screen.

    Args:
        text: The text to display.
        base_font_size: The base (default) font size.
        max_chars: Maximum characters that fit at base size.

    Returns:
        Adjusted font size.
    """
    char_count = len(text)
    if char_count <= max_chars:
        return base_font_size

    # Scale down proportionally, with a minimum of 60% of base size
    scale_factor = max(0.6, max_chars / char_count)
    return max(10, int(base_font_size * scale_factor))


# ── Subtitle Timing Correction ────────────────────────────────

def _correct_subtitle_timing(lines: list[SubtitleLine], min_gap: float = 0.02) -> list[SubtitleLine]:
    """Ensure no overlapping lines and smooth transitions.

    Args:
        lines: List of subtitle lines.
        min_gap: Minimum gap between lines in seconds.

    Returns:
        Corrected list of subtitle lines.
    """
    if not lines:
        return lines

    corrected: list[SubtitleLine] = [lines[0]]

    for i in range(1, len(lines)):
        prev = corrected[-1]
        curr = lines[i]

        # If current starts before previous ends, adjust
        if curr.start < prev.end + min_gap:
            new_start = prev.end + min_gap
            # Only adjust if it doesn't make the line too short
            if new_start < curr.end - 0.1:
                corrected.append(SubtitleLine(
                    text=curr.text,
                    start=new_start,
                    end=curr.end,
                    words=curr.words,
                    is_question=curr.is_question,
                    is_exclamatory=curr.is_exclamatory,
                    emphasis_score=curr.emphasis_score,
                ))
            else:
                # Keep the line but start it at the adjusted time
                corrected.append(SubtitleLine(
                    text=curr.text,
                    start=new_start,
                    end=max(curr.end, new_start + 0.3),
                    words=curr.words,
                    is_question=curr.is_question,
                    is_exclamatory=curr.is_exclamatory,
                    emphasis_score=curr.emphasis_score,
                ))
        else:
            corrected.append(curr)

    return corrected


# ── ASS Header ────────────────────────────────────────────────

def _build_ass_header(settings: Settings, bg_box: bool = False) -> str:
    """Build the ASS file header with styles including advanced effects.

    Args:
        settings: Settings instance.
        bg_box: Whether to include background box style.

    Returns:
        ASS header string.
    """
    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT

    glow_width = settings.SUBTITLE_OUTLINE_WIDTH + 2
    glow_shadow = settings.SUBTITLE_SHADOW_DEPTH + 1

    bg_style = ""
    if bg_box:
        bg_style = f"""Style: BgBox,Arial,20,&H00000000,&H000000FF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,3,0,0,2,10,10,{settings.SUBTITLE_MARGIN_V},1\n"""

    return f"""[Script Info]
Title: YT Shorts Factory Subtitles
ScriptType: v4.00+
PlayResX: {out_w}
PlayResY: {out_h}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
WrapStyle: 0
Collision: Reverse

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, ShadowColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: Highlight,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_HIGHLIGHT_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: Glow,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,&H40FFFFFF,&H60000000,&H40000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{glow_width},{glow_shadow},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: PopDefault,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: Typewriter,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: BounceDefault,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: WaveDefault,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: RainbowDefault,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: NeonDefault,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,&H80FFFFFF,&H80000000,&H40000000,-1,0,0,0,100,100,0,0,1,{glow_width},{glow_shadow},2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: MatrixDefault,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},&H0000FF00,&H000000FF,&H00000000,&H80000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,{settings.SUBTITLE_MARGIN_V},1
Style: Rotate3DDefault,{settings.SUBTITLE_FONT},{settings.SUBTITLE_FONT_SIZE},{settings.SUBTITLE_COLOR},&H000000FF,{settings.SUBTITLE_OUTLINE_COLOR},{settings.SUBTITLE_SHADOW_COLOR},&H00000000,{settings.SUBTITLE_BOLD},0,0,0,100,100,0,0,1,{settings.SUBTITLE_OUTLINE_WIDTH},{settings.SUBTITLE_SHADOW_DEPTH},2,10,10,{settings.SUBTITLE_MARGIN_V},1
{bg_style}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ── Animation Mode Implementations ────────────────────────────

def _generate_karaoke_line(line: SubtitleLine, settings: Settings, gradient: bool = False) -> str:
    """Generate karaoke-style ASS dialogue text with optional gradient highlight.

    Uses {\\kf<cs>} tags for smooth fill-sweep highlighting of each word.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.
        gradient: Whether to use gradient highlight colors.

    Returns:
        ASS dialogue text string.
    """
    parts: list[str] = []

    for i, word in enumerate(line.words):
        duration_cs = max(1, int(round((word.end - word.start) * 100)))
        escaped = ass_escape(word.word)

        if gradient and i > 0:
            # Gradient: interpolate between highlight and base color
            total_words = len(line.words)
            factor = i / max(1, total_words - 1)
            gradient_color = _interpolate_color(
                settings.SUBTITLE_HIGHLIGHT_COLOR,
                settings.SUBTITLE_COLOR,
                factor,
            )
            parts.append(f"{{\\1c{gradient_color}}}{{\\kf{duration_cs}}}{escaped}")
        elif i == 0:
            parts.append(f"{{\\1c{settings.SUBTITLE_COLOR}}}{{\\kf{duration_cs}}}{escaped}")
        else:
            parts.append(f"{{\\kf{duration_cs}}}{escaped}")

    return " ".join(parts)


def _generate_fade_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate fade-in/fade-out subtitle line.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    fade_in_ms = min(150, int(line.duration * 1000 * 0.15))
    fade_out_ms = min(100, int(line.duration * 1000 * 0.10))
    return f"{{\\fad({fade_in_ms},{fade_out_ms})}}{ass_escape(line.text)}"


def _generate_pop_line(line: SubtitleLine, settings: Settings, bg_layer: bool = False) -> str:
    """Generate per-word pop animation.

    Words scale from 80% to 100% with bounce effect.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.
        bg_layer: If True, generate static background layer.

    Returns:
        ASS dialogue text string.
    """
    if bg_layer:
        return f"{{\\alpha&H80&}}{{\\fscx100\\fscy100}}{ass_escape(line.text)}"

    # Pop each word individually
    parts: list[str] = []
    for word in line.words:
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        # Bounce effect: scale 80->110->100
        bounce_ms = min(duration_ms, 150)
        settle_ms = max(0, duration_ms - bounce_ms)

        styled = (
            f"{{\\fscx80\\fscy80\\1c{settings.SUBTITLE_HIGHLIGHT_COLOR}}}"
            f"{escaped}"
            f"{{\\t(0,{bounce_ms},\\fscx110\\fscy110\\1c{settings.SUBTITLE_COLOR})}}"
            f"{{\\t({bounce_ms},{bounce_ms + settle_ms},\\fscx100\\fscy100)}}"
        )
        parts.append(styled)

    return " ".join(parts)


def _generate_glow_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate pulsing glow effect per word.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    parts: list[str] = []
    for word in line.words:
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        # Bright word with outline pulse
        styled = (
            f"{{\\1c{settings.SUBTITLE_HIGHLIGHT_COLOR}\\3c&HFFFFFF&\\4a&HFF&}}"
            f"{escaped}"
            f"{{\\t(0,{duration_ms},\\1c{settings.SUBTITLE_COLOR}\\3c{settings.SUBTITLE_OUTLINE_COLOR}\\4a&H80&)}}"
        )
        parts.append(styled)

    return " ".join(parts)


def _generate_typewriter_line(line: SubtitleLine, settings: Settings, char_index: int = 0) -> tuple[str, int]:
    """Generate character-by-character reveal effect.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.
        char_index: Starting character index for cursor positioning.

    Returns:
        Tuple of (ASS dialogue events string, updated char_index).
    """
    events: list[str] = []
    chars_shown = ""

    for word in line.words:
        for char_idx, char in enumerate(word.word):
            char_frac = char_idx / max(1, len(word.word))
            char_start = word.start + (word.duration * char_frac / len(word.word))
            char_end = word.end
            char_start_ts = ass_timestamp(char_start)
            char_end_ts = ass_timestamp(char_end)
            chars_shown += char

            escaped = ass_escape(chars_shown)
            cursor = "\\_" if char_idx == len(word.word) - 1 else ""
            events.append(
                f"Dialogue: 0,{char_start_ts},{char_end_ts},Typewriter,,0,0,0,,{escaped}{cursor}"
            )

        chars_shown += " "

    return "\n".join(events), char_index + len(chars_shown)


def _generate_bounce_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate bounce animation — word bounces up when active.

    Each word moves up 10 pixels and back down.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    parts: list[str] = []
    for word in line.words:
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        # Bounce up: move Y position up 10px, then back
        bounce_up_ms = min(duration_ms // 2, 100)
        bounce_down_ms = duration_ms - bounce_up_ms

        styled = (
            f"{{\\1c{settings.SUBTITLE_HIGHLIGHT_COLOR}\\pos(0,0)}}"
            f"{escaped}"
            f"{{\\t(0,{bounce_up_ms},\\1c{settings.SUBTITLE_COLOR}\\frx0\\fry0\\frz0\\pos(0,-10))}}"
            f"{{\\t({bounce_up_ms},{bounce_up_ms + bounce_down_ms},\\pos(0,0))}}"
        )
        parts.append(styled)

    return " ".join(parts)


def _generate_wave_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate wave animation — words wave like ocean.

    Each word oscillates vertically in a sine-wave pattern.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    parts: list[str] = []
    total_words = len(line.words)

    for i, word in enumerate(line.words):
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        # Phase offset based on word position for wave effect
        phase = (i / max(1, total_words)) * 2 * math.pi

        # Wave: oscillate Y position using multiple transform steps
        wave_amplitude = 8
        quarter = max(10, duration_ms // 4)

        # Calculate wave offsets at each quarter
        offsets = [
            int(wave_amplitude * math.sin(phase + j * math.pi / 2))
            for j in range(5)
        ]

        styled = (
            f"{{\\1c{settings.SUBTITLE_COLOR}\\pos(0,{offsets[0]})}}"
            f"{escaped}"
            f"{{\\t(0,{quarter},\\pos(0,{offsets[1]}))}}"
            f"{{\\t({quarter},{quarter * 2},\\pos(0,{offsets[2]}))}}"
            f"{{\\t({quarter * 2},{quarter * 3},\\pos(0,{offsets[3]}))}}"
            f"{{\\t({quarter * 3},{duration_ms},\\pos(0,{offsets[4]}))}}"
        )
        parts.append(styled)

    return " ".join(parts)


def _generate_rainbow_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate rainbow color cycling animation.

    Each word cycles through rainbow colors.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    # Rainbow colors in ASS format (BGR)
    rainbow_colors = [
        "&H0000FF",   # Red
        "&H0080FF",   # Orange
        "&H00FFFF",   # Yellow
        "&H00FF00",   # Green
        "&HFF0000",   # Blue
        "&HFF0080",   # Indigo
        "&HFF00FF",   # Violet
    ]

    parts: list[str] = []
    total_words = len(line.words)

    for i, word in enumerate(line.words):
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        # Assign a base rainbow color based on word position
        base_color_idx = i % len(rainbow_colors)
        next_color_idx = (i + 1) % len(rainbow_colors)

        # Cycle to next color during the word
        styled = (
            f"{{\\1c{rainbow_colors[base_color_idx]}}}"
            f"{escaped}"
            f"{{\\t(0,{duration_ms},\\1c{rainbow_colors[next_color_idx]})}}"
        )
        parts.append(styled)

    return " ".join(parts)


def _generate_neon_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate neon glow pulse animation.

    Words pulse between bright and dim with outline glow.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    parts: list[str] = []

    for word in line.words:
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        half = duration_ms // 2

        # Neon pulse: bright -> dim -> bright
        styled = (
            f"{{\\1c&H00FFFF\\3c&H80FFFF\\4a&H00\\bord{settings.SUBTITLE_OUTLINE_WIDTH + 2}}}"
            f"{escaped}"
            f"{{\\t(0,{half},\\1c&H004040\\3c&H408080\\4a&H80\\bord{settings.SUBTITLE_OUTLINE_WIDTH})}}"
            f"{{\\t({half},{duration_ms},\\1c&H00FFFF\\3c&H80FFFF\\4a&H00\\bord{settings.SUBTITLE_OUTLINE_WIDTH + 2})}}"
        )
        parts.append(styled)

    return " ".join(parts)


def _generate_matrix_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate matrix-style reveal animation.

    Characters appear with a green-on-black digital reveal effect.

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    parts: list[str] = []

    for word in line.words:
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        # Matrix reveal: start dark green, flash bright green, settle to green
        styled = (
            f"{{\\1c&H000800\\3c&H000400\\4a&HFF}}"
            f"{escaped}"
            f"{{\\t(0,{min(80, duration_ms // 3)},\\1c&H00FF00\\3c&H008000\\4a&H00)}}"
            f"{{\\t({min(80, duration_ms // 3)},{duration_ms},\\1c&H00C000\\3c&H004000\\4a&H40)}}"
        )
        parts.append(styled)

    return " ".join(parts)


def _generate_3d_rotate_line(line: SubtitleLine, settings: Settings) -> str:
    """Generate 3D perspective rotation animation.

    Words rotate in from a 3D perspective (Y-axis rotation).

    Args:
        line: SubtitleLine to render.
        settings: Settings instance.

    Returns:
        ASS dialogue text string.
    """
    parts: list[str] = []

    for word in line.words:
        duration_ms = max(50, int(word.duration * 1000))
        escaped = ass_escape(word.word)

        # 3D rotate: start rotated 90 degrees, rotate to 0
        rotate_in_ms = min(200, duration_ms)
        settle_ms = max(0, duration_ms - rotate_in_ms)

        styled = (
            f"{{\\1c{settings.SUBTITLE_HIGHLIGHT_COLOR}\\fry90\\fscx50\\fscy50\\alpha&H80&}}"
            f"{escaped}"
            f"{{\\t(0,{rotate_in_ms},\\fry0\\fscx100\\fscy100\\1c{settings.SUBTITLE_COLOR}\\alpha&H00&)}}"
        )
        if settle_ms > 0:
            styled += f"{{\\t({rotate_in_ms},{rotate_in_ms + settle_ms},\\1c{settings.SUBTITLE_COLOR})}}"

        parts.append(styled)

    return " ".join(parts)


# ── Background Box Generation ─────────────────────────────────

def _generate_bg_box_events(line: SubtitleLine, settings: Settings) -> str:
    """Generate ASS events for semi-transparent background box behind text.

    Args:
        line: SubtitleLine to add background for.
        settings: Settings instance.

    Returns:
        ASS dialogue line string for background box.
    """
    start_ts = ass_timestamp(line.start)
    end_ts = ass_timestamp(line.end)
    # Use a drawing for the background box
    text_len = len(line.text)
    box_width = min(text_len * settings.SUBTITLE_FONT_SIZE + 40, settings.OUTPUT_WIDTH - 40)
    box_height = settings.SUBTITLE_FONT_SIZE + 20
    box_x = (settings.OUTPUT_WIDTH - box_width) // 2
    # Position relative to subtitle margin
    box_y = settings.OUTPUT_HEIGHT - settings.SUBTITLE_MARGIN_V - box_height - 10

    drawing = (
        f"{{\\pos({settings.OUTPUT_WIDTH // 2},{box_y + box_height // 2})"
        f"\\1a&HA0&\\1c&H000000&\\bord0\\shad0}}"
        f"{{\\p1}}m 0 0 l {box_width} 0 l {box_width} {box_height} l 0 {box_height}{{\\p0}}"
    )

    return f"Dialogue: -2,{start_ts},{end_ts},BgBox,,0,0,0,,{drawing}\n"


# ── Word Emphasis ─────────────────────────────────────────────

def _apply_word_emphasis(text: str, emphasis_score: float, base_font_size: int) -> str:
    """Apply emphasis styling to text based on emphasis score.

    Args:
        text: ASS-formatted text.
        emphasis_score: 0-1 emphasis score.
        base_font_size: Base font size for scaling.

    Returns:
        Text with emphasis overrides applied.
    """
    if emphasis_score > 0.5:
        # Make emphasized text bigger and bolder
        scale = int(100 + emphasis_score * 30)
        return f"{{\\fscx{scale}\\fscy{scale}\\b1}}{text}{{\\fscx100\\fscy100\\b0}}"
    return text


# ── Main Subtitle Generation ─────────────────────────────────

def generate_subtitles(
    transcription: TranscriptionResult,
    output_path: Path,
    settings: Settings | None = None,
    animation: str | None = None,
    bg_box: bool = False,
    gradient_highlight: bool = False,
) -> Path:
    """Generate an ASS subtitle file with advanced animation effects.

    Supports 11 animation modes:
    - karaoke: Word-by-word fill-sweep highlighting with {\\kf} tags
    - fade: Smooth fade-in/fade-out per line with {\\fad} tags
    - pop: Per-word scale-from-80%-to-100% animation with bounce
    - glow: Pulsing glow/outline effect per word
    - typewriter: Per-character reveal effect
    - bounce: Word bounces up when active
    - wave: Words wave like ocean
    - rainbow: Color cycling through rainbow
    - neon: Neon glow pulse effect
    - matrix: Matrix-style digital reveal
    - 3d_rotate: Perspective rotation
    - none: Plain timed subtitles with glow backdrop

    All modes include a glow/outline backdrop layer for readability on any background.
    Optional background box and gradient highlight.

    Args:
        transcription: TranscriptionResult with word timestamps.
        output_path: Destination path for the ASS file.
        settings: Optional Settings override.
        animation: Override animation mode from settings.
        bg_box: Whether to add semi-transparent background box.
        gradient_highlight: Whether to use gradient highlight for karaoke.

    Returns:
        Path to the written ASS file.
    """
    if settings is None:
        settings = get_settings()

    if not transcription.words:
        logger.warning("No words in transcription; skipping subtitle generation")
        return output_path

    lines = _group_words_into_lines(
        transcription.words,
        max_words=settings.SUBTITLE_MAX_WORDS,
        min_display_time=settings.SUBTITLE_MIN_DISPLAY_TIME,
        overlap=settings.SUBTITLE_OVERLAP,
    )

    # Apply timing correction
    lines = _correct_subtitle_timing(lines)

    if not lines:
        return output_path

    logger.info("Generated %d subtitle lines from %d words", len(lines), len(transcription.words))

    # Build ASS content
    ass_content = _build_ass_header(settings, bg_box=bg_box)
    anim = animation or settings.SUBTITLE_ANIMATION

    for line in lines:
        start_ts = ass_timestamp(line.start)
        end_ts = ass_timestamp(line.end)

        # Background box layer (behind everything)
        if bg_box:
            ass_content += _generate_bg_box_events(line, settings)

        if anim == "karaoke":
            dialogue_text = _generate_karaoke_line(line, settings, gradient=gradient_highlight)
            if line.emphasis_score > 0.5:
                dialogue_text = _apply_word_emphasis(dialogue_text, line.emphasis_score, settings.SUBTITLE_FONT_SIZE)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{dialogue_text}\n"
            # Glow layer behind
            glow_text = _generate_karaoke_line(line, settings)
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{glow_text}\n"

        elif anim == "fade":
            dialogue_text = _generate_fade_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{dialogue_text}\n"
            glow_text = _generate_fade_line(line, settings)
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{glow_text}\n"

        elif anim == "pop":
            # Per-word pop events
            for word in line.words:
                word_start_ts = ass_timestamp(word.start)
                word_end_ts = ass_timestamp(word.end)
                duration_ms = max(50, int((word.end - word.start) * 1000))
                escaped = ass_escape(word.word)

                # Check if this word is emphasized
                is_emphasized = word.word.isupper() and len(word.word) > 1
                highlight = settings.SUBTITLE_HIGHLIGHT_COLOR if is_emphasized else settings.SUBTITLE_COLOR

                bounce_ms = min(duration_ms, 150)
                settle_ms = max(0, duration_ms - bounce_ms)

                styled = (
                    f"{{\\fscx80\\fscy80\\1c{highlight}}}"
                    f"{escaped}"
                    f"{{\\t(0,{bounce_ms},\\fscx110\\fscy110\\1c{settings.SUBTITLE_COLOR})}}"
                    f"{{\\t({bounce_ms},{bounce_ms + settle_ms},\\fscx100\\fscy100)}}"
                )
                ass_content += f"Dialogue: 0,{word_start_ts},{word_end_ts},PopDefault,,0,0,0,,{styled}\n"

            # Static background line (dimmer)
            static_text = f"{{\\alpha&H80&}}{{\\fscx100\\fscy100}}{ass_escape(line.text)}"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Default,,0,0,0,,{static_text}\n"

        elif anim == "glow":
            dialogue_text = _generate_glow_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},Highlight,,0,0,0,,{dialogue_text}\n"
            glow_text = f"{{\\3c&HFFFFFF&\\4a&H80&}}{ass_escape(line.text)}"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{glow_text}\n"

        elif anim == "typewriter":
            events_str, _ = _generate_typewriter_line(line, settings)
            ass_content += events_str + "\n"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{ass_escape(line.text)}\n"

        elif anim == "bounce":
            dialogue_text = _generate_bounce_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},BounceDefault,,0,0,0,,{dialogue_text}\n"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{ass_escape(line.text)}\n"

        elif anim == "wave":
            dialogue_text = _generate_wave_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},WaveDefault,,0,0,0,,{dialogue_text}\n"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{ass_escape(line.text)}\n"

        elif anim == "rainbow":
            dialogue_text = _generate_rainbow_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},RainbowDefault,,0,0,0,,{dialogue_text}\n"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{ass_escape(line.text)}\n"

        elif anim == "neon":
            dialogue_text = _generate_neon_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},NeonDefault,,0,0,0,,{dialogue_text}\n"
            # Extra glow layer for neon
            glow_text = f"{{\\1c&H00FFFF\\3c&H40FFFF\\4a&H40\\bord6}}{ass_escape(line.text)}"
            ass_content += f"Dialogue: -2,{start_ts},{end_ts},NeonDefault,,0,0,0,,{glow_text}\n"

        elif anim == "matrix":
            dialogue_text = _generate_matrix_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},MatrixDefault,,0,0,0,,{dialogue_text}\n"
            # Dark background glow for matrix
            glow_text = f"{{\\1c&H004000\\alpha&HC0&}}{ass_escape(line.text)}"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},MatrixDefault,,0,0,0,,{glow_text}\n"

        elif anim == "3d_rotate":
            dialogue_text = _generate_3d_rotate_line(line, settings)
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},Rotate3DDefault,,0,0,0,,{dialogue_text}\n"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{ass_escape(line.text)}\n"

        else:  # "none"
            ass_content += f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{ass_escape(line.text)}\n"
            ass_content += f"Dialogue: -1,{start_ts},{end_ts},Glow,,0,0,0,,{ass_escape(line.text)}\n"

    # Write the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass_content, encoding="utf-8")
    logger.info("ASS subtitle file written: %s (%d lines)", output_path.name, len(lines))
    return output_path


# ── Subtitle Preview ──────────────────────────────────────────

def subtitle_preview(
    ass_path: Path,
    video_path: Path,
    timestamp: float,
    output_path: Path,
    settings: Settings | None = None,
) -> Path:
    """Render a single frame with subtitles for preview.

    Args:
        ass_path: Path to the ASS subtitle file.
        video_path: Path to the source video.
        timestamp: Time in seconds to render.
        output_path: Destination path for the preview image.
        settings: Optional Settings override.

    Returns:
        Path to the preview image.
    """
    if settings is None:
        settings = get_settings()

    if not ass_path.exists() or not video_path.exists():
        logger.error("ASS or video file not found for preview")
        return output_path

    # Escape ASS path for FFmpeg
    ass_path_str = str(ass_path)
    if "\\" in ass_path_str:
        ass_path_str = ass_path_str.replace("\\", "/")
    ass_filter = f"ass='{ass_path_str.replace(':', '\\:')}'"

    cmd: list[str] = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vf", ass_filter,
        "-vframes", "1",
        "-q:v", "2",
        str(output_path),
    ]

    try:
        run_ffmpeg(cmd, description=f"Subtitle preview at {timestamp:.1f}s")
    except FFmpegError as exc:
        logger.error("Subtitle preview failed: %s", exc)

    return output_path


# ── Export to SRT/VTT ─────────────────────────────────────────

def export_to_srt(
    transcription: TranscriptionResult,
    output_path: Path,
) -> Path:
    """Export transcription as SRT subtitle file.

    Args:
        transcription: TranscriptionResult to export.
        output_path: Destination path for the SRT file.

    Returns:
        Path to the written SRT file.
    """
    from core.transcriber import export_srt
    return export_srt(transcription, output_path)


def export_to_vtt(
    transcription: TranscriptionResult,
    output_path: Path,
) -> Path:
    """Export transcription as WebVTT subtitle file.

    Args:
        transcription: TranscriptionResult to export.
        output_path: Destination path for the VTT file.

    Returns:
        Path to the written VTT file.
    """
    from core.transcriber import export_vtt
    return export_vtt(transcription, output_path)


# ── Burn Subtitles ────────────────────────────────────────────

def burn_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
    settings: Settings | None = None,
) -> Path:
    """Burn ASS subtitles into a video file with quality optimisation.

    Uses hardware-accelerated encoding when available and optimises
    for subtitle rendering quality. Falls back to software encoding
    if hardware fails.

    Args:
        video_path: Path to the source video.
        ass_path: Path to the ASS subtitle file.
        output_path: Destination path for the output video.
        settings: Optional Settings override.

    Returns:
        Path to the output video with burned subtitles.

    Raises:
        FFmpegError: If subtitle burning fails.
    """
    if settings is None:
        settings = get_settings()

    if not video_path.exists():
        raise FFmpegError(f"Input video not found: {video_path}")
    if not ass_path.exists():
        raise FFmpegError(f"ASS subtitle file not found: {ass_path}")

    # Escape path for FFmpeg ass filter
    ass_path_str = str(ass_path)
    if "\\" in ass_path_str:
        ass_path_str = ass_path_str.replace("\\", "/")
    ass_path_escaped = ass_path_str.replace(":", "\\:")
    ass_filter = f"ass='{ass_path_escaped}'"

    video_info = probe_video(video_path)
    duration = video_info.duration

    hw_encoder, hw_preset = detect_hw_encoder()
    video_codec = hw_encoder if hw_encoder else settings.FFMPEG_VIDEO_CODEC
    encoding_preset = hw_preset if hw_encoder else settings.FFMPEG_PRESET

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", ass_filter,
        "-c:v", video_codec,
        "-preset", encoding_preset,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ]

    logger.info("Burning subtitles: %s -> %s (codec=%s)", ass_path.name, output_path.name, video_codec)

    try:
        run_ffmpeg(
            cmd,
            description=f"Burn subtitles from {ass_path.name}",
            show_progress=True,
            total_duration=duration,
            timeout=settings.FFMPEG_TIMEOUT,
        )
    except FFmpegError as exc:
        if hw_encoder:
            logger.warning("HW encoding failed, falling back to software: %s", exc)
            sw_cmd = list(cmd)
            sw_cmd[sw_cmd.index(video_codec)] = settings.FFMPEG_VIDEO_CODEC
            sw_cmd[sw_cmd.index(encoding_preset)] = settings.FFMPEG_PRESET
            run_ffmpeg(sw_cmd, description="Burn subtitles (SW fallback)",
                      show_progress=True, total_duration=duration,
                      timeout=settings.FFMPEG_TIMEOUT)
        else:
            raise

    if not output_path.exists():
        raise FFmpegError(f"Burned subtitle output not created: {output_path}")

    logger.info("Subtitles burned successfully: %s", output_path.name)
    return output_path
