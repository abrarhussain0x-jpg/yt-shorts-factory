"""
core/channel_pattern.py — Channel Pattern System for consistent branding across all shorts.

Provides 10+ built-in channel patterns (templates) that apply consistent branding:
intro/outro, subscribe CTA, lower thirds, watermarks, color themes, subtitle styles,
hook animations, and transition effects. Each pattern is a complete branding preset
that can be customized per-channel.

Built-in Patterns:
  - viral_hype:     Explosive hooks, fast cuts, bold text, neon subtitles
  - chill_vibes:    Smooth fades, warm colors, soft subtitles
  - news_alert:     Breaking news style, ticker, urgency indicators
  - educational:    Clean, structured, chapter markers, key point highlights
  - gaming_clips:   Fast-paced, frame highlights, score overlays
  - motivational:   Epic text, slow reveals, powerful music feel
  - comedy_clip:    Punch timing, reaction zoom, sound effect cues
  - lifestyle:      Aesthetic, minimal branding, clean typography
  - tech_review:    Spec overlays, comparison frames, rating cards
  - custom:         User-defined pattern from JSON config

Each pattern configures:
  - Intro style (stinger duration, animation type, channel name display)
  - Outro style (subscribe CTA, next video, social links)
  - Lower third (name bar, position, colors, duration)
  - Subtitle style (font, color, animation, position, outline)
  - Hook generator (first 3 seconds attention grabber)
  - CTA generator (subscribe/follow at optimal moments)
  - Watermark style (position, opacity, pattern)
  - Color palette (primary, secondary, accent, background)
  - Transition style (fade, slide, zoom, glitch)
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from config.settings import Settings, get_settings, BASE_DIR
from utils.ffmpeg_utils import run_ffmpeg, probe_video, FFmpegError
from utils.file_utils import get_file_size_human
from utils.logger import get_logger
from rich.console import Console

logger = get_logger("channel_pattern")
console = Console()


# ═══════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════

@dataclass
class ColorPalette:
    """Color palette for channel branding."""
    primary: str = "#FFFFFF"        # Main text/highlight color
    secondary: str = "#00FFFF"      # Accent color (karaoke highlight)
    accent: str = "#FF0066"         # CTA/subscribe button color
    background: str = "#000000"     # Background for overlays
    outline: str = "#000000"        # Text outline color
    shadow: str = "#80000000"       # Shadow color


@dataclass
class IntroConfig:
    """Configuration for video intro stinger."""
    enabled: bool = True
    duration: float = 2.0           # Intro duration in seconds
    animation: str = "fade"         # fade, slide_left, slide_right, zoom_in, glitch, typewriter
    show_channel_name: bool = True
    channel_name_size: int = 36
    show_logo: bool = True
    logo_scale: float = 0.20        # Larger logo for intro
    sound_effect: str = ""          # Path to intro sound
    background_color: str = "#000000"
    crossfade_duration: float = 0.5


@dataclass
class OutroConfig:
    """Configuration for video outro card."""
    enabled: bool = True
    duration: float = 4.0           # Outro card duration
    subscribe_text: str = "SUBSCRIBE"
    subscribe_color: str = "#FF0000"
    next_video_text: str = "Watch Next"
    show_social: bool = False
    social_handles: dict[str, str] = field(default_factory=dict)
    show_logo: bool = True
    background_opacity: float = 0.7
    animation: str = "fade"         # fade, slide_up, zoom_out


@dataclass
class LowerThirdConfig:
    """Configuration for lower-third name bar."""
    enabled: bool = False
    duration: float = 5.0
    position: str = "bottom"        # bottom, top, mid
    channel_name: str = ""
    subtitle_text: str = ""
    bg_opacity: float = 0.7
    font_size: int = 24
    show_at_start: bool = True      # Show during first N seconds
    animation: str = "slide_right"  # slide_left, slide_right, fade, pop


@dataclass
class HookConfig:
    """Configuration for attention hook in first 3 seconds."""
    enabled: bool = True
    duration: float = 3.0
    style: str = "text_flash"       # text_flash, zoom_pulse, color_flash, question, bold_statement
    text_template: str = ""         # Auto-filled from transcription
    font_size: int = 40
    position: str = "center"        # center, top, bottom
    flash_rate: float = 0.5         # Flashes per second
    zoom_amount: float = 1.1        # Zoom pulse factor
    color_flash: str = "#FF0000"    # Color for flash effects


@dataclass
class CTAConfig:
    """Configuration for Call-To-Action moments."""
    enabled: bool = True
    timing: str = "optimal"         # optimal, middle, end, multiple
    style: str = "subscribe_popup"  # subscribe_popup, text_overlay, banner, like_remind
    duration: float = 3.0
    subscribe_text: str = "SUBSCRIBE"
    like_text: str = "Like this video!"
    bell_text: str = "Ring the bell!"
    show_count: int = 1             # Number of CTA insertions
    opacity: float = 0.85
    animation: str = "pop"          # pop, slide, fade, bounce


@dataclass
class WatermarkConfig:
    """Configuration for watermark overlay."""
    enabled: bool = True
    style: str = "corner"           # corner, tiled, diagonal, pulse
    opacity: float = 0.85
    position: str = "top-right"
    pulse_rate: float = 0.0         # 0 = no pulse, >0 = pulses per second
    tile_spacing: int = 300         # For tiled style


@dataclass
class SubtitleStyle:
    """Configuration for subtitle styling within a pattern."""
    font: str = "Arial Black"
    font_size: int = 22
    color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000FFFF"
    outline_color: str = "&H00000000"
    outline_width: int = 3
    shadow_depth: int = 2
    animation: str = "karaoke"
    position: str = "bottom"
    max_words: int = 4
    bold: bool = True
    outline_mode: str = "shadow"    # shadow, glow, box, backdrop


@dataclass
class TransitionConfig:
    """Configuration for video transitions."""
    style: str = "fade"             # fade, slide, zoom, glitch, whip_pan
    duration: float = 0.5
    easing: str = "ease_in_out"     # ease_in, ease_out, ease_in_out, linear


@dataclass
class ChannelPattern:
    """Complete channel branding pattern combining all visual elements.

    A ChannelPattern defines every visual aspect of a short video to ensure
    consistent branding across all content. It includes intro/outro animations,
    subtitle styles, hook generators, CTA overlays, watermarks, color themes,
    and transition effects.
    """
    name: str = "custom"
    display_name: str = "Custom Pattern"
    description: str = "User-defined channel pattern"
    category: str = "general"       # general, entertainment, education, gaming, news
    colors: ColorPalette = field(default_factory=ColorPalette)
    intro: IntroConfig = field(default_factory=IntroConfig)
    outro: OutroConfig = field(default_factory=OutroConfig)
    lower_third: LowerThirdConfig = field(default_factory=LowerThirdConfig)
    hook: HookConfig = field(default_factory=HookConfig)
    cta: CTAConfig = field(default_factory=CTAConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    subtitle_style: SubtitleStyle = field(default_factory=SubtitleStyle)
    transition: TransitionConfig = field(default_factory=TransitionConfig)

    def to_dict(self) -> dict:
        """Serialize the pattern to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ChannelPattern:
        """Deserialize a pattern from a dict."""
        colors = ColorPalette(**data.get("colors", {}))
        intro = IntroConfig(**data.get("intro", {}))
        outro = OutroConfig(**data.get("outro", {}))
        lower_third = LowerThirdConfig(**data.get("lower_third", {}))
        hook = HookConfig(**data.get("hook", {}))
        cta = CTAConfig(**data.get("cta", {}))
        watermark = WatermarkConfig(**data.get("watermark", {}))
        subtitle_style = SubtitleStyle(**data.get("subtitle_style", {}))
        transition = TransitionConfig(**data.get("transition", {}))
        return cls(
            name=data.get("name", "custom"),
            display_name=data.get("display_name", "Custom Pattern"),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            colors=colors,
            intro=intro,
            outro=outro,
            lower_third=lower_third,
            hook=hook,
            cta=cta,
            watermark=watermark,
            subtitle_style=subtitle_style,
            transition=transition,
        )


# ═══════════════════════════════════════════════════════════
#  Built-in Channel Patterns
# ═══════════════════════════════════════════════════════════

BUILTIN_PATTERNS: dict[str, ChannelPattern] = {
    "viral_hype": ChannelPattern(
        name="viral_hype",
        display_name="Viral Hype",
        description="Explosive hooks, fast cuts, bold neon text, maximum engagement",
        category="entertainment",
        colors=ColorPalette(
            primary="#FFFFFF", secondary="#00FF00", accent="#FF0066",
            background="#000000", outline="#000000", shadow="#80000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=1.5, animation="zoom_in",
            show_channel_name=True, channel_name_size=42,
            show_logo=True, logo_scale=0.25, background_color="#000000",
        ),
        outro=OutroConfig(
            enabled=True, duration=3.0, subscribe_text="SUBSCRIBE NOW!",
            subscribe_color="#FF0000", next_video_text="Watch Next",
            background_opacity=0.8, animation="zoom_out",
        ),
        lower_third=LowerThirdConfig(
            enabled=True, duration=4.0, position="bottom",
            bg_opacity=0.8, font_size=28, animation="slide_right",
        ),
        hook=HookConfig(
            enabled=True, duration=3.0, style="bold_statement",
            font_size=48, position="center", flash_rate=1.0,
            zoom_amount=1.15, color_flash="#FF0066",
        ),
        cta=CTAConfig(
            enabled=True, timing="middle", style="subscribe_popup",
            duration=2.5, subscribe_text="SUBSCRIBE!", show_count=2,
            opacity=0.9, animation="pop",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.85, position="top-right",
        ),
        subtitle_style=SubtitleStyle(
            font="Arial Black", font_size=24, color="&H00FFFFFF",
            highlight_color="&H0000FF00", outline_color="&H00000000",
            outline_width=4, shadow_depth=3, animation="karaoke",
            position="bottom", max_words=3, bold=True, outline_mode="glow",
        ),
        transition=TransitionConfig(style="glitch", duration=0.3, easing="ease_in_out"),
    ),

    "chill_vibes": ChannelPattern(
        name="chill_vibes",
        display_name="Chill Vibes",
        description="Smooth fades, warm colors, soft subtitles, relaxed pacing",
        category="entertainment",
        colors=ColorPalette(
            primary="#FFF5E6", secondary="#FFB366", accent="#FF8C42",
            background="#1A1A2E", outline="#0A0A1E", shadow="#60000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=3.0, animation="fade",
            show_channel_name=True, channel_name_size=30,
            show_logo=True, logo_scale=0.15, background_color="#1A1A2E",
        ),
        outro=OutroConfig(
            enabled=True, duration=5.0, subscribe_text="Subscribe for more",
            subscribe_color="#FF8C42", next_video_text="More like this",
            background_opacity=0.5, animation="fade",
        ),
        lower_third=LowerThirdConfig(
            enabled=False, duration=6.0, position="bottom",
            bg_opacity=0.5, font_size=22, animation="fade",
        ),
        hook=HookConfig(
            enabled=True, duration=3.0, style="text_flash",
            font_size=36, position="center", flash_rate=0.3,
            zoom_amount=1.05, color_flash="#FFB366",
        ),
        cta=CTAConfig(
            enabled=True, timing="end", style="text_overlay",
            duration=4.0, subscribe_text="Subscribe", show_count=1,
            opacity=0.6, animation="fade",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.5, position="bottom-right",
        ),
        subtitle_style=SubtitleStyle(
            font="Georgia", font_size=20, color="&H00FFF5E6",
            highlight_color="&H00FFB366", outline_color="&H000A0A1E",
            outline_width=2, shadow_depth=2, animation="fade",
            position="bottom", max_words=5, bold=False, outline_mode="shadow",
        ),
        transition=TransitionConfig(style="fade", duration=1.0, easing="ease_in_out"),
    ),

    "news_alert": ChannelPattern(
        name="news_alert",
        display_name="News Alert",
        description="Breaking news style, urgent ticker, professional lower thirds",
        category="news",
        colors=ColorPalette(
            primary="#FFFFFF", secondary="#FF0000", accent="#FF0000",
            background="#1A1A1A", outline="#000000", shadow="#80000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=2.0, animation="slide_left",
            show_channel_name=True, channel_name_size=32,
            show_logo=True, logo_scale=0.12, background_color="#1A1A1A",
        ),
        outro=OutroConfig(
            enabled=True, duration=4.0, subscribe_text="SUBSCRIBE for Updates",
            subscribe_color="#FF0000", next_video_text="Latest News",
            background_opacity=0.85, animation="slide_up",
        ),
        lower_third=LowerThirdConfig(
            enabled=True, duration=6.0, position="bottom",
            bg_opacity=0.9, font_size=26, animation="slide_left",
            show_at_start=True,
        ),
        hook=HookConfig(
            enabled=True, duration=2.5, style="bold_statement",
            font_size=44, position="top", flash_rate=2.0,
            zoom_amount=1.0, color_flash="#FF0000",
        ),
        cta=CTAConfig(
            enabled=True, timing="end", style="banner",
            duration=3.0, subscribe_text="Stay Informed - Subscribe",
            show_count=1, opacity=0.9, animation="slide_up",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.9, position="top-left",
        ),
        subtitle_style=SubtitleStyle(
            font="Arial", font_size=22, color="&H00FFFFFF",
            highlight_color="&H00FF0000", outline_color="&H00000000",
            outline_width=3, shadow_depth=1, animation="typewriter",
            position="bottom", max_words=6, bold=True, outline_mode="box",
        ),
        transition=TransitionConfig(style="slide", duration=0.3, easing="ease_in"),
    ),

    "educational": ChannelPattern(
        name="educational",
        display_name="Educational",
        description="Clean, structured, chapter markers, key point highlights",
        category="education",
        colors=ColorPalette(
            primary="#FFFFFF", secondary="#4CAF50", accent="#2196F3",
            background="#263238", outline="#000000", shadow="#60000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=2.5, animation="fade",
            show_channel_name=True, channel_name_size=28,
            show_logo=True, logo_scale=0.12, background_color="#263238",
        ),
        outro=OutroConfig(
            enabled=True, duration=5.0, subscribe_text="Learn More - Subscribe",
            subscribe_color="#4CAF50", next_video_text="Next Lesson",
            background_opacity=0.6, animation="fade",
        ),
        lower_third=LowerThirdConfig(
            enabled=True, duration=5.0, position="bottom",
            bg_opacity=0.7, font_size=24, animation="slide_right",
            show_at_start=True,
        ),
        hook=HookConfig(
            enabled=True, duration=3.0, style="question",
            font_size=36, position="center", flash_rate=0.0,
            zoom_amount=1.0, color_flash="#4CAF50",
        ),
        cta=CTAConfig(
            enabled=True, timing="end", style="text_overlay",
            duration=4.0, subscribe_text="Subscribe for more lessons",
            show_count=1, opacity=0.7, animation="fade",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.7, position="top-right",
        ),
        subtitle_style=SubtitleStyle(
            font="Arial", font_size=20, color="&H00FFFFFF",
            highlight_color="&H004CAF50", outline_color="&H00000000",
            outline_width=2, shadow_depth=1, animation="pop",
            position="bottom", max_words=5, bold=False, outline_mode="backdrop",
        ),
        transition=TransitionConfig(style="fade", duration=0.5, easing="ease_in_out"),
    ),

    "gaming_clips": ChannelPattern(
        name="gaming_clips",
        display_name="Gaming Clips",
        description="Fast-paced, frame highlights, score overlays, epic moments",
        category="gaming",
        colors=ColorPalette(
            primary="#FFFFFF", secondary="#00FFFF", accent="#9C27B0",
            background="#0D0D0D", outline="#000000", shadow="#80000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=1.5, animation="glitch",
            show_channel_name=True, channel_name_size=38,
            show_logo=True, logo_scale=0.20, background_color="#0D0D0D",
        ),
        outro=OutroConfig(
            enabled=True, duration=3.0, subscribe_text="SUBSCRIBE + BELL",
            subscribe_color="#9C27B0", next_video_text="Next Play",
            background_opacity=0.85, animation="glitch",
        ),
        lower_third=LowerThirdConfig(
            enabled=True, duration=4.0, position="top",
            bg_opacity=0.8, font_size=26, animation="slide_left",
        ),
        hook=HookConfig(
            enabled=True, duration=2.5, style="zoom_pulse",
            font_size=52, position="center", flash_rate=2.0,
            zoom_amount=1.3, color_flash="#9C27B0",
        ),
        cta=CTAConfig(
            enabled=True, timing="middle", style="subscribe_popup",
            duration=2.0, subscribe_text="SUBSCRIBE!", show_count=2,
            opacity=0.9, animation="bounce",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.9, position="top-right",
            pulse_rate=0.5,
        ),
        subtitle_style=SubtitleStyle(
            font="Impact", font_size=26, color="&H00FFFFFF",
            highlight_color="&H00FF00FF", outline_color="&H00000000",
            outline_width=4, shadow_depth=3, animation="bounce",
            position="bottom", max_words=3, bold=True, outline_mode="glow",
        ),
        transition=TransitionConfig(style="glitch", duration=0.2, easing="linear"),
    ),

    "motivational": ChannelPattern(
        name="motivational",
        display_name="Motivational",
        description="Epic text reveals, slow motion feel, powerful quotes",
        category="entertainment",
        colors=ColorPalette(
            primary="#FFD700", secondary="#FF8C00", accent="#FF4500",
            background="#0A0A0A", outline="#000000", shadow="#80000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=3.0, animation="zoom_in",
            show_channel_name=True, channel_name_size=34,
            show_logo=True, logo_scale=0.18, background_color="#0A0A0A",
        ),
        outro=OutroConfig(
            enabled=True, duration=5.0, subscribe_text="Join the Movement",
            subscribe_color="#FFD700", next_video_text="Next Inspiration",
            background_opacity=0.6, animation="zoom_out",
        ),
        lower_third=LowerThirdConfig(
            enabled=False, duration=5.0, position="bottom",
            bg_opacity=0.6, font_size=24, animation="fade",
        ),
        hook=HookConfig(
            enabled=True, duration=3.5, style="bold_statement",
            font_size=46, position="center", flash_rate=0.3,
            zoom_amount=1.2, color_flash="#FFD700",
        ),
        cta=CTAConfig(
            enabled=True, timing="end", style="banner",
            duration=4.0, subscribe_text="SUBSCRIBE for Daily Motivation",
            show_count=1, opacity=0.8, animation="zoom_in",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.75, position="top-right",
        ),
        subtitle_style=SubtitleStyle(
            font="Georgia", font_size=22, color="&H00FFD700",
            highlight_color="&H00FF4500", outline_color="&H00000000",
            outline_width=3, shadow_depth=3, animation="fade",
            position="center", max_words=4, bold=True, outline_mode="shadow",
        ),
        transition=TransitionConfig(style="fade", duration=0.8, easing="ease_in_out"),
    ),

    "comedy_clip": ChannelPattern(
        name="comedy_clip",
        display_name="Comedy Clip",
        description="Punch timing, reaction zoom, sound effect cues, fun vibes",
        category="entertainment",
        colors=ColorPalette(
            primary="#FFFFFF", secondary="#FFEB3B", accent="#FF5722",
            background="#212121", outline="#000000", shadow="#80000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=1.0, animation="pop",
            show_channel_name=False, channel_name_size=32,
            show_logo=True, logo_scale=0.10, background_color="#212121",
        ),
        outro=OutroConfig(
            enabled=True, duration=3.0, subscribe_text="LAUGH & SUBSCRIBE",
            subscribe_color="#FF5722", next_video_text="More Laughs",
            background_opacity=0.7, animation="slide_up",
        ),
        lower_third=LowerThirdConfig(
            enabled=False, duration=3.0, position="bottom",
            bg_opacity=0.7, font_size=24, animation="pop",
        ),
        hook=HookConfig(
            enabled=True, duration=2.0, style="text_flash",
            font_size=44, position="center", flash_rate=3.0,
            zoom_amount=1.0, color_flash="#FFEB3B",
        ),
        cta=CTAConfig(
            enabled=True, timing="middle", style="like_remind",
            duration=2.0, like_text="Like if you laughed!",
            show_count=2, opacity=0.85, animation="bounce",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.8, position="top-right",
        ),
        subtitle_style=SubtitleStyle(
            font="Comic Sans MS", font_size=22, color="&H00FFFFFF",
            highlight_color="&H00FFEB3B", outline_color="&H00000000",
            outline_width=3, shadow_depth=2, animation="pop",
            position="bottom", max_words=4, bold=True, outline_mode="shadow",
        ),
        transition=TransitionConfig(style="whip_pan", duration=0.15, easing="linear"),
    ),

    "lifestyle": ChannelPattern(
        name="lifestyle",
        display_name="Lifestyle",
        description="Aesthetic, minimal branding, clean typography, elegant",
        category="lifestyle",
        colors=ColorPalette(
            primary="#FFFFFF", secondary="#E0E0E0", accent="#B39DDB",
            background="#FAFAFA", outline="#000000", shadow="#40000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=2.5, animation="fade",
            show_channel_name=True, channel_name_size=24,
            show_logo=True, logo_scale=0.10, background_color="#FAFAFA",
        ),
        outro=OutroConfig(
            enabled=True, duration=5.0, subscribe_text="Follow for more",
            subscribe_color="#B39DDB", next_video_text="Next Story",
            background_opacity=0.4, animation="fade",
        ),
        lower_third=LowerThirdConfig(
            enabled=True, duration=5.0, position="bottom",
            bg_opacity=0.4, font_size=20, animation="fade",
        ),
        hook=HookConfig(
            enabled=False, duration=3.0, style="text_flash",
            font_size=32, position="center",
        ),
        cta=CTAConfig(
            enabled=True, timing="end", style="text_overlay",
            duration=4.0, subscribe_text="Follow", show_count=1,
            opacity=0.5, animation="fade",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.4, position="bottom-right",
        ),
        subtitle_style=SubtitleStyle(
            font="Helvetica", font_size=18, color="&H00FFFFFF",
            highlight_color="&H00B39DDB", outline_color="&H00000000",
            outline_width=1, shadow_depth=1, animation="fade",
            position="bottom", max_words=5, bold=False, outline_mode="backdrop",
        ),
        transition=TransitionConfig(style="fade", duration=1.0, easing="ease_in_out"),
    ),

    "tech_review": ChannelPattern(
        name="tech_review",
        display_name="Tech Review",
        description="Spec overlays, comparison frames, rating cards, clean data",
        category="technology",
        colors=ColorPalette(
            primary="#FFFFFF", secondary="#00BCD4", accent="#4CAF50",
            background="#1B1B1B", outline="#000000", shadow="#80000000",
        ),
        intro=IntroConfig(
            enabled=True, duration=2.0, animation="slide_right",
            show_channel_name=True, channel_name_size=28,
            show_logo=True, logo_scale=0.12, background_color="#1B1B1B",
        ),
        outro=OutroConfig(
            enabled=True, duration=4.0, subscribe_text="SUBSCRIBE for Reviews",
            subscribe_color="#4CAF50", next_video_text="Next Review",
            background_opacity=0.8, animation="slide_left",
        ),
        lower_third=LowerThirdConfig(
            enabled=True, duration=5.0, position="bottom",
            bg_opacity=0.85, font_size=22, animation="slide_left",
        ),
        hook=HookConfig(
            enabled=True, duration=3.0, style="question",
            font_size=38, position="center", color_flash="#00BCD4",
        ),
        cta=CTAConfig(
            enabled=True, timing="end", style="banner",
            duration=3.0, subscribe_text="Subscribe for Tech Reviews",
            show_count=1, opacity=0.85, animation="slide_up",
        ),
        watermark=WatermarkConfig(
            enabled=True, style="corner", opacity=0.8, position="top-right",
        ),
        subtitle_style=SubtitleStyle(
            font="Consolas", font_size=20, color="&H00FFFFFF",
            highlight_color="&H0000BCD4", outline_color="&H00000000",
            outline_width=2, shadow_depth=1, animation="typewriter",
            position="bottom", max_words=5, bold=False, outline_mode="box",
        ),
        transition=TransitionConfig(style="slide", duration=0.4, easing="ease_in_out"),
    ),

    "custom": ChannelPattern(
        name="custom",
        display_name="Custom",
        description="User-defined channel pattern (load from JSON)",
        category="general",
    ),
}


# ═══════════════════════════════════════════════════════════
#  Pattern Loading & Saving
# ═══════════════════════════════════════════════════════════

PATTERNS_DIR = BASE_DIR / "assets" / "patterns"


def get_pattern(name: str) -> ChannelPattern:
    """Get a channel pattern by name.

    Checks built-in patterns first, then looks for a JSON file in assets/patterns/.

    Args:
        name: Pattern name (e.g. 'viral_hype', 'custom').

    Returns:
        ChannelPattern instance.

    Raises:
        ValueError: If the pattern name is not found.
    """
    # Check built-in patterns
    if name in BUILTIN_PATTERNS:
        return BUILTIN_PATTERNS[name]

    # Check for JSON pattern file
    json_path = PATTERNS_DIR / f"{name}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return ChannelPattern.from_dict(data)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid pattern JSON file {json_path}: {exc}") from exc

    # Try case-insensitive match
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    for key in BUILTIN_PATTERNS:
        if key.lower() == name_lower:
            return BUILTIN_PATTERNS[key]

    available = sorted(BUILTIN_PATTERNS.keys())
    raise ValueError(
        f"Pattern '{name}' not found. Available patterns: {available}\n"
        f"You can also create custom patterns in assets/patterns/<name>.json"
    )


def list_patterns() -> list[ChannelPattern]:
    """List all available channel patterns.

    Returns:
        List of ChannelPattern instances (built-in + any JSON files found).
    """
    patterns = list(BUILTIN_PATTERNS.values())

    # Also discover JSON pattern files
    PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
    for json_file in PATTERNS_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            pattern = ChannelPattern.from_dict(data)
            if pattern.name not in BUILTIN_PATTERNS:
                patterns.append(pattern)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping invalid pattern file: %s", json_file)

    return patterns


def save_pattern(pattern: ChannelPattern, path: Path | None = None) -> Path:
    """Save a channel pattern to a JSON file.

    Args:
        pattern: ChannelPattern to save.
        path: Optional custom save path. Defaults to assets/patterns/<name>.json.

    Returns:
        Path to the saved JSON file.
    """
    if path is None:
        PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
        path = PATTERNS_DIR / f"{pattern.name}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pattern.to_dict(), indent=2), encoding="utf-8")
    logger.info("Pattern saved: %s -> %s", pattern.name, path)
    return path


# ═══════════════════════════════════════════════════════════
#  Hook Generator — Creates attention-grabbing first 3 seconds
# ═══════════════════════════════════════════════════════════

def generate_hook_overlay(
    input_path: Path,
    output_path: Path,
    pattern: ChannelPattern,
    hook_text: str = "",
    settings: Settings | None = None,
) -> Path:
    """Apply an attention hook overlay to the first few seconds of a video.

    The hook grabs viewer attention in the critical first 3 seconds using
    text flashes, zoom pulses, or color effects depending on the pattern.

    Args:
        input_path: Path to the source video.
        output_path: Destination path.
        pattern: ChannelPattern with hook configuration.
        hook_text: Text to display (auto-filled from transcription if empty).
        settings: Optional Settings override.

    Returns:
        Path to the output video with hook overlay.
    """
    if settings is None:
        settings = get_settings()

    if not pattern.hook.enabled:
        logger.info("Hook disabled in pattern, skipping")
        if output_path and output_path != input_path:
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return output_path
        return input_path

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    video_info = probe_video(input_path)
    duration = video_info.duration
    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT
    hook = pattern.hook
    hook_dur = min(hook.duration, duration * 0.5)

    vfilters: list[str] = []

    if hook.style == "text_flash" and hook_text:
        # Flash text on and off during the hook period
        flash_interval = 1.0 / hook.flash_rate if hook.flash_rate > 0 else 1.0
        drawtext = (
            f"drawtext=text='{hook_text}':"
            f"fontcolor=white:fontsize={hook.font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"enable='between(t\\,0\\,{hook_dur})'"
        )
        vfilters.append(drawtext)

    elif hook.style == "zoom_pulse" and hook.zoom_amount > 1.0:
        # Zoom in and back out during hook
        z = hook.zoom_amount
        total_frames = int(hook_dur * 30)
        half_frames = total_frames // 2
        vfilters.append(
            f"zoompan=z='if(lt(on\\,{half_frames})\\,"
            f"min(zoom+{z-1}/{half_frames}\\,{z})\\,"
            f"max(zoom-{z-1}/{half_frames}\\,1))':"
            f"d={total_frames}:s={out_w}x{out_h}:fps=30"
        )

    elif hook.style == "color_flash":
        # Brief color flash overlay
        alpha = 0.3
        vfilters.append(
            f"colorkey=color={hook.color_flash}:similarity=0.01:blend=0.5:"
            f"enable='between(t\\,0\\,{hook_dur})'"
        )
        # Add a brief solid color overlay that fades out
        vfilters.append(
            f"drawbox=x=0:y=0:w={out_w}:h={out_h}:"
            f"color={hook.color_flash}@0.3:t=fill:"
            f"enable='between(t\\,0\\,{min(hook_dur * 0.3, 1.0)})'"
        )

    elif hook.style == "bold_statement" and hook_text:
        # Large bold text that zooms in slightly
        drawtext = (
            f"drawtext=text='{hook_text}':"
            f"fontcolor={pattern.colors.accent}:fontsize={hook.font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"borderw=4:bordercolor=black:"
            f"enable='between(t\\,0\\,{hook_dur})'"
        )
        vfilters.append(drawtext)

    elif hook.style == "question" and hook_text:
        # Question mark or question text
        drawtext = (
            f"drawtext=text='{hook_text}':"
            f"fontcolor={pattern.colors.secondary}:fontsize={hook.font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/3:"
            f"enable='between(t\\,0\\,{hook_dur})'"
        )
        vfilters.append(drawtext)

    if not vfilters:
        logger.info("No hook filters generated, copying input")
        if output_path and output_path != input_path:
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return output_path
        return input_path

    vf_str = ",".join(vfilters)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", vf_str,
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ]

    try:
        run_ffmpeg(cmd, description=f"Hook overlay ({hook.style})",
                  total_duration=duration, timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.warning("Hook overlay failed: %s, using original", exc)
        if output_path and output_path != input_path:
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return output_path
        return input_path

    if output_path.exists():
        logger.info("Hook overlay applied: %s (%s)", output_path.name, hook.style)
        return output_path

    return input_path


# ═══════════════════════════════════════════════════════════
#  CTA Generator — Subscribe/Like prompts at optimal moments
# ═══════════════════════════════════════════════════════════

def generate_cta_overlay(
    input_path: Path,
    output_path: Path,
    pattern: ChannelPattern,
    settings: Settings | None = None,
    video_duration: float = 0.0,
) -> Path:
    """Apply Call-To-Action overlays (subscribe, like, bell) at optimal moments.

    CTA placement strategy:
    - 'optimal': Mid-point of video (peak engagement)
    - 'middle': Exactly at 50% duration
    - 'end': Last N seconds before outro
    - 'multiple': Insert at 25%, 50%, and 75% marks

    Args:
        input_path: Path to the source video.
        output_path: Destination path.
        pattern: ChannelPattern with CTA configuration.
        settings: Optional Settings override.
        video_duration: Video duration (0 = auto-detect).

    Returns:
        Path to the output video with CTA overlays.
    """
    if settings is None:
        settings = get_settings()

    if not pattern.cta.enabled:
        logger.info("CTA disabled in pattern, skipping")
        if output_path and output_path != input_path:
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return output_path
        return input_path

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    if video_duration <= 0:
        video_info = probe_video(input_path)
        video_duration = video_info.duration

    cta = pattern.cta
    out_w = settings.OUTPUT_WIDTH
    out_h = settings.OUTPUT_HEIGHT
    cta_dur = min(cta.duration, video_duration * 0.2)

    # Calculate CTA insertion points
    cta_points: list[float] = []
    if cta.timing == "middle":
        cta_points = [video_duration * 0.5]
    elif cta.timing == "end":
        cta_points = [max(0, video_duration - cta_dur - pattern.outro.duration)]
    elif cta.timing == "multiple":
        cta_points = [video_duration * 0.25, video_duration * 0.50, video_duration * 0.75]
    else:  # optimal
        # Best engagement is typically at 40-60% through
        cta_points = [video_duration * 0.45]

    # Limit to requested count
    cta_points = cta_points[:cta.show_count]

    vfilters: list[str] = []

    for i, cta_start in enumerate(cta_points):
        cta_end = min(cta_start + cta_dur, video_duration)
        safe_start = max(0, cta_start)

        if cta.style == "subscribe_popup":
            # Subscribe button popup
            box_w, box_h = 280, 54
            box_x = (out_w - box_w) // 2
            box_y = int(out_h * 0.55)

            vfilters.append(
                f"drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}:"
                f"color={pattern.colors.accent}@{cta.opacity}:t=fill:"
                f"enable='between(t\\,{safe_start:.1f}\\,{cta_end:.1f})'"
            )
            vfilters.append(
                f"drawtext=text='{cta.subscribe_text}':"
                f"fontcolor=white:fontsize=26:"
                f"x=(w-text_w)/2:y={box_y + 14}:"
                f"enable='between(t\\,{safe_start:.1f}\\,{cta_end:.1f})'"
            )

        elif cta.style == "text_overlay":
            vfilters.append(
                f"drawtext=text='{cta.subscribe_text}':"
                f"fontcolor={pattern.colors.accent}@{cta.opacity}:fontsize=22:"
                f"x=(w-text_w)/2:y=h*0.65:"
                f"enable='between(t\\,{safe_start:.1f}\\,{cta_end:.1f})'"
            )

        elif cta.style == "banner":
            banner_h = 60
            vfilters.append(
                f"drawbox=x=0:y={out_h - banner_h - 120}:w={out_w}:h={banner_h}:"
                f"color={pattern.colors.background}@{cta.opacity}:t=fill:"
                f"enable='between(t\\,{safe_start:.1f}\\,{cta_end:.1f})'"
            )
            vfilters.append(
                f"drawtext=text='{cta.subscribe_text}':"
                f"fontcolor={pattern.colors.accent}:fontsize=24:"
                f"x=(w-text_w)/2:y={out_h - banner_h - 120 + 18}:"
                f"enable='between(t\\,{safe_start:.1f}\\,{cta_end:.1f})'"
            )

        elif cta.style == "like_remind":
            vfilters.append(
                f"drawtext=text='{cta.like_text}':"
                f"fontcolor=white@{cta.opacity}:fontsize=28:"
                f"x=(w-text_w)/2:y=h*0.6:"
                f"enable='between(t\\,{safe_start:.1f}\\,{cta_end:.1f})'"
            )

    if not vfilters:
        if output_path and output_path != input_path:
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return output_path
        return input_path

    vf_str = ",".join(vfilters)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", vf_str,
        "-c:v", settings.FFMPEG_VIDEO_CODEC,
        "-preset", settings.FFMPEG_PRESET,
        "-crf", str(settings.FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-threads", str(settings.FFMPEG_THREADS),
        str(output_path),
    ]

    try:
        run_ffmpeg(cmd, description=f"CTA overlay ({cta.style}, {len(cta_points)} insertions)",
                  total_duration=video_duration, timeout=settings.FFMPEG_TIMEOUT)
    except FFmpegError as exc:
        logger.warning("CTA overlay failed: %s, using original", exc)
        if output_path and output_path != input_path:
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return output_path
        return input_path

    if output_path.exists():
        logger.info("CTA overlay applied: %s (%d insertions)", output_path.name, len(cta_points))
        return output_path

    return input_path


# ═══════════════════════════════════════════════════════════
#  Apply Full Pattern to Video
# ═══════════════════════════════════════════════════════════

def apply_channel_pattern(
    input_path: Path,
    pattern: ChannelPattern,
    settings: Settings | None = None,
    channel_name: str = "",
    hook_text: str = "",
) -> Path:
    """Apply a complete channel pattern to a video.

    Applies all enabled pattern elements in order:
    1. Hook overlay (first 3 seconds)
    2. Lower third
    3. CTA overlays
    4. Logo/watermark (handled by existing stamp_logo)
    5. Outro card

    Note: Intro stinger is handled separately (before main content).
    Subtitle style is applied during subtitle generation.

    Args:
        input_path: Path to the source video.
        pattern: ChannelPattern to apply.
        settings: Optional Settings override.
        channel_name: Channel name for lower third/outro.
        hook_text: Hook text (from transcription if empty).

    Returns:
        Path to the final output video.
    """
    if settings is None:
        settings = get_settings()

    if not input_path.exists():
        raise FFmpegError(f"Input video not found: {input_path}")

    current = input_path
    intermediates: list[Path] = []

    from utils.file_utils import make_output_path, safe_delete
    safe_name = input_path.stem

    # 1. Hook overlay
    if pattern.hook.enabled:
        hook_out = make_output_path(settings.SHORTS_DIR, safe_name, "hooked")
        intermediates.append(hook_out)
        try:
            current = generate_hook_overlay(current, hook_out, pattern, hook_text, settings)
        except Exception as exc:
            logger.warning("Hook overlay failed (continuing): %s", exc)

    # 2. Lower third
    if pattern.lower_third.enabled and channel_name:
        from core.logo_stamper import add_lower_third
        lt_out = make_output_path(settings.SHORTS_DIR, safe_name, "lower_third")
        intermediates.append(lt_out)
        try:
            current = add_lower_third(
                current, channel_name, lt_out, settings,
                duration=pattern.lower_third.duration,
                bg_opacity=pattern.lower_third.bg_opacity,
                font_size=pattern.lower_third.font_size,
            )
        except Exception as exc:
            logger.warning("Lower third failed (continuing): %s", exc)

    # 3. CTA overlays
    if pattern.cta.enabled:
        cta_out = make_output_path(settings.SHORTS_DIR, safe_name, "cta")
        intermediates.append(cta_out)
        try:
            current = generate_cta_overlay(current, cta_out, pattern, settings)
        except Exception as exc:
            logger.warning("CTA overlay failed (continuing): %s", exc)

    # 4. Outro card
    if pattern.outro.enabled and channel_name:
        from core.logo_stamper import add_outro_card
        outro_out = make_output_path(settings.SHORTS_DIR, safe_name, "outro")
        intermediates.append(outro_out)
        try:
            current = add_outro_card(
                current, outro_out, channel_name, settings,
                duration=pattern.outro.duration,
                subscribe_text=pattern.outro.subscribe_text,
                next_video_text=pattern.outro.next_video_text,
            )
        except Exception as exc:
            logger.warning("Outro card failed (continuing): %s", exc)

    # Cleanup intermediates
    for inter in intermediates:
        if inter != current and inter.exists():
            try:
                safe_delete(inter)
            except OSError:
                pass

    return current


# ═══════════════════════════════════════════════════════════
#  Extract Hook Text from Transcription
# ═══════════════════════════════════════════════════════════

def extract_hook_text(transcription, max_words: int = 8) -> str:
    """Extract the most impactful text from the first seconds of transcription.

    Uses word-level timing to find the first complete phrase, then
    capitalizes and formats it as a hook.

    Args:
        transcription: TranscriptionResult with word timestamps.
        max_words: Maximum number of words for the hook text.

    Returns:
        Hook text string, or empty string if no transcription available.
    """
    if not transcription or not hasattr(transcription, 'words') or not transcription.words:
        return ""

    # Get words from the first few seconds
    first_words = [w for w in transcription.words if w.start < 5.0][:max_words]
    if not first_words:
        first_words = transcription.words[:max_words]

    hook = " ".join(w.word.strip() for w in first_words)
    # Clean up
    hook = hook.strip().rstrip(",.").strip()
    # Capitalize first letter
    if hook:
        hook = hook[0].upper() + hook[1:]

    return hook
