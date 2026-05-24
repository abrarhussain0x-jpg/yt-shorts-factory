"""
config/settings.py — Centralised, validated configuration for yt-shorts-factory.

All settings are environment-variable-driven via pydantic-settings with
strict validation, auto-detection of hardware capabilities, and
automatic directory creation. Every field has a sensible default so
the system works out-of-the-box.

Expanded settings cover: audio enhancement, face tracking, motion detection,
thumbnail generation, content moderation, multi-clip management, advanced
analysis, advanced subtitles, brand kit, performance tuning, advanced export,
hardware acceleration, YouTube API integration, and auto-caption styling.
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


# ── Base directory (project root) ──────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file.

    Validators enforce business rules and auto-detect optimal hardware.
    All paths are resolved to absolute paths on initialization.

    Settings are grouped into logical sections:
      - Whisper: speech recognition
      - Output Resolution: target video dimensions
      - Clip Extraction: duration and energy settings
      - Logo: watermark overlay
      - Subtitles: basic subtitle rendering
      - Subtitle Advanced: enhanced subtitle options
      - Auto Caption Style: dynamic caption rendering
      - FFmpeg: encoding and processing
      - Audio Enhancement: post-processing audio filters
      - Face Tracking: face detection for reframing
      - Motion Detection: motion-based clip selection
      - Thumbnail: thumbnail generation settings
      - Content Moderation: content safety checks
      - Multi Clip: multiple clip extraction rules
      - Advanced Analysis: spectral and speech analysis
      - Brand Kit: branding assets and templates
      - Performance: resource limits
      - Export Advanced: two-pass encoding and effects
      - Platform Variants: export targets
      - Queue / Scheduler: job management
      - Metadata Generation: keyword and hashtag settings
      - YouTube API Integration: metadata enrichment
      - Pipeline: checkpoint and cleanup settings
      - Derived Path Constants: output directories
    """

    # ── Whisper ───────────────────────────────────────────
    WHISPER_MODEL: str = "base"
    WHISPER_LANGUAGE: str = "auto"
    WHISPER_TASK: Literal["transcribe", "translate"] = "transcribe"
    WHISPER_DEVICE: str = "auto"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_TEMPERATURE: float = 0.0
    WHISPER_FALLBACK_MODELS: str = "base,small"
    WHISPER_COMPUTE_TYPE: str = "auto"

    # ── Output Resolution ─────────────────────────────────
    OUTPUT_WIDTH: int = 1080
    OUTPUT_HEIGHT: int = 1920

    # ── Clip Extraction ───────────────────────────────────
    CLIP_DURATION: int = 45
    CLIP_MIN_DURATION: int = 15
    CLIP_MAX_DURATION: int = 180
    CLIP_DURATION_PRESET: str = "standard"  # quick(25s), standard(45s), extended(180s)

    # Duration presets: name -> seconds
    DURATION_PRESETS: dict[str, int] = {
        "quick": 25,       # 25 seconds — TikTok quick hook, fast scroll
        "standard": 45,   # 45 seconds — Sweet spot for Shorts & Reels
        "extended": 180,  # 3 minutes — Long-form Shorts, TikTok full length
        "25s": 25,
        "45s": 45,
        "3min": 180,
        "3minute": 180,
    }
    ENERGY_SAMPLE_INTERVAL: float = 2.0
    ENERGY_SMOOTHING_WINDOW: int = 5
    SCENE_DETECT_THRESHOLD: float = 0.3
    SILENCE_NOISE_FLOOR: str = "-30dB"
    SILENCE_MIN_DURATION: float = 1.0

    # ── Logo ──────────────────────────────────────────────
    LOGO_PATH: str = "assets/logo.png"
    LOGO_POSITION: str = "top-right"
    LOGO_OPACITY: float = 0.85
    LOGO_SCALE: float = 0.12
    LOGO_FADE_DURATION: float = 1.0
    LOGO_MARGIN: int = 20
    LOGO_SAFE_ZONE_TOP: int = 50
    LOGO_SAFE_ZONE_BOTTOM: int = 100

    # ── Subtitles ─────────────────────────────────────────
    SUBTITLE_FONT: str = "Arial Black"
    SUBTITLE_COLOR: str = "&H00FFFF00"
    SUBTITLE_HIGHLIGHT_COLOR: str = "&H0000FFFF"
    SUBTITLE_OUTLINE_COLOR: str = "&H00000000"
    SUBTITLE_SHADOW_COLOR: str = "&H80000000"
    SUBTITLE_FONT_SIZE: int = 22
    SUBTITLE_BOLD: int = 1
    SUBTITLE_OUTLINE_WIDTH: int = 3
    SUBTITLE_SHADOW_DEPTH: int = 2
    SUBTITLE_MAX_WORDS: int = 4
    SUBTITLE_MARGIN_V: int = 80
    SUBTITLE_ANIMATION: str = "karaoke"
    SUBTITLE_LINE_SPACING: int = 8
    SUBTITLE_MIN_DISPLAY_TIME: float = 0.3
    SUBTITLE_OVERLAP: float = 0.05

    # ── Subtitle Advanced ─────────────────────────────────
    SUBTITLE_READING_SPEED_WPM: int = 200
    SUBTITLE_MIN_CHARS_PER_SECOND: float = 5.0
    SUBTITLE_MAX_CHARS_PER_SECOND: float = 25.0
    SUBTITLE_POSITION_STRATEGY: str = "bottom"
    SUBTITLE_OUTLINE_MODE: str = "shadow"

    # ── Auto Caption Style ────────────────────────────────
    AUTO_CAPTION_FONT_AUTO_SCALE: bool = True
    AUTO_CAPTION_MAX_LINES_ON_SCREEN: int = 2
    AUTO_CAPTION_BOX_PADDING: int = 8

    # ── FFmpeg ────────────────────────────────────────────
    FFMPEG_PRESET: str = "fast"
    FFMPEG_CRF: int = 23
    FFMPEG_AUDIO_BITRATE: str = "192k"
    FFMPEG_VIDEO_CODEC: str = "libx264"
    FFMPEG_AUDIO_CODEC: str = "aac"
    FFMPEG_PIXEL_FORMAT: str = "yuv420p"
    FFMPEG_THREADS: int = 0
    FFMPEG_HW_ACCEL: str = "auto"
    FFMPEG_LOGLEVEL: str = "warning"
    FFMPEG_TIMEOUT: int = 600

    # ── Audio Enhancement ─────────────────────────────────
    AUDIO_NOISE_REDUCTION: bool = False
    AUDIO_NOISE_REDUCTION_STRENGTH: str = "medium"
    AUDIO_COMPRESSION: bool = False
    AUDIO_COMPRESSION_THRESHOLD: str = "-20dB"
    AUDIO_COMPRESSION_RATIO: int = 4
    AUDIO_COMPRESSION_ATTACK: int = 5
    AUDIO_COMPRESSION_RELEASE: int = 50
    AUDIO_EQ: bool = False
    AUDIO_EQ_PRESET: str = "voice"
    AUDIO_DEESSER: bool = False
    AUDIO_NORMALIZER: bool = True
    AUDIO_NORMALIZER_TARGET_LUFS: float = -16.0
    AUDIO_NORMALIZER_TRUE_PEAK: float = -1.5
    AUDIO_NORMALIZER_LRA: float = 11.0

    # ── Face Tracking ─────────────────────────────────────
    FACE_TRACKING_MODEL: str = "haar"
    FACE_TRACKING_CONFIDENCE: float = 0.85
    FACE_TRACKING_PADDING: float = 0.2
    FACE_TRACKING_SMOOTHING: float = 0.3
    FACE_TRACKING_MULTI_FACE_STRATEGY: str = "largest"

    # ── Motion Detection ──────────────────────────────────
    MOTION_DETECTION_METHOD: str = "frame_diff"
    MOTION_DETECTION_SENSITIVITY: float = 0.5
    MOTION_DETECTION_MIN_AREA: int = 500
    MOTION_DETECTION_MAX_AREA: int = 500000

    # ── Thumbnail ─────────────────────────────────────────
    THUMBNAIL_COUNT: int = 3
    THUMBNAIL_QUALITY: int = 85
    THUMBNAIL_FORMAT: str = "jpg"
    THUMBNAIL_RESOLUTION: str = "1280x720"
    THUMBNAIL_SELECTION_STRATEGY: str = "evenly-spaced"

    # ── Content Moderation ────────────────────────────────
    CONTENT_MODERATION_ENABLED: bool = False
    CONTENT_MODERATION_SENSITIVITY: str = "medium"
    CONTENT_MODERATION_CATEGORIES: str = "violence,nudity,hate_speech"

    # ── Multi Clip ────────────────────────────────────────
    MULTI_CLIP_MAX_PER_VIDEO: int = 5
    MULTI_CLIP_MIN_GAP_SECONDS: float = 10.0
    MULTI_CLIP_DEDUP_THRESHOLD_SECONDS: float = 5.0

    # ── Advanced Analysis ─────────────────────────────────
    ADVANCED_SPECTRAL_ANALYSIS: bool = False
    ADVANCED_PITCH_DETECTION: bool = False
    ADVANCED_SPEECH_RATE_WPM: bool = False
    ADVANCED_EMPHASIS_DETECTION: bool = False

    # ── Brand Kit ─────────────────────────────────────────
    BRAND_INTRO_CLIP_PATH: str = ""
    BRAND_OUTRO_CLIP_PATH: str = ""
    BRAND_LOWER_THIRD_TEMPLATE: str = ""
    BRAND_COLOR_PALETTE: str = ""

    # ── Performance ───────────────────────────────────────
    PERFORMANCE_MEMORY_LIMIT_MB: int = 4096
    PERFORMANCE_DISK_SPACE_RESERVE_MB: int = 2048
    PERFORMANCE_TEMP_DIR: str = ""

    # ── Export Advanced ───────────────────────────────────
    EXPORT_TWO_PASS_ENCODING: bool = False
    EXPORT_DENOISE_VIDEO: bool = False
    EXPORT_DENOISE_VIDEO_STRENGTH: str = "light"
    EXPORT_SHARPEN: bool = False
    EXPORT_SHARPEN_AMOUNT: float = 1.0
    EXPORT_FILM_GRAIN_AMOUNT: int = 0

    # ── Platform Variants ─────────────────────────────────
    EXPORT_YOUTUBE: bool = True
    EXPORT_TIKTOK: bool = True
    EXPORT_REELS: bool = True

    # ── Queue / Scheduler ─────────────────────────────────
    MAX_CONCURRENT_JOBS: int = 2
    JOB_RETRY_ATTEMPTS: int = 3
    JOB_RETRY_DELAY: int = 30
    WORKER_POLL_INTERVAL: float = 5.0
    WORKER_GRACEFUL_TIMEOUT: int = 300

    # ── Metadata Generation ───────────────────────────────
    GENERATE_METADATA: bool = True
    METADATA_LANGUAGE: str = "en"
    METADATA_MAX_KEYWORDS: int = 15
    METADATA_HASHTAG_COUNT: int = 10

    # ── YouTube API Integration ───────────────────────────
    YOUTUBE_API_KEY: str = ""
    YOUTUBE_API_ENABLED: bool = False
    YOUTUBE_API_QUOTA_DAILY_LIMIT: int = 10000
    YOUTUBE_API_QUOTA_USED: int = 0
    YOUTUBE_API_REGION_CODE: str = "US"

    # ── Channel Pattern ───────────────────────────────────
    CHANNEL_PATTERN: str = ""          # Pattern name: viral_hype, chill_vibes, news_alert, etc.
    CHANNEL_NAME: str = ""             # Channel name for branding overlays
    CHANNEL_HOOK_ENABLED: bool = True  # Enable attention hook in first 3 seconds
    CHANNEL_CTA_ENABLED: bool = True   # Enable subscribe/follow CTA overlays
    CHANNEL_OUTRO_ENABLED: bool = True # Enable outro card with subscribe CTA
    CHANNEL_LOWER_THIRD_ENABLED: bool = False  # Enable lower third name bar

    # ── Turbo Mode ────────────────────────────────────────
    TURBO_MODE: bool = False  # Maximum speed: skip analysis extras, ultrafast encoding, tiny whisper

    # ── Superfast Mode ───────────────────────────────────
    SUPERFAST_MODE: bool = False  # Single-pass FFmpeg, center crop, minimal analysis

    # ── Speed Optimization ────────────────────────────────
    SPEED_PARALLEL_EXPORT: bool = True      # Export platforms in parallel
    SPEED_PARALLEL_BATCH: bool = True       # Process batch URLs in parallel
    SPEED_USE_FASTER_WHISPER: bool = True   # Use faster-whisper if available (4x faster)
    SPEED_AUTO_HW_ACCEL: bool = True        # Auto-detect and use GPU encoding
    SPEED_OPTIMAL_THREADS: bool = True      # Auto-detect optimal thread count
    SPEED_SKIP_ANALYSIS_EXTRAS: bool = False  # Skip spectral/pitch analysis for speed
    SPEED_FAST_SMART_CROP: bool = True      # Use faster smart crop algorithm

    # ── Pipeline ──────────────────────────────────────────
    CLEANUP_INTERMEDIATES: bool = True
    CHECKPOINT_ENABLED: bool = True
    CHECKPOINT_DIR: str = ""

    # ── Derived Path Constants ────────────────────────────
    DOWNLOADS_DIR: Path = BASE_DIR / "output" / "downloads"
    SHORTS_DIR: Path = BASE_DIR / "output" / "shorts"
    YOUTUBE_DIR: Path = BASE_DIR / "output" / "shorts" / "youtube"
    TIKTOK_DIR: Path = BASE_DIR / "output" / "shorts" / "tiktok"
    REELS_DIR: Path = BASE_DIR / "output" / "shorts" / "reels"
    METADATA_DIR: Path = BASE_DIR / "output" / "metadata"
    LOGS_DIR: Path = BASE_DIR / "output" / "logs"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    DB_PATH: Path = BASE_DIR / "output" / "factory.db"
    THUMBNAILS_DIR: Path = BASE_DIR / "output" / "thumbnails"

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_default": True,
    }

    # ══════════════════════════════════════════════════════
    #  Validators
    # ══════════════════════════════════════════════════════

    @field_validator("WHISPER_MODEL")
    @classmethod
    def validate_whisper_model(cls, v: str) -> str:
        """Validate that the Whisper model name is recognised."""
        allowed = {
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
        }
        if v not in allowed:
            raise ValueError(f"WHISPER_MODEL must be one of {sorted(allowed)}, got '{v}'")
        return v

    @field_validator("LOGO_POSITION")
    @classmethod
    def validate_logo_position(cls, v: str) -> str:
        """Validate that logo position is one of the allowed positions."""
        allowed = {"top-left", "top-right", "bottom-left", "bottom-right", "center"}
        if v not in allowed:
            raise ValueError(f"LOGO_POSITION must be one of {allowed}, got '{v}'")
        return v

    @field_validator("LOGO_OPACITY")
    @classmethod
    def validate_logo_opacity(cls, v: float) -> float:
        """Validate logo opacity is in the valid range [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"LOGO_OPACITY must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("LOGO_SCALE")
    @classmethod
    def validate_logo_scale(cls, v: float) -> float:
        """Validate logo scale is within a practical range."""
        if not 0.01 <= v <= 0.5:
            raise ValueError(f"LOGO_SCALE must be between 0.01 and 0.5, got {v}")
        return v

    @field_validator("OUTPUT_WIDTH", "OUTPUT_HEIGHT")
    @classmethod
    def validate_output_dimensions(cls, v: int) -> int:
        """Validate output dimensions are positive, even, and within practical limits."""
        if v <= 0:
            raise ValueError(f"Output dimension must be positive, got {v}")
        if v % 2 != 0:
            raise ValueError(f"Output dimension must be even (required by h264), got {v}")
        if v < 64:
            raise ValueError(f"Output dimension must be at least 64 pixels, got {v}")
        if v > 7680:
            raise ValueError(f"Output dimension must be at most 7680 (8K), got {v}")
        return v

    @field_validator("SUBTITLE_ANIMATION")
    @classmethod
    def validate_subtitle_animation(cls, v: str) -> str:
        """Validate subtitle animation type is recognised."""
        allowed = {"karaoke", "fade", "pop", "glow", "typewriter", "bounce", "wave", "rainbow", "neon", "matrix", "3d_rotate", "none"}
        if v not in allowed:
            raise ValueError(f"SUBTITLE_ANIMATION must be one of {allowed}, got '{v}'")
        return v

    @field_validator("SUBTITLE_POSITION_STRATEGY")
    @classmethod
    def validate_subtitle_position_strategy(cls, v: str) -> str:
        """Validate subtitle position strategy is recognised."""
        allowed = {"bottom", "center", "follow-speaker"}
        if v not in allowed:
            raise ValueError(f"SUBTITLE_POSITION_STRATEGY must be one of {allowed}, got '{v}'")
        return v

    @field_validator("SUBTITLE_OUTLINE_MODE")
    @classmethod
    def validate_subtitle_outline_mode(cls, v: str) -> str:
        """Validate subtitle outline mode is recognised."""
        allowed = {"shadow", "glow", "box", "backdrop"}
        if v not in allowed:
            raise ValueError(f"SUBTITLE_OUTLINE_MODE must be one of {allowed}, got '{v}'")
        return v

    @field_validator("FFMPEG_PRESET")
    @classmethod
    def validate_ffmpeg_preset(cls, v: str) -> str:
        """Validate FFmpeg encoding preset name."""
        allowed = {
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        }
        if v not in allowed:
            raise ValueError(f"FFMPEG_PRESET must be one of {allowed}, got '{v}'")
        return v

    @field_validator("FFMPEG_CRF")
    @classmethod
    def validate_ffmpeg_crf(cls, v: int) -> int:
        """Validate CRF value is within the valid H.264 range."""
        if not 0 <= v <= 51:
            raise ValueError(f"FFMPEG_CRF must be between 0 and 51, got {v}")
        return v

    @field_validator("FFMPEG_HW_ACCEL")
    @classmethod
    def validate_ffmpeg_hw_accel(cls, v: str) -> str:
        """Validate hardware acceleration mode is recognised."""
        allowed = {"auto", "nvenc", "videotoolbox", "vaapi", "qsv", "amf", "none"}
        if v not in allowed:
            raise ValueError(f"FFMPEG_HW_ACCEL must be one of {allowed}, got '{v}'")
        return v

    @field_validator("ENERGY_SAMPLE_INTERVAL")
    @classmethod
    def validate_sample_interval(cls, v: float) -> float:
        """Validate energy sample interval is within practical bounds."""
        if not 0.1 <= v <= 10.0:
            raise ValueError(f"ENERGY_SAMPLE_INTERVAL must be between 0.1 and 10.0, got {v}")
        return v

    @field_validator("SCENE_DETECT_THRESHOLD")
    @classmethod
    def validate_scene_threshold(cls, v: float) -> float:
        """Validate scene detection threshold is between 0 and 1."""
        if not 0.01 <= v <= 1.0:
            raise ValueError(f"SCENE_DETECT_THRESHOLD must be between 0.01 and 1.0, got {v}")
        return v

    @field_validator("SUBTITLE_FONT_SIZE")
    @classmethod
    def validate_font_size(cls, v: int) -> int:
        """Validate subtitle font size is within practical bounds."""
        if not 8 <= v <= 72:
            raise ValueError(f"SUBTITLE_FONT_SIZE must be between 8 and 72, got {v}")
        return v

    @field_validator("AUDIO_NOISE_REDUCTION_STRENGTH")
    @classmethod
    def validate_noise_reduction_strength(cls, v: str) -> str:
        """Validate noise reduction strength level."""
        allowed = {"light", "medium", "heavy"}
        if v not in allowed:
            raise ValueError(f"AUDIO_NOISE_REDUCTION_STRENGTH must be one of {allowed}, got '{v}'")
        return v

    @field_validator("AUDIO_COMPRESSION_RATIO")
    @classmethod
    def validate_compression_ratio(cls, v: int) -> int:
        """Validate audio compression ratio is within a practical range."""
        if not 1 <= v <= 20:
            raise ValueError(f"AUDIO_COMPRESSION_RATIO must be between 1 and 20, got {v}")
        return v

    @field_validator("AUDIO_NORMALIZER_TARGET_LUFS")
    @classmethod
    def validate_normalizer_target_lufs(cls, v: float) -> float:
        """Validate loudness normalisation target is within EBU R128 practical range."""
        if not -30.0 <= v <= 0.0:
            raise ValueError(f"AUDIO_NORMALIZER_TARGET_LUFS must be between -30.0 and 0.0, got {v}")
        return v

    @field_validator("AUDIO_NORMALIZER_TRUE_PEAK")
    @classmethod
    def validate_normalizer_true_peak(cls, v: float) -> float:
        """Validate true-peak ceiling is within a practical range."""
        if not -3.0 <= v <= 0.0:
            raise ValueError(f"AUDIO_NORMALIZER_TRUE_PEAK must be between -3.0 and 0.0, got {v}")
        return v

    @field_validator("AUDIO_NORMALIZER_LRA")
    @classmethod
    def validate_normalizer_lra(cls, v: float) -> float:
        """Validate loudness range target is within a practical range."""
        if not 1.0 <= v <= 20.0:
            raise ValueError(f"AUDIO_NORMALIZER_LRA must be between 1.0 and 20.0, got {v}")
        return v

    @field_validator("FACE_TRACKING_MODEL")
    @classmethod
    def validate_face_tracking_model(cls, v: str) -> str:
        """Validate face tracking model type."""
        allowed = {"haar", "dnn", "mediapipe"}
        if v not in allowed:
            raise ValueError(f"FACE_TRACKING_MODEL must be one of {allowed}, got '{v}'")
        return v

    @field_validator("FACE_TRACKING_CONFIDENCE")
    @classmethod
    def validate_face_tracking_confidence(cls, v: float) -> float:
        """Validate face tracking confidence threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"FACE_TRACKING_CONFIDENCE must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("FACE_TRACKING_PADDING")
    @classmethod
    def validate_face_tracking_padding(cls, v: float) -> float:
        """Validate face tracking padding factor."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"FACE_TRACKING_PADDING must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("FACE_TRACKING_SMOOTHING")
    @classmethod
    def validate_face_tracking_smoothing(cls, v: float) -> float:
        """Validate face tracking smoothing factor."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"FACE_TRACKING_SMOOTHING must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("FACE_TRACKING_MULTI_FACE_STRATEGY")
    @classmethod
    def validate_face_tracking_multi_face_strategy(cls, v: str) -> str:
        """Validate multi-face strategy selection."""
        allowed = {"largest", "center", "all"}
        if v not in allowed:
            raise ValueError(f"FACE_TRACKING_MULTI_FACE_STRATEGY must be one of {allowed}, got '{v}'")
        return v

    @field_validator("MOTION_DETECTION_METHOD")
    @classmethod
    def validate_motion_detection_method(cls, v: str) -> str:
        """Validate motion detection algorithm selection."""
        allowed = {"frame_diff", "optical_flow"}
        if v not in allowed:
            raise ValueError(f"MOTION_DETECTION_METHOD must be one of {allowed}, got '{v}'")
        return v

    @field_validator("MOTION_DETECTION_SENSITIVITY")
    @classmethod
    def validate_motion_detection_sensitivity(cls, v: float) -> float:
        """Validate motion detection sensitivity is in [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"MOTION_DETECTION_SENSITIVITY must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("THUMBNAIL_FORMAT")
    @classmethod
    def validate_thumbnail_format(cls, v: str) -> str:
        """Validate thumbnail image format."""
        allowed = {"jpg", "png", "webp"}
        if v not in allowed:
            raise ValueError(f"THUMBNAIL_FORMAT must be one of {allowed}, got '{v}'")
        return v

    @field_validator("THUMBNAIL_QUALITY")
    @classmethod
    def validate_thumbnail_quality(cls, v: int) -> int:
        """Validate thumbnail quality is within [1, 100]."""
        if not 1 <= v <= 100:
            raise ValueError(f"THUMBNAIL_QUALITY must be between 1 and 100, got {v}")
        return v

    @field_validator("THUMBNAIL_COUNT")
    @classmethod
    def validate_thumbnail_count(cls, v: int) -> int:
        """Validate thumbnail count is positive and practical."""
        if not 1 <= v <= 20:
            raise ValueError(f"THUMBNAIL_COUNT must be between 1 and 20, got {v}")
        return v

    @field_validator("THUMBNAIL_SELECTION_STRATEGY")
    @classmethod
    def validate_thumbnail_selection_strategy(cls, v: str) -> str:
        """Validate thumbnail selection strategy."""
        allowed = {"evenly-spaced", "peak-energy", "face-detected"}
        if v not in allowed:
            raise ValueError(f"THUMBNAIL_SELECTION_STRATEGY must be one of {allowed}, got '{v}'")
        return v

    @field_validator("CONTENT_MODERATION_SENSITIVITY")
    @classmethod
    def validate_content_moderation_sensitivity(cls, v: str) -> str:
        """Validate content moderation sensitivity level."""
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"CONTENT_MODERATION_SENSITIVITY must be one of {allowed}, got '{v}'")
        return v

    @field_validator("MULTI_CLIP_MAX_PER_VIDEO")
    @classmethod
    def validate_multi_clip_max(cls, v: int) -> int:
        """Validate max clips per video is positive and practical."""
        if not 1 <= v <= 50:
            raise ValueError(f"MULTI_CLIP_MAX_PER_VIDEO must be between 1 and 50, got {v}")
        return v

    @field_validator("MULTI_CLIP_MIN_GAP_SECONDS")
    @classmethod
    def validate_multi_clip_min_gap(cls, v: float) -> float:
        """Validate minimum gap between clips is non-negative."""
        if v < 0.0:
            raise ValueError(f"MULTI_CLIP_MIN_GAP_SECONDS must be non-negative, got {v}")
        return v

    @field_validator("MULTI_CLIP_DEDUP_THRESHOLD_SECONDS")
    @classmethod
    def validate_multi_clip_dedup_threshold(cls, v: float) -> float:
        """Validate dedup threshold is positive."""
        if v <= 0.0:
            raise ValueError(f"MULTI_CLIP_DEDUP_THRESHOLD_SECONDS must be positive, got {v}")
        return v

    @field_validator("AUDIO_EQ_PRESET")
    @classmethod
    def validate_audio_eq_preset(cls, v: str) -> str:
        """Validate EQ preset name."""
        allowed = {"voice", "music", "bass_boost", "treble_boost", "flat", "podcast"}
        if v not in allowed:
            raise ValueError(f"AUDIO_EQ_PRESET must be one of {allowed}, got '{v}'")
        return v

    @field_validator("EXPORT_DENOISE_VIDEO_STRENGTH")
    @classmethod
    def validate_denoise_video_strength(cls, v: str) -> str:
        """Validate video denoise strength level."""
        allowed = {"light", "medium", "heavy"}
        if v not in allowed:
            raise ValueError(f"EXPORT_DENOISE_VIDEO_STRENGTH must be one of {allowed}, got '{v}'")
        return v

    @field_validator("EXPORT_SHARPEN_AMOUNT")
    @classmethod
    def validate_sharpen_amount(cls, v: float) -> float:
        """Validate sharpen amount is in a practical range."""
        if not 0.0 <= v <= 5.0:
            raise ValueError(f"EXPORT_SHARPEN_AMOUNT must be between 0.0 and 5.0, got {v}")
        return v

    @field_validator("EXPORT_FILM_GRAIN_AMOUNT")
    @classmethod
    def validate_film_grain_amount(cls, v: int) -> int:
        """Validate film grain amount is within practical range."""
        if not 0 <= v <= 32:
            raise ValueError(f"EXPORT_FILM_GRAIN_AMOUNT must be between 0 and 32, got {v}")
        return v

    @field_validator("PERFORMANCE_MEMORY_LIMIT_MB")
    @classmethod
    def validate_memory_limit(cls, v: int) -> int:
        """Validate memory limit is at least 512 MB."""
        if v < 512:
            raise ValueError(f"PERFORMANCE_MEMORY_LIMIT_MB must be at least 512, got {v}")
        return v

    @field_validator("PERFORMANCE_DISK_SPACE_RESERVE_MB")
    @classmethod
    def validate_disk_space_reserve(cls, v: int) -> int:
        """Validate disk space reserve is at least 100 MB."""
        if v < 100:
            raise ValueError(f"PERFORMANCE_DISK_SPACE_RESERVE_MB must be at least 100, got {v}")
        return v

    @field_validator("AUTO_CAPTION_MAX_LINES_ON_SCREEN")
    @classmethod
    def validate_auto_caption_max_lines(cls, v: int) -> int:
        """Validate auto-caption max lines is practical."""
        if not 1 <= v <= 5:
            raise ValueError(f"AUTO_CAPTION_MAX_LINES_ON_SCREEN must be between 1 and 5, got {v}")
        return v

    @field_validator("AUTO_CAPTION_BOX_PADDING")
    @classmethod
    def validate_auto_caption_box_padding(cls, v: int) -> int:
        """Validate auto-caption box padding is non-negative."""
        if v < 0:
            raise ValueError(f"AUTO_CAPTION_BOX_PADDING must be non-negative, got {v}")
        return v

    @field_validator("SUBTITLE_READING_SPEED_WPM")
    @classmethod
    def validate_reading_speed_wpm(cls, v: int) -> int:
        """Validate reading speed is within practical bounds."""
        if not 50 <= v <= 500:
            raise ValueError(f"SUBTITLE_READING_SPEED_WPM must be between 50 and 500, got {v}")
        return v

    @field_validator("SUBTITLE_MIN_CHARS_PER_SECOND")
    @classmethod
    def validate_min_chars_per_second(cls, v: float) -> float:
        """Validate minimum characters per second for subtitle display."""
        if not 1.0 <= v <= 30.0:
            raise ValueError(f"SUBTITLE_MIN_CHARS_PER_SECOND must be between 1.0 and 30.0, got {v}")
        return v

    @field_validator("SUBTITLE_MAX_CHARS_PER_SECOND")
    @classmethod
    def validate_max_chars_per_second(cls, v: float) -> float:
        """Validate maximum characters per second for subtitle display."""
        if not 5.0 <= v <= 50.0:
            raise ValueError(f"SUBTITLE_MAX_CHARS_PER_SECOND must be between 5.0 and 50.0, got {v}")
        return v

    @field_validator("YOUTUBE_API_QUOTA_DAILY_LIMIT")
    @classmethod
    def validate_youtube_quota_limit(cls, v: int) -> int:
        """Validate YouTube API daily quota limit."""
        if v < 0:
            raise ValueError(f"YOUTUBE_API_QUOTA_DAILY_LIMIT must be non-negative, got {v}")
        return v

    @field_validator("SILENCE_MIN_DURATION")
    @classmethod
    def validate_silence_min_duration(cls, v: float) -> float:
        """Validate silence minimum duration is positive."""
        if v <= 0.0:
            raise ValueError(f"SILENCE_MIN_DURATION must be positive, got {v}")
        return v

    @field_validator("WHISPER_BEAM_SIZE")
    @classmethod
    def validate_whisper_beam_size(cls, v: int) -> int:
        """Validate Whisper beam size is positive."""
        if v < 1:
            raise ValueError(f"WHISPER_BEAM_SIZE must be at least 1, got {v}")
        return v

    @field_validator("MAX_CONCURRENT_JOBS")
    @classmethod
    def validate_max_concurrent_jobs(cls, v: int) -> int:
        """Validate max concurrent jobs is positive."""
        if v < 1:
            raise ValueError(f"MAX_CONCURRENT_JOBS must be at least 1, got {v}")
        return v

    # ══════════════════════════════════════════════════════
    #  Model Validators
    # ══════════════════════════════════════════════════════

    @field_validator("CLIP_DURATION_PRESET")
    @classmethod
    def validate_clip_duration_preset(cls, v: str) -> str:
        """Validate that the clip duration preset is recognised."""
        # Allow any string — it will be resolved at runtime via DURATION_PRESETS
        # but warn if it's not a known preset name
        known = {"quick", "standard", "extended", "25s", "45s", "3min", "3minute", "custom"}
        v_lower = v.lower().strip()
        if v_lower not in known:
            warnings.warn(
                f"CLIP_DURATION_PRESET='{v}' is not a recognised preset. "
                f"Known presets: {sorted(known)}. The raw CLIP_DURATION value will be used.",
                stacklevel=2,
            )
        return v_lower

    @model_validator(mode="after")
    def resolve_duration_preset(self) -> Settings:
        """Resolve CLIP_DURATION_PRESET to CLIP_DURATION if not explicitly overridden.

        If CLIP_DURATION_PRESET is set to a known preset name (quick/standard/extended/25s/45s/3min),
        CLIP_DURATION is automatically set to the corresponding seconds value.
        This runs AFTER validate_clip_duration_range so the resolved value is validated.
        """
        preset = self.CLIP_DURATION_PRESET.lower().strip()
        if preset in self.DURATION_PRESETS:
            resolved = self.DURATION_PRESETS[preset]
            if resolved != self.CLIP_DURATION:
                self.CLIP_DURATION = resolved
        return self

    @model_validator(mode="after")
    def validate_clip_duration_range(self) -> Settings:
        """Ensure CLIP_DURATION sits between MIN and MAX, and MIN < MAX."""
        if not self.CLIP_MIN_DURATION <= self.CLIP_DURATION <= self.CLIP_MAX_DURATION:
            raise ValueError(
                f"CLIP_DURATION ({self.CLIP_DURATION}) must be between "
                f"CLIP_MIN_DURATION ({self.CLIP_MIN_DURATION}) and "
                f"CLIP_MAX_DURATION ({self.CLIP_MAX_DURATION})"
            )
        if self.CLIP_MIN_DURATION >= self.CLIP_MAX_DURATION:
            raise ValueError(
                f"CLIP_MIN_DURATION ({self.CLIP_MIN_DURATION}) must be less than "
                f"CLIP_MAX_DURATION ({self.CLIP_MAX_DURATION})"
            )
        return self

    @model_validator(mode="after")
    def auto_detect_whisper_device(self) -> Settings:
        """Auto-detect best Whisper device: cuda > mps > cpu."""
        if self.WHISPER_DEVICE == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.WHISPER_DEVICE = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.WHISPER_DEVICE = "mps"
                else:
                    self.WHISPER_DEVICE = "cpu"
            except ImportError:
                self.WHISPER_DEVICE = "cpu"
        return self

    @model_validator(mode="after")
    def auto_detect_compute_type(self) -> Settings:
        """Auto-select compute type based on device (float16 for CUDA, int8 otherwise)."""
        if self.WHISPER_COMPUTE_TYPE == "auto":
            if self.WHISPER_DEVICE == "cuda":
                self.WHISPER_COMPUTE_TYPE = "float16"
            else:
                self.WHISPER_COMPUTE_TYPE = "int8"
        return self

    @model_validator(mode="after")
    def resolve_checkpoint_dir(self) -> Settings:
        """Set default checkpoint directory if not specified."""
        if not self.CHECKPOINT_DIR:
            self.CHECKPOINT_DIR = str(self.LOGS_DIR / "checkpoints")
        return self

    @model_validator(mode="after")
    def validate_hw_accel_encoder(self) -> Settings:
        """Test whether the selected HW-accel encoder actually works.

        If FFMPEG_HW_ACCEL is not 'auto' or 'none', we attempt a quick encode
        test to verify the encoder is functional.  If it fails we fall back to
        'none' and emit a warning.
        """
        if self.FFMPEG_HW_ACCEL in ("auto", "none"):
            return self

        encoder_map: dict[str, str] = {
            "nvenc": "h264_nvenc",
            "videotoolbox": "h264_videotoolbox",
            "vaapi": "h264_vaapi",
            "qsv": "h264_qsv",
            "amf": "h264_amf",
        }
        encoder = encoder_map.get(self.FFMPEG_HW_ACCEL, "")
        if not encoder:
            warnings.warn(
                f"Unknown HW_ACCEL mode '{self.FFMPEG_HW_ACCEL}', falling back to software encoding.",
                stacklevel=2,
            )
            self.FFMPEG_HW_ACCEL = "none"
            return self

        # Check that ffmpeg is available
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            return self

        # Try a quick encode test with the encoder
        try:
            result = subprocess.run(
                [
                    ffmpeg_path, "-hide_banner", "-y",
                    "-f", "lavfi", "-i", "color=black:size=64x64:duration=0.1:rate=1",
                    "-c:v", encoder, "-frames:v", "1",
                    "-f", "null", "-",
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                warnings.warn(
                    f"HW encoder '{encoder}' failed test encode (rc={result.returncode}), "
                    f"falling back to software encoding.",
                    stacklevel=2,
                )
                self.FFMPEG_HW_ACCEL = "none"
            else:
                self.FFMPEG_VIDEO_CODEC = encoder
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            warnings.warn(
                f"HW encoder '{encoder}' test failed: {exc}, falling back to software encoding.",
                stacklevel=2,
            )
            self.FFMPEG_HW_ACCEL = "none"

        return self

    @model_validator(mode="after")
    def validate_subtitle_chars_per_second(self) -> Settings:
        """Ensure min chars/second is less than max chars/second."""
        if self.SUBTITLE_MIN_CHARS_PER_SECOND >= self.SUBTITLE_MAX_CHARS_PER_SECOND:
            raise ValueError(
                f"SUBTITLE_MIN_CHARS_PER_SECOND ({self.SUBTITLE_MIN_CHARS_PER_SECOND}) must be "
                f"less than SUBTITLE_MAX_CHARS_PER_SECOND ({self.SUBTITLE_MAX_CHARS_PER_SECOND})"
            )
        return self

    @model_validator(mode="after")
    def validate_brand_kit_paths(self) -> Settings:
        """Validate brand kit file paths exist if specified."""
        for attr_name, attr_val in [
            ("BRAND_INTRO_CLIP_PATH", self.BRAND_INTRO_CLIP_PATH),
            ("BRAND_OUTRO_CLIP_PATH", self.BRAND_OUTRO_CLIP_PATH),
            ("BRAND_LOWER_THIRD_TEMPLATE", self.BRAND_LOWER_THIRD_TEMPLATE),
        ]:
            if attr_val and not Path(attr_val).exists():
                warnings.warn(
                    f"{attr_name}='{attr_val}' does not exist. Branding will be skipped for this asset.",
                    stacklevel=2,
                )
        return self

    @model_validator(mode="after")
    def validate_performance_temp_dir(self) -> Settings:
        """Ensure temp directory exists if specified."""
        if self.PERFORMANCE_TEMP_DIR:
            temp = Path(self.PERFORMANCE_TEMP_DIR)
            try:
                temp.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                warnings.warn(
                    f"Could not create temp directory '{self.PERFORMANCE_TEMP_DIR}': {exc}",
                    stacklevel=2,
                )
                self.PERFORMANCE_TEMP_DIR = ""
        return self

    @model_validator(mode="after")
    def create_output_directories(self) -> Settings:
        """Create all required output directories on startup."""
        for directory in [
            self.DOWNLOADS_DIR,
            self.SHORTS_DIR,
            self.YOUTUBE_DIR,
            self.TIKTOK_DIR,
            self.REELS_DIR,
            self.METADATA_DIR,
            self.LOGS_DIR,
            self.ASSETS_DIR,
            self.THUMBNAILS_DIR,
            Path(self.CHECKPOINT_DIR),
        ]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                warnings.warn(f"Could not create directory {directory}: {exc}")
        return self

    # ══════════════════════════════════════════════════════
    #  Public Methods
    # ══════════════════════════════════════════════════════

    def resolve_duration(self, preset_or_seconds: str | int | None = None) -> int:
        """Resolve a duration preset name or raw seconds to a valid clip duration.

        Accepts:
        - Preset names: 'quick' (25s), 'standard' (45s), 'extended' (180s)
        - Shorthand names: '25s', '45s', '3min', '3minute'
        - Raw integer seconds: 25, 45, 180
        - None: returns the current CLIP_DURATION

        The resolved value is clamped to [CLIP_MIN_DURATION, CLIP_MAX_DURATION].

        Args:
            preset_or_seconds: Preset name, seconds, or None.

        Returns:
            Resolved clip duration in seconds.
        """
        if preset_or_seconds is None:
            return self.CLIP_DURATION

        if isinstance(preset_or_seconds, int):
            duration = preset_or_seconds
        else:
            key = str(preset_or_seconds).lower().strip()
            if key in self.DURATION_PRESETS:
                duration = self.DURATION_PRESETS[key]
            else:
                # Try parsing as integer string
                try:
                    duration = int(key)
                except ValueError:
                    warnings.warn(
                        f"Unknown duration preset '{preset_or_seconds}', "
                        f"using default {self.CLIP_DURATION}s",
                        stacklevel=2,
                    )
                    return self.CLIP_DURATION

        # Clamp to valid range
        duration = max(self.CLIP_MIN_DURATION, min(duration, self.CLIP_MAX_DURATION))
        return duration

    @property
    def current_preset_label(self) -> str:
        """Return a human-readable label for the current clip duration preset.

        Returns:
            String like 'quick (25s)', 'standard (45s)', 'extended (3min)',
            or 'custom (Xs)'.
        """
        _PRESET_LABELS: dict[str, str] = {
            "quick": "quick (25s)",
            "standard": "standard (45s)",
            "extended": "extended (3min)",
        }
        for name, secs in self.DURATION_PRESETS.items():
            if secs == self.CLIP_DURATION and name in _PRESET_LABELS:
                return _PRESET_LABELS[name]
        return f"custom ({self.CLIP_DURATION}s)"

    def apply_turbo(self) -> None:
        """Apply turbo mode overrides for maximum processing speed.

        Reconfigures all settings for the fastest possible pipeline execution:

        - FFmpeg: ultrafast preset, CRF 30, no two-pass
        - Whisper: tiny model (fastest), beam_size=1
        - Analysis: skip spectral centroid, motion energy, emphasis detection
        - Audio: skip noise reduction, compression, de-esser, EQ
        - Face tracking: disabled (use center crop)
        - Subtitles: no animation (simplest rendering)
        - Scene detection: higher threshold (fewer scene changes to process)
        - Energy sampling: wider interval (4s instead of 2s)
        - Thumbnails: minimal (1 thumbnail only)
        - Content moderation: disabled
        - Export: skip two-pass encoding, no denoise, no sharpen, no film grain
        - Speed: skip analysis extras

        These trade quality for speed. The output is still good enough for
        social media, but fine details and accuracy are reduced.
        """
        # FFmpeg — fastest possible encoding
        self.FFMPEG_PRESET = "ultrafast"
        self.FFMPEG_CRF = 30
        self.FFMPEG_TIMEOUT = 300
        self.EXPORT_TWO_PASS_ENCODING = False

        # Whisper — tiny model for speed
        self.WHISPER_MODEL = "tiny"
        self.WHISPER_BEAM_SIZE = 1

        # Analysis — skip expensive signals
        self.ENERGY_SAMPLE_INTERVAL = 4.0  # Wider = fewer FFmpeg calls
        self.SCENE_DETECT_THRESHOLD = 0.5   # Higher = fewer scene changes
        self.ADVANCED_SPECTRAL_ANALYSIS = False
        self.ADVANCED_PITCH_DETECTION = False
        self.ADVANCED_SPEECH_RATE_WPM = False
        self.ADVANCED_EMPHASIS_DETECTION = False

        # Audio — skip enhancement chain
        self.AUDIO_NOISE_REDUCTION = False
        self.AUDIO_COMPRESSION = False
        self.AUDIO_DEESSER = False
        self.AUDIO_EQ = False
        self.AUDIO_NORMALIZER = True  # Keep normalizer — it's fast and important

        # Face tracking — disable (use center crop)
        self.FACE_TRACKING_MODEL = "haar"  # Simplest model

        # Subtitles — simplest animation
        self.SUBTITLE_ANIMATION = "none"

        # Thumbnails — minimal
        self.THUMBNAIL_COUNT = 1

        # Content moderation — skip
        self.CONTENT_MODERATION_ENABLED = False

        # Export — skip quality enhancements
        self.EXPORT_DENOISE_VIDEO = False
        self.EXPORT_SHARPEN = False
        self.EXPORT_FILM_GRAIN_AMOUNT = 0

        # Speed — skip analysis extras
        self.SPEED_SKIP_ANALYSIS_EXTRAS = True

        logger_turbo = __import__("utils.logger", fromlist=["get_logger"]).get_logger("settings")
        logger_turbo.info("TURBO MODE activated — maximum speed, reduced quality")

    def apply_superfast(self) -> None:
        """Apply superfast mode overrides for absolute maximum speed.

        Even faster than turbo: combines all FFmpeg operations into a single pass,
        uses center crop (no face tracking), minimal analysis, and the fastest
        possible encoding settings.

        Speed comparison (typical 10-min video → 45s short):
        - Standard pipeline: ~8-15 min (5-6 FFmpeg passes)
        - Turbo mode: ~4-8 min (same passes, faster settings)
        - Superfast mode: ~2-4 min (1 FFmpeg pass, minimal analysis)

        Key differences from turbo:
        - Single FFmpeg pass (crop + scale + subs + logo + audio in one command)
        - Center crop only (no face tracking at all)
        - Skip letterbox detection
        - Skip motion energy computation
        - Skip spectral centroid computation
        - Wider energy sample interval (8s)
        - Download at 720p only
        - Fade subtitle animation (fastest that still looks good)
        """
        # Apply turbo settings first as base
        self.apply_turbo()

        # Override with even more aggressive settings
        self.SUPERFAST_MODE = True
        self.FFMPEG_PRESET = "ultrafast"
        self.FFMPEG_CRF = 28  # Slightly better than turbo's 30
        self.WHISPER_MODEL = "tiny"
        self.WHISPER_BEAM_SIZE = 1

        # Skip ALL expensive analysis
        self.ENERGY_SAMPLE_INTERVAL = 8.0  # Very wide = minimal FFmpeg calls
        self.SCENE_DETECT_THRESHOLD = 0.8   # Almost skip scene detection
        self.ADVANCED_SPECTRAL_ANALYSIS = False
        self.ADVANCED_PITCH_DETECTION = False
        self.ADVANCED_SPEECH_RATE_WPM = False
        self.ADVANCED_EMPHASIS_DETECTION = False
        self.SPEED_SKIP_ANALYSIS_EXTRAS = True
        self.SPEED_FAST_SMART_CROP = True

        # Face tracking — completely disabled
        self.FACE_TRACKING_MODEL = "haar"

        # Subtitles — fastest animation
        self.SUBTITLE_ANIMATION = "fade"  # Looks good, very fast to render

        # Thumbnails — skip entirely in superfast
        self.THUMBNAIL_COUNT = 0

        # Export — absolute minimum
        self.EXPORT_TWO_PASS_ENCODING = False
        self.EXPORT_DENOISE_VIDEO = False
        self.EXPORT_SHARPEN = False
        self.EXPORT_FILM_GRAIN_AMOUNT = 0

        logger_sf = __import__("utils.logger", fromlist=["get_logger"]).get_logger("settings")
        logger_sf.info("SUPERFAST MODE activated — single-pass FFmpeg, center crop, minimal analysis")

    @property
    def turbo_label(self) -> str:
        """Return a label indicating speed mode status."""
        if self.SUPERFAST_MODE:
            return "SUPERFAST"
        return "TURBO" if self.TURBO_MODE else "NORMAL"

    def validate_all(self) -> list[str]:
        """Run all validation checks and return a list of warning messages.

        This method re-validates all business rules and also checks for
        common configuration issues like missing tools, insufficient disk
        space, and incompatible option combinations.

        Returns:
            List of warning strings. Empty list means all checks passed.
        """
        warnings_list: list[str] = []

        # Check ffmpeg/ffprobe availability
        for tool in ("ffmpeg", "ffprobe"):
            if not shutil.which(tool):
                warnings_list.append(f"{tool} is not installed or not on PATH")

        # Check disk space on the output drive
        try:
            disk_usage = shutil.disk_usage(self.SHORTS_DIR.parent)
            free_mb = disk_usage.free / (1024 * 1024)
            if free_mb < self.PERFORMANCE_DISK_SPACE_RESERVE_MB:
                warnings_list.append(
                    f"Low disk space: {free_mb:.0f} MB free, "
                    f"{self.PERFORMANCE_DISK_SPACE_RESERVE_MB} MB reserved"
                )
        except OSError:
            warnings_list.append("Could not check disk space")

        # Check logo file (resolve relative to BASE_DIR, same as logo_stamper._resolve_logo_path)
        logo = Path(self.LOGO_PATH)
        if not logo.is_absolute():
            logo = BASE_DIR / self.LOGO_PATH
        if not logo.exists():
            warnings_list.append(f"Logo file not found: {logo}")

        # Check brand kit assets
        if self.BRAND_INTRO_CLIP_PATH and not Path(self.BRAND_INTRO_CLIP_PATH).exists():
            warnings_list.append(f"Intro clip not found: {self.BRAND_INTRO_CLIP_PATH}")
        if self.BRAND_OUTRO_CLIP_PATH and not Path(self.BRAND_OUTRO_CLIP_PATH).exists():
            warnings_list.append(f"Outro clip not found: {self.BRAND_OUTRO_CLIP_PATH}")

        # Check two-pass with HW accel
        if self.EXPORT_TWO_PASS_ENCODING and self.FFMPEG_HW_ACCEL not in ("none", "auto"):
            warnings_list.append(
                "Two-pass encoding with hardware acceleration may not be supported "
                "by all encoders"
            )

        # Check YouTube API key if API is enabled
        if self.YOUTUBE_API_ENABLED and not self.YOUTUBE_API_KEY:
            warnings_list.append("YouTube API is enabled but YOUTUBE_API_KEY is empty")

        # Check face tracking with mediapipe
        if self.FACE_TRACKING_MODEL == "mediapipe":
            try:
                import mediapipe  # noqa: F401
            except ImportError:
                warnings_list.append(
                    "FACE_TRACKING_MODEL='mediapipe' but mediapipe is not installed"
                )

        # Check subtitle min/max display time consistency
        if self.SUBTITLE_MIN_DISPLAY_TIME > 1.0:
            warnings_list.append(
                f"SUBTITLE_MIN_DISPLAY_TIME={self.SUBTITLE_MIN_DISPLAY_TIME}s is very long"
            )

        # Check film grain with HW accel
        if self.EXPORT_FILM_GRAIN_AMOUNT > 0 and self.FFMPEG_HW_ACCEL not in ("none", "auto"):
            warnings_list.append(
                "Film grain with hardware acceleration may cause encoding issues"
            )

        return warnings_list

    def export_to_dict(self) -> dict[str, Any]:
        """Export all settings to a plain dictionary.

        Path objects are converted to strings for JSON serialisation.
        Useful for config snapshots, debugging, or passing to external tools.

        Returns:
            Dictionary of all setting names to their current values.
        """
        result: dict[str, Any] = {}
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Path):
                result[field_name] = str(value)
            else:
                result[field_name] = value
        return result

    def import_from_dict(self, data: dict[str, Any]) -> None:
        """Import settings from a dictionary, with validation.

        Only fields that exist in the Settings model are applied.
        Path values given as strings are converted to Path objects.
        After import, all validators are re-run.

        Args:
            data: Dictionary of setting names to values.

        Raises:
            ValueError: If any value fails validation.
        """
        for key, value in data.items():
            if key not in self.model_fields:
                continue
            field_info = self.model_fields[key]
            if field_info.annotation is Path or (
                hasattr(field_info.annotation, "__origin__")
                and field_info.annotation.__origin__ is Path
            ):
                if isinstance(value, str):
                    value = Path(value)
            setattr(self, key, value)

    # ══════════════════════════════════════════════════════
    #  Properties
    # ══════════════════════════════════════════════════════

    @property
    def platform_info(self) -> dict[str, str]:
        """Return a dict of platform/system info for logging."""
        return {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": f"{platform.system()} {platform.release()}",
            "whisper_device": self.WHISPER_DEVICE,
            "whisper_compute": self.WHISPER_COMPUTE_TYPE,
            "whisper_model": self.WHISPER_MODEL,
            "ffmpeg_hw_accel": self.FFMPEG_HW_ACCEL,
            "ffmpeg_video_codec": self.FFMPEG_VIDEO_CODEC,
            "duration_preset": self.current_preset_label,
        }

    @property
    def target_aspect_ratio(self) -> float:
        """Return the target aspect ratio (9:16 = 0.5625)."""
        return self.OUTPUT_WIDTH / self.OUTPUT_HEIGHT

    @property
    def temp_directory(self) -> Path:
        """Return the temp directory path, falling back to system temp."""
        if self.PERFORMANCE_TEMP_DIR:
            return Path(self.PERFORMANCE_TEMP_DIR)
        import tempfile
        return Path(tempfile.gettempdir()) / "yt-shorts-factory"

    @property
    def thumbnail_resolution_tuple(self) -> tuple[int, int]:
        """Parse THUMBNAIL_RESOLUTION into (width, height) tuple."""
        parts = self.THUMBNAIL_RESOLUTION.lower().split("x")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return 1280, 720

    @property
    def content_moderation_categories_list(self) -> list[str]:
        """Parse CONTENT_MODERATION_CATEGORIES into a list."""
        if not self.CONTENT_MODERATION_CATEGORIES:
            return []
        return [c.strip() for c in self.CONTENT_MODERATION_CATEGORIES.split(",") if c.strip()]

    @property
    def brand_color_palette_list(self) -> list[str]:
        """Parse BRAND_COLOR_PALETTE into a list of color values."""
        if not self.BRAND_COLOR_PALETTE:
            return []
        return [c.strip() for c in self.BRAND_COLOR_PALETTE.split(",") if c.strip()]

    @property
    def whisper_fallback_models_list(self) -> list[str]:
        """Parse WHISPER_FALLBACK_MODELS into a list."""
        return [m.strip() for m in self.WHISPER_FALLBACK_MODELS.split(",") if m.strip()]


# ══════════════════════════════════════════════════════════
#  Singleton Access
# ══════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    Uses lru_cache so the .env file is only read once and validators
    only run once per process lifetime.
    """
    return Settings()


def reset_settings() -> None:
    """Clear the cached settings (useful for testing or config reload)."""
    get_settings.cache_clear()
