"""
tests/test_logo_stamper.py — Tests for the logo stamper module.
Tests logo positioning, subtitle zone avoidance, animation presets,
and stamp_logo with missing logo graceful handling.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logo_stamper import (
    BrandKit,
    LogoAnimation,
    _PLATFORM_SAFE_ZONES,
    _calculate_logo_position,
    _avoid_subtitle_zone,
    _get_animation_preset,
    _build_overlay_expression,
    load_brand_kit,
    stamp_logo,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def youtube_safe_zone() -> dict[str, int]:
    """YouTube safe zone values."""
    return _PLATFORM_SAFE_ZONES["youtube"]


@pytest.fixture
def tiktok_safe_zone() -> dict[str, int]:
    """TikTok safe zone values."""
    return _PLATFORM_SAFE_ZONES["tiktok"]


@pytest.fixture
def default_safe_zone() -> dict[str, int]:
    """Default safe zone (YouTube)."""
    return _PLATFORM_SAFE_ZONES["youtube"]


@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock settings for logo stamper."""
    settings = MagicMock()
    settings.OUTPUT_WIDTH = 1080
    settings.OUTPUT_HEIGHT = 1920
    settings.LOGO_MARGIN = 20
    settings.LOGO_OPACITY = 0.85
    settings.LOGO_SCALE = 0.12
    settings.LOGO_POSITION = "top-right"
    settings.LOGO_FADE_DURATION = 1.0
    settings.FFMPEG_VIDEO_CODEC = "libx264"
    settings.FFMPEG_PRESET = "medium"
    settings.FFMPEG_CRF = 23
    settings.FFMPEG_THREADS = 4
    settings.FFMPEG_TIMEOUT = 300
    settings.SUBTITLE_FONT = "Arial"
    settings.SUBTITLE_MARGIN_V = 30
    return settings


# ── Test _calculate_logo_position ─────────────────────────────

class TestCalculateLogoPosition:
    """Tests for the _calculate_logo_position function."""

    def test_top_left_position(self, default_safe_zone: dict) -> None:
        """Top-left position should place logo near top-left with safe zone."""
        x, y = _calculate_logo_position(
            "top-left", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        assert x == default_safe_zone["left"] + 20
        assert y == default_safe_zone["top"] + 20

    def test_top_right_position(self, default_safe_zone: dict) -> None:
        """Top-right position should place logo near top-right with safe zone."""
        x, y = _calculate_logo_position(
            "top-right", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        assert x == 1080 - 100 - default_safe_zone["right"] - 20
        assert y == default_safe_zone["top"] + 20

    def test_bottom_left_position(self, default_safe_zone: dict) -> None:
        """Bottom-left position should place logo near bottom-left."""
        x, y = _calculate_logo_position(
            "bottom-left", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        assert x == default_safe_zone["left"] + 20
        assert y == 1920 - 50 - default_safe_zone["bottom"] - 20

    def test_bottom_right_position(self, default_safe_zone: dict) -> None:
        """Bottom-right position should place logo near bottom-right."""
        x, y = _calculate_logo_position(
            "bottom-right", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        assert x == 1080 - 100 - default_safe_zone["right"] - 20
        assert y == 1920 - 50 - default_safe_zone["bottom"] - 20

    def test_center_position(self, default_safe_zone: dict) -> None:
        """Center position should place logo in the center of the video."""
        x, y = _calculate_logo_position(
            "center", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        assert x == (1080 - 100) // 2
        assert y == (1920 - 50) // 2

    def test_center_top_position(self, default_safe_zone: dict) -> None:
        """Center-top should be horizontally centered, near the top."""
        x, y = _calculate_logo_position(
            "center-top", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        assert x == (1080 - 100) // 2
        assert y == default_safe_zone["top"] + 20

    def test_center_bottom_position(self, default_safe_zone: dict) -> None:
        """Center-bottom should be horizontally centered, near the bottom."""
        x, y = _calculate_logo_position(
            "center-bottom", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        assert x == (1080 - 100) // 2
        assert y == 1920 - 50 - default_safe_zone["bottom"] - 20

    def test_unknown_position_defaults_to_top_right(self, default_safe_zone: dict) -> None:
        """Unknown position names should default to top-right."""
        x, y = _calculate_logo_position(
            "nonexistent", 100, 50, 1080, 1920, default_safe_zone, 20,
        )
        # Should be same as top-right
        expected_x = 1080 - 100 - default_safe_zone["right"] - 20
        expected_y = default_safe_zone["top"] + 20
        assert x == expected_x
        assert y == expected_y

    def test_position_is_non_negative(self, default_safe_zone: dict) -> None:
        """Logo position should never be negative."""
        # Use very large logo that could theoretically push position negative
        x, y = _calculate_logo_position(
            "top-right", 2000, 2000, 1080, 1920, default_safe_zone, 20,
        )
        assert x >= 0
        assert y >= 0

    def test_different_margins(self, default_safe_zone: dict) -> None:
        """Different margin values should affect position correctly."""
        x_small, y_small = _calculate_logo_position(
            "top-left", 100, 50, 1080, 1920, default_safe_zone, 10,
        )
        x_large, y_large = _calculate_logo_position(
            "top-left", 100, 50, 1080, 1920, default_safe_zone, 50,
        )
        # Larger margin should push the logo further from the edge
        assert x_large > x_small
        assert y_large > y_small

    def test_different_platforms_affect_position(self) -> None:
        """Different platforms have different safe zones affecting position."""
        yt_zone = _PLATFORM_SAFE_ZONES["youtube"]
        tt_zone = _PLATFORM_SAFE_ZONES["tiktok"]
        x_yt, y_yt = _calculate_logo_position(
            "top-left", 100, 50, 1080, 1920, yt_zone, 20,
        )
        x_tt, y_tt = _calculate_logo_position(
            "top-left", 100, 50, 1080, 1920, tt_zone, 20,
        )
        # Different safe zones should produce different positions
        assert (x_yt, y_yt) != (x_tt, y_tt)


# ── Test _avoid_subtitle_zone ─────────────────────────────────

class TestAvoidSubtitleZone:
    """Tests for the _avoid_subtitle_zone function."""

    def test_top_position_unchanged(self, youtube_safe_zone: dict) -> None:
        """Top positions should not be changed by subtitle avoidance."""
        result = _avoid_subtitle_zone("top-left", youtube_safe_zone, 1920, 50, 20)
        assert result == "top-left"

    def test_bottom_left_moved_to_top_left(self, youtube_safe_zone: dict) -> None:
        """Bottom-left in subtitle zone should be moved to top-left."""
        # Use a large logo that would overlap with subtitle zone
        result = _avoid_subtitle_zone(
            "bottom-left", youtube_safe_zone, 1920, 100, 20,
        )
        assert result == "top-left"

    def test_bottom_right_moved_to_top_right(self, youtube_safe_zone: dict) -> None:
        """Bottom-right in subtitle zone should be moved to top-right."""
        result = _avoid_subtitle_zone(
            "bottom-right", youtube_safe_zone, 1920, 100, 20,
        )
        assert result == "top-right"

    def test_bottom_position_unchanged_when_no_overlap(self) -> None:
        """Bottom positions should stay if no subtitle zone overlap."""
        # Create a safe zone where subtitle zone is very small (near bottom)
        # and the logo can fit above it.
        # Condition: out_h - safe_zone["bottom"] - margin - logo_h >= out_h - subtitle_zone_bottom
        # => safe_zone["bottom"] + margin + logo_h <= subtitle_zone_bottom
        # With bottom=10, margin=5, logo_h=20 => 35 <= 50 ✓
        safe_zone = {"top": 48, "bottom": 10, "left": 24, "right": 24,
                     "subtitle_zone_top": 0, "subtitle_zone_bottom": 50}
        result = _avoid_subtitle_zone(
            "bottom-left", safe_zone, 2000, 20, 5,
        )
        assert result == "bottom-left"

    def test_center_position_unchanged(self, youtube_safe_zone: dict) -> None:
        """Center position should not be affected by subtitle avoidance."""
        result = _avoid_subtitle_zone("center", youtube_safe_zone, 1920, 50, 20)
        assert result == "center"


# ── Test _get_animation_preset ────────────────────────────────

class TestGetAnimationPreset:
    """Tests for the _get_animation_preset function."""

    def test_fade_in_preset(self) -> None:
        """fade_in preset should return a fade filter expression."""
        result = _get_animation_preset("fade_in", fade_duration=1.0)
        assert "fade" in result
        assert "alpha=1" in result

    def test_slide_in_left_preset(self) -> None:
        """slide_in_left should return a fade filter expression."""
        result = _get_animation_preset("slide_in_left", fade_duration=1.0)
        assert "fade" in result

    def test_slide_in_right_preset(self) -> None:
        """slide_in_right should return a fade filter expression."""
        result = _get_animation_preset("slide_in_right", fade_duration=1.0)
        assert "fade" in result

    def test_slide_in_top_preset(self) -> None:
        """slide_in_top should return a fade filter expression."""
        result = _get_animation_preset("slide_in_top", fade_duration=1.0)
        assert "fade" in result

    def test_slide_in_bottom_preset(self) -> None:
        """slide_in_bottom should return a fade filter expression."""
        result = _get_animation_preset("slide_in_bottom", fade_duration=1.0)
        assert "fade" in result

    def test_scale_up_preset(self) -> None:
        """scale_up preset should contain zoompan and scale filters."""
        result = _get_animation_preset("scale_up", fade_duration=1.0, video_duration=60.0)
        assert "zoompan" in result or "scale" in result

    def test_bounce_in_preset(self) -> None:
        """bounce_in preset should return a fade filter expression."""
        result = _get_animation_preset("bounce_in", fade_duration=1.0)
        assert "fade" in result

    def test_pulse_preset(self) -> None:
        """pulse preset should return a fade filter expression."""
        result = _get_animation_preset("pulse", fade_duration=1.0)
        assert "fade" in result

    def test_breathe_preset(self) -> None:
        """breathe preset should return a fade filter expression."""
        result = _get_animation_preset("breathe", fade_duration=1.0)
        assert "fade" in result

    def test_static_preset(self) -> None:
        """static preset should return 'null' (no animation)."""
        result = _get_animation_preset("static")
        assert result == "null"

    def test_unknown_preset_defaults_to_fade_in(self) -> None:
        """Unknown animation name should default to fade_in behavior."""
        result = _get_animation_preset("nonexistent_animation", fade_duration=1.0)
        assert "fade" in result

    def test_custom_fade_duration(self) -> None:
        """Custom fade duration should be reflected in the filter."""
        result = _get_animation_preset("fade_in", fade_duration=2.5)
        assert "d=2.5" in result

    def test_all_presets_return_strings(self) -> None:
        """All preset names should return a string."""
        for preset in [
            "fade_in", "slide_in_left", "slide_in_right",
            "slide_in_top", "slide_in_bottom", "scale_up",
            "bounce_in", "pulse", "breathe", "static",
        ]:
            result = _get_animation_preset(preset)
            assert isinstance(result, str), f"Preset '{preset}' did not return a string"


# ── Test _build_overlay_expression ────────────────────────────

class TestBuildOverlayExpression:
    """Tests for the _build_overlay_expression function."""

    def test_slide_in_left_x_expression(self) -> None:
        """slide_in_left should have a time-dependent x expression."""
        x_expr, y_expr = _build_overlay_expression(
            "slide_in_left", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert "if" in x_expr or "lt" in x_expr
        assert y_expr == "200"

    def test_slide_in_right_x_expression(self) -> None:
        """slide_in_right should have a time-dependent x expression."""
        x_expr, y_expr = _build_overlay_expression(
            "slide_in_right", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert "if" in x_expr or "lt" in x_expr

    def test_slide_in_top_y_expression(self) -> None:
        """slide_in_top should have a time-dependent y expression."""
        x_expr, y_expr = _build_overlay_expression(
            "slide_in_top", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert x_expr == "500"
        assert "if" in y_expr or "lt" in y_expr

    def test_slide_in_bottom_y_expression(self) -> None:
        """slide_in_bottom should have a time-dependent y expression."""
        x_expr, y_expr = _build_overlay_expression(
            "slide_in_bottom", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert x_expr == "500"
        assert "if" in y_expr or "lt" in y_expr

    def test_pulse_has_sin_expression(self) -> None:
        """pulse preset should use sin/cos for oscillation."""
        x_expr, y_expr = _build_overlay_expression(
            "pulse", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert "sin" in x_expr or "cos" in x_expr

    def test_bounce_in_has_exp_sin(self) -> None:
        """bounce_in should have exponential decay with sine."""
        x_expr, y_expr = _build_overlay_expression(
            "bounce_in", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert "exp" in y_expr or "sin" in y_expr

    def test_static_uses_final_position(self) -> None:
        """Static/fade_in should just use the final position."""
        x_expr, y_expr = _build_overlay_expression(
            "fade_in", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert x_expr == "500"
        assert y_expr == "200"

    def test_breathe_uses_final_position(self) -> None:
        """breathe should use final position (scaling is separate)."""
        x_expr, y_expr = _build_overlay_expression(
            "breathe", 500, 200, 100, 50, 60.0, 1.0,
        )
        assert x_expr == "500"
        assert y_expr == "200"


# ── Test stamp_logo with missing logo ─────────────────────────

class TestStampLogoMissingLogo:
    """Tests for stamp_logo when the logo file is missing."""

    @patch("core.logo_stamper._resolve_logo_path")
    def test_missing_logo_returns_input_when_no_output(
        self, mock_resolve: MagicMock, mock_settings: MagicMock,
    ) -> None:
        """When logo is missing and no output_path, return input_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            input_path.write_text("fake video data")
            mock_resolve.return_value = Path("/nonexistent/logo.png")

            result = stamp_logo(input_path, settings=mock_settings)
            assert result == input_path

    @patch("core.logo_stamper._resolve_logo_path")
    @patch("core.logo_stamper.run_ffmpeg")
    def test_missing_logo_copies_video_when_output_specified(
        self, mock_ffmpeg: MagicMock, mock_resolve: MagicMock, mock_settings: MagicMock,
    ) -> None:
        """When logo is missing and output_path is specified, copy video without logo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output.mp4"
            input_path.write_text("fake video data")
            mock_resolve.return_value = Path("/nonexistent/logo.png")

            # Simulate FFmpeg creating the output file
            def fake_ffmpeg(cmd, **kwargs):
                output_path.write_text("copied video data")
            mock_ffmpeg.side_effect = fake_ffmpeg

            result = stamp_logo(input_path, output_path=output_path, settings=mock_settings)
            assert result == output_path
            # Should have called FFmpeg with -c copy (no re-encode)
            call_args = mock_ffmpeg.call_args
            cmd = call_args[0][0]
            assert "-c" in cmd
            assert "copy" in cmd

    @patch("core.logo_stamper._resolve_logo_path")
    @patch("core.logo_stamper.run_ffmpeg")
    def test_missing_logo_with_brand_kit(
        self, mock_ffmpeg: MagicMock, mock_resolve: MagicMock, mock_settings: MagicMock,
    ) -> None:
        """When brand_kit has missing logo, should handle gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            input_path.write_text("fake video data")
            brand_kit = BrandKit(logo_path="/nonexistent/brand_logo.png")

            # _resolve_logo_path is only used when no brand_kit logo_path
            # But we need to handle the case where brand_kit.logo_path doesn't exist

            def fake_ffmpeg(cmd, **kwargs):
                pass
            mock_ffmpeg.side_effect = fake_ffmpeg

            result = stamp_logo(input_path, brand_kit=brand_kit, settings=mock_settings)
            # Should return input since logo is missing and no output_path
            assert result == input_path


# ── Test BrandKit ─────────────────────────────────────────────

class TestBrandKit:
    """Tests for the BrandKit dataclass."""

    def test_default_values(self) -> None:
        """Default BrandKit should have safe empty values."""
        kit = BrandKit()
        assert kit.logo_path == ""
        assert kit.logo_position == "top-right"
        assert kit.logo_opacity == 0.85
        assert kit.logo_scale == 0.12
        assert kit.logo_animation == "fade_in"
        assert kit.channel_name == ""

    def test_custom_values(self) -> None:
        """BrandKit should accept custom values."""
        kit = BrandKit(
            logo_path="/path/to/logo.png",
            logo_position="bottom-left",
            logo_opacity=0.7,
            logo_scale=0.15,
            logo_animation="bounce_in",
            channel_name="TestChannel",
        )
        assert kit.logo_path == "/path/to/logo.png"
        assert kit.logo_position == "bottom-left"
        assert kit.logo_opacity == 0.7
        assert kit.channel_name == "TestChannel"


# ── Test load_brand_kit ───────────────────────────────────────

class TestLoadBrandKit:
    """Tests for the load_brand_kit function."""

    def test_load_valid_json(self) -> None:
        """A valid JSON brand kit file should be loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kit_path = Path(tmpdir) / "brand_kit.json"
            kit_data = {
                "logo_path": "assets/logo.png",
                "logo_position": "bottom-right",
                "logo_opacity": 0.9,
                "logo_scale": 0.10,
                "logo_animation": "pulse",
                "channel_name": "MyChannel",
                "channel_colors": ["#FF0000", "#00FF00"],
            }
            kit_path.write_text(json.dumps(kit_data), encoding="utf-8")

            result = load_brand_kit(kit_path)
            assert result.logo_path == "assets/logo.png"
            assert result.logo_position == "bottom-right"
            assert result.logo_opacity == 0.9
            assert result.channel_name == "MyChannel"
            assert len(result.channel_colors) == 2

    def test_load_missing_file_raises_error(self) -> None:
        """A missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Brand kit file not found"):
            load_brand_kit(Path("/nonexistent/brand_kit.json"))

    def test_load_invalid_json_raises_error(self) -> None:
        """Invalid JSON should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kit_path = Path(tmpdir) / "bad_kit.json"
            kit_path.write_text("this is not json {{{", encoding="utf-8")

            with pytest.raises(ValueError, match="Invalid brand kit JSON"):
                load_brand_kit(kit_path)

    def test_load_partial_json_uses_defaults(self) -> None:
        """A partial JSON should use defaults for missing fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kit_path = Path(tmpdir) / "partial_kit.json"
            kit_path.write_text(
                json.dumps({"logo_path": "my_logo.png"}),
                encoding="utf-8",
            )

            result = load_brand_kit(kit_path)
            assert result.logo_path == "my_logo.png"
            # Other fields should use defaults
            assert result.logo_position == "top-right"
            assert result.logo_opacity == 0.85

    def test_load_empty_json(self) -> None:
        """An empty JSON object should use all defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kit_path = Path(tmpdir) / "empty_kit.json"
            kit_path.write_text("{}", encoding="utf-8")

            result = load_brand_kit(kit_path)
            assert result.logo_path == ""
            assert result.logo_position == "top-right"


# ── Test LogoAnimation Dataclass ──────────────────────────────

class TestLogoAnimation:
    """Tests for the LogoAnimation dataclass."""

    def test_logo_animation_fields(self) -> None:
        """LogoAnimation should have name and filter_expression fields."""
        anim = LogoAnimation(name="fade_in", filter_expression="fade=t=in:st=0:d=1:alpha=1")
        assert anim.name == "fade_in"
        assert "fade" in anim.filter_expression


# ── Test _PLATFORM_SAFE_ZONES ─────────────────────────────────

class TestPlatformSafeZones:
    """Tests for the platform safe zones configuration."""

    def test_all_platforms_have_required_keys(self) -> None:
        """Each platform safe zone should have all required keys."""
        required_keys = {"top", "bottom", "left", "right", "subtitle_zone_top", "subtitle_zone_bottom"}
        for platform, zone in _PLATFORM_SAFE_ZONES.items():
            assert required_keys.issubset(set(zone.keys())), f"{platform} missing keys"

    def test_all_zones_have_positive_values(self) -> None:
        """All safe zone values should be non-negative."""
        for platform, zone in _PLATFORM_SAFE_ZONES.items():
            for key, value in zone.items():
                assert value >= 0, f"{platform}.{key} is negative: {value}"

    def test_supported_platforms(self) -> None:
        """Expected platforms should exist in the safe zones dict."""
        expected_platforms = {"youtube", "tiktok", "reels", "twitter", "facebook", "snapchat"}
        assert expected_platforms.issubset(set(_PLATFORM_SAFE_ZONES.keys()))
