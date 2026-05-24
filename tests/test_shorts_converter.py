"""
tests/test_shorts_converter.py — Tests for the shorts converter module.
Tests letterbox detection, face tracking, crop smoothing,
filtergraph building, and conversion with mocked FFmpeg.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.shorts_converter import (
    ConverterError,
    PlatformSpec,
    PLATFORM_SPECS,
    CropKeyframe,
    detect_letterbox,
    _track_faces_across_clip,
    _smooth_crop_keyframes,
    _build_crop_filtergraph,
    _build_blur_bg_filtergraph,
    _build_split_screen_filtergraph,
    convert_to_shorts,
)
from core.analyzer import SegmentResult
from config.settings import get_settings


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def mock_settings() -> MagicMock:
    """Mock settings for shorts conversion."""
    settings = MagicMock()
    settings.FFMPEG_PIXEL_FORMAT = "yuv420p"
    settings.FFMPEG_VIDEO_CODEC = "libx264"
    settings.FFMPEG_PRESET = "medium"
    settings.FFMPEG_CRF = 23
    settings.FFMPEG_AUDIO_CODEC = "aac"
    settings.FFMPEG_AUDIO_BITRATE = "128k"
    settings.FFMPEG_THREADS = 4
    settings.FFMPEG_TIMEOUT = 600
    settings.OUTPUT_WIDTH = 1080
    settings.OUTPUT_HEIGHT = 1920
    return settings


@pytest.fixture
def segment() -> SegmentResult:
    """A sample segment for testing."""
    return SegmentResult(
        start_time=10.0,
        end_time=70.0,
        energy_score=0.8,
    )


@pytest.fixture
def mock_video_info() -> MagicMock:
    """Mock video probe result."""
    info = MagicMock()
    info.width = 1920
    info.height = 1080
    info.duration = 120.0
    info.fps = 30.0
    info.has_audio = True
    info.codec = "h264"
    info.bitrate = 5000000
    info.size_bytes = 1000000
    return info


# ── Test detect_letterbox ─────────────────────────────────────

class TestDetectLetterbox:
    """Tests for the detect_letterbox function."""

    def test_detect_letterbox_returns_tuple_of_four(self) -> None:
        """detect_letterbox should return a 4-tuple of (top, bottom, left, right)."""
        # Even on failure, should return (0, 0, 0, 0)
        result = detect_letterbox(Path("nonexistent.mp4"))
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_detect_letterbox_returns_zeros_on_failure(self) -> None:
        """detect_letterbox should return (0,0,0,0) when detection fails."""
        result = detect_letterbox(Path("nonexistent.mp4"))
        assert result == (0, 0, 0, 0)

    def test_detect_letterbox_accepts_sample_time(self) -> None:
        """detect_letterbox should accept a sample_time parameter."""
        result = detect_letterbox(Path("nonexistent.mp4"), sample_time=5.0)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_detect_letterbox_accepts_threshold(self) -> None:
        """detect_letterbox should accept a threshold parameter."""
        result = detect_letterbox(Path("nonexistent.mp4"), threshold=32)
        assert isinstance(result, tuple)
        assert len(result) == 4


# ── Test _track_faces_across_clip ─────────────────────────────

class TestTrackFacesAcrossClip:
    """Tests for the _track_faces_across_clip function."""

    @patch("core.shorts_converter.detect_face_crop_region")
    def test_returns_crop_keyframes(self, mock_detect_face) -> None:
        """Should return a list of CropKeyframe objects."""
        mock_detect_face.return_value = (500, 200)
        keyframes = _track_faces_across_clip(
            Path("fake.mp4"), 0.0, 60.0, 608, 1080, 1920, 1080, sample_count=3,
        )
        assert isinstance(keyframes, list)
        for kf in keyframes:
            assert isinstance(kf, CropKeyframe)

    @patch("core.shorts_converter.detect_face_crop_region")
    def test_falls_back_to_center_crop(self, mock_detect_face) -> None:
        """When no faces detected, should fall back to center crop."""
        mock_detect_face.return_value = (-1, -1)  # No face
        keyframes = _track_faces_across_clip(
            Path("fake.mp4"), 0.0, 60.0, 608, 1080, 1920, 1080, sample_count=3,
        )
        assert len(keyframes) > 0
        # Should be center crop
        center_x = (1920 - 608) // 2
        center_y = (1080 - 1080) // 2
        assert keyframes[0].crop_x == center_x
        assert keyframes[0].crop_y == center_y

    @patch("core.shorts_converter.detect_face_crop_region")
    def test_exception_in_face_detection_falls_back(self, mock_detect_face) -> None:
        """Exceptions in face detection should not crash; fall back to center."""
        mock_detect_face.side_effect = Exception("Face detection failed")
        keyframes = _track_faces_across_clip(
            Path("fake.mp4"), 0.0, 60.0, 608, 1080, 1920, 1080, sample_count=3,
        )
        assert len(keyframes) > 0
        # Should fall back to center
        center_x = (1920 - 608) // 2
        assert keyframes[0].crop_x == center_x

    @patch("core.shorts_converter.detect_face_crop_region")
    def test_keyframes_at_start_and_end(self, mock_detect_face) -> None:
        """Should ensure keyframes exist at start and end of clip."""
        mock_detect_face.return_value = (500, 200)
        keyframes = _track_faces_across_clip(
            Path("fake.mp4"), 10.0, 70.0, 608, 1080, 1920, 1080, sample_count=5,
        )
        # First keyframe should be at or before start_time
        assert keyframes[0].timestamp <= 10.0 + 0.1
        # Last keyframe should be at or after end_time
        assert keyframes[-1].timestamp >= 70.0 - 0.1

    @patch("core.shorts_converter.detect_face_crop_region")
    def test_sample_count_controls_keyframes(self, mock_detect_face) -> None:
        """sample_count should control how many frames are sampled."""
        mock_detect_face.return_value = (500, 200)
        keyframes = _track_faces_across_clip(
            Path("fake.mp4"), 0.0, 60.0, 608, 1080, 1920, 1080, sample_count=2,
        )
        # With 2 samples and possibly start/end padding, should have >= 2 keyframes
        assert len(keyframes) >= 2


# ── Test _smooth_crop_keyframes ───────────────────────────────

class TestSmoothCropKeyframes:
    """Tests for the _smooth_crop_keyframes function."""

    def test_single_keyframe_unchanged(self) -> None:
        """A single keyframe should be returned unchanged."""
        keyframes = [CropKeyframe(timestamp=0.0, crop_x=500, crop_y=200)]
        result = _smooth_crop_keyframes(keyframes)
        assert len(result) == 1
        assert result[0].crop_x == 500
        assert result[0].crop_y == 200

    def test_multiple_keyframes_smoothed(self) -> None:
        """Multiple keyframes should be smoothed (less jittery)."""
        keyframes = [
            CropKeyframe(timestamp=0.0, crop_x=500, crop_y=200),
            CropKeyframe(timestamp=10.0, crop_x=510, crop_y=205),
            CropKeyframe(timestamp=20.0, crop_x=490, crop_y=195),
            CropKeyframe(timestamp=30.0, crop_x=505, crop_y=202),
            CropKeyframe(timestamp=40.0, crop_x=495, crop_y=198),
        ]
        result = _smooth_crop_keyframes(keyframes)
        assert len(result) == 5
        # Smoothed values should be close to the originals
        for i in range(len(result)):
            assert abs(result[i].crop_x - keyframes[i].crop_x) < 30
            assert abs(result[i].crop_y - keyframes[i].crop_y) < 30

    def test_smoothing_window_parameter(self) -> None:
        """Custom smoothing window should be accepted."""
        keyframes = [
            CropKeyframe(timestamp=0.0, crop_x=500, crop_y=200),
            CropKeyframe(timestamp=5.0, crop_x=520, crop_y=220),
            CropKeyframe(timestamp=10.0, crop_x=480, crop_y=180),
        ]
        result_narrow = _smooth_crop_keyframes(keyframes, smoothing_window=1.0)
        result_wide = _smooth_crop_keyframes(keyframes, smoothing_window=5.0)
        # Both should produce valid results
        assert len(result_narrow) == 3
        assert len(result_wide) == 3

    def test_empty_keyframes(self) -> None:
        """Empty keyframes list should return empty list."""
        result = _smooth_crop_keyframes([])
        assert result == []

    def test_two_keyframes(self) -> None:
        """Two keyframes should produce valid smoothed output."""
        keyframes = [
            CropKeyframe(timestamp=0.0, crop_x=500, crop_y=200),
            CropKeyframe(timestamp=60.0, crop_x=600, crop_y=300),
        ]
        result = _smooth_crop_keyframes(keyframes)
        assert len(result) == 2


# ── Test _build_crop_filtergraph ──────────────────────────────

class TestBuildCropFiltergraph:
    """Tests for the _build_crop_filtergraph function."""

    def test_basic_filtergraph_contains_trim(self, mock_settings: MagicMock) -> None:
        """Filtergraph should contain trim filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "trim=start=10.0" in fg

    def test_basic_filtergraph_contains_crop(self, mock_settings: MagicMock) -> None:
        """Filtergraph should contain crop filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "crop=" in fg

    def test_basic_filtergraph_contains_scale(self, mock_settings: MagicMock) -> None:
        """Filtergraph should contain scale filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "scale=" in fg

    def test_filtergraph_contains_fps(self, mock_settings: MagicMock) -> None:
        """Filtergraph should contain fps filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "fps=" in fg

    def test_deinterlace_adds_yadif(self, mock_settings: MagicMock) -> None:
        """Deinterlace flag should add yadif filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            deinterlace=True,
        )
        assert "yadif" in fg

    def test_color_correct_adds_eq(self, mock_settings: MagicMock) -> None:
        """Color correction flag should add eq filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            color_correct=True,
        )
        assert "eq=" in fg

    def test_stabilize_adds_deshake(self, mock_settings: MagicMock) -> None:
        """Stabilize flag should add deshake filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            stabilize=True,
        )
        assert "deshake" in fg

    def test_fade_in_adds_fade_filter(self, mock_settings: MagicMock) -> None:
        """Fade-in should add a fade=t=in filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            fade_in=1.0,
        )
        assert "fade=t=in" in fg

    def test_fade_out_adds_fade_filter(self, mock_settings: MagicMock) -> None:
        """Fade-out should add a fade=t=out filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            fade_out=1.0,
        )
        assert "fade=t=out" in fg

    def test_letterbox_removal_adds_crop(self, mock_settings: MagicMock) -> None:
        """Letterbox removal should add a crop filter for the bars."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            remove_letterbox=(20, 30, 0, 0),
        )
        # Should have a crop filter for letterbox removal
        assert "crop=" in fg

    def test_zoom_effect_adds_zoompan(self, mock_settings: MagicMock) -> None:
        """Zoom effect should add zoompan filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            zoom_effect=True,
        )
        assert "zoompan" in fg

    def test_blur_bg_calls_blur_bg_filtergraph(self, mock_settings: MagicMock) -> None:
        """Blur background flag should delegate to _build_blur_bg_filtergraph."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            blur_bg=True,
        )
        # Blur bg uses filter_complex with overlay
        assert "overlay" in fg or "gblur" in fg

    def test_rotation_adds_rotate(self, mock_settings: MagicMock) -> None:
        """Non-zero rotation should add rotate filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            rotation=90.0,
        )
        assert "rotate" in fg

    def test_zero_rotation_no_rotate(self, mock_settings: MagicMock) -> None:
        """Zero rotation should not add rotate filter."""
        fg = _build_crop_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
            rotation=0.0,
        )
        assert "rotate" not in fg


# ── Test _build_blur_bg_filtergraph ───────────────────────────

class TestBuildBlurBgFiltergraph:
    """Tests for the _build_blur_bg_filtergraph function."""

    def test_blur_bg_contains_overlay(self, mock_settings: MagicMock) -> None:
        """Blur bg filtergraph should contain overlay."""
        fg = _build_blur_bg_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "overlay" in fg

    def test_blur_bg_contains_gblur(self, mock_settings: MagicMock) -> None:
        """Blur bg filtergraph should contain gblur for background blur."""
        fg = _build_blur_bg_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "gblur" in fg

    def test_blur_bg_contains_split(self, mock_settings: MagicMock) -> None:
        """Blur bg filtergraph should split the input stream."""
        fg = _build_blur_bg_filtergraph(
            src_w=1920, src_h=1080, crop_w=608, crop_h=1080,
            crop_x=656, crop_y=0, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "split" in fg


# ── Test _build_split_screen_filtergraph ──────────────────────

class TestBuildSplitScreenFiltergraph:
    """Tests for the _build_split_screen_filtergraph function."""

    def test_split_screen_contains_split(self, mock_settings: MagicMock) -> None:
        """Split screen filtergraph should contain split filter."""
        fg = _build_split_screen_filtergraph(
            src_w=1920, src_h=1080, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "split" in fg

    def test_split_screen_contains_hstack(self, mock_settings: MagicMock) -> None:
        """Split screen filtergraph should contain hstack filter."""
        fg = _build_split_screen_filtergraph(
            src_w=1920, src_h=1080, out_w=1080, out_h=1920,
            start_time=10.0, clip_duration=60.0, settings=mock_settings,
        )
        assert "hstack" in fg


# ── Test convert_to_shorts ────────────────────────────────────

class TestConvertToShorts:
    """Tests for the convert_to_shorts function with mocked FFmpeg."""

    @patch("core.shorts_converter.detect_hw_encoder")
    @patch("core.shorts_converter.run_ffmpeg")
    @patch("core.shorts_converter.detect_face_crop_region")
    @patch("core.shorts_converter.probe_video")
    @patch("core.shorts_converter.detect_letterbox")
    def test_convert_basic_success(
        self,
        mock_letterbox: MagicMock,
        mock_probe: MagicMock,
        mock_face: MagicMock,
        mock_ffmpeg: MagicMock,
        mock_hw: MagicMock,
        segment: SegmentResult,
        mock_video_info: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Basic conversion should succeed with valid inputs."""
        mock_probe.return_value = mock_video_info
        mock_letterbox.return_value = (0, 0, 0, 0)
        mock_face.return_value = (500, 200)
        mock_hw.return_value = (None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output.mp4"
            input_path.write_text("fake video data")

            # Simulate FFmpeg creating the output file
            def fake_ffmpeg(cmd, **kwargs):
                output_path.write_text("x" * 50000)
            mock_ffmpeg.side_effect = fake_ffmpeg

            result = convert_to_shorts(
                input_path, segment, output_path, settings=mock_settings,
                use_face_tracking=False,  # Disable face tracking to simplify
            )
            assert result == output_path
            mock_ffmpeg.assert_called_once()

    def test_convert_missing_input_raises_error(
        self, segment: SegmentResult, mock_settings: MagicMock,
    ) -> None:
        """Missing input file should raise ConverterError."""
        with pytest.raises(ConverterError, match="Input file not found"):
            convert_to_shorts(
                Path("nonexistent.mp4"), segment, Path("output.mp4"),
                settings=mock_settings,
            )

    @patch("core.shorts_converter.detect_hw_encoder")
    @patch("core.shorts_converter.run_ffmpeg")
    @patch("core.shorts_converter.detect_face_crop_region")
    @patch("core.shorts_converter.probe_video")
    @patch("core.shorts_converter.detect_letterbox")
    def test_convert_with_blur_background(
        self,
        mock_letterbox: MagicMock,
        mock_probe: MagicMock,
        mock_face: MagicMock,
        mock_ffmpeg: MagicMock,
        mock_hw: MagicMock,
        segment: SegmentResult,
        mock_video_info: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Conversion with blur_background should use filter_complex."""
        mock_probe.return_value = mock_video_info
        mock_letterbox.return_value = (0, 0, 0, 0)
        mock_face.return_value = (500, 200)
        mock_hw.return_value = (None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output.mp4"
            input_path.write_text("fake video data")

            def fake_ffmpeg(cmd, **kwargs):
                output_path.write_text("x" * 50000)
            mock_ffmpeg.side_effect = fake_ffmpeg

            result = convert_to_shorts(
                input_path, segment, output_path, settings=mock_settings,
                blur_background=True, use_face_tracking=False,
            )
            assert result == output_path
            # Verify filter_complex was used
            call_args = mock_ffmpeg.call_args
            cmd = call_args[0][0]
            assert any("filter_complex" in str(arg) for arg in cmd)

    @patch("core.shorts_converter.detect_hw_encoder")
    @patch("core.shorts_converter.run_ffmpeg")
    @patch("core.shorts_converter.detect_face_crop_region")
    @patch("core.shorts_converter.probe_video")
    @patch("core.shorts_converter.detect_letterbox")
    def test_convert_with_split_screen(
        self,
        mock_letterbox: MagicMock,
        mock_probe: MagicMock,
        mock_face: MagicMock,
        mock_ffmpeg: MagicMock,
        mock_hw: MagicMock,
        segment: SegmentResult,
        mock_video_info: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Conversion with split_screen should use filter_complex."""
        mock_probe.return_value = mock_video_info
        mock_letterbox.return_value = (0, 0, 0, 0)
        mock_face.return_value = (500, 200)
        mock_hw.return_value = (None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output.mp4"
            input_path.write_text("fake video data")

            def fake_ffmpeg(cmd, **kwargs):
                output_path.write_text("x" * 50000)
            mock_ffmpeg.side_effect = fake_ffmpeg

            result = convert_to_shorts(
                input_path, segment, output_path, settings=mock_settings,
                split_screen=True,
            )
            assert result == output_path

    @patch("core.shorts_converter.detect_hw_encoder")
    @patch("core.shorts_converter.run_ffmpeg")
    @patch("core.shorts_converter.detect_face_crop_region")
    @patch("core.shorts_converter.probe_video")
    @patch("core.shorts_converter.detect_letterbox")
    def test_convert_different_platforms(
        self,
        mock_letterbox: MagicMock,
        mock_probe: MagicMock,
        mock_face: MagicMock,
        mock_ffmpeg: MagicMock,
        mock_hw: MagicMock,
        segment: SegmentResult,
        mock_video_info: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Conversion for different platforms should work."""
        mock_probe.return_value = mock_video_info
        mock_letterbox.return_value = (0, 0, 0, 0)
        mock_face.return_value = (500, 200)
        mock_hw.return_value = (None, None)

        for platform in ("youtube_shorts", "tiktok", "instagram_reels"):
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = Path(tmpdir) / "input.mp4"
                output_path = Path(tmpdir) / "output.mp4"
                input_path.write_text("fake video data")

                def fake_ffmpeg(cmd, **kwargs):
                    output_path.write_text("x" * 50000)
                mock_ffmpeg.side_effect = fake_ffmpeg

                result = convert_to_shorts(
                    input_path, segment, output_path, settings=mock_settings,
                    platform=platform, use_face_tracking=False,
                )
                assert result == output_path

    @patch("core.shorts_converter.detect_hw_encoder")
    @patch("core.shorts_converter.probe_video")
    @patch("core.shorts_converter.detect_letterbox")
    def test_convert_invalid_video_dimensions_raises_error(
        self,
        mock_letterbox: MagicMock,
        mock_probe: MagicMock,
        mock_hw: MagicMock,
        segment: SegmentResult,
        mock_settings: MagicMock,
    ) -> None:
        """Zero-dimension video should raise ConverterError."""
        bad_video = MagicMock()
        bad_video.width = 0
        bad_video.height = 0
        bad_video.duration = 120.0
        mock_probe.return_value = bad_video
        mock_letterbox.return_value = (0, 0, 0, 0)
        mock_hw.return_value = (None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output.mp4"
            input_path.write_text("fake video data")

            with pytest.raises(ConverterError, match="Invalid video dimensions"):
                convert_to_shorts(
                    input_path, segment, output_path, settings=mock_settings,
                )

    @patch("core.shorts_converter.detect_hw_encoder")
    @patch("core.shorts_converter.run_ffmpeg")
    @patch("core.shorts_converter.detect_face_crop_region")
    @patch("core.shorts_converter.probe_video")
    @patch("core.shorts_converter.detect_letterbox")
    def test_convert_already_vertical_video(
        self,
        mock_letterbox: MagicMock,
        mock_probe: MagicMock,
        mock_face: MagicMock,
        mock_ffmpeg: MagicMock,
        mock_hw: MagicMock,
        segment: SegmentResult,
        mock_settings: MagicMock,
    ) -> None:
        """An already vertical video should not need cropping."""
        # Vertical video (9:16)
        vertical_video = MagicMock()
        vertical_video.width = 1080
        vertical_video.height = 1920
        vertical_video.duration = 120.0
        mock_probe.return_value = vertical_video
        mock_letterbox.return_value = (0, 0, 0, 0)
        mock_face.return_value = (0, 0)
        mock_hw.return_value = (None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output.mp4"
            input_path.write_text("fake video data")

            def fake_ffmpeg(cmd, **kwargs):
                output_path.write_text("x" * 50000)
            mock_ffmpeg.side_effect = fake_ffmpeg

            result = convert_to_shorts(
                input_path, segment, output_path, settings=mock_settings,
                use_face_tracking=False,
            )
            assert result == output_path


# ── Test PlatformSpec ─────────────────────────────────────────

class TestPlatformSpec:
    """Tests for the PlatformSpec dataclass and PLATFORM_SPECS."""

    def test_youtube_shorts_spec(self) -> None:
        """YouTube Shorts should have 1080x1920 resolution."""
        spec = PLATFORM_SPECS["youtube_shorts"]
        assert spec.width == 1080
        assert spec.height == 1920
        assert spec.aspect_ratio == "9:16"

    def test_tiktok_spec(self) -> None:
        """TikTok should have 1080x1920 resolution."""
        spec = PLATFORM_SPECS["tiktok"]
        assert spec.width == 1080
        assert spec.height == 1920

    def test_instagram_reels_spec(self) -> None:
        """Instagram Reels should have 1080x1920 resolution."""
        spec = PLATFORM_SPECS["instagram_reels"]
        assert spec.width == 1080
        assert spec.height == 1920

    def test_all_specs_have_max_duration(self) -> None:
        """All platform specs should have a positive max_duration."""
        for name, spec in PLATFORM_SPECS.items():
            assert spec.max_duration > 0, f"{name} has invalid max_duration"

    def test_all_specs_have_name(self) -> None:
        """All platform specs should have a non-empty name."""
        for name, spec in PLATFORM_SPECS.items():
            assert len(spec.name) > 0, f"{name} has empty name"

    def test_square_spec(self) -> None:
        """Square format should have equal width and height."""
        spec = PLATFORM_SPECS["square"]
        assert spec.width == spec.height
        assert spec.aspect_ratio == "1:1"
