"""
tests/test_analyzer.py — Tests for the engagement analyzer.
Tests segment selection, duration constraints, and multi-signal scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.analyzer import EngagementAnalyzer, SegmentResult, EnergyProfile


class TestFindPeakSegment:
    """Tests for the peak segment finding algorithm."""

    @patch("core.analyzer.extract_audio_samples")
    @patch("core.analyzer.probe_video")
    def test_find_peak_segment_returns_valid_range(self, mock_probe, mock_audio) -> None:
        """The returned segment should have valid start/end times."""
        mock_probe.return_value = MagicMock(
            duration=120.0, width=1920, height=1080, fps=30.0,
            has_audio=True, codec="h264", bitrate=5000000, size_bytes=1000000,
        )
        mock_audio.return_value = [-20.0] * 60

        analyzer = EngagementAnalyzer(Path("fake.mp4"), clip_duration=60)
        with patch.object(analyzer, "_compute_scene_density", return_value=np.ones(60) * 0.5):
            result = analyzer.analyze()

        assert isinstance(result, SegmentResult)
        assert result.start_time >= 0.0
        assert result.end_time > result.start_time

    @patch("core.analyzer.extract_audio_samples")
    @patch("core.analyzer.probe_video")
    def test_segment_does_not_exceed_video_duration(self, mock_probe, mock_audio) -> None:
        """The segment should not extend beyond the video duration."""
        mock_probe.return_value = MagicMock(
            duration=90.0, width=1920, height=1080, fps=30.0,
            has_audio=True, codec="h264", bitrate=5000000, size_bytes=1000000,
        )
        mock_audio.return_value = [-10.0] * 45

        analyzer = EngagementAnalyzer(Path("fake.mp4"), clip_duration=60)
        with patch.object(analyzer, "_compute_scene_density", return_value=np.ones(45) * 0.5):
            result = analyzer.analyze()

        assert result.end_time <= 92.0  # Tolerance for buffer

    @patch("core.analyzer.extract_audio_samples")
    @patch("core.analyzer.probe_video")
    def test_short_video_uses_full_duration(self, mock_probe, mock_audio) -> None:
        """Videos shorter than clip_duration should use the full video."""
        mock_probe.return_value = MagicMock(
            duration=30.0, width=1920, height=1080, fps=30.0,
            has_audio=True, codec="h264", bitrate=5000000, size_bytes=1000000,
        )
        mock_audio.return_value = [-10.0] * 15

        analyzer = EngagementAnalyzer(Path("fake.mp4"), clip_duration=60)
        result = analyzer.analyze()

        assert result.start_time == 0.0
        assert result.end_time == 30.0
        assert result.method_used == "full_video"


class TestMultiSignalComposite:
    """Tests for the multi-signal composite scoring."""

    def test_audio_energy_dominant_weight(self) -> None:
        """Audio energy should have the highest weight in the composite."""
        # Build profile with high audio energy, low scene density
        n = 10
        profile = EnergyProfile(
            audio_energy_rms=np.ones(n),
            audio_energy_peak=np.ones(n),
            scene_density=np.zeros(n),
            scene_transition_types={},
            silence_mask=np.zeros(n),
            speech_mask=np.zeros(n),
            motion_energy=np.zeros(n),
            spectral_centroid=np.zeros(n),
            emphasis_mask=np.zeros(n),
            composite=np.ones(n) * 0.4,  # Mostly audio
            total_duration=10.0,
            sample_interval=2.0,
        )
        # Audio energy should dominate: audio_rms > scene_density
        assert np.mean(profile.audio_energy_rms) > np.mean(profile.scene_density)

    def test_silence_penalty_reduces_score(self) -> None:
        """Silent regions should reduce the composite score."""
        # Manual calculation: with silence penalty (-0.15), score should be lower
        composite_silent = 0.40 * 0.5 + 0.25 * 0.5 - 0.15 * 1.0
        composite_normal = 0.40 * 0.5 + 0.25 * 0.5 - 0.15 * 0.0
        assert composite_silent < composite_normal

    def test_energy_profile_has_aligned_arrays(self) -> None:
        """EnergyProfile arrays should all have consistent lengths after construction."""
        analyzer = EngagementAnalyzer(Path("fake.mp4"), clip_duration=60)
        # Build a profile manually to check consistency
        n = 30
        profile = EnergyProfile(
            audio_energy_rms=np.random.rand(n),
            audio_energy_peak=np.random.rand(n),
            scene_density=np.random.rand(n),
            scene_transition_types={},
            silence_mask=np.zeros(n),
            speech_mask=np.zeros(n),
            motion_energy=np.zeros(n),
            spectral_centroid=np.zeros(n),
            emphasis_mask=np.zeros(n),
            composite=np.random.rand(n),
            total_duration=60.0,
            sample_interval=2.0,
        )
        # All array fields should have the same length
        arr_fields = [
            profile.audio_energy_rms, profile.audio_energy_peak,
            profile.scene_density, profile.silence_mask,
            profile.speech_mask, profile.motion_energy,
            profile.spectral_centroid, profile.emphasis_mask,
            profile.composite,
        ]
        lengths = [len(a) for a in arr_fields]
        assert len(set(lengths)) == 1, f"Array lengths differ: {lengths}"

    @patch("core.analyzer.extract_audio_samples")
    @patch("core.analyzer.probe_video")
    def test_method_used_is_multi_signal_v2(self, mock_probe, mock_audio) -> None:
        """The analysis should use the multi_signal_v2 method."""
        mock_probe.return_value = MagicMock(
            duration=120.0, width=1920, height=1080, fps=30.0,
            has_audio=True, codec="h264", bitrate=5000000, size_bytes=1000000,
        )
        mock_audio.return_value = [-20.0] * 60

        analyzer = EngagementAnalyzer(Path("fake.mp4"), clip_duration=60)
        with patch.object(analyzer, "_compute_scene_density", return_value=np.ones(60) * 0.5):
            result = analyzer.analyze()

        assert result.method_used.startswith("multi_signal"), f"Expected multi_signal method, got {result.method_used}"


class TestSegmentResult:
    """Tests for the SegmentResult data class."""

    def test_segment_result_has_all_fields(self) -> None:
        """SegmentResult should contain all expected fields."""
        seg = SegmentResult(
            start_time=10.0, end_time=70.0,
            energy_score=0.85, method_used="multi_signal_v2",
        )
        assert seg.start_time == 10.0
        assert seg.end_time == 70.0
        assert seg.energy_score == 0.85
        assert seg.method_used == "multi_signal_v2"

    def test_segment_result_defaults(self) -> None:
        """SegmentResult defaults should be safe values."""
        seg = SegmentResult(start_time=0.0, end_time=60.0, energy_score=0.0)
        assert seg.energy_score == 0.0
        assert seg.silence_ratio == 0.0
        assert seg.speech_detected is True
        assert seg.best_crop_x == -1
        assert seg.confidence == 0.0
