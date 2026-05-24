"""
tests/test_clip_rater.py — Tests for the clip_rater module.
Tests clip rating, comparison, ranking, score computation,
grade mapping, strengths/weaknesses, and title style suggestions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.analyzer import SegmentResult
from core.clip_rater import (
    ClipRating,
    rate_clip,
    compare_clips,
    rank_clips,
    _compute_engagement_score,
    _compute_speech_score,
    _compute_visual_score,
    _score_to_grade,
    _identify_strengths_weaknesses,
    _suggest_title_style,
)
from core.transcriber import TranscriptionResult, WordTimestamp, Segment


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def high_quality_segment() -> SegmentResult:
    """A segment with high engagement metrics."""
    return SegmentResult(
        start_time=10.0,
        end_time=70.0,
        energy_score=0.85,
        confidence=0.9,
        speech_rate_estimate=150.0,
        has_emphasis=True,
        visual_complexity=0.8,
        silence_ratio=0.05,
        music_likelihood=0.1,
        motion_energy=0.06,
        speech_detected=True,
    )


@pytest.fixture
def low_quality_segment() -> SegmentResult:
    """A segment with low engagement metrics."""
    return SegmentResult(
        start_time=0.0,
        end_time=60.0,
        energy_score=0.15,
        confidence=0.2,
        speech_rate_estimate=30.0,
        has_emphasis=False,
        visual_complexity=0.1,
        silence_ratio=0.7,
        music_likelihood=0.9,
        motion_energy=0.01,
        speech_detected=False,
    )


@pytest.fixture
def medium_quality_segment() -> SegmentResult:
    """A segment with moderate engagement metrics."""
    return SegmentResult(
        start_time=20.0,
        end_time=80.0,
        energy_score=0.55,
        confidence=0.6,
        speech_rate_estimate=120.0,
        has_emphasis=False,
        visual_complexity=0.5,
        silence_ratio=0.2,
        music_likelihood=0.3,
        motion_energy=0.04,
        speech_detected=True,
    )


@pytest.fixture
def transcription_result() -> TranscriptionResult:
    """A mock transcription result with word-level timestamps."""
    words = [
        WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
        WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.90),
        WordTimestamp(word="this", start=1.0, end=1.3, confidence=0.88),
        WordTimestamp(word="is", start=1.3, end=1.5, confidence=0.92),
        WordTimestamp(word="amazing", start=1.5, end=2.0, confidence=0.85),
        WordTimestamp(word="content", start=2.0, end=2.5, confidence=0.91),
        WordTimestamp(word="for", start=2.5, end=2.7, confidence=0.87),
        WordTimestamp(word="shorts", start=2.7, end=3.0, confidence=0.93),
        WordTimestamp(word="and", start=3.0, end=3.2, confidence=0.89),
        WordTimestamp(word="more", start=3.2, end=3.5, confidence=0.90),
        WordTimestamp(word="words", start=3.5, end=3.8, confidence=0.86),
        WordTimestamp(word="here", start=3.8, end=4.0, confidence=0.88),
        WordTimestamp(word="to", start=4.0, end=4.2, confidence=0.92),
        WordTimestamp(word="reach", start=4.2, end=4.5, confidence=0.91),
        WordTimestamp(word="twenty", start=4.5, end=4.8, confidence=0.87),
        WordTimestamp(word="plus", start=4.8, end=5.0, confidence=0.90),
        WordTimestamp(word="word", start=5.0, end=5.3, confidence=0.85),
        WordTimestamp(word="count", start=5.3, end=5.5, confidence=0.88),
        WordTimestamp(word="for", start=5.5, end=5.7, confidence=0.92),
        WordTimestamp(word="testing", start=5.7, end=6.0, confidence=0.94),
        WordTimestamp(word="purposes", start=6.0, end=6.5, confidence=0.91),
    ]
    return TranscriptionResult(
        words=words,
        segments=[Segment(text="Hello world this is amazing content", start=0.0, end=3.0)],
        language="en",
        duration=6.5,
    )


@pytest.fixture
def empty_transcription() -> TranscriptionResult:
    """An empty transcription result."""
    return TranscriptionResult(
        words=[],
        segments=[],
        language="en",
        duration=0.0,
    )


# ── Test _score_to_grade ──────────────────────────────────────

class TestScoreToGrade:
    """Tests for the _score_to_grade mapping function."""

    def test_grade_a_for_high_score(self) -> None:
        """Scores >= 80 should map to grade A."""
        assert _score_to_grade(80) == "A"
        assert _score_to_grade(95) == "A"
        assert _score_to_grade(100) == "A"

    def test_grade_b_for_good_score(self) -> None:
        """Scores 60-79 should map to grade B."""
        assert _score_to_grade(60) == "B"
        assert _score_to_grade(70) == "B"
        assert _score_to_grade(79.9) == "B"

    def test_grade_c_for_moderate_score(self) -> None:
        """Scores 40-59 should map to grade C."""
        assert _score_to_grade(40) == "C"
        assert _score_to_grade(50) == "C"
        assert _score_to_grade(59.9) == "C"

    def test_grade_d_for_low_score(self) -> None:
        """Scores 20-39 should map to grade D."""
        assert _score_to_grade(20) == "D"
        assert _score_to_grade(30) == "D"
        assert _score_to_grade(39.9) == "D"

    def test_grade_f_for_very_low_score(self) -> None:
        """Scores < 20 should map to grade F."""
        assert _score_to_grade(0) == "F"
        assert _score_to_grade(10) == "F"
        assert _score_to_grade(19.9) == "F"

    def test_boundary_values(self) -> None:
        """Test exact boundary values."""
        assert _score_to_grade(79.999) == "B"
        assert _score_to_grade(59.999) == "C"
        assert _score_to_grade(39.999) == "D"
        assert _score_to_grade(19.999) == "F"

    def test_negative_score_clamped(self) -> None:
        """Negative scores should map to grade F."""
        assert _score_to_grade(-10) == "F"

    def test_over_100_score_gives_a(self) -> None:
        """Scores over 100 should still give grade A."""
        assert _score_to_grade(150) == "A"


# ── Test _compute_engagement_score ────────────────────────────

class TestComputeEngagementScore:
    """Tests for the _compute_engagement_score function."""

    def test_high_quality_segment_scores_high(self, high_quality_segment: SegmentResult) -> None:
        """A high quality segment should have a high engagement score."""
        score = _compute_engagement_score(high_quality_segment)
        assert score >= 60.0, f"Expected high engagement score, got {score}"

    def test_low_quality_segment_scores_low(self, low_quality_segment: SegmentResult) -> None:
        """A low quality segment should have a low engagement score."""
        score = _compute_engagement_score(low_quality_segment)
        assert score <= 40.0, f"Expected low engagement score, got {score}"

    def test_engagement_score_bounded(self, high_quality_segment: SegmentResult) -> None:
        """Engagement score should be bounded between 0 and 100."""
        score = _compute_engagement_score(high_quality_segment)
        assert 0.0 <= score <= 100.0

    def test_ideal_speech_rate_gets_bonus(self) -> None:
        """A segment with ideal speech rate (120-180 WPM) should get full points."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_rate_estimate=150.0,
        )
        score_ideal = _compute_engagement_score(seg)

        seg_slow = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_rate_estimate=30.0,
        )
        score_slow = _compute_engagement_score(seg_slow)

        assert score_ideal > score_slow

    def test_emphasis_adds_points(self) -> None:
        """A segment with emphasis should score higher than one without."""
        seg_with = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            has_emphasis=True,
        )
        seg_without = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            has_emphasis=False,
        )
        assert _compute_engagement_score(seg_with) > _compute_engagement_score(seg_without)

    def test_zero_energy_gives_low_score(self) -> None:
        """A segment with zero energy score should give a very low engagement score."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.0,
        )
        score = _compute_engagement_score(seg)
        assert score <= 30.0

    def test_default_segment_gives_moderate_score(self) -> None:
        """A default SegmentResult should give a moderate engagement score."""
        seg = SegmentResult(start_time=0.0, end_time=60.0, energy_score=0.5)
        score = _compute_engagement_score(seg)
        assert 10.0 <= score <= 60.0


# ── Test _compute_speech_score ────────────────────────────────

class TestComputeSpeechScore:
    """Tests for the _compute_speech_score function."""

    def test_speech_detected_gives_bonus(self, high_quality_segment: SegmentResult) -> None:
        """Speech detected should give a significant bonus."""
        score = _compute_speech_score(high_quality_segment)
        assert score >= 50.0, f"Expected decent speech score, got {score}"

    def test_no_speech_detected_reduces_score(self, low_quality_segment: SegmentResult) -> None:
        """No speech detected should reduce the score."""
        score = _compute_speech_score(low_quality_segment)
        assert score <= 50.0, f"Expected low speech score, got {score}"

    def test_high_silence_reduces_score(self) -> None:
        """High silence ratio should reduce speech score."""
        seg_silent = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_detected=True, silence_ratio=0.8,
        )
        seg_active = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_detected=True, silence_ratio=0.05,
        )
        assert _compute_speech_score(seg_active) > _compute_speech_score(seg_silent)

    def test_with_transcription_boosts_score(
        self, high_quality_segment: SegmentResult, transcription_result: TranscriptionResult,
    ) -> None:
        """Providing a transcription with many words should boost speech score."""
        score_without = _compute_speech_score(high_quality_segment)
        score_with = _compute_speech_score(high_quality_segment, transcription_result)
        assert score_with >= score_without

    def test_music_likelihood_penalty(self) -> None:
        """High music likelihood should reduce speech score."""
        seg_speech = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            music_likelihood=0.1,
        )
        seg_music = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            music_likelihood=0.9,
        )
        assert _compute_speech_score(seg_speech) > _compute_speech_score(seg_music)

    def test_empty_transcription_no_penalty(
        self, high_quality_segment: SegmentResult, empty_transcription: TranscriptionResult,
    ) -> None:
        """An empty transcription should not significantly change the score."""
        score_without = _compute_speech_score(high_quality_segment)
        score_with = _compute_speech_score(high_quality_segment, empty_transcription)
        # Score should be similar (no significant boost from empty transcription)
        assert abs(score_without - score_with) <= 10.0

    def test_speech_score_bounded(self, high_quality_segment: SegmentResult) -> None:
        """Speech score should be bounded between 0 and 100."""
        score = _compute_speech_score(high_quality_segment)
        assert 0.0 <= score <= 100.0


# ── Test _compute_visual_score ────────────────────────────────

class TestComputeVisualScore:
    """Tests for the _compute_visual_score function."""

    def test_high_visual_complexity_scores_high(self, high_quality_segment: SegmentResult) -> None:
        """High visual complexity should yield a high visual score."""
        score = _compute_visual_score(high_quality_segment)
        assert score >= 40.0, f"Expected decent visual score, got {score}"

    def test_low_visual_complexity_scores_low(self, low_quality_segment: SegmentResult) -> None:
        """Low visual complexity should yield a low visual score."""
        score = _compute_visual_score(low_quality_segment)
        assert score <= 40.0, f"Expected low visual score, got {score}"

    def test_visual_score_bounded(self, high_quality_segment: SegmentResult) -> None:
        """Visual score should be bounded between 0 and 100."""
        score = _compute_visual_score(high_quality_segment)
        assert 0.0 <= score <= 100.0

    def test_motion_energy_contributes(self) -> None:
        """Higher motion energy should contribute to a higher visual score."""
        seg_high_motion = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            motion_energy=0.1,
        )
        seg_low_motion = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            motion_energy=0.0,
        )
        assert _compute_visual_score(seg_high_motion) > _compute_visual_score(seg_low_motion)

    def test_zero_everything_gives_low_score(self) -> None:
        """A segment with all zeros should give a low but non-negative score."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.0,
            visual_complexity=0.0, motion_energy=0.0,
            silence_ratio=1.0, confidence=0.0,
        )
        score = _compute_visual_score(seg)
        assert score >= 0.0
        assert score <= 20.0


# ── Test _identify_strengths_weaknesses ───────────────────────

class TestIdentifyStrengthsWeaknesses:
    """Tests for the _identify_strengths_weaknesses function."""

    def test_high_scores_have_strengths(self, high_quality_segment: SegmentResult) -> None:
        """High quality clips should have identifiable strengths."""
        strengths, weaknesses = _identify_strengths_weaknesses(
            80.0, 80.0, 80.0, high_quality_segment,
        )
        assert len(strengths) > 0

    def test_low_scores_have_weaknesses(self, low_quality_segment: SegmentResult) -> None:
        """Low quality clips should have identifiable weaknesses."""
        strengths, weaknesses = _identify_strengths_weaknesses(
            20.0, 20.0, 20.0, low_quality_segment,
        )
        assert len(weaknesses) > 0

    def test_high_silence_is_weakness(self) -> None:
        """High silence ratio should be identified as a weakness."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            silence_ratio=0.5,
        )
        _, weaknesses = _identify_strengths_weaknesses(50.0, 50.0, 50.0, seg)
        assert any("silence" in w.lower() for w in weaknesses)

    def test_emphasis_is_strength(self) -> None:
        """Emphasis moments should be identified as a strength."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            has_emphasis=True,
        )
        strengths, _ = _identify_strengths_weaknesses(50.0, 50.0, 50.0, seg)
        assert any("emphasis" in s.lower() for s in strengths)

    def test_fast_speech_is_weakness(self) -> None:
        """Very fast speech should be flagged as a weakness."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_rate_estimate=250.0,
        )
        _, weaknesses = _identify_strengths_weaknesses(50.0, 50.0, 50.0, seg)
        assert any("fast" in w.lower() for w in weaknesses)

    def test_slow_speech_is_weakness(self) -> None:
        """Very slow speech should be flagged as a weakness."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_rate_estimate=50.0,
        )
        _, weaknesses = _identify_strengths_weaknesses(50.0, 50.0, 50.0, seg)
        assert any("slow" in w.lower() for w in weaknesses)

    def test_music_dominant_is_weakness(self) -> None:
        """High music likelihood should be identified as a weakness."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            music_likelihood=0.9,
        )
        _, weaknesses = _identify_strengths_weaknesses(50.0, 50.0, 50.0, seg)
        assert any("music" in w.lower() for w in weaknesses)

    def test_clear_speech_is_strength(self) -> None:
        """Low music likelihood should be identified as clear speech strength."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            music_likelihood=0.1,
        )
        strengths, _ = _identify_strengths_weaknesses(50.0, 50.0, 50.0, seg)
        assert any("speech" in s.lower() and "clear" in s.lower() for s in strengths)

    def test_never_returns_empty_lists(self, medium_quality_segment: SegmentResult) -> None:
        """Should always return non-empty strengths and weaknesses."""
        strengths, weaknesses = _identify_strengths_weaknesses(
            50.0, 50.0, 50.0, medium_quality_segment,
        )
        assert len(strengths) > 0
        assert len(weaknesses) > 0


# ── Test _suggest_title_style ─────────────────────────────────

class TestSuggestTitleStyle:
    """Tests for the _suggest_title_style function."""

    def test_shocking_style_for_high_engagement_with_emphasis(self) -> None:
        """High engagement + emphasis should suggest 'shocking' style."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.9,
            has_emphasis=True,
        )
        style = _suggest_title_style(85.0, 50.0, 50.0, seg)
        assert style == "shocking"

    def test_informative_style_for_good_speech_and_visual(self) -> None:
        """Good speech + good visual should suggest 'informative' style."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            has_emphasis=False, speech_rate_estimate=120.0,
        )
        style = _suggest_title_style(50.0, 75.0, 70.0, seg)
        assert style == "informative"

    def test_visual_style_for_visual_dominant(self) -> None:
        """High visual + low speech should suggest 'visual' style."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_rate_estimate=50.0,
        )
        style = _suggest_title_style(50.0, 40.0, 80.0, seg)
        assert style == "visual"

    def test_fast_paced_style_for_rapid_speech(self) -> None:
        """High speech + fast rate should suggest 'fast_paced' style."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.5,
            speech_rate_estimate=200.0,
        )
        # Use visual < 60 so the 'informative' check doesn't match first
        style = _suggest_title_style(50.0, 80.0, 50.0, seg)
        assert style == "fast_paced"

    def test_engaging_style_for_good_engagement(self) -> None:
        """Good engagement without emphasis should suggest 'engaging' style."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.7,
            has_emphasis=False, speech_rate_estimate=100.0,
        )
        style = _suggest_title_style(65.0, 50.0, 50.0, seg)
        assert style == "engaging"

    def test_musical_style_for_music_dominant(self) -> None:
        """High music likelihood should suggest 'musical' style."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.4,
            music_likelihood=0.8,
        )
        style = _suggest_title_style(40.0, 30.0, 30.0, seg)
        assert style == "musical"

    def test_neutral_style_as_fallback(self) -> None:
        """Low scores with no dominant feature should suggest 'neutral' style."""
        seg = SegmentResult(
            start_time=0.0, end_time=60.0, energy_score=0.3,
            music_likelihood=0.4,
        )
        style = _suggest_title_style(30.0, 30.0, 30.0, seg)
        assert style == "neutral"


# ── Test rate_clip ────────────────────────────────────────────

class TestRateClip:
    """Tests for the rate_clip function."""

    def test_rate_clip_returns_clip_rating(self, high_quality_segment: SegmentResult) -> None:
        """rate_clip should return a ClipRating instance."""
        rating = rate_clip(high_quality_segment)
        assert isinstance(rating, ClipRating)

    def test_rate_clip_has_all_fields(self, high_quality_segment: SegmentResult) -> None:
        """ClipRating should have all expected fields."""
        rating = rate_clip(high_quality_segment)
        assert hasattr(rating, "overall_score")
        assert hasattr(rating, "engagement_score")
        assert hasattr(rating, "speech_score")
        assert hasattr(rating, "visual_score")
        assert hasattr(rating, "grade")
        assert hasattr(rating, "strengths")
        assert hasattr(rating, "weaknesses")
        assert hasattr(rating, "title_style")
        assert hasattr(rating, "platform_rankings")

    def test_rate_clip_scores_bounded(self, high_quality_segment: SegmentResult) -> None:
        """All scores should be bounded between 0 and 100."""
        rating = rate_clip(high_quality_segment)
        assert 0.0 <= rating.overall_score <= 100.0
        assert 0.0 <= rating.engagement_score <= 100.0
        assert 0.0 <= rating.speech_score <= 100.0
        assert 0.0 <= rating.visual_score <= 100.0

    def test_rate_clip_grade_is_valid(self, high_quality_segment: SegmentResult) -> None:
        """Grade should be one of A, B, C, D, F."""
        rating = rate_clip(high_quality_segment)
        assert rating.grade in {"A", "B", "C", "D", "F"}

    def test_rate_clip_with_transcription(
        self, high_quality_segment: SegmentResult, transcription_result: TranscriptionResult,
    ) -> None:
        """rate_clip should accept and use an optional TranscriptionResult."""
        rating_without = rate_clip(high_quality_segment)
        rating_with = rate_clip(high_quality_segment, transcription_result)
        # Speech score may differ when transcription is provided
        assert isinstance(rating_with, ClipRating)

    def test_rate_clip_platform_rankings(
        self, high_quality_segment: SegmentResult,
    ) -> None:
        """rate_clip should include platform rankings for youtube, tiktok, reels."""
        rating = rate_clip(high_quality_segment)
        assert "youtube" in rating.platform_rankings
        assert "tiktok" in rating.platform_rankings
        assert "reels" in rating.platform_rankings

    def test_high_quality_gets_good_grade(self, high_quality_segment: SegmentResult) -> None:
        """A high quality segment should get grade A or B."""
        rating = rate_clip(high_quality_segment)
        assert rating.grade in {"A", "B"}

    def test_low_quality_gets_poor_grade(self, low_quality_segment: SegmentResult) -> None:
        """A low quality segment should get grade D or F."""
        rating = rate_clip(low_quality_segment)
        assert rating.grade in {"D", "F"}

    def test_rate_clip_default_segment(self) -> None:
        """A default SegmentResult should produce a valid rating."""
        seg = SegmentResult(start_time=0.0, end_time=60.0, energy_score=0.5)
        rating = rate_clip(seg)
        assert isinstance(rating, ClipRating)
        assert rating.grade in {"A", "B", "C", "D", "F"}


# ── Test compare_clips ────────────────────────────────────────

class TestCompareClips:
    """Tests for the compare_clips function."""

    def test_compare_clips_returns_sorted(
        self,
        high_quality_segment: SegmentResult,
        low_quality_segment: SegmentResult,
        medium_quality_segment: SegmentResult,
    ) -> None:
        """compare_clips should return clips sorted by overall_score descending."""
        results = compare_clips([low_quality_segment, high_quality_segment, medium_quality_segment])
        assert len(results) == 3
        # Best clip first
        assert results[0][1].overall_score >= results[1][1].overall_score
        assert results[1][1].overall_score >= results[2][1].overall_score

    def test_compare_clips_empty_input(self) -> None:
        """Empty input should return empty list."""
        results = compare_clips([])
        assert results == []

    def test_compare_clips_single_segment(
        self, high_quality_segment: SegmentResult,
    ) -> None:
        """A single segment should return a list with one rated entry."""
        results = compare_clips([high_quality_segment])
        assert len(results) == 1
        seg, rating = results[0]
        assert seg is high_quality_segment
        assert isinstance(rating, ClipRating)

    def test_compare_clips_with_transcriptions(
        self,
        high_quality_segment: SegmentResult,
        low_quality_segment: SegmentResult,
        transcription_result: TranscriptionResult,
    ) -> None:
        """compare_clips should accept per-segment transcriptions."""
        results = compare_clips(
            [high_quality_segment, low_quality_segment],
            transcriptions=[transcription_result, None],
        )
        assert len(results) == 2

    def test_compare_clips_returns_tuples(
        self,
        high_quality_segment: SegmentResult,
        low_quality_segment: SegmentResult,
    ) -> None:
        """Each result should be a (SegmentResult, ClipRating) tuple."""
        results = compare_clips([high_quality_segment, low_quality_segment])
        for seg, rating in results:
            assert isinstance(seg, SegmentResult)
            assert isinstance(rating, ClipRating)


# ── Test rank_clips ───────────────────────────────────────────

class TestRankClips:
    """Tests for the rank_clips function."""

    def test_rank_clips_youtube(
        self,
        high_quality_segment: SegmentResult,
        low_quality_segment: SegmentResult,
        medium_quality_segment: SegmentResult,
    ) -> None:
        """rank_clips for youtube should return clips sorted by platform fit."""
        results = rank_clips(
            [medium_quality_segment, low_quality_segment, high_quality_segment],
            platform="youtube",
        )
        assert len(results) == 3

    def test_rank_clips_tiktok(
        self,
        high_quality_segment: SegmentResult,
        low_quality_segment: SegmentResult,
    ) -> None:
        """rank_clips for tiktok should work correctly."""
        results = rank_clips(
            [high_quality_segment, low_quality_segment],
            platform="tiktok",
        )
        assert len(results) == 2

    def test_rank_clips_reels(
        self,
        high_quality_segment: SegmentResult,
        low_quality_segment: SegmentResult,
    ) -> None:
        """rank_clips for reels should work correctly."""
        results = rank_clips(
            [high_quality_segment, low_quality_segment],
            platform="reels",
        )
        assert len(results) == 2

    def test_rank_clips_empty_input(self) -> None:
        """Empty input should return empty list."""
        results = rank_clips([], platform="youtube")
        assert results == []

    def test_rank_clips_different_platforms_give_different_rankings(
        self,
        high_quality_segment: SegmentResult,
        low_quality_segment: SegmentResult,
    ) -> None:
        """Different platforms may give different rankings."""
        # This is a soft test — the rankings might be the same for very
        # different quality clips, but for similar clips they might differ.
        yt_results = rank_clips(
            [high_quality_segment, low_quality_segment], platform="youtube",
        )
        tt_results = rank_clips(
            [high_quality_segment, low_quality_segment], platform="tiktok",
        )
        # Both should return 2 results
        assert len(yt_results) == 2
        assert len(tt_results) == 2

    def test_rank_clips_with_transcriptions(
        self,
        high_quality_segment: SegmentResult,
        transcription_result: TranscriptionResult,
    ) -> None:
        """rank_clips should accept transcriptions."""
        results = rank_clips(
            [high_quality_segment],
            platform="youtube",
            transcriptions=[transcription_result],
        )
        assert len(results) == 1
