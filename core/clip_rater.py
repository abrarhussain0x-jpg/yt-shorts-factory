"""
core/clip_rater.py — Clip quality rating, comparison, and ranking for short-form content.

Rates clips based on engagement, speech quality, and visual appeal.
Supports platform-specific ranking and provides strengths/weaknesses
analysis along with title style suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.analyzer import SegmentResult
from core.transcriber import TranscriptionResult
from config.settings import Settings, get_settings
from utils.logger import get_logger

logger = get_logger("clip_rater")


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class ClipRating:
    """Rating result for a single clip."""

    overall_score: float = 0.0  # 0-100 composite
    engagement_score: float = 0.0  # 0-100
    speech_score: float = 0.0  # 0-100
    visual_score: float = 0.0  # 0-100
    grade: str = "C"  # A/B/C/D/F
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    title_style: str = "neutral"  # suggested title style
    platform_rankings: dict[str, int] = field(default_factory=dict)


# ── Score-to-Grade Mapping ────────────────────────────────────

def _score_to_grade(score: float) -> str:
    """Map a 0-100 score to a letter grade.

    Args:
        score: Numeric score between 0 and 100.

    Returns:
        Letter grade: 'A' (>=80), 'B' (>=60), 'C' (>=40), 'D' (>=20), 'F' (<20).
    """
    if score >= 80:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    elif score >= 20:
        return "D"
    else:
        return "F"


# ── Score Computation ─────────────────────────────────────────

def _compute_engagement_score(segment: SegmentResult) -> float:
    """Compute engagement score from a SegmentResult.

    Combines energy_score, confidence, emphasis, and speech rate
    into a 0-100 engagement score.

    Args:
        segment: The segment analysis result.

    Returns:
        Engagement score between 0 and 100.
    """
    score = 0.0

    # Energy score (0-1) → 0-40 points
    score += segment.energy_score * 40.0

    # Confidence (0-1) → 0-20 points
    score += segment.confidence * 20.0

    # Speech rate: ideal is 120-180 WPM
    if segment.speech_rate_estimate > 0:
        if 120 <= segment.speech_rate_estimate <= 180:
            score += 20.0
        elif 80 <= segment.speech_rate_estimate <= 220:
            score += 12.0
        else:
            score += 5.0

    # Emphasis moments → 0-10 points
    if segment.has_emphasis:
        score += 10.0

    # Visual complexity (0-1) → 0-10 points
    score += segment.visual_complexity * 10.0

    return min(100.0, max(0.0, round(score, 1)))


def _compute_speech_score(
    segment: SegmentResult,
    transcription: Optional[TranscriptionResult] = None,
) -> float:
    """Compute speech quality score.

    Uses segment metadata and optionally transcription details
    to rate speech quality from 0-100.

    Args:
        segment: The segment analysis result.
        transcription: Optional transcription result for detailed analysis.

    Returns:
        Speech quality score between 0 and 100.
    """
    score = 0.0

    # Speech detected bonus
    if segment.speech_detected:
        score += 30.0

    # Low silence ratio → good speech content
    speech_ratio = 1.0 - segment.silence_ratio
    score += speech_ratio * 30.0

    # Speech rate quality
    if segment.speech_rate_estimate > 0:
        if 120 <= segment.speech_rate_estimate <= 180:
            score += 25.0
        elif 80 <= segment.speech_rate_estimate <= 220:
            score += 15.0
        else:
            score += 5.0

    # Music likelihood penalty (less speech-like = lower score)
    score += (1.0 - segment.music_likelihood) * 15.0

    # If transcription provided, factor in word count and confidence
    if transcription is not None and not transcription.is_empty:
        word_count = transcription.word_count
        if word_count >= 20:
            score += 5.0  # bonus for substantial speech
        avg_conf = transcription.average_confidence
        score += avg_conf * 5.0  # bonus for clear speech

    return min(100.0, max(0.0, round(score, 1)))


def _compute_visual_score(segment: SegmentResult) -> float:
    """Compute visual quality score.

    Rates the visual appeal based on scene changes, motion, and
    visual complexity.

    Args:
        segment: The segment analysis result.

    Returns:
        Visual quality score between 0 and 100.
    """
    score = 0.0

    # Visual complexity (0-1) → 0-40 points
    score += segment.visual_complexity * 40.0

    # Motion energy (0-1 range, normalized) → 0-30 points
    motion = min(1.0, max(0.0, segment.motion_energy * 10.0))
    score += motion * 30.0

    # Low silence ratio indicates more visual content
    score += (1.0 - segment.silence_ratio) * 15.0

    # Confidence in the segment
    score += segment.confidence * 15.0

    return min(100.0, max(0.0, round(score, 1)))


# ── Strengths and Weaknesses ──────────────────────────────────

def _identify_strengths_weaknesses(
    engagement: float,
    speech: float,
    visual: float,
    segment: SegmentResult,
) -> tuple[list[str], list[str]]:
    """Identify strengths and weaknesses of a clip.

    Args:
        engagement: Engagement score (0-100).
        speech: Speech score (0-100).
        visual: Visual score (0-100).
        segment: The segment analysis result.

    Returns:
        Tuple of (strengths_list, weaknesses_list).
    """
    strengths: list[str] = []
    weaknesses: list[str] = []

    if engagement >= 70:
        strengths.append("High engagement potential")
    elif engagement < 40:
        weaknesses.append("Low engagement potential")

    if speech >= 70:
        strengths.append("Strong speech content")
    elif speech < 40:
        weaknesses.append("Weak speech content")

    if visual >= 70:
        strengths.append("Visually dynamic")
    elif visual < 40:
        weaknesses.append("Visually static")

    if segment.has_emphasis:
        strengths.append("Emphasis moments detected")

    if segment.silence_ratio > 0.3:
        weaknesses.append("High silence ratio")

    if segment.speech_rate_estimate > 200:
        weaknesses.append("Speech may be too fast")
    elif segment.speech_rate_estimate > 0 and segment.speech_rate_estimate < 80:
        weaknesses.append("Speech may be too slow")

    if segment.music_likelihood > 0.7:
        weaknesses.append("Likely music-dominant (not speech)")
    elif segment.music_likelihood < 0.3:
        strengths.append("Clear speech content (not music)")

    if not strengths:
        strengths.append("No major red flags")

    if not weaknesses:
        weaknesses.append("No significant weaknesses")

    return strengths, weaknesses


# ── Title Style Suggestion ────────────────────────────────────

def _suggest_title_style(
    engagement: float,
    speech: float,
    visual: float,
    segment: SegmentResult,
) -> str:
    """Suggest a title style based on clip characteristics.

    Args:
        engagement: Engagement score (0-100).
        speech: Speech score (0-100).
        visual: Visual score (0-100).
        segment: The segment analysis result.

    Returns:
        Suggested title style string.
    """
    if engagement >= 80 and segment.has_emphasis:
        return "shocking"  # e.g., "You WON'T BELIEVE what happens..."
    elif speech >= 70 and visual >= 60:
        return "informative"  # e.g., "Learn about X in 60 seconds"
    elif visual >= 70 and speech < 50:
        return "visual"  # e.g., "Watch this amazing..."
    elif speech >= 70 and segment.speech_rate_estimate > 150:
        return "fast_paced"  # e.g., "Rapid fire facts about..."
    elif engagement >= 60:
        return "engaging"  # e.g., "This clip will blow your mind"
    elif segment.music_likelihood > 0.5:
        return "musical"  # e.g., "Vibe with this..."
    else:
        return "neutral"


# ── Public API ────────────────────────────────────────────────

def rate_clip(
    segment: SegmentResult,
    transcription: Optional[TranscriptionResult] = None,
    settings: Settings | None = None,
) -> ClipRating:
    """Rate a clip based on its segment analysis and optional transcription.

    Computes engagement, speech, and visual scores, then combines them
    into an overall rating with grade, strengths, weaknesses, and
    a title style suggestion.

    Args:
        segment: The segment analysis result.
        transcription: Optional transcription for speech quality analysis.
        settings: Optional Settings override.

    Returns:
        ClipRating with all computed scores and metadata.
    """
    if settings is None:
        settings = get_settings()

    engagement = _compute_engagement_score(segment)
    speech = _compute_speech_score(segment, transcription)
    visual = _compute_visual_score(segment)

    # Weighted overall: engagement is most important for shorts
    overall = (
        0.45 * engagement
        + 0.30 * speech
        + 0.25 * visual
    )
    overall = min(100.0, max(0.0, round(overall, 1)))

    grade = _score_to_grade(overall)
    strengths, weaknesses = _identify_strengths_weaknesses(
        engagement, speech, visual, segment,
    )
    title_style = _suggest_title_style(engagement, speech, visual, segment)

    # Platform-specific rankings (position in a theoretical top-N list)
    platform_rankings: dict[str, int] = {}
    for platform in ("youtube", "tiktok", "reels"):
        # YouTube favors engagement + speech
        # TikTok favors visual + engagement
        # Reels favors visual + speech
        if platform == "youtube":
            platform_score = 0.40 * engagement + 0.35 * speech + 0.25 * visual
        elif platform == "tiktok":
            platform_score = 0.35 * engagement + 0.20 * speech + 0.45 * visual
        else:  # reels
            platform_score = 0.30 * engagement + 0.35 * speech + 0.35 * visual

        # Convert score to a rank (higher score = lower rank number = better)
        platform_rankings[platform] = max(1, int(101 - platform_score))

    rating = ClipRating(
        overall_score=overall,
        engagement_score=engagement,
        speech_score=speech,
        visual_score=visual,
        grade=grade,
        strengths=strengths,
        weaknesses=weaknesses,
        title_style=title_style,
        platform_rankings=platform_rankings,
    )

    logger.info(
        "Clip rated: overall=%.1f (%s) | engagement=%.1f | speech=%.1f | visual=%.1f | style=%s",
        overall, grade, engagement, speech, visual, title_style,
    )

    return rating


def compare_clips(
    segments: list[SegmentResult],
    transcriptions: Optional[list[Optional[TranscriptionResult]]] = None,
    settings: Settings | None = None,
) -> list[tuple[SegmentResult, ClipRating]]:
    """Compare multiple clips and return them with ratings.

    Rates each segment and returns them sorted by overall score
    (best first).

    Args:
        segments: List of segment results to compare.
        transcriptions: Optional list of transcriptions (one per segment, can be None).
        settings: Optional Settings override.

    Returns:
        List of (segment, rating) tuples sorted by overall_score descending.
    """
    if not segments:
        return []

    if transcriptions is None:
        transcriptions = [None] * len(segments)

    results: list[tuple[SegmentResult, ClipRating]] = []
    for i, segment in enumerate(segments):
        transcription = transcriptions[i] if i < len(transcriptions) else None
        rating = rate_clip(segment, transcription, settings)
        results.append((segment, rating))

    # Sort by overall_score descending
    results.sort(key=lambda x: x[1].overall_score, reverse=True)

    return results


def rank_clips(
    segments: list[SegmentResult],
    platform: str = "youtube",
    transcriptions: Optional[list[Optional[TranscriptionResult]]] = None,
    settings: Settings | None = None,
) -> list[tuple[SegmentResult, ClipRating]]:
    """Rank clips for a specific platform.

    Similar to compare_clips but ranks based on platform-specific
    scoring weights.

    Args:
        segments: List of segment results to rank.
        platform: Target platform ('youtube', 'tiktok', 'reels').
        transcriptions: Optional list of transcriptions.
        settings: Optional Settings override.

    Returns:
        List of (segment, rating) tuples sorted by platform fit descending.
    """
    if not segments:
        return []

    if transcriptions is None:
        transcriptions = [None] * len(segments)

    results: list[tuple[SegmentResult, ClipRating]] = []
    for i, segment in enumerate(segments):
        transcription = transcriptions[i] if i < len(transcriptions) else None
        rating = rate_clip(segment, transcription, settings)
        results.append((segment, rating))

    # Sort by platform ranking (lower rank number = better)
    def platform_sort_key(item: tuple[SegmentResult, ClipRating]) -> int:
        _, rating = item
        return rating.platform_rankings.get(platform, 100)

    results.sort(key=platform_sort_key)

    return results
