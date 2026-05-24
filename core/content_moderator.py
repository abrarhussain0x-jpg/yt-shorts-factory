"""
core/content_moderator.py — Local content moderation for policy compliance.

Provides text, audio, and visual content checking using local-only algorithms
(no API calls). Includes profanity detection, spam detection, violence/adult
keyword detection, misleading content indicators, frame analysis for flashing
lights, and content rating generation. All checks are advisory only.
"""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings
from utils.ffmpeg_utils import probe_video, run_ffmpeg, FFmpegError
from utils.file_utils import safe_delete
from utils.logger import get_logger

logger = get_logger("content_moderator")


# ── Data Classes ──────────────────────────────────────────────

@dataclass
class ModerationResult:
    """Result of text content moderation."""

    is_clean: bool = True
    severity: str = "none"  # none, low, medium, high
    flags: list[str] = field(default_factory=list)
    profanity_detected: bool = False
    spam_detected: bool = False
    violence_detected: bool = False
    adult_detected: bool = False
    misleading_indicators: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class AudioModerationResult:
    """Result of audio content moderation."""

    is_clean: bool = True
    severity: str = "none"
    flags: list[str] = field(default_factory=list)
    silence_ratio: float = 0.0
    volume_spikes: int = 0
    speech_detected: bool = True
    music_likelihood: float = 0.0
    confidence: float = 1.0


@dataclass
class VisualModerationResult:
    """Result of visual content moderation."""

    is_clean: bool = True
    severity: str = "none"
    flags: list[str] = field(default_factory=list)
    flashing_lights: bool = False
    skin_exposure_estimate: float = 0.0  # 0-1 rough estimate
    text_on_screen: bool = False
    dark_frames_ratio: float = 0.0
    brightness_avg: float = 0.5
    confidence: float = 1.0


@dataclass
class FullModerationResult:
    """Complete moderation result combining all checks."""

    text_result: ModerationResult = field(default_factory=ModerationResult)
    audio_result: AudioModerationResult = field(default_factory=AudioModerationResult)
    visual_result: VisualModerationResult = field(default_factory=VisualModerationResult)
    overall_severity: str = "none"
    overall_clean: bool = True
    content_rating: str = "G"
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ContentRating:
    """Content age rating suggestion."""

    rating: str = "G"  # G, PG, PG-13, R
    reason: str = ""
    confidence: float = 1.0


# ── Profanity Word Lists ──────────────────────────────────────

_PROFANITY_WORDS: set[str] = {
    # Common English profanity (mild to strong)
    "damn", "hell", "crap", "ass", "bastard", "bitch", "shit",
    "fuck", "dick", "piss", "cock", "whore", "slut", "cunt",
    "asshole", "bullshit", "goddamn", "motherfucker", "fucker",
    "fucking", "shitty", "dammit", "dumbass", "jackass",
    # Common euphemisms that may need flagging
    "wtf", "omg", "lmao", "stfu", "af", "bs",
}

# Mild profanity that might be acceptable in PG content
_MILD_PROFANITY: set[str] = {
    "damn", "hell", "crap", "ass", "darn", "heck",
    "wtf", "omg", "lmao", "bs",
}

# Strong profanity that escalates rating
_STRONG_PROFANITY: set[str] = {
    "fuck", "shit", "bitch", "cunt", "dick", "cock", "whore",
    "motherfucker", "asshole", "bullshit",
}

# ── Violence Keywords ─────────────────────────────────────────

_VIOLENCE_KEYWORDS: set[str] = {
    "kill", "murder", "attack", "assault", "shoot", "stab",
    "bomb", "explosion", "weapon", "gun", "knife", "blood",
    "violent", "violence", "fight", "beat", "torture", "war",
    "death", "dead", "die", "destroy", "threat", "danger",
    "harm", "hurt", "injury", "wound", "combat", "battle",
    "massacre", "genocide", "terrorist", "terrorism", "execution",
    "beheading", "suicide", "self-harm", "rape", "abuse",
}

# Context words that might make violence keywords acceptable
_VIOLENCE_CONTEXT_ACCEPTABLE: set[str] = {
    "game", "gaming", "movie", "film", "fiction", "story",
    "book", "novel", "video", "character", "play",
}

# ── Adult Content Keywords ────────────────────────────────────

_ADULT_KEYWORDS: set[str] = {
    "porn", "xxx", "nude", "naked", "sex", "sexual", "erotic",
    "orgasm", "fetish", "nsfw", "explicit", "adult", "hardcore",
    "strip", "striptease", "prostitution", "escort",
}

# ── Spam Detection Patterns ───────────────────────────────────

_SPAM_PATTERNS: list[re.Pattern] = [
    re.compile(r"(\b\w+\b)(\s+\1){5,}"),          # Same word 6+ times
    re.compile(r"(\$?\d+\.?\d*)\s*(%|dollars?|bucks?)", re.IGNORECASE),  # Money amounts
    re.compile(r"click\s+(here|below|link)", re.IGNORECASE),  # Click bait
    re.compile(r"free\s+(money|gift|prize|win)", re.IGNORECASE),  # Free stuff scams
    re.compile(r"(subscribe|follow|like).*(subscribe|follow|like).*(subscribe|follow|like)", re.IGNORECASE),
]

# ── Misleading Content Indicators ─────────────────────────────

_MISLEADING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"you\s+won't\s+believe", re.IGNORECASE), "clickbait_phrase"),
    (re.compile(r"shocking\s+(truth|revelation|secret)", re.IGNORECASE), "shocking_claim"),
    (re.compile(r"doctors\s+hate|doctors\s+don't\s+want", re.IGNORECASE), "conspiracy_claim"),
    (re.compile(r"one\s+weird\s+trick", re.IGNORECASE), "scam_pattern"),
    (re.compile(r"100%\s+(guaranteed|proven|effective)", re.IGNORECASE), "absolute_claim"),
    (re.compile(r"miracle\s+(cure|solution|remedy)", re.IGNORECASE), "miracle_claim"),
]


# ── Sensitivity Levels ────────────────────────────────────────

class SensitivityLevel:
    """Sensitivity level configuration for content moderation."""

    def __init__(self, level: str = "moderate") -> None:
        self.level = level
        if level == "strict":
            self.profanity_threshold = 0
            self.violence_threshold = 0
            self.adult_threshold = 0
            self.spam_threshold = 1
            self.skin_exposure_threshold = 0.1
            self.flashing_threshold = 0.05
        elif level == "lenient":
            self.profanity_threshold = 3
            self.violence_threshold = 2
            self.adult_threshold = 1
            self.spam_threshold = 5
            self.skin_exposure_threshold = 0.4
            self.flashing_threshold = 0.15
        else:  # moderate
            self.profanity_threshold = 1
            self.violence_threshold = 1
            self.adult_threshold = 0
            self.spam_threshold = 3
            self.skin_exposure_threshold = 0.25
            self.flashing_threshold = 0.10


# ── Content Moderator Class ───────────────────────────────────

class ContentModerator:
    """Local content moderation for policy compliance.

    Checks text, audio, and visual content for policy violations
    using local algorithms only (no API calls). Provides content
    ratings, warnings, and recommendations.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        sensitivity: str = "moderate",
    ) -> None:
        self.settings = settings or get_settings()
        self.sensitivity = SensitivityLevel(sensitivity)

    def check_text(self, text: str) -> ModerationResult:
        """Check text for policy violations.

        Runs profanity detection, spam detection, violence keyword
        detection, adult content detection, and misleading content
        indicator detection.

        Args:
            text: Text content to check.

        Returns:
            ModerationResult with flags and severity.
        """
        if not text or not text.strip():
            return ModerationResult()

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        word_set = set(words)

        flags: list[str] = []
        severity = "none"
        profanity = False
        spam = False
        violence = False
        adult = False
        misleading: list[str] = []

        # ── Profanity Detection ──────────────────────────
        profanity_count = 0
        strong_profanity_count = 0
        detected_profanity: list[str] = []

        for word in words:
            if word in _PROFANITY_WORDS:
                profanity_count += 1
                detected_profanity.append(word)
                if word in _STRONG_PROFANITY:
                    strong_profanity_count += 1

        if profanity_count > self.sensitivity.profanity_threshold:
            profanity = True
            flags.append(f"profanity: {profanity_count} instance(s) detected")
            if strong_profanity_count > 0:
                flags.append(f"strong_profanity: {strong_profanity_count} instance(s)")

        # ── Spam Detection ───────────────────────────────
        spam_matches = 0
        for pattern in _SPAM_PATTERNS:
            if pattern.search(text):
                spam_matches += 1

        # Also check for repetitive content
        if len(words) > 10:
            unique_ratio = len(word_set) / len(words)
            if unique_ratio < 0.3:
                spam_matches += 1
                flags.append("highly_repetitive: low lexical diversity")

        if spam_matches >= self.sensitivity.spam_threshold:
            spam = True
            flags.append(f"spam_indicators: {spam_matches} pattern(s) detected")

        # ── Violence Keyword Detection ───────────────────
        violence_words = word_set & _VIOLENCE_KEYWORDS
        # Check context for acceptable usage
        context_words = word_set & _VIOLENCE_CONTEXT_ACCEPTABLE
        adjusted_violence_count = max(0, len(violence_words) - len(context_words))

        if adjusted_violence_count > self.sensitivity.violence_threshold:
            violence = True
            flags.append(f"violence_keywords: {', '.join(violence_words)}")

        # ── Adult Content Detection ──────────────────────
        adult_words = word_set & _ADULT_KEYWORDS

        if len(adult_words) > self.sensitivity.adult_threshold:
            adult = True
            flags.append(f"adult_keywords: {', '.join(adult_words)}")

        # ── Misleading Content Indicators ────────────────
        for pattern, indicator in _MISLEADING_PATTERNS:
            if pattern.search(text):
                misleading.append(indicator)

        if misleading:
            flags.append(f"misleading_indicators: {', '.join(misleading)}")

        # ── Determine severity ───────────────────────────
        if adult or strong_profanity_count >= 3:
            severity = "high"
        elif violence or strong_profanity_count >= 1 or spam:
            severity = "medium"
        elif profanity or misleading:
            severity = "low"

        is_clean = severity in ("none", "low")

        # Calculate confidence based on how many signals agree
        signal_count = sum([profanity, spam, violence, adult, bool(misleading)])
        confidence = min(1.0, signal_count * 0.25 + 0.25) if not is_clean else 0.9

        return ModerationResult(
            is_clean=is_clean,
            severity=severity,
            flags=flags,
            profanity_detected=profanity,
            spam_detected=spam,
            violence_detected=violence,
            adult_detected=adult,
            misleading_indicators=misleading,
            confidence=round(confidence, 2),
        )

    def check_audio(self, video_path: Path) -> AudioModerationResult:
        """Check audio content for moderation signals.

        Analyzes audio for silence ratio, volume spikes, and speech presence.

        Args:
            video_path: Path to the video file.

        Returns:
            AudioModerationResult with flags.
        """
        if not video_path.exists():
            logger.error("Video file not found: %s", video_path)
            return AudioModerationResult()

        flags: list[str] = []

        try:
            from utils.ffmpeg_utils import extract_audio_samples, detect_silence, probe_video

            video_info = probe_video(video_path)
            duration = video_info.duration

            # Silence detection
            silence_regions = detect_silence(video_path)
            total_silence = sum(end - start for start, end in silence_regions)
            silence_ratio = total_silence / duration if duration > 0 else 0.0

            if silence_ratio > 0.7:
                flags.append(f"high_silence_ratio: {silence_ratio:.0%} of audio is silent")

            # Volume spike detection
            try:
                samples = extract_audio_samples(video_path, 1.0)
                volume_spikes = 0
                for i in range(1, len(samples)):
                    diff = samples[i] - samples[i - 1]
                    if diff > 30:  # Sudden 30dB increase
                        volume_spikes += 1

                if volume_spikes > 5:
                    flags.append(f"volume_spikes: {volume_spikes} sudden volume changes")
            except Exception:
                volume_spikes = 0

            # Speech detection (rough estimate from silence ratio)
            speech_detected = silence_ratio < 0.5

            # Music likelihood (very rough: non-speech, non-silent audio)
            music_likelihood = max(0.0, 1.0 - silence_ratio - 0.5) if not speech_detected else 0.3

            severity = "none"
            if volume_spikes > 10:
                severity = "medium"
                flags.append("loud_audio: many sudden volume changes")
            elif volume_spikes > 5:
                severity = "low"

            return AudioModerationResult(
                is_clean=severity in ("none", "low"),
                severity=severity,
                flags=flags,
                silence_ratio=round(silence_ratio, 3),
                volume_spikes=volume_spikes,
                speech_detected=speech_detected,
                music_likelihood=round(music_likelihood, 2),
                confidence=0.7,
            )

        except Exception as exc:
            logger.warning("Audio moderation failed: %s", exc)
            return AudioModerationResult(
                flags=[f"audio_analysis_failed: {exc}"],
                confidence=0.0,
            )

    def check_visual(self, video_path: Path) -> VisualModerationResult:
        """Check visual content using frame analysis.

        Analyzes frames for: flashing lights, skin exposure estimate,
        text on screen, and overall brightness.

        Args:
            video_path: Path to the video file.

        Returns:
            VisualModerationResult with flags.
        """
        if not video_path.exists():
            logger.error("Video file not found: %s", video_path)
            return VisualModerationResult()

        flags: list[str] = []

        try:
            from utils.ffmpeg_utils import get_video_thumbnail, probe_video
            import numpy as np

            video_info = probe_video(video_path)
            duration = video_info.duration

            # Sample 5 frames evenly
            frame_count = 5
            timestamps = [duration * (i + 1) / (frame_count + 1) for i in range(frame_count)]

            frames: list[np.ndarray] = []
            for ts in timestamps:
                tmp_path = Path(tempfile.mktemp(suffix=".jpg"))
                try:
                    get_video_thumbnail(video_path, ts, tmp_path)
                    try:
                        from PIL import Image
                        img = Image.open(tmp_path).convert("RGB")
                        frames.append(np.array(img, dtype=float))
                    except ImportError:
                        pass
                except Exception:
                    pass
                finally:
                    safe_delete(tmp_path)

            if not frames:
                return VisualModerationResult(flags=["frame_extraction_failed"])

            # ── Flashing Lights Detection ─────────────────
            flashing = False
            if len(frames) >= 2:
                brightness_changes: list[float] = []
                for i in range(1, len(frames)):
                    prev_brightness = np.mean(frames[i - 1])
                    curr_brightness = np.mean(frames[i])
                    change = abs(curr_brightness - prev_brightness)
                    brightness_changes.append(change)

                avg_change = sum(brightness_changes) / len(brightness_changes) if brightness_changes else 0
                if avg_change > 80:  # Large brightness changes between frames
                    flashing = True
                    flags.append("flashing_lights: rapid brightness changes detected")

            # ── Skin Exposure Estimate (very rough) ───────
            # Use simple color range heuristic for skin tones
            skin_ratios: list[float] = []
            for frame in frames:
                # Skin tone range in RGB (approximate)
                r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                skin_mask = (
                    (r > 60) & (r < 255) &
                    (g > 40) & (g < 230) &
                    (b > 20) & (b < 200) &
                    (r > g) & (r > b) &
                    (r - g > 15) &
                    (r - b > 15) &
                    (abs(r.astype(int) - g.astype(int)) < 100)
                )
                skin_ratio = np.sum(skin_mask) / skin_mask.size
                skin_ratios.append(float(skin_ratio))

            avg_skin = sum(skin_ratios) / len(skin_ratios) if skin_ratios else 0.0

            if avg_skin > self.sensitivity.skin_exposure_threshold:
                flags.append(f"high_skin_exposure: estimated {avg_skin:.0%}")

            # ── Text on Screen Detection ──────────────────
            # Rough estimate: areas with very high contrast edges
            text_on_screen = False
            for frame in frames:
                gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
                # Simple edge detection for text-like patterns
                if gray.shape[0] > 1 and gray.shape[1] > 1:
                    dx = np.abs(np.diff(gray, axis=1))
                    edge_ratio = np.sum(dx > 100) / dx.size
                    if edge_ratio > 0.15:
                        text_on_screen = True
                        break

            if text_on_screen:
                flags.append("text_on_screen: detected text overlay")

            # ── Dark Frames ───────────────────────────────
            dark_count = 0
            brightness_sum = 0.0
            for frame in frames:
                brightness = np.mean(frame)
                brightness_sum += brightness
                if brightness < 30:
                    dark_count += 1

            dark_ratio = dark_count / len(frames)
            avg_brightness = brightness_sum / len(frames) / 255.0

            if dark_ratio > 0.5:
                flags.append("dark_content: majority of frames are very dark")

            # ── Determine Severity ────────────────────────
            severity = "none"
            if avg_skin > self.sensitivity.skin_exposure_threshold:
                severity = "medium"
            elif flashing:
                severity = "medium"
            elif text_on_screen or dark_ratio > 0.3:
                severity = "low"

            return VisualModerationResult(
                is_clean=severity in ("none", "low"),
                severity=severity,
                flags=flags,
                flashing_lights=flashing,
                skin_exposure_estimate=round(avg_skin, 3),
                text_on_screen=text_on_screen,
                dark_frames_ratio=round(dark_ratio, 3),
                brightness_avg=round(avg_brightness, 3),
                confidence=0.6,
            )

        except Exception as exc:
            logger.warning("Visual moderation failed: %s", exc)
            return VisualModerationResult(
                flags=[f"visual_analysis_failed: {exc}"],
                confidence=0.0,
            )

    def moderate_full(
        self,
        video_path: Path,
        transcription: Optional[str] = None,
    ) -> FullModerationResult:
        """Complete moderation check combining text, audio, and visual analysis.

        Args:
            video_path: Path to the video file.
            transcription: Optional transcription text for text moderation.

        Returns:
            FullModerationResult with combined analysis.
        """
        text_result = ModerationResult()
        audio_result = AudioModerationResult()
        visual_result = VisualModerationResult()

        # Text moderation
        if transcription:
            text_result = self.check_text(transcription)

        # Audio moderation
        try:
            audio_result = self.check_audio(video_path)
        except Exception as exc:
            logger.warning("Audio moderation skipped: %s", exc)

        # Visual moderation
        try:
            visual_result = self.check_visual(video_path)
        except Exception as exc:
            logger.warning("Visual moderation skipped: %s", exc)

        # Determine overall severity
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        max_severity = max(
            severity_order.get(text_result.severity, 0),
            severity_order.get(audio_result.severity, 0),
            severity_order.get(visual_result.severity, 0),
        )
        severity_names = {0: "none", 1: "low", 2: "medium", 3: "high"}
        overall_severity = severity_names.get(max_severity, "none")

        # Determine content rating
        rating = self.get_content_rating(video_path, transcription)

        # Generate warnings
        warnings: list[str] = []
        all_flags = text_result.flags + audio_result.flags + visual_result.flags
        for flag in all_flags:
            warnings.append(flag)

        # Generate recommendations
        recommendations: list[str] = []
        if text_result.profanity_detected:
            recommendations.append("Consider editing or muting profanity for broader audience")
        if text_result.violence_detected:
            recommendations.append("Violence-related keywords detected - review content for platform compliance")
        if text_result.adult_detected:
            recommendations.append("Adult content keywords detected - may require age restriction")
        if visual_result.flashing_lights:
            recommendations.append("Flashing lights detected - add epilepsy warning")
        if visual_result.skin_exposure_estimate > self.sensitivity.skin_exposure_threshold:
            recommendations.append("High skin exposure estimated - review for platform guidelines")
        if audio_result.volume_spikes > 5:
            recommendations.append("Audio has sudden volume changes - consider normalization")
        if text_result.spam_detected:
            recommendations.append("Spam-like patterns detected - review description and captions")

        result = FullModerationResult(
            text_result=text_result,
            audio_result=audio_result,
            visual_result=visual_result,
            overall_severity=overall_severity,
            overall_clean=overall_severity in ("none", "low"),
            content_rating=rating.rating,
            warnings=warnings,
            recommendations=recommendations,
        )

        logger.info(
            "Moderation complete: severity=%s, rating=%s, %d warnings, %d recommendations",
            overall_severity, rating.rating, len(warnings), len(recommendations),
        )

        return result

    def get_content_rating(
        self,
        video_path: Path,
        transcription: Optional[str] = None,
    ) -> ContentRating:
        """Suggest a content age rating based on simple heuristics.

        Ratings: G (General), PG (Parental Guidance), PG-13, R (Restricted).

        Args:
            video_path: Path to the video file.
            transcription: Optional transcription text.

        Returns:
            ContentRating with suggested rating and reason.
        """
        reasons: list[str] = []
        rating_score = 0  # 0=G, 1=PG, 2=PG-13, 3=R

        # Check transcription if available
        if transcription:
            text_result = self.check_text(transcription)

            if text_result.adult_detected:
                rating_score = max(rating_score, 3)
                reasons.append("adult content detected")

            if text_result.violence_detected:
                rating_score = max(rating_score, 2)
                reasons.append("violence-related content")

            if text_result.profanity_detected:
                # Check for strong vs mild profanity
                text_lower = transcription.lower()
                has_strong = any(w in text_lower for w in _STRONG_PROFANITY)
                if has_strong:
                    rating_score = max(rating_score, 3)
                    reasons.append("strong profanity")
                else:
                    rating_score = max(rating_score, 1)
                    reasons.append("mild profanity")

            if text_result.misleading_indicators:
                rating_score = max(rating_score, 1)
                reasons.append("potentially misleading claims")

        # Check visual content
        try:
            visual_result = self.check_visual(video_path)

            if visual_result.skin_exposure_estimate > 0.3:
                rating_score = max(rating_score, 2)
                reasons.append("revealing content")

            if visual_result.flashing_lights:
                rating_score = max(rating_score, 1)
                reasons.append("flashing lights (epilepsy risk)")
        except Exception:
            pass

        rating_map = {0: "G", 1: "PG", 2: "PG-13", 3: "R"}
        rating_name = rating_map.get(rating_score, "G")
        reason = "; ".join(reasons) if reasons else "No concerning content detected"

        return ContentRating(
            rating=rating_name,
            reason=reason,
            confidence=0.7 if reasons else 0.9,
        )

    def generate_content_warnings(
        self,
        video_path: Path,
        transcription: Optional[str] = None,
    ) -> list[str]:
        """Generate content warnings for the video.

        Args:
            video_path: Path to the video file.
            transcription: Optional transcription text.

        Returns:
            List of warning strings suitable for display.
        """
        warnings: list[str] = []

        if transcription:
            text_result = self.check_text(transcription)

            if text_result.profanity_detected:
                warnings.append("Contains profanity")
            if text_result.violence_detected:
                warnings.append("Contains violence-related content")
            if text_result.adult_detected:
                warnings.append("Contains adult content")
            if text_result.spam_detected:
                warnings.append("Contains repetitive/promotional content")

        try:
            visual_result = self.check_visual(video_path)
            if visual_result.flashing_lights:
                warnings.append("Flashing lights - may cause seizures")
            if visual_result.skin_exposure_estimate > 0.3:
                warnings.append("May contain revealing content")
        except Exception:
            pass

        return warnings
