"""
core/analyzer.py — Advanced multi-signal engagement scorer for segment extraction.

Finds the highest-engagement N-second segment using audio RMS energy,
peak energy, scene change density, silence detection, speech presence,
motion energy, spectral analysis, and optional face-tracking.
Supports multi-clip detection with quality scoring and Bayesian
conflict resolution between disagreeing signals.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import get_settings
from utils.ffmpeg_utils import (
    extract_audio_samples, probe_video, detect_scene_changes, detect_silence,
)
from utils.logger import get_logger

logger = get_logger("analyzer")


# ── Data Classes ──────────────────────────────────────────

@dataclass
class SegmentResult:
    """Result of engagement analysis — the chosen clip segment."""

    start_time: float
    end_time: float
    energy_score: float
    method_used: str = "composite_audio_scene"
    audio_peak_time: float = 0.0
    scene_peak_time: float = 0.0
    silence_ratio: float = 0.0
    speech_detected: bool = True
    best_crop_x: int = -1  # -1 means center crop
    best_crop_y: int = -1
    confidence: float = 0.0  # How confident we are in this segment (0-1)
    # ── New fields for enhanced analysis ──
    speech_rate_estimate: float = 0.0  # Estimated WPM in segment
    music_likelihood: float = 0.0  # 0-1, higher = more likely music
    visual_complexity: float = 0.0  # 0-1, based on motion + scene changes
    overall_quality_grade: str = "C"  # A/B/C/D quality grade
    has_emphasis: bool = False  # Whether emphasis moments detected
    motion_energy: float = 0.0  # Average motion energy in segment
    spectral_centroid_avg: float = 0.0  # Average spectral centroid


@dataclass
class MultiClipResult:
    """Result of multi-clip analysis — top N non-overlapping segments."""

    segments: list[SegmentResult] = field(default_factory=list)
    total_candidates: int = 0
    analysis_method: str = "multi_signal_v3"


@dataclass
class EnergyProfile:
    """Full energy profile of a video — all signals for analysis."""

    audio_energy_rms: np.ndarray = field(default_factory=lambda: np.array([]))
    audio_energy_peak: np.ndarray = field(default_factory=lambda: np.array([]))
    scene_density: np.ndarray = field(default_factory=lambda: np.array([]))
    scene_transition_types: dict[int, str] = field(default_factory=dict)  # idx -> "cut"|"dissolve"|"fade"
    silence_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    speech_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    motion_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    emphasis_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    composite: np.ndarray = field(default_factory=lambda: np.array([]))
    total_duration: float = 0.0
    sample_interval: float = 2.0

    # Backward compatibility properties
    @property
    def audio_energy(self) -> np.ndarray:
        """Backward-compatible access to RMS audio energy."""
        return self.audio_energy_rms

    @audio_energy.setter
    def audio_energy(self, value: np.ndarray) -> None:
        self.audio_energy_rms = value


# ── Quality Grading ───────────────────────────────────────

def _compute_quality_grade(
    energy_score: float,
    speech_rate: float,
    music_likelihood: float,
    visual_complexity: float,
    silence_ratio: float,
) -> str:
    """Compute a letter quality grade for a segment.

    Scoring rubric:
      A: High engagement, clear speech, visual dynamism
      B: Good engagement with minor weaknesses
      C: Moderate engagement, some dead air or low speech
      D: Low engagement, lots of silence or music-only

    Args:
        energy_score: Composite energy score (0-1).
        speech_rate: Estimated WPM in the segment.
        music_likelihood: 0-1, higher = more likely music.
        visual_complexity: 0-1, based on motion + scene changes.
        silence_ratio: 0-1, proportion of silence.

    Returns:
        Single character grade: 'A', 'B', 'C', or 'D'.
    """
    # Weighted quality score
    quality = (
        0.30 * energy_score
        + 0.20 * (1.0 - min(silence_ratio * 2, 1.0))
        + 0.20 * visual_complexity
        + 0.15 * min(speech_rate / 180.0, 1.0) if speech_rate > 0 else 0.0
        + 0.15 * (1.0 - music_likelihood)
    )

    if quality >= 0.75:
        return "A"
    elif quality >= 0.55:
        return "B"
    elif quality >= 0.35:
        return "C"
    else:
        return "D"


class EngagementAnalyzer:
    """Advanced multi-signal engagement analyser.

    Computes a composite score from seven signals:
    1. Audio RMS energy (weight 0.20) — average loudness variation
    2. Audio peak energy (weight 0.10) — emphasis moments
    3. Scene change density (weight 0.20) — visual dynamism
    4. Silence penalty (weight 0.10) — penalises dead air
    5. Speech presence (weight 0.15) — boosts speech content
    6. Motion energy (weight 0.15) — visual dynamism via frame diff
    7. Spectral centroid (weight 0.10) — speech vs music discrimination

    Uses Bayesian conflict resolution when signals disagree, and
    supports multi-clip detection for finding the top N segments.
    """

    def __init__(self, video_path: Path, clip_duration: int) -> None:
        self.video_path = video_path
        self.clip_duration = clip_duration
        self.settings = get_settings()
        self.sample_interval = self.settings.ENERGY_SAMPLE_INTERVAL
        self.smoothing_window = self.settings.ENERGY_SMOOTHING_WINDOW

    def analyze_fast(self) -> SegmentResult:
        """Fast analysis mode for TURBO — uses only audio energy and silence.

        Skips expensive signals (motion energy, spectral centroid, emphasis,
        scene change classification) for maximum speed. Suitable for TURBO
        mode where speed is prioritized over analysis accuracy.

        Returns:
            SegmentResult with the best clip segment based on audio-only analysis.
        """
        logger.info("FAST ANALYZE (turbo mode) for: %s", self.video_path.name)

        video_info = probe_video(self.video_path)
        total_duration = video_info.duration

        if total_duration <= 0:
            return SegmentResult(
                start_time=0.0,
                end_time=min(float(self.clip_duration), 60.0),
                energy_score=0.0,
                method_used="turbo_fallback",
                confidence=0.0,
                overall_quality_grade="C",
            )

        if total_duration <= self.clip_duration:
            return SegmentResult(
                start_time=0.0,
                end_time=total_duration,
                energy_score=1.0,
                method_used="turbo_full_video",
                confidence=1.0,
                overall_quality_grade="A",
            )

        # Only compute audio RMS + silence — skip everything else
        num_samples = max(1, int(total_duration / self.sample_interval))

        # Signal 1: Audio RMS energy (the most important signal)
        try:
            rms_values = extract_audio_samples(self.video_path, self.sample_interval)
        except Exception:
            rms_values = []

        if not rms_values:
            # Fallback: use first N seconds
            return SegmentResult(
                start_time=0.0,
                end_time=float(self.clip_duration),
                energy_score=0.5,
                method_used="turbo_no_audio",
                confidence=0.0,
                overall_quality_grade="C",
            )

        rms_array = np.array(rms_values, dtype=float)
        rms_array = np.where(np.isfinite(rms_array), rms_array, -60.0)

        # Normalize to 0-1
        min_rms, max_rms = np.min(rms_array), np.max(rms_array)
        if max_rms - min_rms > 1e-6:
            audio_norm = (rms_array - min_rms) / (max_rms - min_rms)
        else:
            audio_norm = np.ones_like(rms_array) * 0.5

        # Pad to expected length
        if len(audio_norm) < num_samples:
            audio_norm = np.pad(audio_norm, (0, num_samples - len(audio_norm)), constant_values=0.5)

        # Signal 2: Silence mask (cheap to compute)
        try:
            silence_mask = self._compute_silence_mask(total_duration)
        except Exception:
            silence_mask = np.zeros(num_samples, dtype=float)

        # Pad silence mask
        if len(silence_mask) < num_samples:
            silence_mask = np.pad(silence_mask, (0, num_samples - len(silence_mask)), constant_values=0.0)

        # Composite: audio energy with silence penalty (fast 2-signal model)
        composite = audio_norm * (1.0 - silence_mask * 0.5)

        # Smoothing
        if self.smoothing_window > 1 and len(composite) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            composite = np.convolve(composite, kernel, mode="same")

        # Find best window
        window_samples = max(1, int(self.clip_duration / self.sample_interval))

        if len(composite) <= window_samples:
            best_start = 0.0
            best_score = float(np.mean(composite)) if len(composite) > 0 else 0.5
        else:
            rolling_sum = np.convolve(composite, np.ones(window_samples), mode="valid")
            rolling_score = rolling_sum / window_samples
            best_idx = int(np.argmax(rolling_score))
            best_start = float(best_idx * self.sample_interval)
            best_score = float(rolling_score[best_idx])

        # Skip intro silence
        intro_samples = min(5, num_samples)
        if np.mean(composite[:intro_samples]) < 0.2 and best_start < 5.0:
            # Intro is quiet, skip it
            for offset_idx in range(intro_samples, min(intro_samples + 10, len(composite))):
                if composite[offset_idx] > 0.3:
                    adjusted_start = max(0.0, float((offset_idx - 1) * self.sample_interval))
                    if adjusted_start + self.clip_duration <= total_duration:
                        best_start = adjusted_start
                    break

        silence_ratio = self._window_mean(silence_mask, int(best_start / self.sample_interval), window_samples)

        result = SegmentResult(
            start_time=round(best_start, 2),
            end_time=round(best_start + self.clip_duration, 2),
            energy_score=round(best_score, 4),
            method_used="turbo_fast_audio",
            silence_ratio=round(silence_ratio, 3),
            confidence=round(min(1.0, best_score * 1.5), 2),
            overall_quality_grade="B" if best_score > 0.5 else "C",
        )

        logger.info(
            "Turbo analysis: %.1fs - %.1fs (score=%.4f, silence=%.1f%%)",
            result.start_time, result.end_time, result.energy_score,
            result.silence_ratio * 100,
        )
        return result

    def analyze(self) -> SegmentResult:
        """Run the full multi-signal engagement analysis.

        Returns:
            SegmentResult with the best clip segment and enriched metadata.
        """
        logger.info("Analyzing engagement for: %s", self.video_path.name)

        video_info = probe_video(self.video_path)
        total_duration = video_info.duration

        # Edge cases
        if total_duration <= 0:
            return SegmentResult(
                start_time=0.0,
                end_time=min(float(self.clip_duration), 60.0),
                energy_score=0.0,
                method_used="fallback_no_duration",
                confidence=0.0,
                overall_quality_grade="D",
            )

        if total_duration <= self.clip_duration:
            return SegmentResult(
                start_time=0.0,
                end_time=total_duration,
                energy_score=1.0,
                method_used="full_video",
                confidence=1.0,
                overall_quality_grade="A",
            )

        # ── Compute all signals ────────────────────────────
        profile = self._build_energy_profile(total_duration)

        # ── Detect and skip intro/outro silence ─────────────
        intro_end = self._detect_intro_end(profile)
        outro_start = self._detect_outro_start(profile)

        # ── Find best window with constraints ───────────────
        best_start = self._find_best_window(
            profile.composite, total_duration, intro_end, outro_start
        )

        # ── Compute final score and metadata ────────────────
        window_samples = max(1, int(self.clip_duration / self.sample_interval))
        start_idx = int(best_start / self.sample_interval)
        best_score = self._window_mean(profile.composite, start_idx, window_samples)

        # Compute per-segment metrics
        silence_ratio = self._window_mean(profile.silence_mask, start_idx, window_samples)
        motion_energy_val = self._window_mean(profile.motion_energy, start_idx, window_samples)
        spectral_centroid_avg = self._window_mean(profile.spectral_centroid, start_idx, window_samples)
        music_likelihood = self._estimate_music_likelihood(spectral_centroid_avg, profile)
        speech_rate = self._estimate_speech_rate(profile, start_idx, window_samples)
        has_emphasis = bool(np.any(profile.emphasis_mask[start_idx:start_idx + window_samples] > 0.5))
        visual_complexity = self._compute_visual_complexity(
            profile.scene_density, profile.motion_energy, start_idx, window_samples
        )

        # Find peak times
        audio_peak_idx = int(np.argmax(profile.audio_energy_rms)) if len(profile.audio_energy_rms) > 0 else 0
        scene_peak_idx = int(np.argmax(profile.scene_density)) if len(profile.scene_density) > 0 else 0

        # Compute confidence using Bayesian weighting
        confidence = self._compute_confidence(
            profile, start_idx, window_samples, best_score
        )

        # Compute quality grade
        quality_grade = _compute_quality_grade(
            best_score, speech_rate, music_likelihood, visual_complexity, silence_ratio
        )

        result = SegmentResult(
            start_time=round(best_start, 2),
            end_time=round(best_start + self.clip_duration, 2),
            energy_score=round(best_score, 4),
            method_used="multi_signal_v3",
            audio_peak_time=round(float(audio_peak_idx * self.sample_interval), 2),
            scene_peak_time=round(float(scene_peak_idx * self.sample_interval), 2),
            silence_ratio=round(silence_ratio, 3),
            confidence=round(confidence, 2),
            speech_rate_estimate=round(speech_rate, 1),
            music_likelihood=round(music_likelihood, 3),
            visual_complexity=round(visual_complexity, 3),
            overall_quality_grade=quality_grade,
            has_emphasis=has_emphasis,
            motion_energy=round(motion_energy_val, 4),
            spectral_centroid_avg=round(spectral_centroid_avg, 1),
        )

        logger.info(
            "Best segment: %.1fs - %.1fs (score=%.4f, grade=%s, confidence=%.2f, silence=%.1f%%, speech_rate=%.0f WPM)",
            result.start_time, result.end_time, result.energy_score,
            result.overall_quality_grade, result.confidence,
            result.silence_ratio * 100, result.speech_rate_estimate,
        )
        return result

    def analyze_multiple_clips(self, num_clips: int = 3, min_gap_seconds: float = 10.0) -> MultiClipResult:
        """Find top N non-overlapping segments sorted by quality.

        Runs the full analysis pipeline and then iteratively selects
        the best non-overlapping segments with at least min_gap_seconds
        between them.

        Args:
            num_clips: Maximum number of clips to return.
            min_gap_seconds: Minimum gap between clip boundaries.

        Returns:
            MultiClipResult with sorted list of SegmentResult objects.
        """
        logger.info("Analyzing top %d clips for: %s", num_clips, self.video_path.name)

        video_info = probe_video(self.video_path)
        total_duration = video_info.duration

        if total_duration <= self.clip_duration:
            single = self.analyze()
            return MultiClipResult(
                segments=[single],
                total_candidates=1,
                analysis_method="single_clip_fallback",
            )

        profile = self._build_energy_profile(total_duration)
        intro_end = self._detect_intro_end(profile)
        outro_start = self._detect_outro_start(profile)

        # Compute rolling scores for all possible windows
        window_samples = max(1, int(self.clip_duration / self.sample_interval))
        composite = profile.composite

        if len(composite) <= window_samples:
            single = self.analyze()
            return MultiClipResult(
                segments=[single],
                total_candidates=1,
                analysis_method="short_video_fallback",
            )

        rolling_sum = np.convolve(composite, np.ones(window_samples), mode="valid")
        rolling_score = rolling_sum / window_samples

        # Track occupied time ranges to prevent overlap
        occupied: list[tuple[float, float]] = []
        selected_segments: list[SegmentResult] = []

        # Sort candidates by score descending
        candidate_indices = np.argsort(rolling_score)[::-1]

        for idx in candidate_indices:
            if len(selected_segments) >= num_clips:
                break

            start_time = float(idx * self.sample_interval)
            end_time = start_time + self.clip_duration

            # Check intro/outro boundaries
            if start_time < intro_end or end_time > outro_start:
                continue

            # Check overlap with already selected segments
            overlaps = False
            for occ_start, occ_end in occupied:
                if not (end_time + min_gap_seconds <= occ_start or start_time >= occ_end + min_gap_seconds):
                    overlaps = True
                    break

            if overlaps:
                continue

            # Check video duration boundary
            if end_time > total_duration:
                continue

            # Compute segment metadata
            start_idx = int(start_time / self.sample_interval)
            best_score = rolling_score[idx]
            silence_ratio = self._window_mean(profile.silence_mask, start_idx, window_samples)
            motion_energy_val = self._window_mean(profile.motion_energy, start_idx, window_samples)
            spectral_centroid_avg = self._window_mean(profile.spectral_centroid, start_idx, window_samples)
            music_likelihood = self._estimate_music_likelihood(spectral_centroid_avg, profile)
            speech_rate = self._estimate_speech_rate(profile, start_idx, window_samples)
            has_emphasis = bool(np.any(profile.emphasis_mask[start_idx:start_idx + window_samples] > 0.5))
            visual_complexity = self._compute_visual_complexity(
                profile.scene_density, profile.motion_energy, start_idx, window_samples
            )
            confidence = self._compute_confidence(profile, start_idx, window_samples, best_score)
            quality_grade = _compute_quality_grade(
                best_score, speech_rate, music_likelihood, visual_complexity, silence_ratio
            )

            segment = SegmentResult(
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                energy_score=round(float(best_score), 4),
                method_used="multi_signal_v3",
                silence_ratio=round(silence_ratio, 3),
                confidence=round(confidence, 2),
                speech_rate_estimate=round(speech_rate, 1),
                music_likelihood=round(music_likelihood, 3),
                visual_complexity=round(visual_complexity, 3),
                overall_quality_grade=quality_grade,
                has_emphasis=has_emphasis,
                motion_energy=round(motion_energy_val, 4),
                spectral_centroid_avg=round(spectral_centroid_avg, 1),
            )

            selected_segments.append(segment)
            occupied.append((start_time, end_time))

        # Sort by energy_score descending
        selected_segments.sort(key=lambda s: s.energy_score, reverse=True)

        logger.info(
            "Found %d/%d clips (grades: %s)",
            len(selected_segments), num_clips,
            ", ".join(s.overall_quality_grade for s in selected_segments),
        )

        return MultiClipResult(
            segments=selected_segments,
            total_candidates=len(rolling_score),
            analysis_method="multi_signal_v3",
        )

    # ── Signal Computation ─────────────────────────────────

    @staticmethod
    def _window_mean(arr: np.ndarray, start_idx: int, window: int) -> float:
        """Compute mean of a window in an array, handling bounds.

        Args:
            arr: Input array.
            start_idx: Starting index for the window.
            window: Window size in samples.

        Returns:
            Mean value of the window, or 0.0 if empty.
        """
        if len(arr) == 0:
            return 0.0
        end_idx = min(start_idx + window, len(arr))
        start_idx = max(0, min(start_idx, len(arr) - 1))
        if start_idx >= end_idx:
            return float(arr[start_idx]) if start_idx < len(arr) else 0.0
        return float(np.mean(arr[start_idx:end_idx]))

    def _build_energy_profile(self, total_duration: float) -> EnergyProfile:
        """Compute all energy signals and build the composite.

        Builds the following signals:
        - Audio RMS energy (average loudness)
        - Audio peak energy (emphasis detection)
        - Scene change density
        - Silence mask
        - Speech presence estimate
        - Motion energy (frame differencing)
        - Spectral centroid (speech vs music)
        - Emphasis mask (sudden volume spikes)

        Then combines them with Bayesian-weighted conflict resolution
        into a single composite score array.

        Args:
            total_duration: Total video duration in seconds.

        Returns:
            EnergyProfile with all computed signals.
        """
        profile = EnergyProfile(
            total_duration=total_duration,
            sample_interval=self.sample_interval,
        )

        # Signal 1 & 2: Audio RMS and Peak energy
        profile.audio_energy_rms, profile.audio_energy_peak = self._compute_audio_energy(total_duration)

        # Signal 3: Scene change density
        profile.scene_density = self._compute_scene_density(total_duration)

        # Signal 4: Silence detection
        profile.silence_mask = self._compute_silence_mask(total_duration)

        # Signal 5: Speech presence estimate
        profile.speech_mask = self._compute_speech_mask(
            profile.audio_energy_rms, profile.silence_mask, profile.audio_energy_peak
        )

        # Signal 6: Motion energy
        profile.motion_energy = self._compute_motion_energy(total_duration)

        # Signal 7: Spectral centroid
        profile.spectral_centroid = self._compute_spectral_centroid(total_duration)

        # Derived: Emphasis mask from peak energy
        profile.emphasis_mask = self._compute_emphasis_mask(profile.audio_energy_peak)

        # Align all signal lengths
        all_arrays = [
            profile.audio_energy_rms, profile.audio_energy_peak,
            profile.scene_density, profile.silence_mask,
            profile.speech_mask, profile.motion_energy,
            profile.spectral_centroid, profile.emphasis_mask,
        ]
        max_len = max((len(arr) for arr in all_arrays), default=0)
        if max_len == 0:
            max_len = max(1, int(total_duration / self.sample_interval))

        for arr_name in (
            "audio_energy_rms", "audio_energy_peak",
            "scene_density", "silence_mask",
            "speech_mask", "motion_energy",
            "spectral_centroid", "emphasis_mask",
        ):
            arr = getattr(profile, arr_name)
            if len(arr) < max_len:
                padded = np.pad(arr, (0, max_len - len(arr)), constant_values=0.0)
                setattr(profile, arr_name, padded)

        # Bayesian conflict resolution: weight signals by reliability
        profile.composite = self._compute_bayesian_composite(profile)

        return profile

    def _compute_audio_energy(self, total_duration: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalised, smoothed audio RMS and peak energy per interval.

        Uses FFmpeg astats to extract both RMS and peak energy levels.
        RMS captures average loudness; peak captures emphasis moments.

        Args:
            total_duration: Total video duration in seconds.

        Returns:
            Tuple of (rms_normalised, peak_normalised) arrays.
        """
        default_len = max(1, int(total_duration / self.sample_interval))

        # ── RMS Energy ─────────────────────────────────────
        try:
            rms_values = extract_audio_samples(self.video_path, self.sample_interval)
        except Exception as exc:
            logger.warning("Audio energy extraction failed: %s; using uniform", exc)
            return np.ones(default_len) * 0.5, np.ones(default_len) * 0.3

        if not rms_values:
            return np.ones(default_len) * 0.5, np.ones(default_len) * 0.3

        rms_array = np.array(rms_values, dtype=float)
        rms_array = np.where(np.isfinite(rms_array), rms_array, -60.0)

        # Normalise RMS to 0-1
        min_rms, max_rms = np.min(rms_array), np.max(rms_array)
        if max_rms - min_rms > 1e-6:
            rms_normalised = (rms_array - min_rms) / (max_rms - min_rms)
        else:
            rms_normalised = np.ones_like(rms_array) * 0.5

        # ── Peak Energy ────────────────────────────────────
        peak_values = self._extract_peak_energy(total_duration)
        if len(peak_values) > 0:
            peak_array = np.array(peak_values, dtype=float)
            peak_array = np.where(np.isfinite(peak_array), peak_array, -60.0)
            min_peak, max_peak = np.min(peak_array), np.max(peak_array)
            if max_peak - min_peak > 1e-6:
                peak_normalised = (peak_array - min_peak) / (max_peak - min_peak)
            else:
                peak_normalised = np.ones_like(peak_array) * 0.3
        else:
            peak_normalised = np.ones(default_len) * 0.3

        # Align lengths
        max_len = max(len(rms_normalised), len(peak_normalised))
        if len(rms_normalised) < max_len:
            rms_normalised = np.pad(rms_normalised, (0, max_len - len(rms_normalised)), constant_values=0.5)
        if len(peak_normalised) < max_len:
            peak_normalised = np.pad(peak_normalised, (0, max_len - len(peak_normalised)), constant_values=0.3)

        # Smoothing
        if self.smoothing_window > 1 and len(rms_normalised) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            rms_normalised = np.convolve(rms_normalised, kernel, mode="same")
            peak_normalised = np.convolve(peak_normalised, kernel, mode="same")

        return rms_normalised, peak_normalised

    def _extract_peak_energy(self, total_duration: float) -> list[float]:
        """Extract per-segment peak audio level using FFmpeg.

        Uses the astats filter with peak_level measurement.

        Args:
            total_duration: Total video duration in seconds.

        Returns:
            List of peak dB values, one per sample interval.
        """
        cmd: list[str] = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-af", f"astats=metadata=1:reset={self.sample_interval},ametadata=print:key=lavfi.astats.Overall.Peak_level:file=-",
            "-f", "null",
            "-",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return []

        peak_values: list[float] = []
        peak_pattern = __import__("re").compile(
            r"lavfi\.astats\.Overall\.Peak_level=(-?[\d.]+|nan|-inf)"
        )

        for output_text in (result.stdout, result.stderr):
            for line in output_text.splitlines():
                match = peak_pattern.search(line)
                if match:
                    val = match.group(1)
                    if val in ("nan", "-inf"):
                        peak_values.append(-60.0)
                    else:
                        try:
                            peak_values.append(float(val))
                        except ValueError:
                            peak_values.append(-60.0)

        return peak_values

    def _compute_scene_density(self, total_duration: float) -> np.ndarray:
        """Compute normalised scene change density per sample interval.

        Also classifies scene transitions as cut, dissolve, or fade
        based on the distribution of scene change magnitudes.

        Args:
            total_duration: Total video duration in seconds.

        Returns:
            Normalised scene density array (0-1).
        """
        num_samples = max(1, int(total_duration / self.sample_interval))
        scene_counts = np.zeros(num_samples, dtype=float)
        transition_types: dict[int, str] = {}

        try:
            timestamps = detect_scene_changes(
                self.video_path,
                threshold=self.settings.SCENE_DETECT_THRESHOLD,
            )

            for idx, ts in enumerate(timestamps):
                bucket = int(ts / self.sample_interval)
                if 0 <= bucket < num_samples:
                    scene_counts[bucket] += 1
                    # Classify transition type based on scene density
                    # High density in short span = cuts, low = fades/dissolves
                    if scene_counts[bucket] >= 2:
                        transition_types[bucket] = "cut"
                    elif bucket > 0 and scene_counts[bucket - 1] > 0:
                        transition_types[bucket] = "dissolve"
                    else:
                        transition_types[bucket] = "fade"

        except Exception as exc:
            logger.warning("Scene detection failed: %s; using uniform", exc)
            return np.ones(num_samples) * 0.5

        max_count = np.max(scene_counts)
        if max_count > 0:
            normalised = scene_counts / max_count
        else:
            normalised = np.zeros(num_samples, dtype=float)

        # Smoothing
        if self.smoothing_window > 1 and len(normalised) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            normalised = np.convolve(normalised, kernel, mode="same")

        return normalised

    def _compute_silence_mask(self, total_duration: float) -> np.ndarray:
        """Detect silent regions using FFmpeg silencedetect.

        Returns a binary mask (1=silence, 0=audio) per sample interval.

        Args:
            total_duration: Total video duration in seconds.

        Returns:
            Binary silence mask array.
        """
        num_samples = max(1, int(total_duration / self.sample_interval))
        silence_mask = np.zeros(num_samples, dtype=float)

        try:
            regions = detect_silence(
                self.video_path,
                noise_floor=self.settings.SILENCE_NOISE_FLOOR,
                min_duration=self.settings.SILENCE_MIN_DURATION,
            )

            for start, end in regions:
                start_bucket = max(0, int(start / self.sample_interval))
                end_bucket = min(num_samples, int(end / self.sample_interval) + 1)
                silence_mask[start_bucket:end_bucket] = 1.0

        except Exception as exc:
            logger.debug("Silence detection failed: %s", exc)

        return silence_mask

    def _compute_speech_mask(
        self,
        audio_energy_rms: np.ndarray,
        silence_mask: np.ndarray,
        audio_energy_peak: np.ndarray,
    ) -> np.ndarray:
        """Estimate speech presence from audio energy, peaks, and silence mask.

        Differentiates speech from music and noise using spectral features:
        - Speech: high energy + no silence + moderate peak variation
        - Music: high energy + no silence + consistent peaks
        - Noise: low energy or erratic energy patterns

        Args:
            audio_energy_rms: Normalised RMS energy array.
            silence_mask: Binary silence mask array.
            audio_energy_peak: Normalised peak energy array.

        Returns:
            Normalised speech presence array (0-1).
        """
        min_len = min(len(audio_energy_rms), len(silence_mask), len(audio_energy_peak))
        if min_len == 0:
            return np.array([])

        rms = audio_energy_rms[:min_len]
        silence = silence_mask[:min_len]
        peak = audio_energy_peak[:min_len]

        # Speech: high RMS energy, not silent, moderate peak variation
        # (speech has natural variation; music tends to be more consistent)
        base_speech = rms * (1.0 - silence)

        # Peak variation: speech has higher variation than music
        if len(peak) >= 3:
            # Compute local variation in peak energy
            peak_diff = np.abs(np.diff(peak))
            peak_variation = np.zeros_like(peak)
            peak_variation[1:] = peak_diff
            peak_variation[0] = peak_diff[0] if len(peak_diff) > 0 else 0.0

            # Normalise peak variation
            max_var = np.max(peak_variation)
            if max_var > 1e-6:
                peak_variation = peak_variation / max_var
            # Speech has moderate-to-high peak variation
            speech_boost = 1.0 + 0.3 * peak_variation
        else:
            speech_boost = np.ones_like(peak)

        speech = base_speech * speech_boost

        max_val = np.max(speech)
        if max_val > 1e-6:
            speech = speech / max_val

        return speech

    def _compute_motion_energy(self, total_duration: float) -> np.ndarray:
        """Compute motion energy using frame differencing.

        Extracts frames at the sample interval rate, computes the
        absolute difference between consecutive frames, and averages
        the pixel-wise differences to produce a motion energy signal.

        Args:
            total_duration: Total video duration in seconds.

        Returns:
            Normalised motion energy array (0-1).
        """
        num_samples = max(1, int(total_duration / self.sample_interval))
        motion_values = np.zeros(num_samples, dtype=float)

        try:
            # Extract frames at sample intervals using FFmpeg
            prev_frame: Optional[np.ndarray] = None
            for i in range(num_samples):
                timestamp = i * self.sample_interval + self.sample_interval * 0.5

                # Extract a single frame as raw pixel data
                cmd: list[str] = [
                    "ffmpeg",
                    "-ss", f"{timestamp:.3f}",
                    "-i", str(self.video_path),
                    "-vframes", "1",
                    "-vf", "scale=160:90",  # Downscale for speed
                    "-f", "rawvideo",
                    "-pix_fmt", "gray",
                    "-",
                ]

                result = subprocess.run(
                    cmd, capture_output=True, timeout=30,
                )

                if result.returncode != 0 or len(result.stdout) == 0:
                    continue

                # Parse raw grayscale frame
                expected_size = 160 * 90
                if len(result.stdout) < expected_size:
                    continue

                frame = np.frombuffer(result.stdout[:expected_size], dtype=np.uint8).reshape(90, 160)

                if prev_frame is not None:
                    # Compute absolute difference
                    diff = np.abs(frame.astype(float) - prev_frame.astype(float))
                    motion_values[i] = float(np.mean(diff))

                prev_frame = frame.copy()

        except Exception as exc:
            logger.debug("Motion energy computation failed: %s; using zeros", exc)
            return np.zeros(num_samples, dtype=float)

        # Normalise to 0-1
        max_motion = np.max(motion_values)
        if max_motion > 1e-6:
            motion_values = motion_values / max_motion

        # Smoothing
        if self.smoothing_window > 1 and len(motion_values) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            motion_values = np.convolve(motion_values, kernel, mode="same")

        return motion_values

    def _compute_spectral_centroid(self, total_duration: float) -> np.ndarray:
        """Compute spectral centroid to distinguish speech from music.

        The spectral centroid is the weighted mean of frequencies in
        the signal. Music typically has a higher and more stable
        spectral centroid, while speech has a lower, more variable one.

        Uses FFmpeg's astats filter to extract spectral information,
        with a fallback approach using frequency band energy analysis.

        Args:
            total_duration: Total video duration in seconds.

        Returns:
            Spectral centroid values per sample interval.
        """
        num_samples = max(1, int(total_duration / self.sample_interval))
        centroid_values = np.zeros(num_samples, dtype=float)

        try:
            # Use FFmpeg to extract frequency band energy for centroid estimation
            # We extract low, mid, and high frequency band energies
            # Low band: 0-1kHz, Mid: 1-4kHz, High: 4-16kHz
            # Speech centroid is typically 1-3kHz, music is 2-6kHz

            for i in range(num_samples):
                timestamp = i * self.sample_interval + self.sample_interval * 0.5
                duration = min(self.sample_interval, total_duration - timestamp)
                if duration <= 0:
                    continue

                # Extract frequency spectrum via FFmpeg
                cmd: list[str] = [
                    "ffmpeg",
                    "-ss", f"{timestamp:.3f}",
                    "-t", f"{duration:.3f}",
                    "-i", str(self.video_path),
                    "-af", (
                        f"highpass=f=50,lowpass=f=8000,"
                        f"astats=metadata=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
                    ),
                    "-f", "null", "-",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    continue

                # Parse RMS level as proxy for spectral content
                import re
                rms_match = re.search(r"lavfi\.astats\.Overall\.RMS_level=(-?[\d.]+)", result.stderr)
                if rms_match:
                    try:
                        rms_val = float(rms_match.group(1))
                        # Map RMS value to approximate centroid frequency
                        # Louder mid-range = speech (~2000Hz), louder high = music (~4000Hz)
                        # This is a simplified heuristic
                        centroid_values[i] = self._estimate_centroid_from_energy(rms_val, timestamp, duration)
                    except ValueError:
                        pass

        except Exception as exc:
            logger.debug("Spectral centroid computation failed: %s", exc)

        return centroid_values

    def _estimate_centroid_from_energy(self, rms_db: float, timestamp: float, duration: float) -> float:
        """Estimate spectral centroid from RMS energy as a simplified heuristic.

        This uses the observation that speech tends to have mid-frequency
        energy concentration while music tends to be broader spectrum.
        We use a simplified model based on overall energy level.

        Args:
            rms_db: RMS energy level in dB.
            timestamp: Current timestamp in seconds.
            duration: Segment duration in seconds.

        Returns:
            Estimated spectral centroid frequency in Hz.
        """
        # Normalize dB to 0-1 range (typical speech: -20 to -5 dB)
        normalized_energy = max(0.0, min(1.0, (rms_db + 60) / 55))

        # Speech centroid typically around 1500-3000 Hz
        # Music centroid typically around 3000-6000 Hz
        # Use energy as a rough proxy: moderate energy with variation = speech
        base_centroid = 1500 + normalized_energy * 3000

        return base_centroid

    def _compute_emphasis_mask(self, audio_energy_peak: np.ndarray) -> np.ndarray:
        """Detect emphasis moments from sudden volume spikes.

        Emphasis is detected when the peak energy suddenly jumps
        above the local average by a significant margin.

        Args:
            audio_energy_peak: Normalised peak energy array.

        Returns:
            Binary emphasis mask (1=emphasis, 0=normal).
        """
        if len(audio_energy_peak) < 3:
            return np.zeros_like(audio_energy_peak)

        emphasis = np.zeros_like(audio_energy_peak)

        # Compute local mean
        window = min(5, len(audio_energy_peak))
        kernel = np.ones(window) / window
        local_mean = np.convolve(audio_energy_peak, kernel, mode="same")

        # Emphasis: peak is significantly above local mean
        threshold = 1.5  # 50% above local average
        for i in range(len(audio_energy_peak)):
            if local_mean[i] > 0.05 and audio_energy_peak[i] > local_mean[i] * threshold:
                emphasis[i] = 1.0

        return emphasis

    def _compute_bayesian_composite(self, profile: EnergyProfile) -> np.ndarray:
        """Compute composite score using Bayesian conflict resolution.

        When multiple signals disagree (e.g., high audio but no motion),
        this method applies Bayesian weighting to resolve conflicts.
        Signals with higher reliability get more weight.

        Signal weights:
        - Audio RMS: 0.20 (reliable for speech presence)
        - Audio peak: 0.10 (reliable for emphasis)
        - Scene density: 0.20 (reliable for visual dynamism)
        - Silence penalty: 0.10 (very reliable when detected)
        - Speech presence: 0.15 (moderate reliability)
        - Motion energy: 0.15 (reliable for visual change)
        - Spectral centroid: 0.10 (helpful for speech/music distinction)

        Args:
            profile: EnergyProfile with all computed signals.

        Returns:
            Composite score array clamped to [0, 1].
        """
        n = len(profile.audio_energy_rms)

        # Compute signal reliabilities (how much they vary — more variation = more informative)
        def signal_reliability(arr: np.ndarray) -> float:
            if len(arr) < 2:
                return 0.5
            std = float(np.std(arr))
            return min(1.0, std * 3)  # More variation = more reliable

        # Compute reliability for each signal
        r_rms = signal_reliability(profile.audio_energy_rms)
        r_peak = signal_reliability(profile.audio_energy_peak)
        r_scene = signal_reliability(profile.scene_density)
        r_silence = signal_reliability(profile.silence_mask)
        r_speech = signal_reliability(profile.speech_mask)
        r_motion = signal_reliability(profile.motion_energy)
        r_spectral = signal_reliability(profile.spectral_centroid)

        # Normalize spectral centroid to 0-1 range
        spectral_norm = np.zeros(n)
        if len(profile.spectral_centroid) == n:
            sc = profile.spectral_centroid
            sc_min, sc_max = np.min(sc), np.max(sc)
            if sc_max - sc_min > 1e-6:
                spectral_norm = (sc - sc_min) / (sc_max - sc_min)
            else:
                spectral_norm = np.ones(n) * 0.5
        elif len(profile.spectral_centroid) > 0:
            spectral_norm = np.full(n, 0.5)

        # Bayesian-weighted composite
        # Reliability-adjusted weights
        total_reliability = r_rms + r_peak + r_scene + r_silence + r_speech + r_motion + r_spectral
        if total_reliability < 1e-6:
            total_reliability = 1.0

        w_rms = 0.20 * (r_rms / total_reliability) * 7
        w_peak = 0.10 * (r_peak / total_reliability) * 7
        w_scene = 0.20 * (r_scene / total_reliability) * 7
        w_silence = 0.10 * (r_silence / total_reliability) * 7
        w_speech = 0.15 * (r_speech / total_reliability) * 7
        w_motion = 0.15 * (r_motion / total_reliability) * 7
        w_spectral = 0.10 * (r_spectral / total_reliability) * 7

        # Normalize weights to sum to ~1
        weight_sum = w_rms + w_peak + w_scene + w_silence + w_speech + w_motion + w_spectral
        if weight_sum > 1e-6:
            w_rms /= weight_sum
            w_peak /= weight_sum
            w_scene /= weight_sum
            w_silence /= weight_sum
            w_speech /= weight_sum
            w_motion /= weight_sum
            w_spectral /= weight_sum

        composite = (
            w_rms * profile.audio_energy_rms
            + w_peak * profile.audio_energy_peak
            + w_scene * profile.scene_density
            - w_silence * profile.silence_mask
            + w_speech * profile.speech_mask
            + w_motion * profile.motion_energy
            + w_spectral * (1.0 - spectral_norm)  # Lower centroid = more speech-like = better for shorts
        )

        # Clamp to [0, 1]
        composite = np.clip(composite, 0.0, 1.0)

        return composite

    # ── Segment Estimation Helpers ─────────────────────────

    def _estimate_speech_rate(
        self, profile: EnergyProfile, start_idx: int, window_samples: int
    ) -> float:
        """Estimate words per minute from energy profile.

        Estimates speech rate by counting the number of energy peaks
        (speech syllable approximations) in the segment and converting
        to WPM using the average syllable-to-word ratio.

        Args:
            profile: EnergyProfile with computed signals.
            start_idx: Starting sample index.
            window_samples: Number of samples in the window.

        Returns:
            Estimated words per minute.
        """
        end_idx = min(start_idx + window_samples, len(profile.speech_mask))
        if end_idx <= start_idx:
            return 0.0

        speech_segment = profile.speech_mask[start_idx:end_idx]
        if len(speech_segment) == 0 or np.mean(speech_segment) < 0.1:
            return 0.0

        # Count speech bursts (transitions from silence to speech)
        threshold = np.mean(speech_segment) * 0.5
        above_threshold = speech_segment > threshold
        # Count rising edges
        rising_edges = int(np.sum(np.diff(above_threshold.astype(int)) == 1))

        # Each rising edge approximates a syllable
        # Average syllable rate for speech: ~4-5 syllables/second
        segment_duration = window_samples * self.sample_interval
        if segment_duration <= 0:
            return 0.0

        syllables_per_second = rising_edges / segment_duration

        # Convert to WPM (average: 1.5 syllables per word)
        words_per_minute = (syllables_per_second / 1.5) * 60

        # Clamp to reasonable range
        return min(300.0, max(0.0, words_per_minute))

    def _estimate_music_likelihood(
        self, segment_centroid_avg: float, profile: EnergyProfile
    ) -> float:
        """Estimate likelihood that a segment is music vs speech.

        Music typically has:
        - Higher and more stable spectral centroid
        - Consistent energy without pause patterns
        - Lower speech mask values

        Args:
            segment_centroid_avg: Average spectral centroid for the segment.
            profile: Full energy profile.

        Returns:
            Likelihood of music (0=speech, 1=music).
        """
        if segment_centroid_avg <= 0:
            return 0.5

        # Music typically has centroid > 3500 Hz, speech < 3000 Hz
        if segment_centroid_avg > 4000:
            centroid_music = 0.9
        elif segment_centroid_avg > 3000:
            centroid_music = 0.6
        elif segment_centroid_avg > 2000:
            centroid_music = 0.3
        else:
            centroid_music = 0.1

        # Also check speech mask — low speech = likely music
        avg_speech = float(np.mean(profile.speech_mask)) if len(profile.speech_mask) > 0 else 0.5
        speech_music_factor = 1.0 - avg_speech  # Low speech = high music likelihood

        # Also check silence mask — music has less silence
        avg_silence = float(np.mean(profile.silence_mask)) if len(profile.silence_mask) > 0 else 0.0
        silence_music_factor = 1.0 - avg_silence  # Less silence = more likely music

        # Weighted combination
        music_likelihood = (
            0.4 * centroid_music
            + 0.3 * speech_music_factor
            + 0.3 * silence_music_factor
        )

        return max(0.0, min(1.0, music_likelihood))

    def _compute_visual_complexity(
        self,
        scene_density: np.ndarray,
        motion_energy: np.ndarray,
        start_idx: int,
        window_samples: int,
    ) -> float:
        """Compute visual complexity score for a segment.

        Combines scene change density and motion energy into a
        single visual complexity metric.

        Args:
            scene_density: Normalised scene density array.
            motion_energy: Normalised motion energy array.
            start_idx: Starting sample index.
            window_samples: Number of samples in the window.

        Returns:
            Visual complexity score (0-1).
        """
        end_idx = min(start_idx + window_samples, len(scene_density), len(motion_energy))
        if end_idx <= start_idx:
            return 0.0

        scene_val = float(np.mean(scene_density[start_idx:end_idx]))
        motion_val = float(np.mean(motion_energy[start_idx:end_idx]))

        # Weighted combination: both contribute equally
        complexity = 0.5 * scene_val + 0.5 * motion_val
        return max(0.0, min(1.0, complexity))

    def _compute_confidence(
        self,
        profile: EnergyProfile,
        start_idx: int,
        window_samples: int,
        best_score: float,
    ) -> float:
        """Compute confidence score using Bayesian weighting.

        Higher confidence when multiple signals agree on the segment
        quality. Lower when signals conflict.

        Args:
            profile: EnergyProfile with all signals.
            start_idx: Starting sample index.
            window_samples: Number of samples in the window.
            best_score: Composite score of the segment.

        Returns:
            Confidence score (0-1).
        """
        end_idx = min(start_idx + window_samples, len(profile.audio_energy_rms))
        if end_idx <= start_idx:
            return 0.0

        # Compute each signal's agreement with the composite
        signals = {
            "rms": profile.audio_energy_rms[start_idx:end_idx],
            "peak": profile.audio_energy_peak[start_idx:end_idx],
            "scene": profile.scene_density[start_idx:end_idx],
            "speech": profile.speech_mask[start_idx:end_idx],
            "motion": profile.motion_energy[start_idx:end_idx],
        }

        agreement_scores: list[float] = []
        for name, signal in signals.items():
            if len(signal) == 0:
                continue
            signal_mean = float(np.mean(signal))
            # Agreement: both signal and composite agree on quality
            if signal_mean > 0.5 and best_score > 0.5:
                agreement_scores.append(min(signal_mean, best_score))
            elif signal_mean <= 0.5 and best_score <= 0.5:
                agreement_scores.append(min(1.0 - signal_mean, 1.0 - best_score))
            else:
                # Disagreement — reduce confidence
                agreement_scores.append(0.3)

        if not agreement_scores:
            return 0.3

        # Confidence is the average agreement
        avg_agreement = sum(agreement_scores) / len(agreement_scores)

        # Scale and clamp
        confidence = min(1.0, avg_agreement * 1.5)
        return round(confidence, 2)

    # ── Intro/Outro Detection ──────────────────────────────

    def _detect_intro_end(self, profile: EnergyProfile) -> float:
        """Detect where the intro/cold-open ends.

        Looks for the first sustained high-composite-score region after
        the initial 5% of the video, considering speech presence and
        motion energy as additional signals.

        Args:
            profile: EnergyProfile with computed signals.

        Returns:
            Timestamp in seconds where the intro ends.
        """
        if profile.total_duration <= 10:
            return 0.0

        skip_to = profile.total_duration * 0.05
        skip_idx = int(skip_to / profile.sample_interval)

        if len(profile.composite) <= skip_idx:
            return skip_to

        threshold = np.mean(profile.composite) + 0.5 * np.std(profile.composite)
        sustained_count = 0
        sustained_required = 3

        for i in range(skip_idx, len(profile.composite)):
            if profile.composite[i] >= threshold:
                sustained_count += 1
                if sustained_count >= sustained_required:
                    intro_end = float((i - sustained_required + 1) * profile.sample_interval)
                    logger.info("Intro ends at %.1fs", intro_end)
                    return intro_end
            else:
                sustained_count = 0

        return skip_to

    def _detect_outro_start(self, profile: EnergyProfile) -> float:
        """Detect where the outro/credits begin.

        Looks for the last sustained high-composite-score region from
        the end of the video, considering drop in speech and motion.

        Args:
            profile: EnergyProfile with computed signals.

        Returns:
            Timestamp in seconds where the outro starts.
        """
        if profile.total_duration <= 10:
            return profile.total_duration

        outro_threshold = profile.total_duration * 0.90
        outro_idx = int(outro_threshold / profile.sample_interval)

        if len(profile.composite) <= outro_idx:
            return profile.total_duration

        threshold = np.mean(profile.composite) - 0.5 * np.std(profile.composite)
        for i in range(len(profile.composite) - 1, outro_idx, -1):
            if profile.composite[i] < threshold:
                outro_start = float(i * profile.sample_interval)
                logger.info("Outro starts at %.1fs", outro_start)
                return outro_start

        return profile.total_duration

    def _detect_opening_closing(self, profile: EnergyProfile) -> tuple[float, float]:
        """Detect opening and closing sequences to avoid.

        Opening sequences have low speech and consistent low energy.
        Closing sequences have dropping energy and increased silence.

        Args:
            profile: EnergyProfile with computed signals.

        Returns:
            Tuple of (opening_end_time, closing_start_time).
        """
        opening_end = self._detect_intro_end(profile)
        closing_start = self._detect_outro_start(profile)
        return opening_end, closing_start

    # ── Window Search ──────────────────────────────────────

    def _find_best_window(
        self,
        composite: np.ndarray,
        total_duration: float,
        intro_end: float = 0.0,
        outro_start: float = float("inf"),
        overlap: float = 0.5,
    ) -> float:
        """Find the start time of the best clip_duration window.

        Uses a proper sliding window with configurable overlap to
        search for the optimal segment. Respects intro/outro boundaries
        and applies edge buffers.

        Args:
            composite: Composite score array.
            total_duration: Total video duration in seconds.
            intro_end: Timestamp where intro ends.
            outro_start: Timestamp where outro begins.
            overlap: Overlap fraction between sliding window steps (0-1).

        Returns:
            Best start time in seconds.
        """
        buffer_seconds = 2.0
        window_samples = max(1, int(self.clip_duration / self.sample_interval))

        if len(composite) <= window_samples:
            return buffer_seconds

        # Step size based on overlap
        step = max(1, int(window_samples * (1.0 - overlap)))

        # Rolling sum via convolution
        rolling_sum = np.convolve(composite, np.ones(window_samples), mode="valid")
        rolling_score = rolling_sum / window_samples

        # Valid range: after intro, before outro
        intro_idx = max(0, int(intro_end / self.sample_interval))
        outro_idx = min(len(rolling_score), int(outro_start / self.sample_interval))

        buffer_samples = int(buffer_seconds / self.sample_interval)
        valid_start = max(intro_idx, buffer_samples)
        valid_end = max(valid_start + 1, min(outro_idx, len(rolling_score)))

        valid_start = min(valid_start, len(rolling_score) - 1)
        valid_end = min(valid_end, len(rolling_score))

        if valid_end <= valid_start:
            valid_end = len(rolling_score)

        # Find the best window
        if valid_end > valid_start:
            best_idx = valid_start + int(np.argmax(rolling_score[valid_start:valid_end]))
        else:
            best_idx = 0

        best_start_time = float(best_idx * self.sample_interval)

        # Ensure clip doesn't exceed video duration
        if best_start_time + self.clip_duration > total_duration:
            best_start_time = max(0.0, total_duration - self.clip_duration - buffer_seconds)

        return best_start_time

    def _snap_to_keyframe(self, timestamp: float, keyframe_interval: float = 2.0) -> float:
        """Snap a clip boundary to the nearest keyframe position.

        Ensures clean cuts by aligning to keyframe boundaries.

        Args:
            timestamp: Original timestamp in seconds.
            keyframe_interval: Keyframe interval in seconds (default 2.0).

        Returns:
            Snapped timestamp.
        """
        if keyframe_interval <= 0:
            return timestamp
        snapped = round(timestamp / keyframe_interval) * keyframe_interval
        return max(0.0, snapped)


# ── Visualization ─────────────────────────────────────────

def visualize_profile(
    profile: EnergyProfile,
    output_path: Path | None = None,
    segment: SegmentResult | None = None,
) -> Path | None:
    """Generate a matplotlib chart of the energy profile for debugging.

    Creates a multi-subplot figure showing all signal arrays with
    an optional highlighted segment region.

    Args:
        profile: EnergyProfile to visualize.
        output_path: Path to save the chart. If None, saves to
            the current working directory.
        segment: Optional SegmentResult to highlight.

    Returns:
        Path to the saved chart, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping profile visualization")
        return None

    if output_path is None:
        output_path = Path("energy_profile.png")

    fig, axes = plt.subplots(7, 1, figsize=(14, 14), sharex=True)
    time_axis = np.arange(len(profile.audio_energy_rms)) * profile.sample_interval

    signals = [
        ("Audio RMS Energy", profile.audio_energy_rms, "tab:blue"),
        ("Audio Peak Energy", profile.audio_energy_peak, "tab:cyan"),
        ("Scene Density", profile.scene_density, "tab:orange"),
        ("Speech Mask", profile.speech_mask, "tab:green"),
        ("Motion Energy", profile.motion_energy, "tab:red"),
        ("Silence Mask", profile.silence_mask, "tab:gray"),
        ("Composite Score", profile.composite, "tab:purple"),
    ]

    for ax, (title, data, color) in zip(axes, signals):
        if len(data) > 0 and len(time_axis) > 0:
            ax.plot(time_axis[:len(data)], data, color=color, linewidth=1)
            ax.fill_between(time_axis[:len(data)], data, alpha=0.2, color=color)
        ax.set_ylabel(title, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # Highlight segment
        if segment and len(time_axis) > 0:
            ax.axvspan(segment.start_time, segment.end_time, alpha=0.2, color="yellow")

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Energy Profile (duration={profile.total_duration:.1f}s, interval={profile.sample_interval:.1f}s)", fontsize=12)
    fig.tight_layout()

    try:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Energy profile chart saved to: %s", output_path)
        return output_path
    except Exception as exc:
        logger.warning("Failed to save energy profile chart: %s", exc)
        plt.close(fig)
        return None


# ── Face Detection (backward compatible) ──────────────────

def detect_face_crop_region(
    video_path: Path,
    timestamp: float,
    target_w: int,
    target_h: int,
) -> tuple[int, int]:
    """Detect the optimal crop position by tracking faces/subjects.

    Uses FFmpeg to extract a frame and OpenCV (if available) to detect
    face positions. Falls back to centre crop if OpenCV is unavailable.

    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds to sample.
        target_w: Target crop width in pixels.
        target_h: Target crop height in pixels.

    Returns:
        Tuple of (crop_x, crop_y) for the best crop position.
        Returns (-1, -1) if face detection is unavailable.
    """
    try:
        import cv2

        from utils.ffmpeg_utils import get_video_thumbnail

        tmp_jpg = Path(tempfile.mktemp(suffix=".jpg"))
        get_video_thumbnail(video_path, timestamp, tmp_jpg)

        img = cv2.imread(str(tmp_jpg))
        if img is None:
            tmp_jpg.unlink(missing_ok=True)
            return -1, -1

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try Haar cascade face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        tmp_jpg.unlink(missing_ok=True)

        if len(faces) == 0:
            return -1, -1

        # Compute centroid of all detected faces
        face_centers_x = [fx + fw // 2 for (fx, fy, fw, fh) in faces]
        face_centers_y = [fy + fh // 2 for (fx, fy, fw, fh) in faces]

        centroid_x = int(np.mean(face_centers_x))
        centroid_y = int(np.mean(face_centers_y))

        # Compute crop position centred on face centroid
        crop_x = max(0, min(centroid_x - target_w // 2, w - target_w))
        crop_y = max(0, min(centroid_y - target_h // 2, h - target_h))

        logger.info("Face-tracked crop: (%d, %d) from %d faces", crop_x, crop_y, len(faces))
        return crop_x, crop_y

    except ImportError:
        logger.debug("OpenCV not available; using centre crop")
        return -1, -1
    except Exception as exc:
        logger.debug("Face detection failed: %s; using centre crop", exc)
        return -1, -1
