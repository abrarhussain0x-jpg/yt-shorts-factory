"""
core/audio_enhancer.py — Audio enhancement for video content.

Provides noise reduction, loudness normalization, dynamic range
compression, de-essing, speech enhancement, vocal isolation,
audio fades, and automatic level adjustment. All processing is
done via FFmpeg subprocess calls — no external APIs required.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.ffmpeg_utils import probe_video, run_ffmpeg, FFmpegError
from utils.logger import get_logger

logger = get_logger("audio_enhancer")


# ── Custom Exceptions ─────────────────────────────────────

class AudioEnhanceError(Exception):
    """Base exception for audio enhancement failures."""
    pass


class AudioAnalysisError(AudioEnhanceError):
    """Raised when audio quality analysis fails."""
    pass


class AudioProcessError(AudioEnhanceError):
    """Raised when an audio processing step fails."""
    pass


# ── Data Classes ──────────────────────────────────────────

@dataclass
class AudioQualityReport:
    """Full audio quality analysis report."""

    loudness_lufs: float = -70.0  # Integrated loudness (EBU R128)
    true_peak: float = -9.0  # True peak in dBTP
    noise_floor_db: float = -60.0  # Estimated noise floor
    dynamic_range: float = 0.0  # Loudness range (LRA) in LU
    speech_clarity_score: float = 0.0  # 0-1, estimated clarity
    has_clipping: bool = False  # Whether clipping is detected
    has_music: bool = False  # Whether music is detected
    sample_rate: int = 0
    channels: int = 0
    audio_codec: str = ""
    duration: float = 0.0
    bit_depth: int = 0
    rms_level_db: float = -60.0  # Overall RMS level

    @property
    def quality_summary(self) -> str:
        """Return a human-readable quality summary."""
        issues: list[str] = []
        if self.has_clipping:
            issues.append("clipping detected")
        if self.loudness_lufs < -30:
            issues.append("very low loudness")
        elif self.loudness_lufs > -5:
            issues.append("very high loudness")
        if self.noise_floor_db > -35:
            issues.append("high noise floor")
        if self.dynamic_range < 3:
            issues.append("compressed dynamic range")
        if self.speech_clarity_score < 0.3:
            issues.append("poor speech clarity")

        if not issues:
            return "Good audio quality"
        return "Issues: " + ", ".join(issues)


# ── AudioEnhancer Class ───────────────────────────────────

class AudioEnhancer:
    """Audio enhancement pipeline using FFmpeg subprocess calls.

    Provides a complete audio processing chain:
    analyze -> denoise -> compress -> normalize -> de-ess -> enhance

    Each step is optional and independently configurable. All
    processing is done via FFmpeg with no external APIs.
    """

    def __init__(self, timeout: int = 600) -> None:
        """Initialize the audio enhancer.

        Args:
            timeout: Maximum FFmpeg subprocess timeout in seconds.
        """
        self.timeout = timeout

    def _run_ffmpeg_audio(self, args: list[str], description: str = "") -> subprocess.CompletedProcess:
        """Run an FFmpeg audio processing command.

        Args:
            args: FFmpeg arguments (without 'ffmpeg' prefix).
            description: Description for logging.

        Returns:
            CompletedProcess result.

        Raises:
            AudioProcessError: If FFmpeg fails.
        """
        cmd = ["ffmpeg", "-hide_banner", "-y"] + args
        logger.debug("FFmpeg audio command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                raise AudioProcessError(
                    f"FFmpeg audio processing failed: {description}\n"
                    f"stderr: {result.stderr[-500:]}"
                )
            return result
        except subprocess.TimeoutExpired:
            raise AudioProcessError(
                f"FFmpeg audio processing timed out ({self.timeout}s): {description}"
            )
        except FileNotFoundError:
            raise AudioProcessError("FFmpeg not found on PATH")

    def analyze_audio_quality(self, video_path: Path) -> AudioQualityReport:
        """Perform a full audio quality analysis of a video file.

        Uses FFmpeg's ebur128 filter for loudness measurement,
        astats for level analysis, and silencedetect for noise
        floor estimation.

        Args:
            video_path: Path to the video file.

        Returns:
            AudioQualityReport with detailed quality metrics.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            AudioAnalysisError: If analysis fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        report = AudioQualityReport()

        # Get basic audio info from probe
        try:
            video_info = probe_video(video_path)
            report.sample_rate = video_info.sample_rate
            report.channels = video_info.channels
            report.audio_codec = video_info.audio_codec
            report.duration = video_info.duration
        except Exception as exc:
            logger.warning("Video probe failed during audio analysis: %s", exc)

        # ── EBU R128 Loudness Analysis ──────────────────────
        try:
            report.loudness_lufs, report.true_peak, report.dynamic_range = (
                self._measure_ebur128(video_path)
            )
        except Exception as exc:
            logger.warning("EBU R128 loudness measurement failed: %s", exc)

        # ── Audio Statistics ────────────────────────────────
        try:
            stats = self._measure_audio_stats(video_path)
            report.rms_level_db = stats.get("rms_level", -60.0)
            report.noise_floor_db = stats.get("noise_floor", -60.0)
            report.has_clipping = stats.get("has_clipping", False)
            report.bit_depth = stats.get("bit_depth", 0)
        except Exception as exc:
            logger.warning("Audio stats measurement failed: %s", exc)

        # ── Music Detection ────────────────────────────────
        try:
            report.has_music = self._detect_music(video_path)
        except Exception as exc:
            logger.debug("Music detection failed: %s", exc)

        # ── Speech Clarity Estimation ──────────────────────
        try:
            report.speech_clarity_score = self._estimate_speech_clarity(video_path, report)
        except Exception as exc:
            logger.debug("Speech clarity estimation failed: %s", exc)

        logger.info(
            "Audio quality: LUFS=%.1f, peak=%.1f dBTP, LRA=%.1f LU, "
            "noise_floor=%.1f dB, clarity=%.2f, clipping=%s, music=%s",
            report.loudness_lufs, report.true_peak, report.dynamic_range,
            report.noise_floor_db, report.speech_clarity_score,
            report.has_clipping, report.has_music,
        )

        return report

    def _measure_ebur128(self, video_path: Path) -> tuple[float, float, float]:
        """Measure EBU R128 loudness metrics.

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of (integrated_loudness_lufs, true_peak_dbtp, loudness_range_lu).
        """
        cmd: list[str] = [
            "-i", str(video_path),
            "-af", "ebur128",
            "-f", "null",
            "-",
        ]

        result = self._run_ffmpeg_audio(cmd, "EBU R128 loudness measurement")

        # Parse EBU R128 output
        stderr = result.stderr
        loudness_lufs = -70.0
        true_peak = -9.0
        lra = 0.0

        # Look for integrated loudness
        i_match = re.search(r"I:\s+(-?[\d.]+)\s+LUFS", stderr)
        if i_match:
            loudness_lufs = float(i_match.group(1))

        # Look for true peak
        tp_match = re.search(r"Peak:\s+(-?[\d.]+)\s+dBTP", stderr)
        if tp_match:
            true_peak = float(tp_match.group(1))

        # Look for loudness range
        lra_match = re.search(r"LRA:\s+([\d.]+)\s+LU", stderr)
        if lra_match:
            lra = float(lra_match.group(1))

        return loudness_lufs, true_peak, lra

    def _measure_audio_stats(self, video_path: Path) -> dict[str, any]:
        """Measure audio statistics using FFmpeg astats filter.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict with audio statistics.
        """
        cmd: list[str] = [
            "-i", str(video_path),
            "-af", "astats=metadata=1:reset=5,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null",
            "-",
        ]

        result = self._run_ffmpeg_audio(cmd, "Audio statistics measurement")

        rms_values: list[float] = []
        rms_pattern = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?[\d.]+|nan|-inf)")

        for line in (result.stdout + result.stderr).splitlines():
            match = rms_pattern.search(line)
            if match:
                val = match.group(1)
                if val not in ("nan", "-inf"):
                    try:
                        rms_values.append(float(val))
                    except ValueError:
                        pass

        # Estimate noise floor as the minimum RMS value
        noise_floor = min(rms_values) if rms_values else -60.0

        # Detect clipping by checking for peak values near 0 dB
        cmd_clip: list[str] = [
            "-i", str(video_path),
            "-af", "astats=metadata=1,ametadata=print:key=lavfi.astats.Overall.Peak_level:file=-",
            "-f", "null",
            "-",
        ]

        clip_result = self._run_ffmpeg_audio(cmd_clip, "Clipping detection")
        peak_values: list[float] = []
        peak_pattern = re.compile(r"lavfi\.astats\.Overall\.Peak_level=(-?[\d.]+|nan|-inf)")

        for line in (clip_result.stdout + clip_result.stderr).splitlines():
            match = peak_pattern.search(line)
            if match:
                val = match.group(1)
                if val not in ("nan", "-inf"):
                    try:
                        peak_values.append(float(val))
                    except ValueError:
                        pass

        has_clipping = any(p > -0.5 for p in peak_values) if peak_values else False
        rms_level = sum(rms_values) / len(rms_values) if rms_values else -60.0

        return {
            "rms_level": rms_level,
            "noise_floor": noise_floor,
            "has_clipping": has_clipping,
            "bit_depth": 0,  # Would need ffprobe for this
        }

    def _detect_music(self, video_path: Path) -> bool:
        """Detect whether the audio contains music.

        Uses spectral analysis heuristics: music tends to have
        consistent energy across frequency bands with low silence
        ratio, while speech has more variation and pauses.

        Args:
            video_path: Path to the video file.

        Returns:
            True if music is likely present.
        """
        # Use silence detection to estimate silence ratio
        # Music has very low silence ratio, speech has moderate
        cmd: list[str] = [
            "-i", str(video_path),
            "-af", "silencedetect=noise=-35dB:d=0.3",
            "-f", "null",
            "-",
        ]

        result = self._run_ffmpeg_audio(cmd, "Music detection via silence analysis")

        silence_starts: list[float] = []
        silence_ends: list[float] = []

        for line in result.stderr.splitlines():
            start_match = re.search(r"silence_start:\s*([\d.]+)", line)
            if start_match:
                silence_starts.append(float(start_match.group(1)))
            end_match = re.search(r"silence_end:\s*([\d.]+)", line)
            if end_match:
                silence_ends.append(float(end_match.group(1)))

        # Compute silence duration
        total_silence = 0.0
        for i in range(min(len(silence_starts), len(silence_ends))):
            total_silence += silence_ends[i] - silence_starts[i]

        # Get total duration
        try:
            info = probe_video(video_path)
            duration = info.duration
        except Exception:
            duration = 60.0  # Default

        silence_ratio = total_silence / max(duration, 1.0)

        # Music typically has <5% silence; speech typically 15-40%
        return silence_ratio < 0.08

    def _estimate_speech_clarity(self, video_path: Path, report: AudioQualityReport) -> float:
        """Estimate speech clarity score from audio metrics.

        Combines loudness, noise floor, dynamic range, and clipping
        status into a single clarity score.

        Args:
            video_path: Path to the video file.
            report: Partially filled AudioQualityReport.

        Returns:
            Speech clarity score (0-1).
        """
        score = 0.5  # Start neutral

        # Good loudness for speech: -20 to -10 LUFS
        if -20 <= report.loudness_lufs <= -10:
            score += 0.2
        elif -25 <= report.loudness_lufs <= -5:
            score += 0.1
        elif report.loudness_lufs < -30 or report.loudness_lufs > -3:
            score -= 0.2

        # Low noise floor is better
        if report.noise_floor_db < -50:
            score += 0.15
        elif report.noise_floor_db < -40:
            score += 0.05
        elif report.noise_floor_db > -30:
            score -= 0.2

        # No clipping is better
        if report.has_clipping:
            score -= 0.3

        # Moderate dynamic range is best for speech
        if 5 <= report.dynamic_range <= 20:
            score += 0.1
        elif report.dynamic_range < 3:
            score -= 0.1

        # No music interference is better for speech clarity
        if not report.has_music:
            score += 0.05

        return max(0.0, min(1.0, score))

    # ── Audio Processing Methods ───────────────────────────

    def reduce_noise(
        self,
        video_path: Path,
        output_path: Path,
        strength: str = "medium",
    ) -> Path:
        """Reduce noise using FFmpeg's afftdn filter.

        The afftdn (FFT denoise) filter analyzes the noise profile
        and reduces noise while preserving speech content.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.
            strength: Noise reduction strength. One of:
                "light" (5dB reduction), "medium" (12dB reduction),
                "heavy" (20dB reduction), or a numeric dB value.

        Returns:
            Path to the output file with reduced noise.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If processing fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Map strength to noise reduction amount
        strength_map = {
            "light": "nr=5",
            "medium": "nr=12",
            "heavy": "nr=20",
        }

        if strength in strength_map:
            nr_param = strength_map[strength]
        else:
            # Try to parse as numeric
            try:
                nr_val = float(strength)
                nr_param = f"nr={nr_val}"
            except ValueError:
                nr_param = "nr=12"  # Default to medium

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", f"afftdn={nr_param}:nf=-50",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, f"Noise reduction (strength={strength})")

        if not output_path.exists():
            raise AudioProcessError(f"Noise reduction output not found: {output_path}")

        logger.info("Noise reduction applied (strength=%s): %s", strength, output_path.name)
        return output_path

    def normalize_loudness(
        self,
        video_path: Path,
        output_path: Path,
        target_lufs: float = -16.0,
    ) -> Path:
        """Normalize loudness using EBU R128 loudnorm filter.

        Performs two-pass loudness normalization:
        1. First pass: measure integrated loudness
        2. Second pass: apply correction to reach target

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.
            target_lufs: Target integrated loudness in LUFS.
                Default -16 (YouTube standard).

        Returns:
            Path to the output file with normalized loudness.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If normalization fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pass 1: Measure loudness
        cmd_measure: list[str] = [
            "-i", str(video_path),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
            "-f", "null",
            "-",
        ]

        measure_result = self._run_ffmpeg_audio(cmd_measure, "Loudness measurement pass")

        # Parse measurement results
        measured_data: dict[str, float] = {}
        try:
            # Find the JSON block in stderr
            json_start = measure_result.stderr.rfind("{")
            json_end = measure_result.stderr.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = measure_result.stderr[json_start:json_end]
                measured_data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse loudnorm measurement; using single-pass")

        # Pass 2: Apply normalization
        if measured_data and "input_i" in measured_data:
            # Two-pass normalization with measured values
            input_i = measured_data.get("input_i", target_lufs)
            input_tp = measured_data.get("input_tp", 0.0)
            input_lra = measured_data.get("input_lra", 11.0)
            input_thresh = measured_data.get("input_thresh", -70.0)
            target_offset = measured_data.get("target_offset", 0.0)

            normalize_filter = (
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                f"measured_I={input_i}:measured_TP={input_tp}:"
                f"measured_LRA={input_lra}:measured_thresh={input_thresh}:"
                f"offset={target_offset}:linear=true"
            )
        else:
            # Single-pass normalization
            normalize_filter = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"

        cmd_normalize: list[str] = [
            "-i", str(video_path),
            "-af", normalize_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd_normalize, f"Loudness normalization (target={target_lufs} LUFS)")

        if not output_path.exists():
            raise AudioProcessError(f"Loudness normalization output not found: {output_path}")

        logger.info("Loudness normalized to %.1f LUFS: %s", target_lufs, output_path.name)
        return output_path

    def compress_dynamic_range(
        self,
        video_path: Path,
        output_path: Path,
        threshold: float = -20.0,
        ratio: float = 4.0,
    ) -> Path:
        """Apply dynamic range compression using FFmpeg's acompressor.

        Compresses the dynamic range to make quiet sounds louder and
        loud sounds quieter, improving audibility on mobile devices.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.
            threshold: Threshold in dB where compression starts. Default -20.
            ratio: Compression ratio. Default 4:1.

        Returns:
            Path to the output file with compressed audio.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If compression fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        compress_filter = (
            f"acompressor=threshold={threshold}dB:ratio={ratio}:"
            f"attack=5:release=50:makeup=2"
        )

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", compress_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, f"Dynamic range compression (threshold={threshold}, ratio={ratio})")

        if not output_path.exists():
            raise AudioProcessError(f"Compression output not found: {output_path}")

        logger.info("Dynamic range compressed: %s", output_path.name)
        return output_path

    def de_ess(self, video_path: Path, output_path: Path) -> Path:
        """Remove sibilance (harsh 's' sounds) from audio.

        Uses FFmpeg's highpass and lowpass filters to create a
        de-essing effect by reducing the 4-8 kHz frequency range
        where sibilance is most prominent.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.

        Returns:
            Path to the output file with reduced sibilance.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If processing fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # De-essing: split into bands, reduce sibilant range, recombine
        # Simple approach: use a dynamic filter that reduces 4-8kHz when it's too loud
        de_ess_filter = (
            "highpass=f=80,"
            "lowpass=f=12000,"
            # Reduce sibilance frequencies with a notch-like approach
            "equalizer=f=6000:t=q:w=2:g=-3dB,"
            "equalizer=f=4500:t=q:w=2:g=-2dB,"
            "equalizer=f=7500:t=q:w=2:g=-2dB"
        )

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", de_ess_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, "De-essing")

        if not output_path.exists():
            raise AudioProcessError(f"De-ess output not found: {output_path}")

        logger.info("De-essing applied: %s", output_path.name)
        return output_path

    def enhance_speech(self, video_path: Path, output_path: Path) -> Path:
        """Apply a speech enhancement chain.

        Applies the full enhancement chain optimized for speech:
        1. High-pass filter (remove low-frequency rumble)
        2. Noise reduction
        3. De-essing
        4. Compression
        5. EQ boost for speech frequencies (1-4 kHz)
        6. Loudness normalization

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.

        Returns:
            Path to the output file with enhanced speech.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If processing fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Speech enhancement chain
        speech_filter = (
            # 1. High-pass filter: remove rumble below 80 Hz
            "highpass=f=80,"
            # 2. Low-pass filter: remove hiss above 12 kHz
            "lowpass=f=12000,"
            # 3. Noise reduction (moderate)
            "afftdn=nr=8:nf=-45,"
            # 4. De-essing: reduce sibilant frequencies
            "equalizer=f=6000:t=q:w=2:g=-3dB,"
            # 5. Speech EQ: boost 1-4 kHz range for clarity
            "equalizer=f=2000:t=q:w=1:g=3dB,"
            "equalizer=f=3000:t=q:w=1:g=2dB,"
            # 6. Compression: make speech consistent
            "acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=1,"
            # 7. Loudness normalization for platforms
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        )

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", speech_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, "Speech enhancement chain")

        if not output_path.exists():
            raise AudioProcessError(f"Speech enhancement output not found: {output_path}")

        logger.info("Speech enhancement applied: %s", output_path.name)
        return output_path

    def remove_background_music(
        self,
        video_path: Path,
        output_path: Path,
    ) -> Path:
        """Attempt vocal isolation to remove background music.

        Uses a simple mid-side processing approach to isolate the
        center-panned content (typically vocals) from the sides
        (typically instruments/music). This is a basic approach
        and works best with stereo content.

        For mono content, uses frequency-based filtering to suppress
        musical content.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.

        Returns:
            Path to the output file with reduced background music.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If processing fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if audio is stereo
        try:
            info = probe_video(video_path)
            is_stereo = info.channels >= 2
        except Exception:
            is_stereo = False

        if is_stereo:
            # Center-channel extraction (vocals are typically center-panned)
            # Uses channel splitting and mid-side decoding
            vocal_filter = (
                # Extract center channel: L+R (mid) minus L-R (side)
                "pan=1c|c0=0.5*FL+0.5*FR,"
                # Boost speech frequencies
                "highpass=f=80,"
                "lowpass=f=10000,"
                "equalizer=f=2000:t=q:w=1:g=3dB,"
                # Light noise reduction
                "afftdn=nr=5:nf=-45,"
                # Normalize
                "loudnorm=I=-16:TP=-1.5:LRA=11"
            )
        else:
            # Mono content: use frequency-based approach
            # Suppress frequencies where music typically dominates
            vocal_filter = (
                "highpass=f=120,"
                "lowpass=f=8000,"
                # Suppress low bass (typical music range)
                "equalizer=f=150:t=q:w=1:g=-3dB,"
                # Boost speech mid-range
                "equalizer=f=2000:t=q:w=1:g=4dB,"
                "equalizer=f=3000:t=q:w=1:g=3dB,"
                # Noise reduction
                "afftdn=nr=8:nf=-45,"
                # Normalize
                "loudnorm=I=-16:TP=-1.5:LRA=11"
            )

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", vocal_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ac", "2",  # Output as stereo for compatibility
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, "Background music removal / vocal isolation")

        if not output_path.exists():
            raise AudioProcessError(f"Music removal output not found: {output_path}")

        logger.info("Background music reduction applied: %s", output_path.name)
        return output_path

    def add_audio_fade(
        self,
        video_path: Path,
        output_path: Path,
        fade_in: float = 0.3,
        fade_out: float = 0.3,
    ) -> Path:
        """Add fade-in and fade-out to the audio track.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.
            fade_in: Fade-in duration in seconds. Default 0.3.
            fade_out: Fade-out duration in seconds. Default 0.3.

        Returns:
            Path to the output file with audio fades.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If processing fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get duration for fade-out calculation
        try:
            info = probe_video(video_path)
            duration = info.duration
        except Exception:
            duration = 60.0

        fade_out_start = max(0.0, duration - fade_out)

        fade_filter = f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start:.3f}:d={fade_out}"

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", fade_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, f"Audio fade (in={fade_in}s, out={fade_out}s)")

        if not output_path.exists():
            raise AudioProcessError(f"Audio fade output not found: {output_path}")

        logger.info("Audio fade applied (in=%.1fs, out=%.1fs): %s", fade_in, fade_out, output_path.name)
        return output_path

    def auto_level(
        self,
        video_path: Path,
        output_path: Path,
    ) -> Path:
        """Automatic audio level adjustment.

        Analyzes the audio and applies the optimal combination of:
        1. Noise reduction (if noise floor is high)
        2. Compression (if dynamic range is too wide)
        3. Loudness normalization (always)
        4. De-essing (if sibilance detected)
        5. Audio fade (always, for clean transitions)

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the processed video.

        Returns:
            Path to the output file with auto-leveled audio.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If processing fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Analyze first
        try:
            report = self.analyze_audio_quality(video_path)
        except Exception as exc:
            logger.warning("Audio analysis failed; using default chain: %s", exc)
            report = AudioQualityReport()

        # Build adaptive filter chain
        filters: list[str] = []

        # Always: high-pass to remove rumble
        filters.append("highpass=f=80")

        # Conditional: noise reduction
        if report.noise_floor_db > -45:
            nr_amount = min(15, max(5, int((report.noise_floor_db + 60) * 0.5)))
            filters.append(f"afftdn=nr={nr_amount}:nf=-50")
            logger.info("Auto-level: applying noise reduction (nr=%d)", nr_amount)

        # Conditional: de-essing if needed
        # Check for high energy in sibilance range
        if report.rms_level_db > -15:
            filters.append("equalizer=f=6000:t=q:w=2:g=-2dB")
            logger.info("Auto-level: applying de-essing")

        # Conditional: compression
        if report.dynamic_range > 15:
            filters.append("acompressor=threshold=-20dB:ratio=3:attack=5:release=50:makeup=1")
            logger.info("Auto-level: applying compression (LRA=%.1f)", report.dynamic_range)
        elif report.dynamic_range < 5:
            # Very compressed already, slight expansion
            filters.append("acompressor=threshold=-25dB:ratio=2:attack=10:release=100:makeup=0")
            logger.info("Auto-level: applying light compression")

        # Always: loudness normalization
        target_lufs = -16.0
        filters.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11")

        # Always: fade in/out
        try:
            info = probe_video(video_path)
            duration = info.duration
        except Exception:
            duration = 60.0

        fade_in = 0.3
        fade_out = 0.3
        fade_out_start = max(0.0, duration - fade_out)
        filters.append(f"afade=t=in:st=0:d={fade_in}")
        filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out}")

        # Combine all filters
        audio_filter = ",".join(filters)

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", audio_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, "Auto-level audio adjustment")

        if not output_path.exists():
            raise AudioProcessError(f"Auto-level output not found: {output_path}")

        logger.info("Auto-level applied: %s", output_path.name)
        return output_path

    def enhance_pipeline(
        self,
        video_path: Path,
        output_path: Path,
        denoise: bool = True,
        compress: bool = True,
        normalize: bool = True,
        deess: bool = False,
        fade: bool = True,
        target_lufs: float = -16.0,
    ) -> Path:
        """Run a configurable audio enhancement pipeline.

        Applies selected processing steps in the optimal order:
        1. Noise reduction
        2. De-essing
        3. Dynamic range compression
        4. Loudness normalization
        5. Audio fade in/out

        Each step is optional and the pipeline only applies
        enabled steps.

        Args:
            video_path: Path to the source video.
            output_path: Destination path for the final output.
            denoise: Whether to apply noise reduction.
            compress: Whether to apply dynamic range compression.
            normalize: Whether to normalize loudness.
            deess: Whether to apply de-essing.
            fade: Whether to add audio fades.
            target_lufs: Target loudness in LUFS.

        Returns:
            Path to the final output file.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            AudioProcessError: If any processing step fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the complete filter chain
        filters: list[str] = []

        # Step 1: High-pass (always, remove rumble)
        filters.append("highpass=f=80")

        # Step 2: Noise reduction
        if denoise:
            filters.append("afftdn=nr=10:nf=-50")

        # Step 3: De-essing
        if deess:
            filters.append("equalizer=f=6000:t=q:w=2:g=-3dB")
            filters.append("equalizer=f=4500:t=q:w=2:g=-2dB")

        # Step 4: Compression
        if compress:
            filters.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=50:makeup=1")

        # Step 5: Loudness normalization
        if normalize:
            filters.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11")

        # Step 6: Fade
        if fade:
            try:
                info = probe_video(video_path)
                duration = info.duration
            except Exception:
                duration = 60.0

            fade_in = 0.3
            fade_out = 0.3
            fade_out_start = max(0.0, duration - fade_out)
            filters.append(f"afade=t=in:st=0:d={fade_in}")
            filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out}")

        audio_filter = ",".join(filters)

        cmd: list[str] = [
            "-i", str(video_path),
            "-af", audio_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        self._run_ffmpeg_audio(cmd, "Audio enhancement pipeline")

        if not output_path.exists():
            raise AudioProcessError(f"Enhancement pipeline output not found: {output_path}")

        logger.info("Audio enhancement pipeline complete: %s", output_path.name)
        return output_path
