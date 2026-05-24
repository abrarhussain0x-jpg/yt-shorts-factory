"""
utils/ffmpeg_utils.py — All FFmpeg subprocess wrappers.

Provides typed, error-handled wrappers for every FFmpeg operation
used by the pipeline: probing, audio extraction, encoding, progress parsing,
audio/video processing, integrity validation, dedup detection, and batch
execution. Every function validates inputs, handles timeouts, and provides
detailed error context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from utils.logger import get_logger

logger = get_logger("ffmpeg_utils")


# ── Custom Exception ───────────────────────────────────────
class FFmpegError(Exception):
    """Raised when an FFmpeg subprocess exits with a non-zero return code."""

    def __init__(self, message: str, stderr_output: str = "", return_code: int = -1) -> None:
        self.stderr_output = stderr_output
        self.return_code = return_code
        super().__init__(f"{message}\nFFmpeg stderr (last 500 chars):\n{stderr_output[-500:]}")


# ── Data Classes ───────────────────────────────────────────
@dataclass
class VideoInfo:
    """Parsed metadata about a video file from ffprobe."""

    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0
    has_audio: bool = False
    audio_codec: str = ""
    video_codec: str = ""
    bitrate: int = 0
    size_bytes: int = 0
    sample_rate: int = 0
    channels: int = 0
    aspect_ratio: float = 0.0
    bit_depth: int = 0
    color_space: str = ""
    color_range: str = ""
    frame_count: int = 0
    has_subtitles: bool = False
    subtitle_count: int = 0

    @property
    def is_landscape(self) -> bool:
        """Return True if the video is wider than it is tall."""
        return self.width > self.height

    @property
    def is_portrait(self) -> bool:
        """Return True if the video is taller than it is wide."""
        return self.height > self.width

    @property
    def is_shorts_format(self) -> bool:
        """Return True if the video matches YouTube Shorts 9:16 1080x1920."""
        return self.height == 1920 and self.width == 1080


@dataclass
class FFmpegProgress:
    """Real-time progress info from FFmpeg stderr parsing."""

    percent: float = 0.0
    speed: str = "N/A"
    eta: str = "N/A"
    current_time: float = 0.0
    bitrate: str = ""
    frame: int = 0


@dataclass
class BatchResult:
    """Result of a batch FFmpeg operation."""

    successes: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def success_count(self) -> int:
        """Return the number of successful operations."""
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        """Return the number of failed operations."""
        return len(self.failures)


# ══════════════════════════════════════════════════════════
#  Core FFmpeg Functions
# ══════════════════════════════════════════════════════════

def check_ffmpeg() -> bool:
    """Verify that ffmpeg and ffprobe are on PATH and version >= 4.0.

    Returns:
        True if both tools are present and meet the version requirement.

    Raises:
        RuntimeError: If ffmpeg or ffprobe is missing or too old.
    """
    for tool in ("ffmpeg", "ffprobe"):
        try:
            result = subprocess.run(
                [tool, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            first_line = result.stdout.split("\n")[0] if result.stdout else ""
            match = re.search(r"(\d+)\.(\d+)", first_line)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                if major < 4:
                    raise RuntimeError(
                        f"{tool} version {major}.{minor} is too old (need >= 4.0). "
                        f"Please upgrade:\n"
                        f"  macOS:   brew upgrade ffmpeg\n"
                        f"  Ubuntu:  sudo apt install ffmpeg\n"
                        f"  Windows: choco install ffmpeg"
                    )
            else:
                logger.warning("Could not parse %s version from: %s", tool, first_line)
        except FileNotFoundError:
            raise RuntimeError(
                f"{tool} is not installed or not on PATH. Install it:\n"
                f"  macOS:   brew install ffmpeg\n"
                f"  Ubuntu:  sudo apt install ffmpeg\n"
                f"  Windows: choco install ffmpeg"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{tool} -version timed out. Check your installation.")

    logger.debug("FFmpeg and FFprobe are available and meet version requirements.")
    return True


def probe_video(path: Path) -> VideoInfo:
    """Probe a video file and return typed metadata.

    Extracts comprehensive video information including bit depth, color space,
    color range, frame count, and subtitle information in addition to the
    standard width, height, duration, fps, etc.

    Args:
        path: Path to the video file to probe.

    Returns:
        VideoInfo dataclass with width, height, duration, fps, bit_depth,
        color_space, color_range, frame_count, has_subtitles, subtitle_count, etc.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FFmpegError: If ffprobe fails to probe the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd: list[str] = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]

    logger.debug("Probing video: %s", path.name)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise FFmpegError(f"ffprobe failed for {path}", result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        raise FFmpegError(f"ffprobe timed out for {path}")

    info = VideoInfo()

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise FFmpegError(f"ffprobe returned invalid JSON for {path}")

    # ── Parse streams ────────────────────────────────────
    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "video":
            info.width = int(stream.get("width", 0))
            info.height = int(stream.get("height", 0))
            info.video_codec = stream.get("codec_name", "")

            # Parse FPS from r_frame_rate (e.g. "30/1")
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            try:
                num, den = r_frame_rate.split("/")
                info.fps = float(num) / float(den) if float(den) != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                info.fps = 0.0

            # Compute aspect ratio
            if info.height > 0:
                info.aspect_ratio = info.width / info.height

            # Check for display aspect ratio override
            dar = stream.get("display_aspect_ratio", "")
            if dar and ":" in dar:
                try:
                    dw, dh = dar.split(":")
                    info.aspect_ratio = float(dw) / float(dh)
                except (ValueError, ZeroDivisionError):
                    pass

            # Bit depth
            info.bit_depth = int(stream.get("bits_per_raw_sample", 0)) or int(
                stream.get("pix_fmt", "").rstrip("lebe0123456789") and "8"
            )
            pix_fmt = stream.get("pix_fmt", "")
            if "10" in pix_fmt:
                info.bit_depth = 10
            elif "12" in pix_fmt:
                info.bit_depth = 12
            elif "16" in pix_fmt:
                info.bit_depth = 16
            elif info.bit_depth == 0:
                info.bit_depth = 8

            # Color space and range
            info.color_space = stream.get("color_space", "") or stream.get("colorspace", "")
            info.color_range = stream.get("color_range", "")

            # Frame count
            nb_frames = stream.get("nb_frames", "")
            if nb_frames:
                try:
                    info.frame_count = int(nb_frames)
                except ValueError:
                    info.frame_count = 0

        elif codec_type == "audio":
            info.has_audio = True
            info.audio_codec = stream.get("codec_name", "")
            info.sample_rate = int(stream.get("sample_rate", 0))
            info.channels = int(stream.get("channels", 0))

        elif codec_type == "subtitle":
            info.has_subtitles = True
            info.subtitle_count += 1

    # ── Parse format ─────────────────────────────────────
    fmt = data.get("format", {})
    info.duration = float(fmt.get("duration", 0.0))
    info.bitrate = int(fmt.get("bit_rate", 0))
    info.size_bytes = int(fmt.get("size", 0))

    # If nb_frames not in stream, estimate from duration and fps
    if info.frame_count == 0 and info.duration > 0 and info.fps > 0:
        info.frame_count = int(info.duration * info.fps)

    logger.debug(
        "Probed %s: %dx%d (%.3f AR), %.2fs, %.1ffps, audio=%s, vcodec=%s, "
        "bit_depth=%d, color_space=%s, frames=%d, subtitles=%d",
        path.name, info.width, info.height, info.aspect_ratio, info.duration,
        info.fps, info.has_audio, info.video_codec,
        info.bit_depth, info.color_space, info.frame_count, info.subtitle_count,
    )

    return info


def extract_audio_samples(path: Path, sample_interval: float) -> list[float]:
    """Extract per-segment RMS audio level from a video file.

    Uses FFmpeg's astats filter to measure RMS dB at each sample interval.
    Falls back to overall statistics if per-segment extraction fails.

    Args:
        path: Path to the video file.
        sample_interval: Seconds between energy samples.

    Returns:
        List of RMS dB values, one per sample interval.

    Raises:
        FFmpegError: If ffmpeg fails during analysis.
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be positive, got {sample_interval}")

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-af", f"astats=metadata=1:reset={sample_interval},ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f", "null",
        "-",
    ]

    logger.debug("Extracting audio samples (%.1fs interval): %s", sample_interval, path.name)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise FFmpegError("Audio sample extraction timed out (600s)")

    rms_values: list[float] = []
    rms_pattern = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?[\d.]+|nan|-inf)")

    for output_text in (result.stdout, result.stderr):
        for line in output_text.splitlines():
            match = rms_pattern.search(line)
            if match:
                val = match.group(1)
                if val in ("nan", "-inf"):
                    rms_values.append(-60.0)
                else:
                    try:
                        rms_values.append(float(val))
                    except ValueError:
                        rms_values.append(-60.0)

    logger.debug("Extracted %d audio RMS samples from %s", len(rms_values), path.name)
    return rms_values


def run_ffmpeg(
    args: list[str],
    description: str = "",
    show_progress: bool = False,
    total_duration: float = 0.0,
    progress_callback: Optional[Callable[[FFmpegProgress], None]] = None,
    timeout: int = 600,
    use_progress_pipe: bool = False,
) -> subprocess.CompletedProcess:
    """Execute an FFmpeg command with full error handling and optional progress.

    Supports two progress reporting modes:
    - Standard: Parses progress from stderr (default).
    - Progress pipe: Uses -progress pipe:1 for structured key=value output,
      which is more reliable and provides finer-grained updates.

    Args:
        args: Full argument list. If first element is not 'ffmpeg', it's prepended.
        description: Human-readable description for logging.
        show_progress: Whether to parse progress from stderr.
        total_duration: Total duration in seconds for progress calculation.
        progress_callback: Optional callable(FFmpegProgress) invoked with updates.
        timeout: Maximum runtime in seconds (default 600).
        use_progress_pipe: Use -progress pipe:1 for structured progress output.

    Returns:
        subprocess.CompletedProcess with returncode, stdout, stderr.

    Raises:
        FFmpegError: If FFmpeg exits with non-zero return code.
    """
    if not args:
        raise FFmpegError("Empty FFmpeg command")

    if args[0] != "ffmpeg":
        args = ["ffmpeg"] + args

    # Add -hide_banner and -y (overwrite output) if not already present
    if "-hide_banner" not in args:
        args.insert(1, "-hide_banner")
    if "-y" not in args:
        args.insert(1, "-y")

    # Add -progress pipe:1 if requested and not already present
    if use_progress_pipe and "-progress" not in args:
        args.insert(1, "pipe:1")
        args.insert(1, "-progress")

    logger.debug("FFmpeg command: %s", " ".join(args))
    if description:
        logger.info("FFmpeg: %s", description)

    start_time = time.time()

    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except FileNotFoundError:
        raise FFmpegError("ffmpeg binary not found on PATH")
    except OSError as exc:
        raise FFmpegError(f"Failed to start ffmpeg: {exc}")

    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
    speed_pattern = re.compile(r"speed=\s*([\d.]+)x")
    bitrate_pattern = re.compile(r"bitrate=\s*([\d.]+\w+/s)")
    frame_pattern = re.compile(r"frame=\s*(\d+)")

    # Patterns for -progress pipe:1 output
    progress_time_pattern = re.compile(r"^out_time_us=(\d+)$", re.MULTILINE)
    progress_speed_pattern = re.compile(r"^speed=([\d.]+)x$", re.MULTILINE)
    progress_bitrate_pattern = re.compile(r"^bitrate=([\d.]+\w+/s)$", re.MULTILINE)
    progress_frame_pattern = re.compile(r"^frame=(\d+)$", re.MULTILINE)

    stderr_lines: list[str] = []
    stdout_lines: list[str] = []

    def _parse_progress_from_line(line: str) -> None:
        """Parse progress from a single stderr line."""
        time_match = time_pattern.search(line)
        if time_match:
            hours = int(time_match.group(1))
            minutes = int(time_match.group(2))
            seconds = float(time_match.group(3))
            current_time = hours * 3600 + minutes * 60 + seconds
            percent = min(100.0, (current_time / total_duration) * 100.0) if total_duration > 0 else 0.0

            speed_match = speed_pattern.search(line)
            speed = speed_match.group(1) + "x" if speed_match else "N/A"

            bitrate_match = bitrate_pattern.search(line)
            bitrate_val = bitrate_match.group(1) if bitrate_match else ""

            frame_match = frame_pattern.search(line)
            frame = int(frame_match.group(1)) if frame_match else 0

            remaining = total_duration - current_time if total_duration > 0 else 0
            try:
                speed_val = float(speed_match.group(1)) if speed_match else 1.0
                eta_secs = remaining / speed_val if speed_val > 0 else 0
                eta = f"{int(eta_secs // 60)}m{int(eta_secs % 60)}s"
            except (ValueError, ZeroDivisionError):
                eta = "N/A"

            if progress_callback:
                progress_callback(FFmpegProgress(
                    percent=percent, speed=speed, eta=eta,
                    current_time=current_time, bitrate=bitrate_val, frame=frame,
                ))

    def _parse_progress_pipe_output(stdout_text: str) -> None:
        """Parse structured progress from -progress pipe:1 output."""
        time_match = progress_time_pattern.search(stdout_text)
        if time_match:
            current_time = int(time_match.group(1)) / 1_000_000
            percent = min(100.0, (current_time / total_duration) * 100.0) if total_duration > 0 else 0.0

            speed_match = progress_speed_pattern.search(stdout_text)
            speed = speed_match.group(1) + "x" if speed_match else "N/A"

            bitrate_match = progress_bitrate_pattern.search(stdout_text)
            bitrate_val = bitrate_match.group(1) if bitrate_match else ""

            frame_match = progress_frame_pattern.search(stdout_text)
            frame = int(frame_match.group(1)) if frame_match else 0

            remaining = total_duration - current_time if total_duration > 0 else 0
            try:
                speed_val = float(speed_match.group(1)) if speed_match else 1.0
                eta_secs = remaining / speed_val if speed_val > 0 else 0
                eta = f"{int(eta_secs // 60)}m{int(eta_secs % 60)}s"
            except (ValueError, ZeroDivisionError):
                eta = "N/A"

            if progress_callback:
                progress_callback(FFmpegProgress(
                    percent=percent, speed=speed, eta=eta,
                    current_time=current_time, bitrate=bitrate_val, frame=frame,
                ))

    # Read stdout and stderr
    if process.stdout is not None:
        for line in process.stdout:
            stdout_lines.append(line.rstrip())
            if use_progress_pipe and show_progress:
                _parse_progress_pipe_output(line)

    if process.stderr is not None:
        for line in process.stderr:
            stderr_lines.append(line.rstrip())
            if show_progress and not use_progress_pipe:
                _parse_progress_from_line(line)

    try:
        return_code = process.wait(timeout=max(1, timeout - (time.time() - start_time)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise FFmpegError(
            f"FFmpeg timed out after {timeout}s: {description}",
            "\n".join(stderr_lines[-20:]),
        )

    stderr_output = "\n".join(stderr_lines)

    if return_code != 0:
        logger.error("FFmpeg failed (rc=%d): %s", return_code, description)
        error_lines = [l for l in stderr_lines if l.strip() and not l.startswith("frame=")]
        error_summary = "\n".join(error_lines[-5:]) if error_lines else stderr_output[-500:]
        raise FFmpegError(
            f"FFmpeg failed with return code {return_code}: {description}",
            error_summary,
            return_code,
        )

    elapsed = time.time() - start_time
    logger.debug("FFmpeg completed in %.1fs: %s", elapsed, description)

    return subprocess.CompletedProcess(
        args=args, returncode=return_code,
        stdout="\n".join(stdout_lines), stderr=stderr_output,
    )


def extract_audio_wav(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """Extract audio from a video file as mono 16 kHz WAV.

    Args:
        video_path: Path to the source video.
        output_path: Destination path for the WAV file.
        sample_rate: Target sample rate in Hz (default 16000 for Whisper).

    Returns:
        Path to the written WAV file.

    Raises:
        FileNotFoundError: If video doesn't exist.
        FFmpegError: If extraction fails.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        str(output_path),
    ]

    logger.debug("Extracting audio WAV: %s -> %s", video_path.name, output_path.name)
    run_ffmpeg(cmd, description=f"Extract audio WAV from {video_path.name}")

    if not output_path.exists():
        raise FFmpegError(f"WAV extraction completed but file not found: {output_path}")

    return output_path


def get_video_thumbnail(
    path: Path,
    timestamp: float,
    output_path: Path,
) -> Path:
    """Extract a single frame from a video as a JPEG thumbnail.

    Args:
        path: Path to the source video.
        timestamp: Time in seconds to extract the frame at.
        output_path: Destination path for the JPEG thumbnail.

    Returns:
        Path to the written JPEG file.

    Raises:
        FileNotFoundError: If video doesn't exist.
        FFmpegError: If thumbnail extraction fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", str(path),
        "-vframes", "1",
        "-q:v", "2",
        str(output_path),
    ]

    logger.debug("Extracting thumbnail at %.1fs: %s", timestamp, path.name)
    run_ffmpeg(cmd, description=f"Extract thumbnail from {path.name} at {timestamp:.1f}s")

    if not output_path.exists():
        raise FFmpegError(f"Thumbnail was not created at {output_path}")

    return output_path


# ══════════════════════════════════════════════════════════
#  New Frame Extraction Functions
# ══════════════════════════════════════════════════════════

def get_video_frames(path: Path, timestamps: list[float], output_dir: Path | None = None) -> list[Path]:
    """Extract multiple frames at given timestamps from a video.

    Creates JPEG images named frame_001.jpg, frame_002.jpg, etc. in the
    output directory.

    Args:
        path: Path to the source video file.
        timestamps: List of timestamps in seconds to extract frames at.
        output_dir: Directory for output frames. Defaults to a temp directory.

    Returns:
        List of Paths to the extracted frame images, in the same order as timestamps.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        FFmpegError: If frame extraction fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not timestamps:
        return []

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="frames_"))

    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []
    for idx, ts in enumerate(timestamps, start=1):
        out_path = output_dir / f"frame_{idx:03d}.jpg"
        try:
            get_video_thumbnail(path, ts, out_path)
            result_paths.append(out_path)
        except FFmpegError as exc:
            logger.warning("Failed to extract frame at %.2fs from %s: %s", ts, path.name, exc)
            continue

    logger.debug("Extracted %d/%d frames from %s", len(result_paths), len(timestamps), path.name)
    return result_paths


# ══════════════════════════════════════════════════════════
#  Audio Processing Functions
# ══════════════════════════════════════════════════════════

def extract_audio_segment(
    path: Path,
    start: float,
    duration: float,
    output_path: Path,
) -> Path:
    """Extract a segment of audio from a video file.

    Args:
        path: Path to the source video.
        start: Start time in seconds.
        duration: Duration of the segment in seconds.
        output_path: Destination path for the audio segment.

    Returns:
        Path to the written audio file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If extraction fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if start < 0:
        raise ValueError(f"Start time must be non-negative, got {start}")
    if duration <= 0:
        raise ValueError(f"Duration must be positive, got {duration}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-ss", str(start),
        "-t", str(duration),
        "-vn",
        "-acodec", "copy",
        str(output_path),
    ]

    logger.debug("Extracting audio segment %.1fs+%.1fs: %s", start, duration, path.name)
    run_ffmpeg(cmd, description=f"Extract audio segment from {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Audio segment extraction failed: {output_path}")

    return output_path


def normalize_audio_loudness(
    path: Path,
    output_path: Path,
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
    lra: float = 11.0,
) -> Path:
    """Normalize audio loudness to EBU R128 standards using the loudnorm filter.

    Two-pass normalization for maximum accuracy: first pass analyses the
    source loudness, second pass applies the correction.

    Args:
        path: Path to the source audio/video file.
        output_path: Destination path for the normalized file.
        target_lufs: Target integrated loudness in LUFS (default -16).
        true_peak: Maximum true peak in dBTP (default -1.5).
        lra: Target loudness range in LU (default 11).

    Returns:
        Path to the normalized output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If normalization fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio/video file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: analyze loudness
    cmd_pass1: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-af", (
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}"
            ":print_format=json"
        ),
        "-f", "null",
        "-",
    ]

    logger.debug("Pass 1: Analyzing loudness for %s", path.name)
    result = run_ffmpeg(cmd_pass1, description=f"Loudnorm analysis for {path.name}")

    # Parse the JSON loudness stats from stderr
    loudnorm_data: dict[str, str] = {}
    json_match = re.search(r"\{[^}]+\}", result.stderr, re.DOTALL)
    if json_match:
        try:
            loudnorm_data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.warning("Could not parse loudnorm JSON, using single-pass normalization")

    if loudnorm_data:
        # Second pass: apply measured normalization
        measured_i = loudnorm_data.get("input_i", target_lufs)
        measured_tp = loudnorm_data.get("input_tp", true_peak)
        measured_lra = loudnorm_data.get("input_lra", lra)
        measured_thresh = loudnorm_data.get("input_thresh", "-70.0")
        target_offset = loudnorm_data.get("target_offset", "0.0")

        filter_str = (
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}"
            f":measured_I={measured_i}:measured_TP={measured_tp}"
            f":measured_LRA={measured_lra}:measured_thresh={measured_thresh}"
            f":offset={target_offset}:linear=true:print_format=json"
        )

        cmd_pass2: list[str] = [
            "ffmpeg",
            "-i", str(path),
            "-af", filter_str,
            "-c:v", "copy",
            str(output_path),
        ]

        logger.debug("Pass 2: Applying loudnorm correction to %s", path.name)
        run_ffmpeg(cmd_pass2, description=f"Loudnorm correction for {path.name}")
    else:
        # Fallback: single-pass normalization
        cmd_single: list[str] = [
            "ffmpeg",
            "-i", str(path),
            "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}",
            "-c:v", "copy",
            str(output_path),
        ]

        logger.debug("Single-pass loudnorm for %s", path.name)
        run_ffmpeg(cmd_single, description=f"Single-pass loudnorm for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Audio normalization output not found: {output_path}")

    return output_path


def apply_noise_reduction(
    path: Path,
    output_path: Path,
    strength: str = "medium",
) -> Path:
    """Apply noise reduction using FFmpeg's afftdn filter.

    The afftdn (FFT denoise) filter reduces stationary background noise.
    Strength maps to the noise reduction amount (nr parameter).

    Args:
        path: Path to the source audio/video file.
        output_path: Destination path for the processed file.
        strength: Noise reduction strength: 'light' (nr=6), 'medium' (nr=12),
                  or 'heavy' (nr=20).

    Returns:
        Path to the processed output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If processing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio/video file not found: {path}")

    strength_map: dict[str, int] = {"light": 6, "medium": 12, "heavy": 20}
    nr_value = strength_map.get(strength, 12)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-af", f"afftdn=nf=-25:nr={nr_value}:tn=1",
        "-c:v", "copy",
        str(output_path),
    ]

    logger.debug("Applying noise reduction (strength=%s, nr=%d): %s", strength, nr_value, path.name)
    run_ffmpeg(cmd, description=f"Noise reduction ({strength}) for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Noise reduction output not found: {output_path}")

    return output_path


def apply_audio_compressor(
    path: Path,
    output_path: Path,
    threshold: str = "-20dB",
    ratio: int = 4,
    attack: int = 5,
    release: int = 50,
) -> Path:
    """Apply dynamic range compression to audio using FFmpeg's acompressor filter.

    Reduces the volume of loud sounds or amplifies quiet sounds by narrowing
    the dynamic range.

    Args:
        path: Path to the source audio/video file.
        output_path: Destination path for the processed file.
        threshold: Threshold level in dB (default '-20dB').
        ratio: Compression ratio (default 4:1).
        attack: Attack time in milliseconds (default 5).
        release: Release time in milliseconds (default 50).

    Returns:
        Path to the processed output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If processing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio/video file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-af", f"acompressor=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}",
        "-c:v", "copy",
        str(output_path),
    ]

    logger.debug("Applying audio compressor (threshold=%s, ratio=%d): %s", threshold, ratio, path.name)
    run_ffmpeg(cmd, description=f"Audio compression for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Audio compression output not found: {output_path}")

    return output_path


def apply_deesser(path: Path, output_path: Path) -> Path:
    """Apply de-essing to reduce sibilance in speech using FFmpeg's deesser filter.

    De-essing reduces harsh 's', 'sh', and 'ch' sounds that are common
    in voice recordings, especially with condenser microphones.

    Args:
        path: Path to the source audio/video file.
        output_path: Destination path for the processed file.

    Returns:
        Path to the processed output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If processing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio/video file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-af", "deesser=i=0.4:m=0.5:f=0.5",
        "-c:v", "copy",
        str(output_path),
    ]

    logger.debug("Applying de-esser: %s", path.name)
    run_ffmpeg(cmd, description=f"De-esser for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"De-esser output not found: {output_path}")

    return output_path


# ══════════════════════════════════════════════════════════
#  Video Processing Functions
# ══════════════════════════════════════════════════════════

def apply_video_denoise(
    path: Path,
    output_path: Path,
    strength: str = "light",
) -> Path:
    """Apply video denoising using hqdn3d (light/medium) or nlmeans (heavy).

    hqdn3d is a fast, high-quality denoising filter suitable for light to
    moderate noise. nlmeans provides superior quality for heavy noise but
    is significantly slower.

    Args:
        path: Path to the source video file.
        output_path: Destination path for the processed file.
        strength: Denoising strength: 'light', 'medium', or 'heavy'.

    Returns:
        Path to the processed output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If processing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if strength == "heavy":
        # nlmeans for heavy denoising (slow but high quality)
        filter_str = "nlmeans=s=3.0:p=7:r=15"
    elif strength == "medium":
        filter_str = "hqdn3d=4.0:3.0:6.0:4.5"
    else:
        # light
        filter_str = "hqdn3d=2.0:1.5:3.0:2.5"

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-vf", filter_str,
        "-c:a", "copy",
        str(output_path),
    ]

    logger.debug("Applying video denoise (strength=%s): %s", strength, path.name)
    run_ffmpeg(cmd, description=f"Video denoise ({strength}) for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Video denoise output not found: {output_path}")

    return output_path


def apply_video_sharpen(
    path: Path,
    output_path: Path,
    amount: float = 1.0,
) -> Path:
    """Apply video sharpening using FFmpeg's unsharp filter.

    Enhances edges and fine details in the video. Use sparingly as
    over-sharpening can introduce artifacts.

    Args:
        path: Path to the source video file.
        output_path: Destination path for the processed file.
        amount: Sharpening amount from 0.0 to 5.0 (default 1.0).

    Returns:
        Path to the processed output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If processing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not 0.0 <= amount <= 5.0:
        raise ValueError(f"Sharpen amount must be between 0.0 and 5.0, got {amount}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    luma_amount = amount
    chroma_amount = amount * 0.5

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-vf", f"unsharp=5:5:{luma_amount:.1f}:5:5:{chroma_amount:.1f}",
        "-c:a", "copy",
        str(output_path),
    ]

    logger.debug("Applying video sharpen (amount=%.1f): %s", amount, path.name)
    run_ffmpeg(cmd, description=f"Video sharpen for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Video sharpen output not found: {output_path}")

    return output_path


def add_film_grain(
    path: Path,
    output_path: Path,
    amount: int = 4,
) -> Path:
    """Add film grain effect using FFmpeg's noise filter.

    Adds a subtle, cinematic grain effect that can make digital video
    appear more film-like and can help mask compression artifacts.

    Args:
        path: Path to the source video file.
        output_path: Destination path for the processed file.
        amount: Grain intensity from 0 to 32 (default 4). 0 disables grain.

    Returns:
        Path to the processed output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If processing fails.
        ValueError: If amount is out of range.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not 0 <= amount <= 32:
        raise ValueError(f"Film grain amount must be between 0 and 32, got {amount}")

    if amount == 0:
        # No grain requested, just copy
        shutil.copy2(str(path), str(output_path))
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-vf", f"noise=c0s={amount}:allf=t",
        "-c:a", "copy",
        str(output_path),
    ]

    logger.debug("Adding film grain (amount=%d): %s", amount, path.name)
    run_ffmpeg(cmd, description=f"Film grain for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Film grain output not found: {output_path}")

    return output_path


# ══════════════════════════════════════════════════════════
#  Detection Functions
# ══════════════════════════════════════════════════════════

def detect_scene_changes(
    path: Path,
    threshold: float = 0.3,
    timeout: int = 300,
) -> list[float]:
    """Detect scene change timestamps in a video file.

    Args:
        path: Path to the video file.
        threshold: Scene detection threshold (0-1, default 0.3).
        timeout: Maximum runtime in seconds.

    Returns:
        List of timestamps (in seconds) where scene changes occur.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Scene detection timed out for %s", path.name)
        return []

    timestamps: list[float] = []
    pts_pattern = re.compile(r"pts_time:(\d+\.?\d*)")
    for line in result.stderr.splitlines():
        match = pts_pattern.search(line)
        if match:
            timestamps.append(float(match.group(1)))

    logger.debug("Detected %d scene changes in %s", len(timestamps), path.name)
    return timestamps


def detect_silence(
    path: Path,
    noise_floor: str = "-30dB",
    min_duration: float = 1.0,
    timeout: int = 300,
) -> list[tuple[float, float]]:
    """Detect silent regions in a video file.

    Args:
        path: Path to the video file.
        noise_floor: Noise threshold (e.g. '-30dB').
        min_duration: Minimum silence duration in seconds.
        timeout: Maximum runtime in seconds.

    Returns:
        List of (start, end) tuples for silent regions.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-af", f"silencedetect=noise={noise_floor}:d={min_duration}",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Silence detection timed out for %s", path.name)
        return []

    starts: list[float] = []
    ends: list[float] = []
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([\d.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))
        end_match = re.search(r"silence_end:\s*([\d.]+)", line)
        if end_match:
            ends.append(float(end_match.group(1)))

    regions: list[tuple[float, float]] = []
    for i in range(len(starts)):
        end = ends[i] if i < len(ends) else float("inf")
        regions.append((starts[i], end))

    logger.debug("Detected %d silent regions in %s", len(regions), path.name)
    return regions


def detect_black_frames(
    path: Path,
    threshold: float = 0.98,
    min_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """Detect black frame regions in a video file.

    Uses FFmpeg's blackdetect filter to find sequences of black frames,
    which typically indicate transitions, intros/outros, or missing content.

    Args:
        path: Path to the video file.
        threshold: Black threshold (0-1, default 0.98). Higher = stricter.
        min_duration: Minimum black duration in seconds (default 0.5).

    Returns:
        List of (start, end) tuples for black frame regions.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FFmpegError: If detection fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-vf", f"blackdetect=d={min_duration}:pic_th={threshold}",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("Black frame detection timed out for %s", path.name)
        return []

    regions: list[tuple[float, float]] = []
    pattern = re.compile(r"blackdetect:start:([\d.]+)\s+end:([\d.]+)")
    for line in result.stderr.splitlines():
        match = pattern.search(line)
        if match:
            start = float(match.group(1))
            end = float(match.group(2))
            regions.append((start, end))

    logger.debug("Detected %d black frame regions in %s", len(regions), path.name)
    return regions


def detect_sustained_audio(
    path: Path,
    min_duration: float = 2.0,
    threshold: str = "-30dB",
) -> list[tuple[float, float]]:
    """Detect regions with sustained (non-silent) audio.

    Complements detect_silence by finding the inverse — regions where
    audio is present above the threshold for at least min_duration seconds.

    Args:
        path: Path to the video file.
        min_duration: Minimum sustained audio duration in seconds (default 2.0).
        threshold: Audio level threshold (e.g. '-30dB').

    Returns:
        List of (start, end) tuples for sustained audio regions.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FFmpegError: If detection fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    # Use silencedetect and invert the results
    silence_regions = detect_silence(path, noise_floor=threshold, min_duration=0.1)

    # Get total duration
    info = probe_video(path)
    if info.duration <= 0:
        return []

    # Invert silence regions to get audio regions
    audio_regions: list[tuple[float, float]] = []
    prev_end = 0.0

    for silence_start, silence_end in silence_regions:
        if silence_start > prev_end:
            gap_duration = silence_start - prev_end
            if gap_duration >= min_duration:
                audio_regions.append((prev_end, silence_start))
        prev_end = min(silence_end, info.duration) if silence_end != float("inf") else info.duration

    # Check trailing audio after last silence
    if prev_end < info.duration:
        trailing_duration = info.duration - prev_end
        if trailing_duration >= min_duration:
            audio_regions.append((prev_end, info.duration))

    logger.debug("Detected %d sustained audio regions in %s", len(audio_regions), path.name)
    return audio_regions


# ══════════════════════════════════════════════════════════
#  HW Encoder Detection
# ══════════════════════════════════════════════════════════

def detect_hw_encoder() -> tuple[str, str]:
    """Detect available hardware video encoders.

    Checks for NVENC, VideoToolbox, VAAPI, QSV, AMF, and Media Foundation
    encoders in priority order. Returns the first available encoder along
    with a recommended preset.

    Returns:
        Tuple of (encoder_name, preset). Both empty string if none available.
    """
    encoders_to_try: list[tuple[str, str]] = [
        ("h264_nvenc", "p4"),
        ("hevc_nvenc", "p4"),
        ("h264_videotoolbox", ""),
        ("h264_vaapi", ""),
        ("h264_qsv", "medium"),
        ("h264_amf", "balanced"),
        ("h264_mf", ""),
    ]

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        for enc_name, preset in encoders_to_try:
            if enc_name in result.stdout:
                logger.info("Detected hardware encoder: %s", enc_name)
                return enc_name, preset
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return "", ""


# ══════════════════════════════════════════════════════════
#  Video Integrity & Analysis Functions
# ══════════════════════════════════════════════════════════

def get_video_bitrate(path: Path) -> int:
    """Get the actual bitrate of a video file in bits per second.

    Args:
        path: Path to the video file.

    Returns:
        Bitrate in bits per second, or 0 if unable to determine.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    try:
        info = probe_video(path)
        return info.bitrate
    except FFmpegError:
        # Fallback: estimate from file size and duration
        try:
            size_bytes = path.stat().st_size
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                if duration > 0:
                    return int((size_bytes * 8) / duration)
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass

    return 0


def validate_video_integrity(path: Path) -> bool:
    """Check if a video file is valid and not corrupt.

    Runs ffprobe to verify the file can be parsed and ffmpeg to decode
    a few frames as an integrity check.

    Args:
        path: Path to the video file to validate.

    Returns:
        True if the video appears valid, False otherwise.
    """
    if not path.exists():
        logger.warning("File does not exist: %s", path)
        return False

    if path.stat().st_size == 0:
        logger.warning("File is empty: %s", path)
        return False

    # Step 1: ffprobe can read the file
    try:
        info = probe_video(path)
        if info.duration <= 0 and info.frame_count == 0:
            logger.warning("Video has no duration or frames: %s", path.name)
            return False
    except FFmpegError:
        logger.warning("ffprobe cannot parse file: %s", path.name)
        return False

    # Step 2: ffmpeg can decode at least the first frame
    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
        "-vframes", "1",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("FFmpeg cannot decode first frame: %s", path.name)
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Video integrity check timed out: %s", path.name)
        return False

    logger.debug("Video integrity check passed: %s", path.name)
    return True


def get_keyframe_timestamps(path: Path) -> list[float]:
    """Get keyframe (I-frame) positions for making clean cuts.

    Extracts the presentation timestamp (PTS) of every keyframe in the
    video. Cutting at keyframe boundaries avoids re-encoding and ensures
    glitch-free playback.

    Args:
        path: Path to the video file.

    Returns:
        List of keyframe timestamps in seconds, sorted ascending.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FFmpegError: If keyframe extraction fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd: list[str] = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "frame=pts_time,key_frame",
        "-of", "csv=p=0",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise FFmpegError(f"Keyframe extraction timed out for {path}")

    if result.returncode != 0:
        raise FFmpegError(f"ffprobe keyframe extraction failed for {path}", result.stderr, result.returncode)

    timestamps: list[float] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        # Format: pts_time,key_frame=1
        if len(parts) >= 2 and "1" in parts[1]:
            try:
                ts = float(parts[0])
                timestamps.append(ts)
            except ValueError:
                continue

    timestamps.sort()
    logger.debug("Found %d keyframes in %s", len(timestamps), path.name)
    return timestamps


# ══════════════════════════════════════════════════════════
#  Video Manipulation Functions
# ══════════════════════════════════════════════════════════

def concat_videos(video_paths: list[Path], output_path: Path) -> Path:
    """Concatenate multiple video files into one.

    Uses FFmpeg's concat demuxer for lossless concatenation. All videos
    must have the same codecs and parameters for best results.

    Args:
        video_paths: Ordered list of video file paths to concatenate.
        output_path: Destination path for the concatenated video.

    Returns:
        Path to the concatenated output file.

    Raises:
        FileNotFoundError: If any input file doesn't exist.
        FFmpegError: If concatenation fails.
        ValueError: If the video_paths list is empty.
    """
    if not video_paths:
        raise ValueError("video_paths must not be empty")

    for vp in video_paths:
        if not vp.exists():
            raise FileNotFoundError(f"Video file not found: {vp}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary concat list file
    concat_content = "\n".join(
        f"file '{vp.resolve()}'" for vp in video_paths
    )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="concat_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(concat_content)

        cmd: list[str] = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", tmp_path,
            "-c", "copy",
            str(output_path),
        ]

        logger.debug("Concatenating %d videos -> %s", len(video_paths), output_path.name)
        run_ffmpeg(cmd, description=f"Concatenate {len(video_paths)} videos")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not output_path.exists():
        raise FFmpegError(f"Concat output not found: {output_path}")

    return output_path


def add_silence_padding(
    path: Path,
    output_path: Path,
    start_padding: float = 0.0,
    end_padding: float = 0.5,
) -> Path:
    """Add silence padding at the start and/or end of a video.

    Generates silent black frames for the specified durations and
    concatenates them with the original video.

    Args:
        path: Path to the source video file.
        output_path: Destination path for the padded video.
        start_padding: Seconds of silence to add at the start (default 0.0).
        end_padding: Seconds of silence to add at the end (default 0.5).

    Returns:
        Path to the padded output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If padding fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if start_padding < 0 or end_padding < 0:
        raise ValueError("Padding values must be non-negative")

    if start_padding == 0 and end_padding == 0:
        shutil.copy2(str(path), str(output_path))
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe_video(path)
    width = info.width or 1080
    height = info.height or 1920
    fps = info.fps or 30
    sample_rate = info.sample_rate or 44100

    segments: list[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="padding_"))

    try:
        # Create start padding if needed
        if start_padding > 0:
            start_pad = tmp_dir / "start_pad.mp4"
            cmd: list[str] = [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"color=c=black:s={width}x{height}:d={start_padding}:r={fps}",
                "-f", "lavfi",
                "-i", f"anullsrc=r={sample_rate}:cl=stereo",
                "-t", str(start_padding),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                "-shortest",
                str(start_pad),
            ]
            run_ffmpeg(cmd, description="Generate start silence padding")
            segments.append(start_pad)

        segments.append(path)

        # Create end padding if needed
        if end_padding > 0:
            end_pad = tmp_dir / "end_pad.mp4"
            cmd = [
                "ffmpeg",
                "-f", "lavfi",
                "-i", f"color=c=black:s={width}x{height}:d={end_padding}:r={fps}",
                "-f", "lavfi",
                "-i", f"anullsrc=r={sample_rate}:cl=stereo",
                "-t", str(end_padding),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                "-shortest",
                str(end_pad),
            ]
            run_ffmpeg(cmd, description="Generate end silence padding")
            segments.append(end_pad)

        # Concatenate all segments
        concat_videos(segments, output_path)

    finally:
        # Cleanup temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not output_path.exists():
        raise FFmpegError(f"Silence padding output not found: {output_path}")

    return output_path


def add_fade_effects(
    path: Path,
    output_path: Path,
    fade_in: float = 0.3,
    fade_out: float = 0.3,
) -> Path:
    """Add fade-in and fade-out effects to a video.

    Applies both video and audio fades simultaneously for smooth
    transitions at the start and end of the video.

    Args:
        path: Path to the source video file.
        output_path: Destination path for the processed file.
        fade_in: Duration of fade-in in seconds (default 0.3).
        fade_out: Duration of fade-out in seconds (default 0.3).

    Returns:
        Path to the processed output file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If processing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if fade_in < 0 or fade_out < 0:
        raise ValueError("Fade durations must be non-negative")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe_video(path)
    duration = info.duration

    filters: list[str] = []
    audio_filters: list[str] = []

    if fade_in > 0:
        filters.append(f"fade=t=in:st=0:d={fade_in}")
        audio_filters.append(f"afade=t=in:st=0:d={fade_in}")

    if fade_out > 0 and duration > 0:
        fade_out_start = max(0, duration - fade_out)
        filters.append(f"fade=t=out:st={fade_out_start:.3f}:d={fade_out}")
        audio_filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out}")

    if not filters and not audio_filters:
        shutil.copy2(str(path), str(output_path))
        return output_path

    vf = ",".join(filters)
    af = ",".join(audio_filters)

    cmd: list[str] = [
        "ffmpeg",
        "-i", str(path),
    ]

    if vf:
        cmd.extend(["-vf", vf])
    if af:
        cmd.extend(["-af", af])

    cmd.append(str(output_path))

    logger.debug("Adding fade effects (in=%.1fs, out=%.1fs): %s", fade_in, fade_out, path.name)
    run_ffmpeg(cmd, description=f"Fade effects for {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Fade effects output not found: {output_path}")

    return output_path


def extract_clip_precise(
    path: Path,
    start: float,
    end: float,
    output_path: Path,
    keyframe_cut: bool = True,
) -> Path:
    """Extract a clip with frame-accurate precision.

    When keyframe_cut is True, the start time is adjusted to the nearest
    preceding keyframe to avoid re-encoding. When False, re-encoding is
    used for exact timestamp cuts.

    Args:
        path: Path to the source video file.
        start: Start time in seconds.
        end: End time in seconds.
        output_path: Destination path for the extracted clip.
        keyframe_cut: If True, adjust start to nearest keyframe for stream copy.

    Returns:
        Path to the extracted clip.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FFmpegError: If extraction fails.
        ValueError: If start >= end or times are negative.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if start < 0:
        raise ValueError(f"Start time must be non-negative, got {start}")
    if end <= start:
        raise ValueError(f"End time must be greater than start, got start={start} end={end}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual_start = start

    if keyframe_cut:
        # Adjust start to the nearest preceding keyframe
        try:
            keyframes = get_keyframe_timestamps(path)
            # Find the last keyframe before or at start
            preceding_keyframes = [kf for kf in keyframes if kf <= start]
            if preceding_keyframes:
                actual_start = preceding_keyframes[-1]
                logger.debug(
                    "Adjusted clip start from %.3fs to keyframe at %.3fs",
                    start, actual_start,
                )
        except FFmpegError:
            logger.warning("Could not get keyframes, using exact cut")

    duration = end - actual_start

    if keyframe_cut:
        # Stream copy for speed (no re-encode)
        cmd: list[str] = [
            "ffmpeg",
            "-ss", str(actual_start),
            "-i", str(path),
            "-t", str(duration),
            "-c", "copy",
            str(output_path),
        ]
    else:
        # Re-encode for frame accuracy
        cmd = [
            "ffmpeg",
            "-ss", str(actual_start),
            "-i", str(path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            str(output_path),
        ]

    logger.debug(
        "Extracting clip %.3fs-%.3fs (keyframe=%s): %s",
        actual_start, end, keyframe_cut, path.name,
    )
    run_ffmpeg(cmd, description=f"Extract clip from {path.name}")

    if not output_path.exists():
        raise FFmpegError(f"Clip extraction output not found: {output_path}")

    return output_path


# ══════════════════════════════════════════════════════════
#  Dedup Detection
# ══════════════════════════════════════════════════════════

def compute_perceptual_hash(path: Path, timestamp: float = 0) -> str:
    """Compute a perceptual hash of a video frame for dedup detection.

    Extracts a frame at the given timestamp, resizes it to 8x8 grayscale,
    and computes an MD5 hash. Videos with identical or near-identical
    frames at the same timestamp will have the same or very similar hashes.

    Args:
        path: Path to the video file.
        timestamp: Time in seconds to sample the frame (default 0).

    Returns:
        Hex string of the perceptual hash.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FFmpegError: If frame extraction fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    # Extract a small grayscale frame and hash it
    cmd: list[str] = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", str(path),
        "-vframes", "1",
        "-vf", "scale=8:8:force_original_aspect_ratio=disable,format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise FFmpegError(f"Perceptual hash frame extraction failed for {path.name}")
    except subprocess.TimeoutExpired:
        raise FFmpegError(f"Perceptual hash extraction timed out for {path.name}")

    # Hash the raw pixel data
    return hashlib.md5(result.stdout).hexdigest()


# ══════════════════════════════════════════════════════════
#  FFmpegBatch Class
# ══════════════════════════════════════════════════════════

class FFmpegBatch:
    """Run multiple FFmpeg commands in sequence with shared error handling.

    Provides a convenient way to execute a batch of FFmpeg operations
    with automatic error collection, timing, and summary reporting.

    Example:
        batch = FFmpegBatch()
        batch.add(extract_audio_wav, video_path, wav_path)
        batch.add(normalize_audio_loudness, wav_path, norm_path)
        result = batch.execute()
        print(f"Successes: {result.success_count}, Failures: {result.failure_count}")
    """

    def __init__(self, stop_on_error: bool = False) -> None:
        """Initialise the batch executor.

        Args:
            stop_on_error: If True, stop executing on the first error.
                          If False, continue with remaining commands.
        """
        self.stop_on_error = stop_on_error
        self._commands: list[tuple[Callable, tuple, dict]] = []

    def add(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Add an FFmpeg operation to the batch.

        Args:
            func: FFmpeg utility function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        """
        self._commands.append((func, args, kwargs))

    def execute(self) -> BatchResult:
        """Execute all queued FFmpeg operations in sequence.

        Returns:
            BatchResult with lists of successes and failures.
        """
        result = BatchResult()
        start_time = time.time()

        for func, args, kwargs in self._commands:
            func_name = getattr(func, "__name__", str(func))
            try:
                logger.debug("Batch: executing %s", func_name)
                func(*args, **kwargs)
                result.successes.append(func_name)
                logger.debug("Batch: %s succeeded", func_name)
            except (FFmpegError, FileNotFoundError, ValueError, OSError) as exc:
                error_msg = f"{func_name}: {exc}"
                result.failures.append((func_name, str(exc)))
                logger.error("Batch: %s failed: %s", func_name, exc)
                if self.stop_on_error:
                    logger.warning("Batch: stopping on first error")
                    break

        result.total_duration = time.time() - start_time
        logger.info(
            "Batch complete: %d succeeded, %d failed in %.1fs",
            result.success_count, result.failure_count, result.total_duration,
        )
        return result

    @property
    def command_count(self) -> int:
        """Return the number of queued commands."""
        return len(self._commands)
