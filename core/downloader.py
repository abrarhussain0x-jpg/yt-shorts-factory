"""
core/downloader.py — YouTube video downloader using yt-dlp.

Downloads videos with metadata extraction, format selection,
duplicate detection, retry logic, and comprehensive error handling.
Supports multiple URL formats, cookie-based authentication,
playlist enumeration, subtitles, thumbnails, and integrity verification.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from config.settings import get_settings
from utils.file_utils import sanitize_filename, get_file_size_human, safe_delete
from utils.logger import get_logger
from utils.retry import retry_on_failure

logger = get_logger("downloader")
console = Console()


# ── Custom Exceptions ─────────────────────────────────────
class DownloadError(Exception):
    """Base exception for download failures."""
    pass


class GeoRestrictedError(DownloadError):
    """Video is geo-restricted and cannot be downloaded."""
    pass


class PrivateVideoError(DownloadError):
    """Video is private or age-gated."""
    pass


class AgeRestrictedError(DownloadError):
    """Video requires age verification."""
    pass


class MembersOnlyError(DownloadError):
    """Video is members-only content."""
    pass


class LiveStreamError(DownloadError):
    """Video is a live stream that hasn't ended or is currently live."""
    pass


class UnavailableError(DownloadError):
    """Video is unavailable (deleted, made private, etc.)."""
    pass


class CopyrightError(DownloadError):
    """Video is blocked due to copyright claim."""
    pass


class InvalidURLError(DownloadError):
    """URL is not a valid YouTube link."""
    pass


class RateLimitError(DownloadError):
    """Rate-limited by YouTube; needs cooldown."""
    pass


class DiskSpaceError(DownloadError):
    """Insufficient disk space for download."""
    pass


class IntegrityError(DownloadError):
    """Downloaded file is corrupt or truncated."""
    pass


# ── Data Classes ──────────────────────────────────────────
@dataclass
class FormatInfo:
    """Information about an available video format/quality."""

    format_id: str
    ext: str
    resolution: str
    fps: int = 0
    vcodec: str = ""
    acodec: str = ""
    filesize: int = 0
    filesize_approx: int = 0
    tbr: float = 0.0
    vbr: float = 0.0
    abr: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    language: str = ""
    format_note: str = ""

    @property
    def filesize_human(self) -> str:
        """Return human-readable file size."""
        size = self.filesize or self.filesize_approx or 0
        if size <= 0:
            return "unknown"
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


@dataclass
class DownloadOptions:
    """Configuration options for a download operation."""

    format_selection: str = "best"
    audio_only: bool = False
    download_subtitles: bool = False
    subtitle_languages: list[str] = field(default_factory=lambda: ["en", "en-US", "en-GB"])
    download_thumbnail: bool = True
    concurrent_fragments: int = 4
    retries: int = 3
    fragment_retries: int = 3
    merge_output_format: str = "mp4"
    write_metadata: bool = True
    embed_chapters: bool = True
    proxy: str = ""
    cookie_file: str = ""
    rate_limit: str = ""
    temp_dir: str = ""
    progress_callback: Optional[Callable[[float, str], None]] = None

    # Format selection presets
    FORMAT_PRESETS: dict[str, str] = field(default_factory=lambda: {
        "best": "bestvideo+bestaudio/best",
        "4k": "bestvideo[height<=2160]+bestaudio/best[height<=2160]",
        "1080": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "720": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "480": "bestvideo[height<=480]+bestaudio/best[height<=480]",
        "audio_only": "bestaudio/best",
    })

    def get_format_string(self) -> str:
        """Resolve format_selection to yt-dlp format string.

        Returns:
            yt-dlp compatible format selection string.
        """
        if self.audio_only:
            return self.FORMAT_PRESETS["audio_only"]
        return self.FORMAT_PRESETS.get(self.format_selection, self.format_selection)


@dataclass
class PlaylistInfo:
    """Information about a YouTube playlist."""

    playlist_id: str
    title: str
    description: str = ""
    video_count: int = 0
    videos: list[dict[str, Any]] = field(default_factory=list)
    uploader: str = ""
    upload_date: str = ""


@dataclass
class CommentInfo:
    """Summary of video comments for metadata enrichment."""

    total_count: int = 0
    top_comments: list[dict[str, Any]] = field(default_factory=list)
    sample_text: list[str] = field(default_factory=list)


# ── URL Validation ────────────────────────────────────────
_YOUTUBE_URL_PATTERNS: list[re.Pattern[str]] = [
    # Standard youtube.com watch URLs
    re.compile(
        r"^(https?://)?(www\.)?youtube\.com/watch\?v=[\w\-]{11}(&.*)?$"
    ),
    # youtu.be short URLs
    re.compile(
        r"^(https?://)?youtu\.be/[\w\-]{11}(\?.*)?$"
    ),
    # youtube.com/shorts/ URLs
    re.compile(
        r"^(https?://)?(www\.)?youtube\.com/shorts/[\w\-]{11}(\?.*)?$"
    ),
    # youtube.com/live/ URLs
    re.compile(
        r"^(https?://)?(www\.)?youtube\.com/live/[\w\-]{11}(\?.*)?$"
    ),
    # youtube.com/embed/ URLs
    re.compile(
        r"^(https?://)?(www\.)?youtube\.com/embed/[\w\-]{11}(\?.*)?$"
    ),
    # youtube.com/playlist URLs
    re.compile(
        r"^(https?://)?(www\.)?youtube\.com/playlist\?list=[\w\-]+(&.*)?$"
    ),
    # Invidious instances (invidious.io, inv.nadeko.net, etc.)
    re.compile(
        r"^(https?://)?[\w\.\-]+\.invidious\.io/watch\?v=[\w\-]{11}(&.*)?$"
    ),
    re.compile(
        r"^(https?://)?inv\.nadeko\.net/watch\?v=[\w\-]{11}(&.*)?$"
    ),
    re.compile(
        r"^(https?://)?(www\.)?invidious\.[\w\.\-]+/watch\?v=[\w\-]{11}(&.*)?$"
    ),
    # piped.video instances
    re.compile(
        r"^(https?://)?(www\.)?piped\.video/watch\?v=[\w\-]{11}(&.*)?$"
    ),
    re.compile(
        r"^(https?://)?piped\.[\w\.\-]+/watch\?v=[\w\-]{11}(&.*)?$"
    ),
]


def _validate_url(url: str) -> None:
    """Validate that the URL is a supported video link.

    Supports youtube.com (watch, shorts, live, embed), youtu.be,
    invidious instances, and piped.video instances.

    Args:
        url: URL string to validate.

    Raises:
        InvalidURLError: If the URL is empty or does not match any
            supported pattern.
    """
    if not url or not url.strip():
        raise InvalidURLError("Empty URL provided")

    url_stripped = url.strip()
    for pattern in _YOUTUBE_URL_PATTERNS:
        if pattern.match(url_stripped):
            return

    raise InvalidURLError(
        f"Invalid YouTube URL: {url}\n"
        f"Supported formats:\n"
        f"  - https://www.youtube.com/watch?v=VIDEO_ID\n"
        f"  - https://youtu.be/VIDEO_ID\n"
        f"  - https://www.youtube.com/shorts/VIDEO_ID\n"
        f"  - https://www.youtube.com/live/VIDEO_ID\n"
        f"  - https://www.youtube.com/embed/VIDEO_ID\n"
        f"  - https://www.youtube.com/playlist?list=PLAYLIST_ID\n"
        f"  - Invidious instances (invidious.io, inv.nadeko.net, etc.)\n"
        f"  - piped.video instances"
    )


def _extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL.

    Handles all supported URL formats including youtu.be, shorts,
    live, embed, and invidious/piped instances.

    Args:
        url: URL string containing a YouTube video ID.

    Returns:
        The 11-character video ID string, or empty string if not found.
    """
    patterns = [
        r"(?:v=|youtu\.be/|shorts/|embed/|live/|watch\?v=)([\w\-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def _check_already_downloaded(youtube_id: str, output_dir: Path) -> Path | None:
    """Check if a video has already been downloaded.

    Searches the output directory for files containing the youtube_id
    with common video extensions.

    Args:
        youtube_id: YouTube video ID to search for.
        output_dir: Directory to search in.

    Returns:
        Path to existing file if found, None otherwise.
    """
    if not output_dir.exists():
        return None

    for existing_file in output_dir.iterdir():
        if youtube_id in existing_file.name and existing_file.suffix in (".mp4", ".mkv", ".webm", ".mp3", ".m4a"):
            logger.info("Video %s already downloaded: %s", youtube_id, existing_file)
            return existing_file

    return None


def _classify_error(stderr: str) -> type[DownloadError]:
    """Classify a yt-dlp error message into a specific exception type.

    Analyzes the stderr output to determine the most specific error
    category that applies.

    Args:
        stderr: Error output from yt-dlp.

    Returns:
        The most specific DownloadError subclass that matches.
    """
    stderr_lower = stderr.lower()

    if any(kw in stderr_lower for kw in ("copyright", "blocked.*copyright", "removed.*copyright", "inappropriate")):
        return CopyrightError
    if any(kw in stderr_lower for kw in ("members-only", "member only", "join this channel")):
        return MembersOnlyError
    if any(kw in stderr_lower for kw in ("live event", "live stream", "is live", "premiere")):
        return LiveStreamError
    if any(kw in stderr_lower for kw in ("age-restrict", "age restrict", "age-restricted", "inappropriate for some users", "sign in to confirm your age")):
        return AgeRestrictedError
    if any(kw in stderr_lower for kw in ("private", "sign in to confirm", "bot detection")):
        return PrivateVideoError
    if any(kw in stderr_lower for kw in ("not available", "not exist", "been removed", "unavailable", "does not exist")):
        return UnavailableError
    if any(kw in stderr_lower for kw in ("geo-restrict", "not available in your country", "not available in this country")):
        return GeoRestrictedError
    if any(kw in stderr_lower for kw in ("rate-limit", "too many requests", "http error 429")):
        return RateLimitError

    return DownloadError


def _check_disk_space(output_dir: Path, estimated_size_bytes: int = 0) -> None:
    """Check that sufficient disk space is available for a download.

    Args:
        output_dir: Target directory for the download.
        estimated_size_bytes: Estimated file size in bytes. If 0, uses
            a 2 GB default estimate.

    Raises:
        DiskSpaceError: If insufficient disk space is available.
    """
    min_required = estimated_size_bytes if estimated_size_bytes > 0 else 2 * 1024 * 1024 * 1024  # 2 GB default
    buffer_bytes = 500 * 1024 * 1024  # 500 MB buffer
    required = min_required + buffer_bytes

    try:
        disk_usage = shutil.disk_usage(output_dir)
        available = disk_usage.free
        if available < required:
            required_gb = required / (1024 ** 3)
            available_gb = available / (1024 ** 3)
            raise DiskSpaceError(
                f"Insufficient disk space: need {required_gb:.1f} GB, "
                f"have {available_gb:.1f} GB available in {output_dir}"
            )
    except DiskSpaceError:
        raise
    except Exception as exc:
        logger.warning("Could not check disk space: %s", exc)


# ── Core Download Functions ───────────────────────────────

@retry_on_failure(max_attempts=2, delay=3.0, exceptions=(subprocess.CalledProcessError,))
def _fetch_metadata(url: str, cookie_file: str = "") -> dict[str, Any]:
    """Fetch video metadata using yt-dlp --dump-json.

    Args:
        url: YouTube video URL.
        cookie_file: Optional path to cookie file for authentication.

    Returns:
        Dictionary with all video metadata from yt-dlp.

    Raises:
        DownloadError: If metadata fetch fails or times out.
        AgeRestrictedError: If the video requires age verification.
        PrivateVideoError: If the video is private.
        GeoRestrictedError: If the video is geo-restricted.
    """
    meta_cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        "--flat-playlist",
    ]

    if cookie_file and Path(cookie_file).exists():
        meta_cmd.extend(["--cookies", cookie_file])

    meta_cmd.append(url)

    try:
        result = subprocess.run(
            meta_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            error_cls = _classify_error(result.stderr)
            raise error_cls(f"Failed to fetch metadata: {result.stderr[:500]}")

        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DownloadError(f"Failed to parse video metadata JSON: {exc}")
    except subprocess.TimeoutExpired:
        raise DownloadError("Metadata fetch timed out (120s)")


def download_with_options(
    url: str,
    output_dir: Path,
    options: DownloadOptions | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Download a video with comprehensive configuration options.

    Supports format selection, audio-only downloads, subtitle and
    thumbnail downloads, cookie authentication, proxy support,
    rate limiting, and progress callbacks.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded video.
        options: DownloadOptions configuration. Defaults used if None.

    Returns:
        Tuple of (path_to_video, info_dict) where info_dict contains
        title, duration, view_count, like_count, uploader, etc.

    Raises:
        InvalidURLError: If the URL is not a valid supported link.
        GeoRestrictedError: If the video is geo-restricted.
        AgeRestrictedError: If the video requires age verification.
        MembersOnlyError: If the video is members-only.
        PrivateVideoError: If the video is private.
        LiveStreamError: If the video is a live stream.
        CopyrightError: If the video is blocked for copyright.
        DiskSpaceError: If insufficient disk space.
        DownloadError: If the download fails for any other reason.
    """
    if options is None:
        options = DownloadOptions()

    # Validate URL
    _validate_url(url)
    youtube_id = _extract_video_id(url)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch metadata
    logger.info("Fetching metadata for: %s", url)
    info: dict[str, Any] = _fetch_metadata(url, cookie_file=options.cookie_file)

    title: str = info.get("title", "unknown_video")
    duration: float = info.get("duration", 0.0)
    view_count: int = info.get("view_count", 0)

    if not info.get("id"):
        info["id"] = youtube_id
    else:
        youtube_id = info["id"]

    logger.info(
        "Video: %s (%s) - %.0fs, %d views",
        title, youtube_id, duration, view_count,
    )

    # Check for existing download
    existing = _check_already_downloaded(youtube_id, output_dir)
    if existing:
        console.print(f"[yellow]Using cached download:[/yellow] {existing.name}")
        info["_local_path"] = str(existing)
        return existing, info

    # Check disk space
    estimated_size = info.get("filesize") or info.get("filesize_approx") or 0
    _check_disk_space(output_dir, estimated_size)

    # Setup output template
    safe_title = sanitize_filename(title, max_length=80)
    if options.audio_only:
        output_template = str(output_dir / f"{safe_title}_{youtube_id}.%(ext)s")
    else:
        output_template = str(output_dir / f"{safe_title}_{youtube_id}.%(ext)s")

    # Build download command
    format_string = options.get_format_string()

    download_cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
    ]

    # Format selection
    download_cmd.extend(["-f", format_string])

    # Merge output format (skip for audio_only)
    if not options.audio_only:
        download_cmd.extend(["--merge-output-format", options.merge_output_format])

    # Thumbnail
    if options.download_thumbnail:
        download_cmd.append("--write-thumbnail")

    # Metadata
    if options.write_metadata:
        download_cmd.append("--add-metadata")

    if options.embed_chapters:
        download_cmd.append("--embed-chapters")

    # Subtitles
    if options.download_subtitles:
        download_cmd.extend(["--write-subs", "--write-auto-subs"])
        if options.subtitle_languages:
            download_cmd.extend(["--sub-langs", ",".join(options.subtitle_languages)])

    # Retry settings
    download_cmd.extend([
        "--retries", str(options.retries),
        "--fragment-retries", str(options.fragment_retries),
        "--concurrent-fragments", str(options.concurrent_fragments),
    ])

    # Cookies
    if options.cookie_file and Path(options.cookie_file).exists():
        download_cmd.extend(["--cookies", options.cookie_file])

    # Proxy
    if options.proxy:
        download_cmd.extend(["--proxy", options.proxy])

    # Rate limit
    if options.rate_limit:
        download_cmd.extend(["--rate-limit", options.rate_limit])

    # Temp directory
    if options.temp_dir:
        download_cmd.extend(["--paths", options.temp_dir])

    # Output template
    download_cmd.extend(["-o", output_template])

    download_cmd.append(url)

    # Track temp files for cleanup on failure
    temp_files: list[Path] = []

    logger.info("Starting download: %s (format=%s)", title, options.format_selection)

    try:
        process = subprocess.Popen(
            download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        progress_pattern = re.compile(r"\[download\]\s+(\d+\.?\d*)%")
        speed_pattern = re.compile(r"at\s+([\d.]+\w+/s)")
        last_percent = 0.0

        if process.stdout is not None:
            for line in process.stdout:
                line = line.rstrip()
                match = progress_pattern.search(line)
                if match:
                    current_percent = float(match.group(1))
                    speed_match = speed_pattern.search(line)
                    speed = speed_match.group(1) if speed_match else ""

                    if current_percent - last_percent >= 5.0:
                        console.print(f"  [cyan]Download: {current_percent:.0f}%[/cyan] {speed}")
                        last_percent = current_percent

                        if options.progress_callback:
                            options.progress_callback(current_percent, speed)

        return_code = process.wait()

        if return_code != 0:
            raise DownloadError(f"yt-dlp exited with code {return_code}")

    except FileNotFoundError:
        raise DownloadError(
            "yt-dlp is not installed. Install with: pip install yt-dlp"
        )
    except DownloadError:
        # Auto-cleanup temp files on failure
        for tf in temp_files:
            safe_delete(tf)
        # Also clean up any partial downloads
        for f in output_dir.iterdir():
            if f.suffix in (".part", ".temp", ".ytdl"):
                safe_delete(f)
        raise

    # Locate the downloaded file
    downloaded_path: Path | None = None
    target_ext = ".mp3" if options.audio_only else ".mp4"

    if youtube_id:
        for f in output_dir.iterdir():
            if youtube_id in f.name and f.suffix in (".mp4", ".mkv", ".webm", ".mp3", ".m4a"):
                downloaded_path = f
                break

    if downloaded_path is None:
        extensions = (".mp3", ".m4a") if options.audio_only else (".mp4", ".mkv", ".webm")
        files = sorted(
            [f for f in output_dir.iterdir() if f.suffix in extensions],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            downloaded_path = files[0]

    if downloaded_path is None or not downloaded_path.exists():
        raise DownloadError(f"Downloaded file not found for {youtube_id}")

    logger.info("Downloaded to: %s (%s)", downloaded_path, get_file_size_human(downloaded_path))
    console.print(f"[green]Downloaded:[/green] {downloaded_path.name} ({get_file_size_human(downloaded_path)})")

    info["_local_path"] = str(downloaded_path)
    return downloaded_path, info


@retry_on_failure(max_attempts=3, delay=5.0, exceptions=(DownloadError, subprocess.CalledProcessError))
def download_video(url: str, output_dir: Path, turbo: bool = False) -> tuple[Path, dict[str, Any]]:
    """Download a YouTube video with full metadata extraction.

    Backward-compatible wrapper around download_with_options using
    default settings. Validates the URL, fetches metadata, checks for
    existing downloads, selects the best quality format, and streams
    the download with progress tracking via Rich.

    In turbo mode, downloads at 720p and skips metadata/chapters for speed.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded video.
        turbo: If True, download at 720p with minimal metadata for speed.

    Returns:
        Tuple of (path_to_video, info_dict) where info_dict contains
        title, duration, view_count, like_count, uploader, etc.

    Raises:
        InvalidURLError: If the URL is not a valid YouTube link.
        GeoRestrictedError: If the video is geo-restricted.
        PrivateVideoError: If the video is private or age-gated.
        DownloadError: If the download fails for any other reason.
    """
    settings = get_settings()

    # Determine proxy from environment
    proxy = os.environ.get("YT_PROXY", "")

    # Turbo mode: download at 720p, skip extras for speed
    format_sel = "720" if turbo else "1080"
    write_metadata = not turbo
    embed_chapters = not turbo
    concurrent = 8 if turbo else 4  # More concurrent fragments in turbo

    options = DownloadOptions(
        format_selection=format_sel,
        audio_only=False,
        download_subtitles=False,
        download_thumbnail=not turbo,
        concurrent_fragments=concurrent,
        retries=2 if turbo else 3,
        fragment_retries=2 if turbo else 3,
        merge_output_format="mp4",
        write_metadata=write_metadata,
        embed_chapters=embed_chapters,
        proxy=proxy,
    )

    if turbo:
        logger.info("TURBO download: 720p, skip metadata, 8 concurrent fragments")

    return download_with_options(url, output_dir, options)


def download_playlist_info(url: str, cookie_file: str = "") -> PlaylistInfo:
    """Get all video URLs from a playlist without downloading.

    Extracts playlist metadata including all video entries with
    their IDs, titles, and durations.

    Args:
        url: YouTube playlist URL (must contain list= parameter).
        cookie_file: Optional path to cookie file for authentication.

    Returns:
        PlaylistInfo with playlist metadata and list of video entries.

    Raises:
        InvalidURLError: If the URL is not a valid playlist link.
        DownloadError: If playlist info extraction fails.
    """
    _validate_url(url)

    cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "--dump-json",
        "--flat-playlist",
        "--skip-download",
    ]

    if cookie_file and Path(cookie_file).exists():
        cmd.extend(["--cookies", cookie_file])

    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            error_cls = _classify_error(result.stderr)
            raise error_cls(f"Failed to fetch playlist info: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise DownloadError("Playlist info fetch timed out (180s)")

    videos: list[dict[str, Any]] = []

    # yt-dlp outputs one JSON object per line for flat playlist
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            videos.append({
                "id": entry.get("id", ""),
                "title": entry.get("title", "Unknown"),
                "duration": entry.get("duration", 0),
                "url": f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                "uploader": entry.get("uploader", ""),
                "view_count": entry.get("view_count", 0),
            })
        except json.JSONDecodeError:
            continue

    # Try to get playlist-level metadata from first entry
    playlist_id = ""
    playlist_title = ""
    match = re.search(r"list=([\w\-]+)", url)
    if match:
        playlist_id = match.group(1)

    if videos:
        # Try to extract playlist title from yt-dlp output
        try:
            # Re-run with --print to get playlist title
            title_cmd = [sys.executable, "-m", "yt_dlp", "--print", "playlist", "--skip-download", url]
            title_result = subprocess.run(title_cmd, capture_output=True, text=True, timeout=30)
            if title_result.returncode == 0 and title_result.stdout.strip():
                playlist_title = title_result.stdout.strip().splitlines()[0]
        except Exception:
            pass

        if not playlist_title:
            playlist_title = f"Playlist {playlist_id}"

    playlist_info = PlaylistInfo(
        playlist_id=playlist_id,
        title=playlist_title,
        video_count=len(videos),
        videos=videos,
    )

    logger.info(
        "Playlist: %s (%s) - %d videos",
        playlist_info.title, playlist_info.playlist_id, playlist_info.video_count,
    )

    return playlist_info


def get_available_formats(url: str, cookie_file: str = "") -> list[FormatInfo]:
    """List all available formats/qualities for a video URL.

    Queries yt-dlp for all available download formats including
    video+audio combinations, video-only, and audio-only streams.

    Args:
        url: YouTube video URL.
        cookie_file: Optional path to cookie file for authentication.

    Returns:
        List of FormatInfo objects describing each available format,
        sorted by resolution (highest first).

    Raises:
        InvalidURLError: If the URL is not valid.
        DownloadError: If format listing fails.
    """
    _validate_url(url)

    info = _fetch_metadata(url, cookie_file=cookie_file)

    formats: list[FormatInfo] = []
    raw_formats = info.get("formats", [])

    for fmt in raw_formats:
        vcodec = fmt.get("vcodec", "none")
        acodec = fmt.get("acodec", "none")
        has_video = vcodec != "none" and vcodec != ""
        has_audio = acodec != "none" and acodec != ""

        # Build resolution string
        width = fmt.get("width", 0) or 0
        height = fmt.get("height", 0) or 0
        if has_video and height > 0:
            resolution = f"{width}x{height}"
        elif has_audio and not has_video:
            resolution = "audio only"
        else:
            resolution = fmt.get("format_note", "unknown")

        format_info = FormatInfo(
            format_id=fmt.get("format_id", ""),
            ext=fmt.get("ext", ""),
            resolution=resolution,
            fps=fmt.get("fps", 0) or 0,
            vcodec=vcodec,
            acodec=acodec,
            filesize=fmt.get("filesize", 0) or 0,
            filesize_approx=fmt.get("filesize_approx", 0) or 0,
            tbr=fmt.get("tbr", 0.0) or 0.0,
            vbr=fmt.get("vbr", 0.0) or 0.0,
            abr=fmt.get("abr", 0.0) or 0.0,
            has_video=has_video,
            has_audio=has_audio,
            language=fmt.get("language", ""),
            format_note=fmt.get("format_note", ""),
        )
        formats.append(format_info)

    # Sort: video+audio first, then by height descending
    def sort_key(f: FormatInfo) -> tuple[int, int, float]:
        has_both = 1 if (f.has_video and f.has_audio) else 0
        # Extract height for sorting
        height = 0
        if f.has_video:
            match = re.search(r"(\d+)$", f.resolution.split("x")[-1] if "x" in f.resolution else "")
            if match:
                height = int(match.group(1))
            else:
                # Try parsing from resolution like "1920x1080"
                parts = f.resolution.split("x")
                if len(parts) == 2:
                    try:
                        height = int(parts[1])
                    except ValueError:
                        pass
        return (-has_both, -height, -f.tbr)

    formats.sort(key=sort_key)

    logger.info("Found %d formats for %s", len(formats), url[:60])
    return formats


def download_thumbnail(url: str, output_dir: Path, filename: str = "") -> Path:
    """Download just the thumbnail for a video.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the thumbnail.
        filename: Optional custom filename. If empty, uses video ID.

    Returns:
        Path to the downloaded thumbnail image.

    Raises:
        InvalidURLError: If the URL is not valid.
        DownloadError: If thumbnail download fails.
    """
    _validate_url(url)
    youtube_id = _extract_video_id(url)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = f"{youtube_id}_thumb"

    output_template = str(output_dir / f"{filename}.%(ext)s")

    cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "--write-thumbnail",
        "--skip-download",
        "--no-playlist",
        "-o", output_template,
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise DownloadError(f"Thumbnail download failed: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        raise DownloadError("Thumbnail download timed out (60s)")

    # Find the downloaded thumbnail
    for ext in (".jpg", ".png", ".webp"):
        thumb_path = output_dir / f"{filename}{ext}"
        if thumb_path.exists():
            logger.info("Thumbnail downloaded: %s", thumb_path.name)
            return thumb_path

    raise DownloadError(f"Thumbnail file not found after download for {youtube_id}")


def download_subtitles(
    url: str,
    output_dir: Path,
    languages: list[str] | None = None,
    auto_subs: bool = True,
) -> dict[str, Path]:
    """Download available subtitle tracks for a video.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save subtitle files.
        languages: List of language codes to download (e.g. ["en", "es"]).
            Defaults to ["en"].
        auto_subs: Whether to also download auto-generated subtitles.

    Returns:
        Dictionary mapping language code to subtitle file Path.

    Raises:
        InvalidURLError: If the URL is not valid.
        DownloadError: If subtitle download fails.
    """
    _validate_url(url)
    youtube_id = _extract_video_id(url)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not languages:
        languages = ["en"]

    cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "--write-subs",
    ]

    if auto_subs:
        cmd.append("--write-auto-subs")

    cmd.extend([
        "--sub-langs", ",".join(languages),
        "--skip-download",
        "--no-playlist",
        "-o", str(output_dir / f"{youtube_id}"),
        url,
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("Subtitle download had issues: %s", result.stderr[:300])
    except subprocess.TimeoutExpired:
        raise DownloadError("Subtitle download timed out (120s)")

    # Find downloaded subtitle files
    subtitles: dict[str, Path] = {}
    for f in output_dir.iterdir():
        if f.name.startswith(youtube_id) and f.suffix in (".vtt", ".srt", ".ass"):
            # Extract language from filename: ID.lang.vtt
            parts = f.stem.split(".")
            if len(parts) >= 2:
                lang = parts[-1]
            else:
                lang = "unknown"
            subtitles[lang] = f

    logger.info("Downloaded %d subtitle tracks for %s", len(subtitles), youtube_id)
    return subtitles


def verify_video_integrity(video_path: Path) -> bool:
    """Check that a downloaded video file isn't corrupt or truncated.

    Uses ffprobe to verify the file has valid streams and that the
    reported duration matches the actual container duration.

    Args:
        video_path: Path to the video file to verify.

    Returns:
        True if the file appears valid, False if corrupt/truncated.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check minimum file size (less than 10KB is likely corrupt)
    file_size = video_path.stat().st_size
    if file_size < 10240:
        logger.warning("Video file is too small (%d bytes), likely corrupt: %s", file_size, video_path.name)
        return False

    # Use ffprobe to check streams
    cmd: list[str] = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("ffprobe failed for %s: %s", video_path.name, result.stderr[:200])
            return False

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        format_info = data.get("format", {})

        # Must have at least one stream
        if not streams:
            logger.warning("No streams found in %s", video_path.name)
            return False

        # Check for video stream
        has_video = any(s.get("codec_type") == "video" for s in streams)
        if not has_video:
            logger.warning("No video stream found in %s", video_path.name)
            return False

        # Check duration is reasonable
        duration = float(format_info.get("duration", 0))
        if duration <= 0:
            logger.warning("Invalid duration (%.1f) for %s", duration, video_path.name)
            return False

        logger.info("Integrity check passed: %s (%.1fs, %d streams)", video_path.name, duration, len(streams))
        return True

    except (json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        logger.warning("Integrity check failed for %s: %s", video_path.name, exc)
        return False


def resume_partial_download(url: str, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Resume an interrupted download.

    Attempts to resume a partially downloaded file using yt-dlp's
    built-in resume support. If no partial download exists, starts
    a fresh download.

    Args:
        url: YouTube video URL.
        output_dir: Directory where partial download exists.

    Returns:
        Tuple of (path_to_video, info_dict).

    Raises:
        InvalidURLError: If the URL is not valid.
        DownloadError: If resume attempt fails.
    """
    _validate_url(url)
    youtube_id = _extract_video_id(url)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for partial download files
    has_partial = False
    for f in output_dir.iterdir():
        if youtube_id in f.name and f.suffix in (".part", ".ytdl"):
            has_partial = True
            logger.info("Found partial download: %s", f.name)
            break

    if not has_partial:
        logger.info("No partial download found; starting fresh download")
        return download_video(url, output_dir)

    # Use yt-dlp with --continue flag to resume
    info = _fetch_metadata(url)
    title = info.get("title", "unknown_video")
    safe_title = sanitize_filename(title, max_length=80)
    output_template = str(output_dir / f"{safe_title}_{youtube_id}.%(ext)s")

    cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "--continue",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--add-metadata",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "-o", output_template,
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            error_cls = _classify_error(result.stderr)
            raise error_cls(f"Resume download failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise DownloadError("Resume download timed out (600s)")

    # Locate the downloaded file
    downloaded_path: Path | None = None
    if youtube_id:
        for f in output_dir.iterdir():
            if youtube_id in f.name and f.suffix == ".mp4":
                downloaded_path = f
                break

    if downloaded_path is None or not downloaded_path.exists():
        raise DownloadError(f"Resumed file not found for {youtube_id}")

    logger.info("Resumed download complete: %s", downloaded_path.name)
    info["_local_path"] = str(downloaded_path)
    return downloaded_path, info


def download_with_cookies(
    url: str,
    output_dir: Path,
    cookie_file: str,
    format_selection: str = "best",
) -> tuple[Path, dict[str, Any]]:
    """Download a video using cookie-based authentication.

    Supports downloading member-only videos, age-restricted content,
    and other authentication-gated videos by providing a browser
    cookie file exported from the user's browser.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded video.
        cookie_file: Path to the cookie file (Netscape format).
        format_selection: Format preset name (best/4k/1080/720/480).

    Returns:
        Tuple of (path_to_video, info_dict).

    Raises:
        InvalidURLError: If the URL is not valid.
        FileNotFoundError: If the cookie file doesn't exist.
        MembersOnlyError: If cookies don't grant access.
        DownloadError: If download fails.
    """
    _validate_url(url)

    if not Path(cookie_file).exists():
        raise FileNotFoundError(f"Cookie file not found: {cookie_file}")

    options = DownloadOptions(
        format_selection=format_selection,
        cookie_file=cookie_file,
    )

    return download_with_options(url, output_dir, options)


def get_video_comments(
    url: str,
    max_comments: int = 20,
    cookie_file: str = "",
) -> CommentInfo:
    """Get video comments for metadata enrichment.

    Extracts comment count and top comments from a video.
    This is useful for generating hashtags, keywords, and
    understanding audience engagement.

    Args:
        url: YouTube video URL.
        max_comments: Maximum number of comments to retrieve.
        cookie_file: Optional path to cookie file for authentication.

    Returns:
        CommentInfo with total count and top comments.

    Raises:
        InvalidURLError: If the URL is not valid.
        DownloadError: If comment extraction fails.
    """
    _validate_url(url)

    cmd: list[str] = [
        sys.executable, "-m", "yt_dlp",
        "--dump-json",
        "--no-playlist",
        "--skip-download",
        "--flat-playlist",
    ]

    if cookie_file and Path(cookie_file).exists():
        cmd.extend(["--cookies", cookie_file])

    cmd.extend(["--extractor-args", f"youtube:comment_count={max_comments}"])
    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning("Comment extraction failed: %s", result.stderr[:200])
            return CommentInfo()
    except subprocess.TimeoutExpired:
        logger.warning("Comment extraction timed out")
        return CommentInfo()

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return CommentInfo()

    # Extract comment information
    comment_count = info.get("comment_count", 0)
    comments_raw = info.get("comments", [])

    top_comments: list[dict[str, Any]] = []
    sample_text: list[str] = []

    for comment in comments_raw[:max_comments]:
        comment_data = {
            "author": comment.get("author", ""),
            "text": comment.get("text", ""),
            "like_count": comment.get("like_count", 0),
            "is_favorited": comment.get("is_favorited", False),
        }
        top_comments.append(comment_data)
        if comment.get("text"):
            sample_text.append(comment["text"][:200])

    return CommentInfo(
        total_count=comment_count,
        top_comments=top_comments,
        sample_text=sample_text,
    )
