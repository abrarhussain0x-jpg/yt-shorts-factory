"""
utils/file_utils.py — Safe file operations, cleanup, and naming helpers.

Provides sanitised filenames, timestamped output paths, safe deletion,
intermediate cleanup, human-readable file sizes, hash computation,
duplicate detection, disk usage monitoring, atomic writes, JSON safety,
temp directory management, file organisation, and video file validation.
All operations are defensive and never raise on missing files.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from utils.logger import get_logger

logger = get_logger("file_utils")


# ══════════════════════════════════════════════════════════
#  Filename Helpers
# ══════════════════════════════════════════════════════════

def sanitize_filename(name: str, max_length: int = 120) -> str:
    """Sanitise a string for safe use as a filename.

    Removes or replaces characters illegal across OS, collapses whitespace,
    and truncates to max_length while preserving file extensions.

    Args:
        name: Raw filename string to sanitise.
        max_length: Maximum character length for the result.

    Returns:
        Cleaned, truncated filename string with no illegal characters.
    """
    # Coerce non-string inputs (e.g., Path objects) to string
    if not isinstance(name, str):
        name = str(name)

    if not name or not name.strip():
        return "untitled"

    # Remove illegal characters: \ / * ? : " < > | and control chars
    sanitized = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", name)

    # Replace runs of whitespace or underscores with a single underscore
    sanitized = re.sub(r"[\s_]+", "_", sanitized)

    # Strip leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")

    # Handle empty result after sanitization
    if not sanitized:
        return "untitled"

    # Truncate to max_length, preserving extension if present
    if len(sanitized) > max_length:
        if "." in sanitized:
            last_dot = sanitized.rfind(".")
            if 0 < last_dot and len(sanitized) - last_dot <= 10:
                ext = sanitized[last_dot:]
                base = sanitized[:last_dot]
                sanitized = base[: max_length - len(ext)] + ext
            else:
                sanitized = sanitized[:max_length]
        else:
            sanitized = sanitized[:max_length]

    return sanitized


def make_output_path(
    base_dir: Path,
    title: str,
    suffix: str,
    ext: str = "mp4",
) -> Path:
    """Construct a timestamped, sanitised output file path.

    Format: {base_dir}/{sanitised_title}_{suffix}_{YYYYMMDD_HHMMSS}.{ext}
    Creates the base_dir if it doesn't exist.

    Args:
        base_dir: Parent directory for the output file.
        title: Video title used for the filename prefix.
        suffix: Label suffix (e.g. 'yt_short', 'tiktok').
        ext: File extension without dot (default 'mp4').

    Returns:
        Absolute Path to the output file.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = sanitize_filename(title, max_length=80)
    filename = f"{safe_title}_{suffix}_{timestamp}.{ext}"
    return base_dir / filename


def compute_output_basename(title: str, youtube_id: str = "") -> str:
    """Compute a standardized base filename for a video.

    Args:
        title: Video title.
        youtube_id: Optional YouTube video ID for uniqueness.

    Returns:
        Sanitised base filename string.
    """
    safe = sanitize_filename(title, max_length=60)
    if youtube_id:
        return f"{safe}_{youtube_id}"
    return safe


# ══════════════════════════════════════════════════════════
#  File Deletion & Cleanup
# ══════════════════════════════════════════════════════════

def safe_delete(path: Path) -> bool:
    """Delete a file if it exists; never raises.

    Args:
        path: Path to the file to delete.

    Returns:
        True if the file was deleted, False otherwise.
    """
    try:
        if path.exists():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            logger.debug("Deleted: %s", path)
            return True
    except OSError as exc:
        logger.warning("Failed to delete %s: %s", path, exc)
    return False


def cleanup_intermediates(paths: list[Path]) -> int:
    """Delete a list of intermediate files, returning the count cleaned.

    Args:
        paths: List of file paths to attempt deletion on.

    Returns:
        Number of files successfully deleted.
    """
    cleaned = 0
    for p in paths:
        if safe_delete(p):
            cleaned += 1
    if paths:
        logger.info("Cleaned up %d/%d intermediate files", cleaned, len(paths))
    return cleaned


def clean_old_files(
    directory: Path,
    max_age_days: int = 30,
    pattern: str = "*",
) -> int:
    """Remove files older than a specified number of days.

    Scans the directory for files matching the pattern and deletes those
    whose modification time exceeds max_age_days. Does not recurse into
    subdirectories by default.

    Args:
        directory: Directory to scan for old files.
        max_age_days: Maximum age in days (default 30). Files older are deleted.
        pattern: Glob pattern to match files (default '*' for all files).

    Returns:
        Number of files successfully deleted.
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return 0

    if max_age_days < 0:
        raise ValueError(f"max_age_days must be non-negative, got {max_age_days}")

    now = datetime.now().timestamp()
    cutoff = now - (max_age_days * 86400)
    deleted_count = 0

    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        try:
            mtime = file_path.stat().st_mtime
            if mtime < cutoff:
                if safe_delete(file_path):
                    deleted_count += 1
        except OSError as exc:
            logger.warning("Could not check mtime for %s: %s", file_path, exc)

    if deleted_count > 0:
        logger.info("Cleaned %d files older than %d days from %s", deleted_count, max_age_days, directory)

    return deleted_count


# ══════════════════════════════════════════════════════════
#  File Size Helpers
# ══════════════════════════════════════════════════════════

def get_file_size_human(path: Path) -> str:
    """Return a human-readable file size string.

    Args:
        path: Path to the file.

    Returns:
        Human-readable size (e.g. '1.2 GB', '345 MB') or 'N/A'.
    """
    try:
        if not path.exists():
            return "N/A"
        size = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
    except OSError:
        return "N/A"


def get_file_size_bytes(path: Path) -> int:
    """Return file size in bytes, or 0 if file doesn't exist.

    Args:
        path: Path to the file.

    Returns:
        File size in bytes.
    """
    try:
        return path.stat().st_size if path.exists() else 0
    except OSError:
        return 0


def get_disk_usage(directory: Path) -> dict[str, int | float]:
    """Get disk usage statistics for the filesystem containing the directory.

    Args:
        directory: Path to check disk usage for.

    Returns:
        Dictionary with keys 'total_bytes', 'used_bytes', 'free_bytes',
        'free_mb', 'used_percent'.
    """
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            directory = directory.parent

    try:
        usage = shutil.disk_usage(directory)
        total = usage.total
        used = usage.used
        free = usage.free
        return {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "free_mb": round(free / (1024 * 1024), 1),
            "used_percent": round((used / total) * 100, 1) if total > 0 else 0.0,
        }
    except OSError as exc:
        logger.warning("Could not get disk usage for %s: %s", directory, exc)
        return {
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "free_mb": 0.0,
            "used_percent": 0.0,
        }


def ensure_min_disk_space(path: Path, required_mb: int) -> bool:
    """Check and warn about disk space availability.

    Args:
        path: Path on the filesystem to check.
        required_mb: Minimum required free space in megabytes.

    Returns:
        True if sufficient disk space is available, False otherwise.
    """
    usage = get_disk_usage(path)
    free_mb = usage["free_mb"]

    if free_mb < required_mb:
        logger.warning(
            "Insufficient disk space: %.0f MB free, %d MB required at %s",
            free_mb, required_mb, path,
        )
        return False

    return True


# ══════════════════════════════════════════════════════════
#  Directory Helpers
# ══════════════════════════════════════════════════════════

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure.

    Returns:
        The same path (for chaining).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create directory %s: %s", path, exc)
    return path


# ══════════════════════════════════════════════════════════
#  Hash Computation
# ══════════════════════════════════════════════════════════

def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file for dedup detection.

    Reads the file in chunks to handle large files without excessive
    memory usage.

    Args:
        path: Path to the file to hash.
        chunk_size: Size of chunks to read at a time in bytes (default 8192).

    Returns:
        Hexadecimal MD5 hash string, or empty string if file not found.
    """
    if not path.exists():
        logger.warning("File not found for MD5: %s", path)
        return ""

    if not path.is_file():
        logger.warning("Path is not a file for MD5: %s", path)
        return ""

    md5_hash = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except OSError as exc:
        logger.error("Failed to compute MD5 for %s: %s", path, exc)
        return ""


def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file for integrity verification.

    SHA-256 is cryptographically secure and suitable for verifying file
    integrity. Reads in chunks to handle large files.

    Args:
        path: Path to the file to hash.
        chunk_size: Size of chunks to read at a time in bytes (default 8192).

    Returns:
        Hexadecimal SHA-256 hash string, or empty string if file not found.
    """
    if not path.exists():
        logger.warning("File not found for SHA-256: %s", path)
        return ""

    if not path.is_file():
        logger.warning("Path is not a file for SHA-256: %s", path)
        return ""

    sha256_hash = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except OSError as exc:
        logger.error("Failed to compute SHA-256 for %s: %s", path, exc)
        return ""


# ══════════════════════════════════════════════════════════
#  Duplicate Detection
# ══════════════════════════════════════════════════════════

def find_duplicate_files(
    directory: Path,
    pattern: str = "*",
) -> dict[str, list[Path]]:
    """Find files with the same size and hash (potential duplicates).

    Uses a two-phase approach: first groups files by size, then only
    computes MD5 hashes for groups with the same size, making it
    efficient for large directories.

    Args:
        directory: Directory to scan for duplicates.
        pattern: Glob pattern to match files (default '*' for all).

    Returns:
        Dictionary mapping hash strings to lists of Paths with that hash.
        Only entries with 2+ files are included.
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return {}

    # Phase 1: Group files by size
    size_groups: dict[int, list[Path]] = {}
    for file_path in directory.rglob(pattern):
        if file_path.is_file():
            try:
                size = file_path.stat().st_size
                if size > 0:
                    size_groups.setdefault(size, []).append(file_path)
            except OSError:
                continue

    # Phase 2: For size groups with multiple files, compute hashes
    hash_groups: dict[str, list[Path]] = {}
    for size, files in size_groups.items():
        if len(files) < 2:
            continue
        for file_path in files:
            file_hash = compute_md5(file_path)
            if file_hash:
                hash_groups.setdefault(file_hash, []).append(file_path)

    # Only return groups with 2+ files
    duplicates = {
        h: paths for h, paths in hash_groups.items() if len(paths) >= 2
    }

    if duplicates:
        total_dupes = sum(len(v) for v in duplicates.values())
        logger.info("Found %d duplicate groups (%d files total) in %s", len(duplicates), total_dupes, directory)

    return duplicates


# ══════════════════════════════════════════════════════════
#  File Rotation
# ══════════════════════════════════════════════════════════

def rotate_file(
    path: Path,
    max_size_mb: float = 100.0,
    keep: int = 5,
) -> Path:
    """Rotate a file if it exceeds the maximum size.

    When the file exceeds max_size_mb, it is renamed with a numeric
    suffix (.1, .2, etc.) and older rotations are shifted up. The
    oldest rotation beyond 'keep' is deleted.

    Args:
        path: Path to the file to rotate.
        max_size_mb: Maximum file size in MB before rotation (default 100).
        keep: Number of rotated copies to keep (default 5).

    Returns:
        Path to the current (possibly new) file.

    Raises:
        ValueError: If keep is less than 1.
    """
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")

    if not path.exists():
        return path

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb < max_size_mb:
        return path

    # Delete the oldest rotation
    oldest = Path(f"{path}.{keep}")
    if oldest.exists():
        safe_delete(oldest)

    # Shift existing rotations up by one
    for i in range(keep - 1, 0, -1):
        src = Path(f"{path}.{i}")
        dst = Path(f"{path}.{i + 1}")
        if src.exists():
            try:
                src.rename(dst)
            except OSError as exc:
                logger.warning("Failed to rotate %s -> %s: %s", src, dst, exc)

    # Rotate the current file to .1
    rotated = Path(f"{path}.1")
    try:
        path.rename(rotated)
        logger.info("Rotated %s (%.1f MB) -> %s", path.name, size_mb, rotated.name)
    except OSError as exc:
        logger.warning("Failed to rotate %s: %s", path, exc)

    return path


# ══════════════════════════════════════════════════════════
#  Atomic File Operations
# ══════════════════════════════════════════════════════════

def atomic_write(path: Path, content: str | bytes, mode: str = "text") -> None:
    """Write a file atomically using write-to-temp-then-rename.

    Prevents partial writes from corrupting the target file by first
    writing to a temporary file in the same directory, then atomically
    renaming it to the target path.

    Args:
        path: Target file path.
        content: Content to write (string for text mode, bytes for binary mode).
        mode: Write mode: 'text' or 'binary' (default 'text').

    Raises:
        ValueError: If mode is not 'text' or 'binary'.
        OSError: If the write or rename fails.
    """
    if mode not in ("text", "binary"):
        raise ValueError(f"mode must be 'text' or 'binary', got '{mode}'")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".tmp_",
        suffix=path.suffix,
    )

    try:
        if mode == "text":
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(content)

        # Atomic rename (POSIX) or best-effort (Windows)
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json_safe(path: Path, default: Any = None) -> dict:
    """Safely read a JSON file with fallback default.

    Returns the default value if the file doesn't exist, is empty,
    or contains invalid JSON.

    Args:
        path: Path to the JSON file.
        default: Default value to return on failure (default None).

    Returns:
        Parsed JSON data (typically dict or list), or default on failure.
    """
    if not path.exists():
        logger.debug("JSON file not found: %s", path)
        return default if default is not None else {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read JSON from %s: %s", path, exc)
        return default if default is not None else {}


def write_json_safe(path: Path, data: Any, indent: int = 2) -> None:
    """Safely write JSON using temp file and atomic rename.

    Serialises data to JSON and writes it atomically to prevent
    corruption from partial writes.

    Args:
        path: Target JSON file path.
        data: Data to serialise as JSON.
        indent: JSON indentation level (default 2).

    Raises:
        OSError: If the write or rename fails.
        TypeError: If the data is not JSON-serialisable.
    """
    content = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    atomic_write(path, content, mode="text")


# ══════════════════════════════════════════════════════════
#  Temp Directory Context Manager
# ══════════════════════════════════════════════════════════

@contextmanager
def temp_directory(prefix: str = "") -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up on exit.

    Context manager that creates a temp directory and ensures it is
    removed when the context exits, even if an exception occurs.

    Args:
        prefix: Prefix for the temp directory name.

    Yields:
        Path to the temporary directory.

    Example:
        with temp_directory(prefix="export_") as tmp:
            output = tmp / "video.mp4"
            # ... use output ...
        # tmp directory is now cleaned up
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix or "yt_shorts_"))
    logger.debug("Created temp directory: %s", tmp_dir)
    try:
        yield tmp_dir
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug("Removed temp directory: %s", tmp_dir)
        except Exception as exc:
            logger.warning("Failed to clean temp directory %s: %s", tmp_dir, exc)


# ══════════════════════════════════════════════════════════
#  File Copy with Metadata
# ══════════════════════════════════════════════════════════

def copy_with_metadata(src: Path, dst: Path) -> Path:
    """Copy a file preserving timestamps and metadata.

    Uses shutil.copy2 which preserves file metadata including
    modification time, access time, and permission bits.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        Path to the copied file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        OSError: If the copy fails.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(str(src), str(dst))
        logger.debug("Copied (with metadata): %s -> %s", src.name, dst.name)
    except OSError as exc:
        logger.error("Failed to copy %s -> %s: %s", src, dst, exc)
        raise

    return dst


# ══════════════════════════════════════════════════════════
#  Video File Validation
# ══════════════════════════════════════════════════════════

# Common video file extensions
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv", ".ts", ".mts",
    ".m2ts", ".vob", ".rm", ".rmvb", ".asf", ".divx", ".f4v",
})


def get_video_duration_fallback(path: Path) -> float:
    """Estimate video duration from filename or file header.

    Attempts to parse duration from the filename (e.g. 'clip_120s.mp4')
    and falls back to a basic header scan. This is a fallback when
    ffprobe is unavailable.

    Args:
        path: Path to the video file.

    Returns:
        Estimated duration in seconds, or 0.0 if unable to determine.
    """
    if not path.exists():
        return 0.0

    # Try to parse duration from filename (e.g. "video_60s.mp4" or "clip_120sec.mp4")
    name = path.stem
    duration_match = re.search(r"(\d+)\s*(?:s|sec|seconds?)\b", name, re.IGNORECASE)
    if duration_match:
        try:
            return float(duration_match.group(1))
        except ValueError:
            pass

    # Try mm_ss pattern (e.g. "clip_2_30.mp4")
    time_match = re.search(r"(\d+)_(\d+)\b", name)
    if time_match:
        try:
            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))
            if seconds < 60 and minutes < 300:
                return float(minutes * 60 + seconds)
        except ValueError:
            pass

    return 0.0


def validate_video_file(path: Path) -> tuple[bool, str]:
    """Check if a file is a valid video file.

    Validates based on file extension, file size, and optionally
    by checking if ffprobe can parse it.

    Args:
        path: Path to the file to validate.

    Returns:
        Tuple of (is_valid, message). is_valid is True if the file
        appears to be a valid video. message contains details about
        the validation result.
    """
    if not path.exists():
        return False, f"File does not exist: {path}"

    if not path.is_file():
        return False, f"Path is not a regular file: {path}"

    # Check file size
    try:
        size = path.stat().st_size
        if size == 0:
            return False, f"File is empty (0 bytes): {path}"
        if size < 1024:
            return False, f"File is too small ({size} bytes) to be a valid video: {path}"
    except OSError as exc:
        return False, f"Cannot read file stats: {exc}"

    # Check file extension
    ext = path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return False, f"Unrecognized video extension '{ext}': {path}"

    # Try to probe with ffprobe for deeper validation
    try:
        from utils.ffmpeg_utils import probe_video, FFmpegError
        info = probe_video(path)
        if info.duration <= 0 and info.frame_count == 0:
            return False, f"Video has no duration or frames: {path}"
        return True, f"Valid video: {info.width}x{info.height}, {info.duration:.1f}s, {info.video_codec}"
    except ImportError:
        # ffmpeg_utils not available, rely on extension check
        return True, f"Valid extension ({ext}), size={size} bytes"
    except Exception as exc:
        return False, f"ffprobe validation failed: {exc}"


# ══════════════════════════════════════════════════════════
#  File Organisation
# ══════════════════════════════════════════════════════════

def organize_files_by_date(
    directory: Path,
    pattern: str = "*",
) -> dict[str, list[Path]]:
    """Group files by their modification date.

    Creates a dictionary mapping date strings (YYYY-MM-DD) to lists
    of files modified on that date.

    Args:
        directory: Directory to scan.
        pattern: Glob pattern to match files (default '*' for all).

    Returns:
        Dictionary mapping date strings to lists of file Paths.
    """
    if not directory.exists() or not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return {}

    grouped: dict[str, list[Path]] = {}

    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        try:
            mtime = file_path.stat().st_mtime
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            grouped.setdefault(date_str, []).append(file_path)
        except OSError as exc:
            logger.warning("Could not read mtime for %s: %s", file_path, exc)

    # Sort each group by modification time
    for date_key in grouped:
        grouped[date_key].sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)

    logger.debug("Organized %d files into %d date groups in %s",
                 sum(len(v) for v in grouped.values()), len(grouped), directory)

    return grouped
