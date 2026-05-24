"""
tests/test_downloader.py — Tests for the downloader and file utility functions.
Tests URL validation, filename sanitisation, and duplicate detection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.downloader import _validate_url, InvalidURLError, _check_already_downloaded
from utils.file_utils import sanitize_filename


class TestSanitizeFilename:
    """Tests for the sanitize_filename utility function."""

    def test_sanitize_filename_removes_special_chars(self) -> None:
        """Special characters like \\ / * ? : \" < > | should be removed."""
        result = sanitize_filename('video: "best" moments? part 1/2 *must* see|')
        assert ":" not in result
        assert '"' not in result
        assert "?" not in result
        assert "/" not in result
        assert "*" not in result
        assert "|" not in result

    def test_sanitize_filename_truncates_long_names(self) -> None:
        """Long names should be truncated to max_length."""
        long_name = "a" * 200
        result = sanitize_filename(long_name, max_length=50)
        assert len(result) <= 50

    def test_sanitize_filename_replaces_whitespace_runs(self) -> None:
        """Multiple spaces/underscores should collapse to a single underscore."""
        result = sanitize_filename("hello   world__test")
        assert "   " not in result
        assert "__" not in result
        assert "hello_world_test" == result

    def test_sanitize_filename_strips_leading_trailing(self) -> None:
        """Leading/trailing underscores and dots should be stripped."""
        result = sanitize_filename("_hello_world_.")
        assert result == "hello_world"

    def test_sanitize_filename_handles_empty_string(self) -> None:
        """Empty string should return 'untitled'."""
        result = sanitize_filename("")
        assert result == "untitled"

    def test_sanitize_filename_preserves_extension(self) -> None:
        """Short extension should be preserved when truncating."""
        result = sanitize_filename("a" * 150 + ".mp4", max_length=50)
        assert result.endswith(".mp4")
        assert len(result) <= 50

    def test_sanitize_filename_handles_control_chars(self) -> None:
        """Control characters (0x00-0x1f) should be removed."""
        result = sanitize_filename("hello\x00world\x01test")
        assert "\x00" not in result
        assert "\x01" not in result


class TestValidateUrl:
    """Tests for URL validation in the downloader."""

    def test_validate_url_accepts_youtube_watch_urls(self) -> None:
        """Standard youtube.com/watch?v= URLs should be accepted."""
        # Should not raise
        _validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_validate_url_accepts_youtu_be_urls(self) -> None:
        """Short youtu.be URLs should be accepted."""
        _validate_url("https://youtu.be/dQw4w9WgXcQ")

    def test_validate_url_accepts_youtube_shorts_urls(self) -> None:
        """YouTube Shorts URLs should be accepted."""
        _validate_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")

    def test_validate_url_rejects_non_youtube(self) -> None:
        """Non-YouTube URLs should be rejected."""
        with pytest.raises(InvalidURLError):
            _validate_url("https://www.vimeo.com/12345")

    def test_validate_url_rejects_random_string(self) -> None:
        """Random strings should be rejected."""
        with pytest.raises(InvalidURLError):
            _validate_url("not a url at all")

    def test_validate_url_rejects_google_url(self) -> None:
        """Google.com should be rejected (not YouTube)."""
        with pytest.raises(InvalidURLError):
            _validate_url("https://www.google.com/search?q=test")


class TestDuplicateDetection:
    """Tests for the duplicate download detection function."""

    def test_check_already_downloaded_returns_none_for_empty_dir(self) -> None:
        """Should return None if no files match the youtube_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _check_already_downloaded("abc123", Path(tmpdir))
            assert result is None

    def test_check_already_downloaded_finds_existing(self) -> None:
        """Should find and return path to an existing download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "my_video_abc123.mp4"
            existing.write_text("fake video data")
            result = _check_already_downloaded("abc123", Path(tmpdir))
            assert result is not None
            assert result == existing

    def test_check_already_downloaded_ignores_non_mp4(self) -> None:
        """Should ignore files that are not .mp4."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "my_video_abc123.jpg"
            existing.write_text("fake image data")
            result = _check_already_downloaded("abc123", Path(tmpdir))
            assert result is None
